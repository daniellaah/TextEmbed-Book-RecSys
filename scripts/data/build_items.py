#!/usr/bin/env python3
"""Build structured items.jsonl from Amazon Books metadata."""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(value: Any) -> str:
    """Normalize text with strip + whitespace collapse."""
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    return WHITESPACE_RE.sub(" ", text)


def normalize_list(values: Any) -> list[str]:
    """Normalize list elements and drop empty/null elements."""
    if values is None:
        return []
    if not isinstance(values, list):
        values = [values]

    cleaned: list[str] = []
    for element in values:
        text = normalize_text(element)
        if text:
            cleaned.append(text)
    return cleaned


def normalize_author(author_raw: Any) -> str:
    if isinstance(author_raw, dict):
        return normalize_text(author_raw.get("name", ""))
    if isinstance(author_raw, list):
        return ", ".join(normalize_list(author_raw))
    return normalize_text(author_raw)


def normalize_description(description_raw: Any) -> str:
    if isinstance(description_raw, list):
        return " ".join(normalize_list(description_raw))
    return normalize_text(description_raw)


def build_structured_item(raw: dict[str, Any]) -> dict[str, str]:
    features_cleaned = normalize_list(raw.get("features"))
    categories_cleaned = normalize_list(raw.get("categories"))

    return {
        "item_id": normalize_text(raw.get("parent_asin")),
        "title": normalize_text(raw.get("title")),
        "subtitle": normalize_text(raw.get("subtitle")),
        "author": normalize_author(raw.get("author")),
        "description": normalize_description(raw.get("description")),
        "features": "; ".join(features_cleaned) if features_cleaned else "",
        "categories": " > ".join(categories_cleaned) if categories_cleaned else "",
    }


def completeness_score(item: dict[str, str]) -> tuple[int, int]:
    fields = ["title", "subtitle", "author", "description", "features", "categories"]
    non_empty = sum(1 for key in fields if item.get(key, ""))
    char_count = sum(len(item.get(key, "")) for key in fields)
    return non_empty, char_count


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build structured items.jsonl.")
    parser.add_argument(
        "--input",
        default="data/raw/meta_Books.jsonl",
        help="Input metadata jsonl path.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/items.jsonl",
        help="Output structured items jsonl path.",
    )
    parser.add_argument(
        "--report",
        default="reports/build_items_report.json",
        help="Output report json path.",
    )
    parser.add_argument(
        "--tmp-db",
        default="data/processed/.tmp_build_items.sqlite3",
        help="Temporary sqlite path for dedup.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    report_path = Path(args.report)
    tmp_db_path = Path(args.tmp_db)

    ensure_parent_dir(output_path)
    ensure_parent_dir(report_path)
    ensure_parent_dir(tmp_db_path)

    if tmp_db_path.exists():
        tmp_db_path.unlink()

    conn = sqlite3.connect(tmp_db_path)
    try:
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = OFF;")
        conn.execute("PRAGMA temp_store = MEMORY;")
        conn.execute(
            """
            CREATE TABLE items (
                item_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                subtitle TEXT NOT NULL,
                author TEXT NOT NULL,
                description TEXT NOT NULL,
                features TEXT NOT NULL,
                categories TEXT NOT NULL,
                score_non_empty INTEGER NOT NULL,
                score_chars INTEGER NOT NULL
            );
            """
        )

        upsert_sql = """
        INSERT INTO items (
            item_id, title, subtitle, author, description, features, categories,
            score_non_empty, score_chars
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(item_id) DO UPDATE SET
            title = excluded.title,
            subtitle = excluded.subtitle,
            author = excluded.author,
            description = excluded.description,
            features = excluded.features,
            categories = excluded.categories,
            score_non_empty = excluded.score_non_empty,
            score_chars = excluded.score_chars
        WHERE
            (
                CASE WHEN excluded.title != '' THEN 1 ELSE 0 END
            ) > (
                CASE WHEN items.title != '' THEN 1 ELSE 0 END
            )
            OR (
                (
                    CASE WHEN excluded.title != '' THEN 1 ELSE 0 END
                ) = (
                    CASE WHEN items.title != '' THEN 1 ELSE 0 END
                )
                AND (
                    excluded.score_non_empty > items.score_non_empty
                    OR (
                        excluded.score_non_empty = items.score_non_empty
                        AND excluded.score_chars > items.score_chars
                    )
                )
            );
        """

        input_rows_total = 0
        parse_error_rows = 0
        rows_missing_item_id = 0
        rows_missing_title = 0
        rows_with_item_id = 0

        with input_path.open("r", encoding="utf-8") as f:
            for line in f:
                input_rows_total += 1
                line = line.strip()
                if not line:
                    parse_error_rows += 1
                    continue

                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    parse_error_rows += 1
                    continue
                if not isinstance(raw, dict):
                    parse_error_rows += 1
                    continue

                item = build_structured_item(raw)
                if not item["item_id"]:
                    rows_missing_item_id += 1
                    continue

                if not item["title"]:
                    rows_missing_title += 1

                rows_with_item_id += 1
                score_non_empty, score_chars = completeness_score(item)
                conn.execute(
                    upsert_sql,
                    (
                        item["item_id"],
                        item["title"],
                        item["subtitle"],
                        item["author"],
                        item["description"],
                        item["features"],
                        item["categories"],
                        score_non_empty,
                        score_chars,
                    ),
                )

        conn.commit()

        unique_item_ids_before_title_filter = conn.execute(
            "SELECT COUNT(*) FROM items"
        ).fetchone()[0]
        duplicates_removed = rows_with_item_id - unique_item_ids_before_title_filter
        dedup_rate = (
            duplicates_removed / rows_with_item_id if rows_with_item_id else 0.0
        )

        rows_written = 0
        dropped_missing_title_after_dedup = 0
        with output_path.open("w", encoding="utf-8") as out_f:
            cursor = conn.execute(
                """
                SELECT item_id, title, subtitle, author, description, features, categories
                FROM items
                ORDER BY item_id
                """
            )
            for row in cursor:
                item = {
                    "item_id": row[0],
                    "title": row[1],
                    "subtitle": row[2],
                    "author": row[3],
                    "description": row[4],
                    "features": row[5],
                    "categories": row[6],
                }
                if not item["title"]:
                    dropped_missing_title_after_dedup += 1
                    continue

                out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
                rows_written += 1

        item_id_unique_rate = 1.0 if rows_written > 0 else 0.0
        item_id_non_empty_rate = 1.0 if rows_written > 0 else 0.0
        title_non_empty_rate = 1.0 if rows_written > 0 else 0.0

        report = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "input": {
                "path": str(input_path),
                "rows_total": input_rows_total,
                "parse_error_rows": parse_error_rows,
                "rows_with_item_id": rows_with_item_id,
                "rows_missing_item_id": rows_missing_item_id,
                "rows_missing_title": rows_missing_title,
                "missing_item_id_rate": (
                    rows_missing_item_id / input_rows_total if input_rows_total else 0.0
                ),
                "missing_title_rate": (
                    rows_missing_title / input_rows_total if input_rows_total else 0.0
                ),
                "missing_title_rate_over_item_id_rows": (
                    rows_missing_title / rows_with_item_id if rows_with_item_id else 0.0
                ),
                "missing_title_rate_over_all_rows": (
                    rows_missing_title / input_rows_total if input_rows_total else 0.0
                ),
            },
            "dedup": {
                "unique_item_ids_before_title_filter": unique_item_ids_before_title_filter,
                "duplicates_removed": duplicates_removed,
                "dedup_rate": dedup_rate,
            },
            "output": {
                "path": str(output_path),
                "rows_written": rows_written,
                "dropped_missing_title_after_dedup": dropped_missing_title_after_dedup,
                "item_id_unique_rate": item_id_unique_rate,
                "key_field_non_empty_rate": {
                    "item_id": item_id_non_empty_rate,
                    "title": title_non_empty_rate,
                },
            },
            "rules": {
                "list_rendering": {
                    "features": "'; '",
                    "categories": "' > '",
                    "description_list_join": "' '",
                },
                "dedup_keep_rule": "for same item_id: prefer non-empty title first, then higher (non_empty_field_count, total_char_count)",
                "drop_rule": "drop rows with empty item_id or empty title",
            },
        }
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    finally:
        conn.close()
        if tmp_db_path.exists():
            tmp_db_path.unlink()


if __name__ == "__main__":
    main()
