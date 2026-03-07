#!/usr/bin/env python3
"""Build cleaned interactions.jsonl from raw Books reviews."""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def parse_rating(value: Any) -> float:
    if value is None:
        raise ValueError("missing rating")
    rating = float(value)
    if rating != rating:  # NaN check
        raise ValueError("invalid rating nan")
    return rating


def parse_timestamp(value: Any) -> int:
    if value is None:
        raise ValueError("missing timestamp")
    if isinstance(value, bool):
        raise ValueError("invalid timestamp bool")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError("invalid timestamp float")
        return int(value)

    text = str(value).strip()
    if not text:
        raise ValueError("invalid timestamp empty")
    return int(text)


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_valid_items(items_input: Path) -> tuple[set[str], dict[str, int]]:
    valid_items: set[str] = set()
    stats = {
        "rows_total": 0,
        "parse_error_rows": 0,
        "non_object_rows": 0,
        "rows_missing_item_id": 0,
    }

    with items_input.open("r", encoding="utf-8") as f:
        for line in f:
            stats["rows_total"] += 1
            line = line.strip()
            if not line:
                stats["parse_error_rows"] += 1
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                stats["parse_error_rows"] += 1
                continue
            if not isinstance(raw, dict):
                stats["non_object_rows"] += 1
                continue

            item_id = normalize_text(raw.get("item_id"))
            if not item_id:
                stats["rows_missing_item_id"] += 1
                continue

            valid_items.add(item_id)

    stats["unique_item_ids"] = len(valid_items)
    return valid_items, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build cleaned interactions jsonl.")
    parser.add_argument(
        "--books-input",
        default="data/raw/Books.jsonl",
        help="Input Books raw jsonl path.",
    )
    parser.add_argument(
        "--items-input",
        default="data/processed/items.jsonl",
        help="Input cleaned items jsonl path.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/interactions.jsonl",
        help="Output interactions jsonl path.",
    )
    parser.add_argument(
        "--report",
        default="reports/build_interactions_report.json",
        help="Output build report path.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed reserved for deterministic pipeline configs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    books_input = Path(args.books_input)
    items_input = Path(args.items_input)
    output_path = Path(args.output)
    report_path = Path(args.report)
    tmp_stats_db = output_path.parent / ".tmp_interactions_stats.sqlite3"

    ensure_parent_dir(output_path)
    ensure_parent_dir(report_path)

    valid_items, item_stats = load_valid_items(items_input)

    if tmp_stats_db.exists():
        tmp_stats_db.unlink()

    conn = sqlite3.connect(tmp_stats_db)
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = OFF;")
    conn.execute(
        "CREATE TABLE unique_users (user_id TEXT PRIMARY KEY NOT NULL)"
    )
    conn.execute(
        "CREATE TABLE unique_items (item_id TEXT PRIMARY KEY NOT NULL)"
    )

    books_rows_total = 0
    books_parse_error_rows = 0
    books_non_object_rows = 0

    rows_missing_user_id = 0
    rows_missing_item_id = 0
    rows_missing_rating = 0
    rows_missing_timestamp = 0
    rows_invalid_rating = 0
    rows_invalid_timestamp = 0
    rows_item_not_in_items = 0

    rows_written = 0
    timestamp_min: int | None = None
    timestamp_max: int | None = None

    user_buffer: list[tuple[str]] = []
    item_buffer: list[tuple[str]] = []
    buffer_flush_size = 10000

    def flush_unique_buffers() -> None:
        if user_buffer:
            conn.executemany(
                "INSERT OR IGNORE INTO unique_users (user_id) VALUES (?)",
                user_buffer,
            )
            user_buffer.clear()
        if item_buffer:
            conn.executemany(
                "INSERT OR IGNORE INTO unique_items (item_id) VALUES (?)",
                item_buffer,
            )
            item_buffer.clear()

    try:
        with books_input.open("r", encoding="utf-8") as in_f, output_path.open(
            "w", encoding="utf-8"
        ) as out_f:
            for line in in_f:
                books_rows_total += 1
                line = line.strip()
                if not line:
                    books_parse_error_rows += 1
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    books_parse_error_rows += 1
                    continue
                if not isinstance(raw, dict):
                    books_non_object_rows += 1
                    continue

                user_id = normalize_text(raw.get("user_id"))
                item_id = normalize_text(raw.get("parent_asin"))
                raw_rating = raw.get("rating")
                raw_timestamp = raw.get("timestamp")

                if not user_id:
                    rows_missing_user_id += 1
                    continue
                if not item_id:
                    rows_missing_item_id += 1
                    continue
                if raw_rating is None:
                    rows_missing_rating += 1
                    continue
                if raw_timestamp is None:
                    rows_missing_timestamp += 1
                    continue

                try:
                    rating = parse_rating(raw_rating)
                except (TypeError, ValueError):
                    rows_invalid_rating += 1
                    continue

                try:
                    timestamp = parse_timestamp(raw_timestamp)
                except (TypeError, ValueError):
                    rows_invalid_timestamp += 1
                    continue

                if item_id not in valid_items:
                    rows_item_not_in_items += 1
                    continue

                record = {
                    "user_id": user_id,
                    "item_id": item_id,
                    "rating": rating,
                    "timestamp": timestamp,
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                rows_written += 1

                if timestamp_min is None or timestamp < timestamp_min:
                    timestamp_min = timestamp
                if timestamp_max is None or timestamp > timestamp_max:
                    timestamp_max = timestamp

                user_buffer.append((user_id,))
                item_buffer.append((item_id,))
                if len(user_buffer) >= buffer_flush_size:
                    flush_unique_buffers()

        flush_unique_buffers()
        conn.commit()

        unique_users = conn.execute(
            "SELECT COUNT(*) FROM unique_users"
        ).fetchone()[0]
        unique_items = conn.execute(
            "SELECT COUNT(*) FROM unique_items"
        ).fetchone()[0]
    finally:
        conn.close()
        if tmp_stats_db.exists():
            tmp_stats_db.unlink()

    dropped_rows = books_rows_total - rows_written
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "seed": args.seed,
            "books_input": str(books_input),
            "items_input": str(items_input),
            "output": str(output_path),
        },
        "items_catalog": item_stats,
        "books_input_stats": {
            "rows_total": books_rows_total,
            "parse_error_rows": books_parse_error_rows,
            "non_object_rows": books_non_object_rows,
            "rows_missing_user_id": rows_missing_user_id,
            "rows_missing_item_id": rows_missing_item_id,
            "rows_missing_rating": rows_missing_rating,
            "rows_missing_timestamp": rows_missing_timestamp,
            "rows_invalid_rating": rows_invalid_rating,
            "rows_invalid_timestamp": rows_invalid_timestamp,
            "rows_item_not_in_items": rows_item_not_in_items,
        },
        "output_stats": {
            "rows_written": rows_written,
            "unique_users": unique_users,
            "unique_items": unique_items,
            "timestamp_min": timestamp_min,
            "timestamp_max": timestamp_max,
        },
        "rates": {
            "drop_rate_over_all_rows": (
                dropped_rows / books_rows_total if books_rows_total else 0.0
            ),
            "item_not_in_items_rate_over_all_rows": (
                rows_item_not_in_items / books_rows_total if books_rows_total else 0.0
            ),
            "parse_error_rate_over_all_rows": (
                books_parse_error_rows / books_rows_total if books_rows_total else 0.0
            ),
        },
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
