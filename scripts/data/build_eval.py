#!/usr/bin/env python3
"""Build eval queries from interactions.jsonl."""

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
    if rating != rating:
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


def default_report_output_for_queries(queries_output: str) -> str:
    queries_path = Path(queries_output)
    stem = normalize_text(queries_path.stem)
    if not stem:
        stem = "eval"
    return str(Path("reports") / f"build_eval_report_{stem}.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build eval query set from interactions.")
    parser.add_argument(
        "--interactions-input",
        default="data/processed/interactions.jsonl",
        help="Input interactions jsonl path.",
    )
    parser.add_argument(
        "--queries-output",
        default="data/processed/eval.jsonl",
        help="Output eval queries jsonl path.",
    )
    parser.add_argument(
        "--report-output",
        default="",
        help=(
            "Output build report json path. "
            "If omitted, auto uses reports/build_eval_report_<queries_output_stem>.json"
        ),
    )
    parser.add_argument(
        "--rating-threshold",
        type=float,
        default=4.0,
        help="Positive sample threshold: rating >= rating_threshold.",
    )
    parser.add_argument(
        "--min-user-pos",
        type=int,
        default=1,
        help="Minimum positive interactions per user after filtering.",
    )
    parser.add_argument(
        "--min-item-pos",
        type=int,
        default=1,
        help="Minimum positive interactions per item after filtering.",
    )
    parser.add_argument(
        "--query-history-n",
        type=int,
        default=1,
        help="How many most-recent positives before target to use as query.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Reserved for deterministic protocol metadata.",
    )
    args = parser.parse_args()
    if args.min_user_pos < 1:
        parser.error("--min-user-pos must be >= 1")
    if args.min_item_pos < 1:
        parser.error("--min-item-pos must be >= 1")
    if args.query_history_n < 1:
        parser.error("--query-history-n must be >= 1")
    if not normalize_text(args.report_output):
        args.report_output = default_report_output_for_queries(args.queries_output)
    return args


def build_kcore(conn: sqlite3.Connection, min_user_pos: int, min_item_pos: int) -> dict[str, int]:
    iterations = 0
    total_deleted_by_user_filter = 0
    total_deleted_by_item_filter = 0

    if min_user_pos <= 1 and min_item_pos <= 1:
        return {
            "enabled": 0,
            "iterations": 0,
            "total_deleted_by_user_filter": 0,
            "total_deleted_by_item_filter": 0,
        }

    while True:
        iterations += 1
        deleted_in_iter = 0

        if min_user_pos > 1:
            conn.execute("DROP TABLE IF EXISTS bad_users")
            conn.execute(
                """
                CREATE TEMP TABLE bad_users AS
                SELECT user_id
                FROM positives
                GROUP BY user_id
                HAVING COUNT(*) < ?
                """,
                (min_user_pos,),
            )
            bad_user_count = conn.execute("SELECT COUNT(*) FROM bad_users").fetchone()[0]
            if bad_user_count > 0:
                deleted_user_rows = conn.execute(
                    "DELETE FROM positives WHERE user_id IN (SELECT user_id FROM bad_users)"
                ).rowcount
                total_deleted_by_user_filter += deleted_user_rows
                deleted_in_iter += deleted_user_rows

        if min_item_pos > 1:
            conn.execute("DROP TABLE IF EXISTS bad_items")
            conn.execute(
                """
                CREATE TEMP TABLE bad_items AS
                SELECT item_id
                FROM positives
                GROUP BY item_id
                HAVING COUNT(*) < ?
                """,
                (min_item_pos,),
            )
            bad_item_count = conn.execute("SELECT COUNT(*) FROM bad_items").fetchone()[0]
            if bad_item_count > 0:
                deleted_item_rows = conn.execute(
                    "DELETE FROM positives WHERE item_id IN (SELECT item_id FROM bad_items)"
                ).rowcount
                total_deleted_by_item_filter += deleted_item_rows
                deleted_in_iter += deleted_item_rows

        conn.commit()
        if deleted_in_iter == 0:
            break

    return {
        "enabled": 1,
        "iterations": iterations,
        "total_deleted_by_user_filter": total_deleted_by_user_filter,
        "total_deleted_by_item_filter": total_deleted_by_item_filter,
    }


def main() -> None:
    args = parse_args()
    interactions_input = Path(args.interactions_input)
    queries_output = Path(args.queries_output)
    report_output = Path(args.report_output)
    tmp_db = queries_output.parent / ".tmp_build_eval.sqlite3"

    ensure_parent_dir(queries_output)
    ensure_parent_dir(report_output)
    ensure_parent_dir(tmp_db)

    if tmp_db.exists():
        tmp_db.unlink()

    conn = sqlite3.connect(tmp_db)
    try:
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = OFF;")
        conn.execute("PRAGMA temp_store = MEMORY;")
        conn.execute(
            """
            CREATE TABLE positives (
                input_order INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                item_id TEXT NOT NULL,
                timestamp INTEGER NOT NULL
            );
            """
        )
        conn.execute("CREATE INDEX idx_positives_user ON positives(user_id)")
        conn.execute("CREATE INDEX idx_positives_item ON positives(item_id)")
        conn.execute("CREATE INDEX idx_positives_user_ts ON positives(user_id, timestamp, input_order)")

        input_rows_total = 0
        parse_error_rows = 0
        non_object_rows = 0
        rows_missing_user_id = 0
        rows_missing_item_id = 0
        rows_missing_rating = 0
        rows_missing_timestamp = 0
        rows_invalid_rating = 0
        rows_invalid_timestamp = 0
        rows_negative_or_below_threshold = 0
        rows_positive_loaded = 0

        with interactions_input.open("r", encoding="utf-8") as in_f:
            for line in in_f:
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
                    non_object_rows += 1
                    continue

                user_id = normalize_text(raw.get("user_id"))
                item_id = normalize_text(raw.get("item_id"))
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

                if rating < args.rating_threshold:
                    rows_negative_or_below_threshold += 1
                    continue

                conn.execute(
                    "INSERT INTO positives (user_id, item_id, timestamp) VALUES (?, ?, ?)",
                    (user_id, item_id, timestamp),
                )
                rows_positive_loaded += 1

        conn.commit()

        positive_users_pre = conn.execute(
            "SELECT COUNT(DISTINCT user_id) FROM positives"
        ).fetchone()[0]
        positive_items_pre = conn.execute(
            "SELECT COUNT(DISTINCT item_id) FROM positives"
        ).fetchone()[0]
        positive_samples_pre = conn.execute("SELECT COUNT(*) FROM positives").fetchone()[0]

        kcore_stats = build_kcore(conn, args.min_user_pos, args.min_item_pos)

        positive_users_post = conn.execute(
            "SELECT COUNT(DISTINCT user_id) FROM positives"
        ).fetchone()[0]
        positive_items_post = conn.execute(
            "SELECT COUNT(DISTINCT item_id) FROM positives"
        ).fetchone()[0]
        positive_samples_post = conn.execute("SELECT COUNT(*) FROM positives").fetchone()[0]

        conn.execute("DROP TABLE IF EXISTS ranked")
        conn.execute(
            """
            CREATE TEMP TABLE ranked AS
            SELECT
                user_id,
                item_id,
                timestamp,
                input_order,
                ROW_NUMBER() OVER (
                    PARTITION BY user_id
                    ORDER BY timestamp DESC, input_order DESC
                ) AS rn_desc
            FROM positives;
            """
        )
        conn.execute("CREATE INDEX idx_ranked_user_rn ON ranked(user_id, rn_desc)")
        conn.execute("DROP TABLE IF EXISTS target")
        conn.execute(
            """
            CREATE TEMP TABLE target AS
            SELECT user_id, item_id AS target_item_id, timestamp AS target_timestamp
            FROM ranked
            WHERE rn_desc = 1;
            """
        )
        conn.execute("CREATE INDEX idx_target_user ON target(user_id)")
        conn.execute("DROP TABLE IF EXISTS query")
        conn.execute(
            """
            CREATE TEMP TABLE query AS
            SELECT user_id, item_id, timestamp, input_order
            FROM ranked
            WHERE rn_desc BETWEEN 2 AND ?;
            """,
            (args.query_history_n + 1,),
        )
        conn.execute("CREATE INDEX idx_query_user_ts ON query(user_id, timestamp, input_order)")

        users_with_target = conn.execute("SELECT COUNT(*) FROM target").fetchone()[0]
        users_with_any_query = conn.execute(
            "SELECT COUNT(DISTINCT user_id) FROM query"
        ).fetchone()[0]

        query_rows_written = 0
        users_dropped_due_to_no_history = 0
        total_query_items = 0

        with queries_output.open("w", encoding="utf-8") as out_f:
            cursor = conn.execute(
                """
                SELECT
                    t.user_id,
                    t.target_item_id,
                    q.item_id
                FROM target t
                LEFT JOIN query q ON q.user_id = t.user_id
                ORDER BY t.user_id ASC, q.timestamp ASC, q.input_order ASC;
                """
            )

            current_user: str | None = None
            current_target: str | None = None
            current_queries: list[str] = []

            def flush_current() -> None:
                nonlocal query_rows_written, users_dropped_due_to_no_history, total_query_items
                if current_user is None or current_target is None:
                    return
                if not current_queries:
                    users_dropped_due_to_no_history += 1
                    return
                out_f.write(
                    json.dumps(
                        {
                            "user_id": current_user,
                            "query_item_ids": current_queries,
                            "target_item_id": current_target,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                query_rows_written += 1
                total_query_items += len(current_queries)

            for user_id, target_item_id, query_item_id in cursor:
                if current_user != user_id:
                    flush_current()
                    current_user = user_id
                    current_target = target_item_id
                    current_queries = []
                if query_item_id:
                    current_queries.append(query_item_id)
            flush_current()

        report = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "config": {
                "seed": args.seed,
                "interactions_input": str(interactions_input),
                "queries_output": str(queries_output),
                "report_output": str(report_output),
                "rating_threshold": args.rating_threshold,
                "min_user_pos": args.min_user_pos,
                "min_item_pos": args.min_item_pos,
                "query_history_n": args.query_history_n,
            },
            "input_stats": {
                "rows_total": input_rows_total,
                "parse_error_rows": parse_error_rows,
                "non_object_rows": non_object_rows,
                "rows_missing_user_id": rows_missing_user_id,
                "rows_missing_item_id": rows_missing_item_id,
                "rows_missing_rating": rows_missing_rating,
                "rows_missing_timestamp": rows_missing_timestamp,
                "rows_invalid_rating": rows_invalid_rating,
                "rows_invalid_timestamp": rows_invalid_timestamp,
                "rows_below_rating_threshold": rows_negative_or_below_threshold,
            },
            "positive_stats": {
                "rows_positive_loaded": rows_positive_loaded,
                "users_pre_filter": positive_users_pre,
                "items_pre_filter": positive_items_pre,
                "samples_pre_filter": positive_samples_pre,
                "users_post_filter": positive_users_post,
                "items_post_filter": positive_items_post,
                "samples_post_filter": positive_samples_post,
            },
            "kcore": kcore_stats,
            "eval_stats": {
                "users_with_target": users_with_target,
                "users_with_any_query_before_drop": users_with_any_query,
                "users_dropped_due_to_no_history": users_dropped_due_to_no_history,
                "eval_query_rows_written": query_rows_written,
                "avg_query_length": (
                    total_query_items / query_rows_written if query_rows_written else 0.0
                ),
                "query_user_coverage_over_post_filter_users": (
                    query_rows_written / positive_users_post if positive_users_post else 0.0
                ),
            },
        }
        report_output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    finally:
        conn.close()
        if tmp_db.exists():
            tmp_db.unlink()


if __name__ == "__main__":
    main()
