#!/usr/bin/env python3
"""Build a smaller items subset based on item_ids used in eval.jsonl."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def default_output_for_eval_input(eval_input: str) -> str:
    stem = normalize_text(Path(eval_input).stem)
    if not stem:
        stem = "eval"
    return str(Path("data/processed") / f"items_subset_{stem}.jsonl")


def default_report_for_eval_input(eval_input: str) -> str:
    stem = normalize_text(Path(eval_input).stem)
    if not stem:
        stem = "eval"
    return str(Path("reports") / f"build_items_subset_report_from_{stem}.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build items subset from eval query/target item ids."
    )
    parser.add_argument(
        "--eval-input",
        default="data/processed/eval.jsonl",
        help="Input eval jsonl path.",
    )
    parser.add_argument(
        "--items-input",
        default="data/processed/items.jsonl",
        help="Input full items jsonl path.",
    )
    parser.add_argument(
        "--output",
        default="",
        help=(
            "Output subset items jsonl path. "
            "If omitted, auto uses data/processed/items_subset_<eval_input_stem>.jsonl"
        ),
    )
    parser.add_argument(
        "--report",
        default="",
        help=(
            "Output report json path. "
            "If omitted, auto uses reports/build_items_subset_report_from_<eval_input_stem>.json"
        ),
    )
    args = parser.parse_args()
    if not normalize_text(args.output):
        args.output = default_output_for_eval_input(args.eval_input)
    if not normalize_text(args.report):
        args.report = default_report_for_eval_input(args.eval_input)
    return args


def collect_wanted_item_ids(eval_input: Path) -> tuple[set[str], dict[str, int]]:
    wanted_item_ids: set[str] = set()

    stats = {
        "rows_total": 0,
        "parse_error_rows": 0,
        "non_object_rows": 0,
        "rows_missing_user_id": 0,
        "rows_missing_target_item_id": 0,
        "rows_invalid_query_item_ids": 0,
        "rows_valid": 0,
        "query_item_ids_total": 0,
        "query_item_ids_non_empty": 0,
    }

    with eval_input.open("r", encoding="utf-8") as f:
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

            user_id = normalize_text(raw.get("user_id"))
            target_item_id = normalize_text(raw.get("target_item_id"))
            query_item_ids = raw.get("query_item_ids")

            if not user_id:
                stats["rows_missing_user_id"] += 1
                continue
            if not target_item_id:
                stats["rows_missing_target_item_id"] += 1
                continue
            if not isinstance(query_item_ids, list):
                stats["rows_invalid_query_item_ids"] += 1
                continue

            stats["rows_valid"] += 1
            wanted_item_ids.add(target_item_id)

            for item in query_item_ids:
                stats["query_item_ids_total"] += 1
                item_id = normalize_text(item)
                if not item_id:
                    continue
                stats["query_item_ids_non_empty"] += 1
                wanted_item_ids.add(item_id)

    stats["wanted_unique_item_ids"] = len(wanted_item_ids)
    return wanted_item_ids, stats


def main() -> None:
    args = parse_args()
    eval_input = Path(args.eval_input)
    items_input = Path(args.items_input)
    output_path = Path(args.output)
    report_path = Path(args.report)

    ensure_parent_dir(output_path)
    ensure_parent_dir(report_path)

    wanted_item_ids, eval_stats = collect_wanted_item_ids(eval_input)

    items_scan_stats = {
        "rows_total": 0,
        "parse_error_rows": 0,
        "non_object_rows": 0,
        "rows_missing_item_id": 0,
        "rows_valid_with_item_id": 0,
        "duplicate_item_id_rows": 0,
    }

    written_item_ids: set[str] = set()

    with items_input.open("r", encoding="utf-8") as in_f, output_path.open(
        "w", encoding="utf-8"
    ) as out_f:
        for line in in_f:
            items_scan_stats["rows_total"] += 1
            line = line.strip()
            if not line:
                items_scan_stats["parse_error_rows"] += 1
                continue

            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                items_scan_stats["parse_error_rows"] += 1
                continue
            if not isinstance(raw, dict):
                items_scan_stats["non_object_rows"] += 1
                continue

            item_id = normalize_text(raw.get("item_id"))
            if not item_id:
                items_scan_stats["rows_missing_item_id"] += 1
                continue

            items_scan_stats["rows_valid_with_item_id"] += 1
            if item_id not in wanted_item_ids:
                continue
            if item_id in written_item_ids:
                items_scan_stats["duplicate_item_id_rows"] += 1
                continue

            out_f.write(json.dumps(raw, ensure_ascii=False) + "\n")
            written_item_ids.add(item_id)

    missing_item_ids = wanted_item_ids - written_item_ids

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "eval_input": str(eval_input),
            "items_input": str(items_input),
            "output": str(output_path),
            "report": str(report_path),
        },
        "eval_stats": eval_stats,
        "items_scan_stats": items_scan_stats,
        "output_stats": {
            "rows_written": len(written_item_ids),
            "missing_item_ids_from_items": len(missing_item_ids),
            "coverage_over_wanted_item_ids": (
                len(written_item_ids) / len(wanted_item_ids) if wanted_item_ids else 0.0
            ),
            "reduction_ratio_vs_items_input_valid_rows": (
                1.0
                - (len(written_item_ids) / items_scan_stats["rows_valid_with_item_id"])
                if items_scan_stats["rows_valid_with_item_id"]
                else 0.0
            ),
        },
    }

    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
