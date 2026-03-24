#!/usr/bin/env python3
"""Run non-embedding retrieval baselines on eval.jsonl."""

from __future__ import annotations

import argparse
import heapq
import json
import math
import multiprocessing as mp
import random
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.baselines.baseline_utils import (
    ensure_parent_dir,
    file_sha256,
    find_rank,
    generate_run_id_local,
    load_items_metadata,
    load_positive_popularity,
    metric_value,
    normalize_text,
    parse_topk_list,
)

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
TFIDF_TERM_POOL_SIZE = 256
_TFIDF_POSTINGS_BY_TERM: dict[str, list[tuple[int, float]]] = {}
_TFIDF_ITEM_IDS: list[str] = []
_TFIDF_POOL_SIZE = 0

def init_tfidf_pool_builder(
    postings_by_term: dict[str, list[tuple[int, float]]],
    item_ids: list[str],
    pool_size: int,
) -> None:
    global _TFIDF_POSTINGS_BY_TERM, _TFIDF_ITEM_IDS, _TFIDF_POOL_SIZE
    _TFIDF_POSTINGS_BY_TERM = postings_by_term
    _TFIDF_ITEM_IDS = item_ids
    _TFIDF_POOL_SIZE = pool_size


def build_tfidf_query_pools_chunk(
    chunk: list[tuple[str, dict[str, float]]],
) -> dict[str, list[str]]:
    query_pools: dict[str, list[str]] = {}
    for query_item_id, query_weights in chunk:
        if not query_weights:
            continue
        scores: dict[int, float] = {}
        for term, query_weight in query_weights.items():
            for row_id, doc_weight in _TFIDF_POSTINGS_BY_TERM.get(term, []):
                scores[row_id] = scores.get(row_id, 0.0) + query_weight * doc_weight
        ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:_TFIDF_POOL_SIZE]
        query_pools[query_item_id] = [_TFIDF_ITEM_IDS[row_id] for row_id, _ in ranked]
    return query_pools


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run retrieval baselines from eval.jsonl.")
    parser.add_argument(
        "--baseline",
        required=True,
        choices=["random", "global_popular", "category_random", "category_popular", "tfidf"],
        help="Baseline retrieval method.",
    )
    parser.add_argument(
        "--items-input",
        default="data/processed/items.jsonl",
        help="Input items jsonl path.",
    )
    parser.add_argument(
        "--interactions-input",
        default="data/processed/interactions.jsonl",
        help="Input interactions jsonl path.",
    )
    parser.add_argument(
        "--eval-input",
        default="data/processed/eval.jsonl",
        help="Input eval queries jsonl path.",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/baselines",
        help="Root output directory for baseline artifacts.",
    )
    parser.add_argument(
        "--topk",
        default="10,50",
        help="Comma-separated metric K list, e.g. '10,50'.",
    )
    parser.add_argument(
        "--max-query",
        type=int,
        default=0,
        help="Maximum number of valid eval queries to evaluate. 0 means no limit.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic baselines.",
    )
    parser.add_argument(
        "--rating-threshold",
        type=float,
        default=4.0,
        help="Positive interaction threshold used for popularity counting.",
    )
    parser.add_argument(
        "--run-id",
        default="",
        help="Optional run id. Default is local timestamp YYYYMMDDHHMMSSmmm.",
    )
    parser.add_argument(
        "--text-fields",
        default="title,author,categories",
        help="Comma-separated item fields used by tfidf baseline.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Worker processes used when precomputing tfidf query pools. 1 disables multiprocessing.",
    )
    args = parser.parse_args()
    if args.max_query < 0:
        parser.error("--max-query must be >= 0")
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    args.topk_list = parse_topk_list(args.topk)
    args.text_field_list = [x.strip() for x in args.text_fields.split(",") if x.strip()]
    if args.baseline == "tfidf" and not args.text_field_list:
        parser.error("--text-fields cannot be empty for tfidf baseline")
    return args


def sort_popular_item_ids(item_ids: list[str], popularity: dict[str, int]) -> list[str]:
    return sorted(item_ids, key=lambda item_id: (-popularity.get(item_id, 0), item_id))


def build_random_order(item_ids: list[str], seed: int) -> list[str]:
    item_ids_copy = list(item_ids)
    rng = random.Random(seed)
    rng.shuffle(item_ids_copy)
    return item_ids_copy


def take_topk_excluding(
    *,
    ordered_item_ids: list[str],
    excluded_item_ids: set[str],
    top_k: int,
) -> list[str]:
    selected: list[str] = []
    if top_k <= 0:
        return selected
    for item_id in ordered_item_ids:
        if item_id in excluded_item_ids:
            continue
        selected.append(item_id)
        if len(selected) >= top_k:
            break
    return selected


def take_combined_topk_excluding(
    *,
    primary_item_ids: list[str],
    fallback_item_ids: list[str],
    excluded_item_ids: set[str],
    top_k: int,
) -> list[str]:
    selected: list[str] = []
    selected_set: set[str] = set()
    if top_k <= 0:
        return []
    for item_ids in (primary_item_ids, fallback_item_ids):
        for item_id in item_ids:
            if item_id in excluded_item_ids or item_id in selected_set:
                continue
            selected.append(item_id)
            selected_set.add(item_id)
            if len(selected) >= top_k:
                return selected
    return selected


def build_predictions(
    *,
    ordered_item_ids: list[str],
    top_k: int,
    popularity: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    selected = ordered_item_ids[:top_k]
    return [
        {
            "rank": idx + 1,
            "item_id": item_id,
            "score": float(popularity.get(item_id, 0)) if popularity is not None else 0.0,
        }
        for idx, item_id in enumerate(selected)
    ]


def build_short_pools(
    *,
    item_ids: list[str],
    item_ids_by_category: dict[str, list[str]],
    popularity: dict[str, int],
    seed: int,
    pool_size: int,
) -> tuple[list[str], list[str], dict[str, list[str]], dict[str, list[str]]]:
    global_popular_pool = sort_popular_item_ids(item_ids, popularity)[:pool_size]
    global_random_pool = build_random_order(item_ids, seed)[:pool_size]
    category_popular_pools = {
        category: sort_popular_item_ids(category_item_ids, popularity)[:pool_size]
        for category, category_item_ids in item_ids_by_category.items()
    }
    category_random_pools = {
        category: build_random_order(category_item_ids, seed)[:pool_size]
        for category, category_item_ids in item_ids_by_category.items()
    }
    return global_popular_pool, global_random_pool, category_popular_pools, category_random_pools


def tokenize_text(text: str) -> list[str]:
    tokens = [token.lower() for token in TOKEN_RE.findall(text)]
    return [token for token in tokens if len(token) > 1]


def render_item_text(raw: dict[str, Any], text_fields: list[str]) -> str:
    return " ".join(normalize_text(raw.get(field)) for field in text_fields if normalize_text(raw.get(field)))


def iter_valid_eval_rows(
    *,
    eval_input: Path,
    valid_item_ids: set[str],
):
    with eval_input.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(raw, dict):
                continue

            user_id = normalize_text(raw.get("user_id"))
            target_item_id = normalize_text(raw.get("target_item_id"))
            query_item_ids_raw = raw.get("query_item_ids")
            if not user_id or not target_item_id or not isinstance(query_item_ids_raw, list):
                continue

            cleaned_query_item_ids: list[str] = []
            seen_query_item_ids: set[str] = set()
            for item in query_item_ids_raw:
                item_id = normalize_text(item)
                if not item_id or item_id in seen_query_item_ids:
                    continue
                seen_query_item_ids.add(item_id)
                cleaned_query_item_ids.append(item_id)

            if not cleaned_query_item_ids:
                continue
            if target_item_id in seen_query_item_ids or target_item_id not in valid_item_ids:
                continue
            if any(query_item_id not in valid_item_ids for query_item_id in cleaned_query_item_ids):
                continue

            yield {
                "user_id": user_id,
                "target_item_id": target_item_id,
                "query_item_ids": cleaned_query_item_ids,
                "last_query_item_id": cleaned_query_item_ids[-1],
            }


def build_tfidf_resources(
    *,
    items_input: Path,
    item_id_to_row: dict[str, int],
    eval_input: Path,
    valid_item_ids: set[str],
    text_fields: list[str],
    max_query: int,
) -> tuple[dict[str, dict[str, float]], dict[str, list[tuple[int, float]]], dict[str, int]]:
    query_item_ids: set[str] = set()
    valid_eval_rows = 0
    for row in iter_valid_eval_rows(eval_input=eval_input, valid_item_ids=valid_item_ids):
        query_item_ids.add(row["last_query_item_id"])
        valid_eval_rows += 1
        if max_query > 0 and valid_eval_rows >= max_query:
            break
    query_texts: dict[str, str] = {}
    with items_input.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(raw, dict):
                continue
            item_id = normalize_text(raw.get("item_id"))
            if item_id in query_item_ids:
                query_texts[item_id] = render_item_text(raw, text_fields)

    query_term_counts: dict[str, Counter[str]] = {}
    query_vocabulary: set[str] = set()
    missing_query_text_item_ids = 0
    for item_id in query_item_ids:
        counts = Counter(tokenize_text(query_texts.get(item_id, "")))
        if not counts:
            missing_query_text_item_ids += 1
        query_term_counts[item_id] = counts
        query_vocabulary.update(counts)

    postings_heaps: dict[str, list[tuple[float, int]]] = {}
    df_by_term: dict[str, int] = {}
    indexed_item_rows = 0
    with items_input.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(raw, dict):
                continue
            item_id = normalize_text(raw.get("item_id"))
            row_id = item_id_to_row.get(item_id)
            if row_id is None:
                continue
            term_counts = Counter(tokenize_text(render_item_text(raw, text_fields)))
            filtered_term_counts = {
                term: count for term, count in term_counts.items() if term in query_vocabulary
            }
            if not filtered_term_counts:
                continue
            indexed_item_rows += 1
            for term, count in filtered_term_counts.items():
                df_by_term[term] = df_by_term.get(term, 0) + 1
                tf_weight = 1.0 + math.log(float(count))
                heap = postings_heaps.setdefault(term, [])
                if len(heap) < TFIDF_TERM_POOL_SIZE:
                    heapq.heappush(heap, (tf_weight, row_id))
                elif tf_weight > heap[0][0]:
                    heapq.heapreplace(heap, (tf_weight, row_id))

    num_docs = len(item_id_to_row)
    idf_by_term = {
        term: math.log((1.0 + float(num_docs)) / (1.0 + float(df))) + 1.0
        for term, df in df_by_term.items()
    }
    postings_by_term = {
        term: [
            (row_id, tf_weight * idf_by_term[term])
            for tf_weight, row_id in sorted(postings, key=lambda x: (-x[0], x[1]))
        ]
        for term, postings in postings_heaps.items()
    }
    query_weights_by_item_id = {
        item_id: {
            term: (1.0 + math.log(float(count))) * idf_by_term[term]
            for term, count in term_counts.items()
            if term in idf_by_term
        }
        for item_id, term_counts in query_term_counts.items()
    }
    tfidf_stats = {
        "text_fields": text_fields,
        "query_item_ids_with_text": len(query_item_ids) - missing_query_text_item_ids,
        "query_item_ids_missing_text": missing_query_text_item_ids,
        "query_vocabulary_size": len(query_vocabulary),
        "indexed_terms": len(postings_by_term),
        "indexed_item_rows": indexed_item_rows,
        "term_postings_limit": TFIDF_TERM_POOL_SIZE,
    }
    return query_weights_by_item_id, postings_by_term, tfidf_stats


def build_tfidf_query_pools(
    *,
    query_weights_by_item_id: dict[str, dict[str, float]],
    postings_by_term: dict[str, list[tuple[int, float]]],
    item_ids: list[str],
    pool_size: int,
    workers: int,
) -> dict[str, list[str]]:
    items = list(query_weights_by_item_id.items())
    if workers == 1 or len(items) <= 512:
        init_tfidf_pool_builder(postings_by_term, item_ids, pool_size)
        return build_tfidf_query_pools_chunk(items)

    chunk_size = max(256, math.ceil(len(items) / float(workers * 8)))
    chunks = [items[idx : idx + chunk_size] for idx in range(0, len(items), chunk_size)]
    query_pools: dict[str, list[str]] = {}
    ctx = mp.get_context("fork")
    with ctx.Pool(
        processes=workers,
        initializer=init_tfidf_pool_builder,
        initargs=(postings_by_term, item_ids, pool_size),
    ) as pool:
        for chunk_result in pool.imap_unordered(build_tfidf_query_pools_chunk, chunks):
            query_pools.update(chunk_result)
    return query_pools


def main() -> None:
    args = parse_args()

    items_input = Path(args.items_input)
    interactions_input = Path(args.interactions_input)
    eval_input = Path(args.eval_input)

    item_ids, category_by_item_id, item_ids_by_category, item_stats = load_items_metadata(items_input)
    valid_item_ids = set(item_ids)
    item_id_to_row = {item_id: idx for idx, item_id in enumerate(item_ids)}
    popularity, popularity_stats = load_positive_popularity(
        interactions_input,
        valid_item_ids=valid_item_ids,
        rating_threshold=float(args.rating_threshold),
    )

    run_id = normalize_text(args.run_id) or generate_run_id_local()
    run_output_dir = Path(args.output_root) / args.baseline / run_id
    predictions_output = run_output_dir / "predictions.jsonl"
    report_output = run_output_dir / "report.json"
    info_output = run_output_dir / "info.json"
    ensure_parent_dir(predictions_output)
    ensure_parent_dir(report_output)
    ensure_parent_dir(info_output)

    ks = args.topk_list
    max_k = max(ks)
    pool_size = max(128, max_k + 8)
    (
        global_popular_pool,
        global_random_pool,
        category_popular_pools,
        category_random_pools,
    ) = build_short_pools(
        item_ids=item_ids,
        item_ids_by_category=item_ids_by_category,
        popularity=popularity,
        seed=args.seed,
        pool_size=pool_size,
    )
    query_weights_by_item_id: dict[str, dict[str, float]] = {}
    postings_by_term: dict[str, list[tuple[int, float]]] = {}
    tfidf_query_pools: dict[str, list[str]] = {}
    tfidf_stats = {
        "text_fields": args.text_field_list,
        "query_item_ids_with_text": 0,
        "query_item_ids_missing_text": 0,
        "query_vocabulary_size": 0,
        "indexed_terms": 0,
        "indexed_item_rows": 0,
        "query_fallback_missing_text": 0,
        "query_pool_count": 0,
    }
    if args.baseline == "tfidf":
        query_weights_by_item_id, postings_by_term, built_tfidf_stats = build_tfidf_resources(
            items_input=items_input,
            item_id_to_row=item_id_to_row,
            eval_input=eval_input,
            valid_item_ids=valid_item_ids,
            text_fields=args.text_field_list,
            max_query=args.max_query,
        )
        tfidf_query_pools = build_tfidf_query_pools(
            query_weights_by_item_id=query_weights_by_item_id,
            postings_by_term=postings_by_term,
            item_ids=item_ids,
            pool_size=pool_size,
            workers=args.workers,
        )
        tfidf_stats.update(built_tfidf_stats)
        tfidf_stats["query_pool_count"] = len(tfidf_query_pools)
    sum_hit = {k: 0.0 for k in ks}
    sum_mrr = {k: 0.0 for k in ks}
    sum_ndcg = {k: 0.0 for k in ks}

    input_rows_total = 0
    parse_error_rows = 0
    non_object_rows = 0
    rows_missing_user_id = 0
    rows_missing_target_item_id = 0
    rows_invalid_query_list = 0

    dropped_target_not_in_items = 0
    dropped_query_contains_target = 0
    dropped_query_item_not_in_items = 0
    dropped_query_empty_after_clean = 0
    dropped_no_candidates_after_filter = 0

    category_rows = 0
    category_fallback_missing_category = 0
    category_fallback_short_pool = 0
    category_rows_with_matched_category_pool = 0

    valid_eval_rows = 0

    with eval_input.open("r", encoding="utf-8") as in_f, predictions_output.open(
        "w", encoding="utf-8"
    ) as pred_f:
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
            target_item_id = normalize_text(raw.get("target_item_id"))
            query_item_ids_raw = raw.get("query_item_ids")

            if not user_id:
                rows_missing_user_id += 1
                continue
            if not target_item_id:
                rows_missing_target_item_id += 1
                continue
            if not isinstance(query_item_ids_raw, list):
                rows_invalid_query_list += 1
                continue

            cleaned_query_item_ids: list[str] = []
            seen_query_item_ids: set[str] = set()
            for item in query_item_ids_raw:
                item_id = normalize_text(item)
                if not item_id:
                    continue
                if item_id in seen_query_item_ids:
                    continue
                seen_query_item_ids.add(item_id)
                cleaned_query_item_ids.append(item_id)

            if not cleaned_query_item_ids:
                dropped_query_empty_after_clean += 1
                continue
            if target_item_id in seen_query_item_ids:
                dropped_query_contains_target += 1
                continue
            if target_item_id not in valid_item_ids:
                dropped_target_not_in_items += 1
                continue

            missing_query_item = False
            for query_item_id in cleaned_query_item_ids:
                if query_item_id not in valid_item_ids:
                    missing_query_item = True
                    break
            if missing_query_item:
                dropped_query_item_not_in_items += 1
                continue

            excluded_item_ids = set(cleaned_query_item_ids)

            if args.baseline == "random":
                predictions = build_predictions(
                    ordered_item_ids=take_topk_excluding(
                        ordered_item_ids=global_random_pool,
                        excluded_item_ids=excluded_item_ids,
                        top_k=max_k,
                    ),
                    top_k=max_k,
                )
            elif args.baseline == "global_popular":
                predictions = build_predictions(
                    ordered_item_ids=take_topk_excluding(
                        ordered_item_ids=global_popular_pool,
                        excluded_item_ids=excluded_item_ids,
                        top_k=max_k,
                    ),
                    top_k=max_k,
                    popularity=popularity,
                )
            elif args.baseline == "tfidf":
                predictions = build_predictions(
                    ordered_item_ids=take_topk_excluding(
                        ordered_item_ids=tfidf_query_pools.get(cleaned_query_item_ids[-1], []),
                        excluded_item_ids=excluded_item_ids,
                        top_k=max_k,
                    ),
                    top_k=max_k,
                )
                if not predictions:
                    tfidf_stats["query_fallback_missing_text"] += 1
                    predictions = build_predictions(
                        ordered_item_ids=take_topk_excluding(
                            ordered_item_ids=global_popular_pool,
                            excluded_item_ids=excluded_item_ids,
                            top_k=max_k,
                        ),
                        top_k=max_k,
                        popularity=popularity,
                    )
            else:
                category_rows += 1
                last_query_item_id = cleaned_query_item_ids[-1]
                query_category = category_by_item_id.get(last_query_item_id, "")
                if query_category:
                    if args.baseline == "category_random":
                        primary_pool = category_random_pools.get(query_category, [])
                        fallback_pool = global_random_pool
                        popularity_for_predictions = None
                    else:
                        primary_pool = category_popular_pools.get(query_category, [])
                        fallback_pool = global_popular_pool
                        popularity_for_predictions = popularity
                    primary_filtered_item_ids = take_topk_excluding(
                        ordered_item_ids=primary_pool,
                        excluded_item_ids=excluded_item_ids,
                        top_k=max_k,
                    )
                    combined_item_ids = take_combined_topk_excluding(
                        primary_item_ids=primary_pool,
                        fallback_item_ids=fallback_pool,
                        excluded_item_ids=excluded_item_ids,
                        top_k=max_k,
                    )
                    if len(primary_filtered_item_ids) == max_k:
                        category_rows_with_matched_category_pool += 1
                    else:
                        category_fallback_short_pool += 1
                    predictions = build_predictions(
                        ordered_item_ids=combined_item_ids,
                        top_k=max_k,
                        popularity=popularity_for_predictions,
                    )
                else:
                    category_fallback_missing_category += 1
                    if args.baseline == "category_random":
                        predictions = build_predictions(
                            ordered_item_ids=take_topk_excluding(
                                ordered_item_ids=global_random_pool,
                                excluded_item_ids=excluded_item_ids,
                                top_k=max_k,
                            ),
                            top_k=max_k,
                        )
                    else:
                        predictions = build_predictions(
                            ordered_item_ids=take_topk_excluding(
                                ordered_item_ids=global_popular_pool,
                                excluded_item_ids=excluded_item_ids,
                                top_k=max_k,
                            ),
                            top_k=max_k,
                            popularity=popularity,
                        )

            if not predictions:
                dropped_no_candidates_after_filter += 1
                continue

            target_rank = find_rank(predictions, target_item_id)
            valid_eval_rows += 1

            for k in ks:
                hit, mrr, ndcg = metric_value(target_rank, k)
                sum_hit[k] += hit
                sum_mrr[k] += mrr
                sum_ndcg[k] += ndcg

            pred_f.write(
                json.dumps(
                    {
                        "user_id": user_id,
                        "query_item_ids": cleaned_query_item_ids,
                        "target_item_id": target_item_id,
                        "target_rank": target_rank,
                        "predictions": predictions,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

            if args.max_query > 0 and valid_eval_rows >= args.max_query:
                break

    metrics: dict[str, dict[str, float]] = {}
    for k in ks:
        denom = float(valid_eval_rows) if valid_eval_rows else 1.0
        metrics[f"@{k}"] = {
            "hit_rate": sum_hit[k] / denom,
            "recall": sum_hit[k] / denom,
            "mrr": sum_mrr[k] / denom,
            "ndcg": sum_ndcg[k] / denom,
        }

    generated_at = datetime.now(timezone.utc).isoformat()
    report = {
        "generated_at_utc": generated_at,
        "baseline": args.baseline,
        "config": {
            "baseline": args.baseline,
            "items_input": str(items_input.resolve()),
            "interactions_input": str(interactions_input.resolve()),
            "eval_input": str(eval_input.resolve()),
            "output_root": str(Path(args.output_root).resolve()),
            "run_output_dir": str(run_output_dir.resolve()),
            "predictions_output": str(predictions_output.resolve()),
            "report_output": str(report_output.resolve()),
            "info_output": str(info_output.resolve()),
            "topk": ks,
            "max_query": args.max_query,
            "seed": args.seed,
            "rating_threshold": args.rating_threshold,
            "text_fields": args.text_field_list,
            "workers": args.workers,
            "run_id": run_id,
        },
        "input_stats": {
            "eval_rows_total": input_rows_total,
            "eval_parse_error_rows": parse_error_rows,
            "eval_non_object_rows": non_object_rows,
            "eval_rows_missing_user_id": rows_missing_user_id,
            "eval_rows_missing_target_item_id": rows_missing_target_item_id,
            "eval_rows_invalid_query_item_ids": rows_invalid_query_list,
            "items": item_stats,
            "interactions": popularity_stats,
        },
        "filter_stats": {
            "dropped_target_not_in_items": dropped_target_not_in_items,
            "dropped_query_contains_target": dropped_query_contains_target,
            "dropped_query_item_not_in_items": dropped_query_item_not_in_items,
            "dropped_query_empty_after_clean": dropped_query_empty_after_clean,
            "dropped_no_candidates_after_filter": dropped_no_candidates_after_filter,
        },
        "baseline_stats": {
            "category_rows": category_rows,
            "category_rows_with_matched_category_pool": category_rows_with_matched_category_pool,
            "category_fallback_missing_category": category_fallback_missing_category,
            "category_fallback_short_pool": category_fallback_short_pool,
            "candidate_pool_size": pool_size,
            "tfidf": tfidf_stats,
        },
        "eval_stats": {
            "valid_eval_rows": valid_eval_rows,
            "kept_rate_over_input": (
                valid_eval_rows / input_rows_total if input_rows_total else 0.0
            ),
        },
        "metrics": metrics,
    }
    report_output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    info = {
        "generated_at_utc": generated_at,
        "baseline": args.baseline,
        "run_id": run_id,
        "inputs": {
            "items_input": {
                "path": str(items_input.resolve()),
                "sha256": file_sha256(items_input),
            },
            "interactions_input": {
                "path": str(interactions_input.resolve()),
                "sha256": file_sha256(interactions_input),
            },
            "eval_input": {
                "path": str(eval_input.resolve()),
                "sha256": file_sha256(eval_input),
            },
        },
        "config": report["config"],
        "outputs": {
            "run_output_dir": str(run_output_dir.resolve()),
            "predictions_output": str(predictions_output.resolve()),
            "report_output": str(report_output.resolve()),
            "info_output": str(info_output.resolve()),
        },
    }
    info_output.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
