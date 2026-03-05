#!/usr/bin/env python3
"""Run offline retrieval evaluation on eval query set."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.retrieval.ann_utils import build_faiss_index, load_embeddings, load_item_ids


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def parse_topk_list(value: str) -> list[int]:
    tokens = [x.strip() for x in value.split(",") if x.strip()]
    if not tokens:
        raise ValueError("topk list cannot be empty")
    ks: list[int] = []
    for token in tokens:
        k = int(token)
        if k <= 0:
            raise ValueError("topk values must be > 0")
        ks.append(k)
    return sorted(set(ks))


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


EMBEDDING_DIM_FILE_RE = re.compile(r"^item_embeddings_(\d+)\.npy$")


def collect_embedding_dim_files(embedding_dir: Path) -> dict[int, Path]:
    files: dict[int, Path] = {}
    for path in embedding_dir.iterdir():
        if not path.is_file():
            continue
        match = EMBEDDING_DIM_FILE_RE.fullmatch(path.name)
        if not match:
            continue
        dim = int(match.group(1))
        files[dim] = path
    return dict(sorted(files.items(), key=lambda x: x[0]))


def load_embedding_run_config(embedding_dir: Path) -> dict[str, Any] | None:
    config_path = embedding_dir / "config.json"
    if not config_path.exists():
        return None
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(raw, dict):
        return None
    return raw


def infer_embedding_identity(embedding_dir: Path) -> dict[str, str]:
    embedding_run_id = normalize_text(embedding_dir.name)
    model_dir_name = normalize_text(embedding_dir.parent.name if embedding_dir.parent else "")
    model_name_guess = model_dir_name.replace("__", "/") if model_dir_name else ""
    experiment_id = ""

    run_cfg = load_embedding_run_config(embedding_dir)
    if run_cfg:
        experiment_id = normalize_text(run_cfg.get("experiment_id"))
        model_cfg = run_cfg.get("model")
        if isinstance(model_cfg, dict):
            model_name = normalize_text(model_cfg.get("name"))
            if model_name:
                model_name_guess = model_name
                model_dir_name = model_name.replace("/", "__")

    # Backward compatibility for legacy layout: embeddings/<model>/<experiment>/<run_id>
    if not run_cfg and embedding_dir.parent and embedding_dir.parent.parent:
        maybe_experiment = normalize_text(embedding_dir.parent.name)
        maybe_model_dir = normalize_text(embedding_dir.parent.parent.name)
        if maybe_experiment.startswith("exp_") and maybe_model_dir:
            experiment_id = maybe_experiment
            model_dir_name = maybe_model_dir
            model_name_guess = maybe_model_dir.replace("__", "/")

    return {
        "model_dir_name": model_dir_name,
        "model_name_guess": model_name_guess,
        "experiment_id": experiment_id,
        "embedding_run_id": embedding_run_id,
    }


def resolve_embedding_files(
    embedding_dir: Path,
    embedding_dim_arg: str,
) -> list[tuple[int | None, Path]]:
    dim_files = collect_embedding_dim_files(embedding_dir)
    legacy_file = embedding_dir / "item_embeddings.npy"
    has_legacy = legacy_file.exists()

    if embedding_dim_arg == "all":
        if dim_files:
            return [(dim, path) for dim, path in dim_files.items()]
        if has_legacy:
            return [(None, legacy_file)]
        raise FileNotFoundError(
            f"No embedding files found in {embedding_dir}. "
            "Expected item_embeddings_<dim>.npy (or legacy item_embeddings.npy)."
        )

    if embedding_dim_arg == "max":
        if dim_files:
            max_dim = max(dim_files)
            return [(max_dim, dim_files[max_dim])]
        if has_legacy:
            return [(None, legacy_file)]
        raise FileNotFoundError(
            f"No embedding files found in {embedding_dir}. "
            "Expected item_embeddings_<dim>.npy (or legacy item_embeddings.npy)."
        )

    requested_dim = int(embedding_dim_arg)
    if dim_files:
        path = dim_files.get(requested_dim)
        if path is None:
            raise FileNotFoundError(
                f"Requested --embedding-dim={requested_dim} not found in {embedding_dir}. "
                f"Available dims: {sorted(dim_files.keys())}"
            )
        return [(requested_dim, path)]
    if has_legacy:
        raise FileNotFoundError(
            f"Requested --embedding-dim={requested_dim}, but {embedding_dir} only has legacy "
            "item_embeddings.npy without dimension suffix."
        )
    raise FileNotFoundError(
        f"No embedding files found in {embedding_dir}. "
        "Expected item_embeddings_<dim>.npy (or legacy item_embeddings.npy)."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run retrieval evaluation from eval.jsonl.")
    parser.add_argument(
        "--eval-input",
        default="data/processed/eval.jsonl",
        help="Input eval queries jsonl path.",
    )
    parser.add_argument(
        "--embedding-dir",
        required=True,
        help=(
            "Embedding artifact directory containing item_ids.jsonl and "
            "item_embeddings_<dim>.npy (or legacy item_embeddings.npy)."
        ),
    )
    parser.add_argument(
        "--embedding-dim",
        default="max",
        help=(
            "Embedding dimension to evaluate. "
            "Use integer like 128/1024, 'max' for largest available, or 'all' for all dimensions."
        ),
    )
    parser.add_argument(
        "--output-root",
        default="outputs/eval",
        help="Root output dir. Files are written to <output_root>/<eval_run_id>/.",
    )
    parser.add_argument(
        "--eval-run-id",
        default="",
        help="Optional eval run id. Default is local timestamp YYYYMMDDHHMMSS.",
    )
    parser.add_argument(
        "--max-query",
        type=int,
        default=0,
        help="Maximum number of valid eval queries to evaluate. 0 means no limit.",
    )
    parser.add_argument(
        "--topk",
        default="10,50",
        help="Comma-separated metric K list, e.g. '10,50'.",
    )
    parser.add_argument(
        "--query-history-n",
        type=int,
        default=0,
        help="Number of most recent query history items to use; 0 means use all.",
    )
    parser.add_argument(
        "--query-pooling",
        choices=["mean", "max", "last"],
        default="mean",
        help="How to pool multi-item query history into one query embedding.",
    )
    parser.add_argument(
        "--query-retrieval-mode",
        choices=["pooling", "merging"],
        default="pooling",
        help="Query retrieval mode: one pooled query vs per-query retrieval + merge.",
    )
    parser.add_argument(
        "--per-query-topk",
        type=int,
        default=20,
        help="Per-query topK used only when --query-retrieval-mode=merging.",
    )
    parser.add_argument(
        "--merge-fusion",
        choices=["max", "rrf"],
        default="max",
        help="Fusion method used only when --query-retrieval-mode=merging.",
    )
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=60,
        help="RRF constant K used when --merge-fusion=rrf.",
    )
    parser.add_argument(
        "--recency-weighting",
        choices=["none", "linear", "exp"],
        default="none",
        help="Recency weighting strategy over query history in merging mode.",
    )
    parser.add_argument(
        "--recency-alpha",
        type=float,
        default=1.0,
        help="Exponential decay alpha used when --recency-weighting=exp.",
    )
    parser.add_argument(
        "--index-type",
        choices=["flat", "hnsw"],
        default="flat",
        help="ANN index type.",
    )
    parser.add_argument(
        "--hnsw-m",
        type=int,
        default=32,
        help="HNSW M parameter (used only when --index-type hnsw).",
    )
    parser.add_argument(
        "--hnsw-ef-search",
        type=int,
        default=64,
        help="HNSW efSearch parameter (used only when --index-type hnsw).",
    )
    parser.add_argument(
        "--hnsw-ef-construction",
        type=int,
        default=200,
        help="HNSW efConstruction parameter (used only when --index-type hnsw).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic runtime metadata.",
    )
    args = parser.parse_args()
    if args.max_query < 0:
        parser.error("--max-query must be >= 0")
    if args.query_history_n < 0:
        parser.error("--query-history-n must be >= 0")
    if args.per_query_topk <= 0:
        parser.error("--per-query-topk must be > 0")
    if args.rrf_k <= 0:
        parser.error("--rrf-k must be > 0")
    if args.recency_alpha < 0:
        parser.error("--recency-alpha must be >= 0")
    embedding_dim_arg = normalize_text(args.embedding_dim).lower()
    if embedding_dim_arg not in {"all", "max"}:
        try:
            value = int(embedding_dim_arg)
        except ValueError:
            parser.error("--embedding-dim must be an integer, 'max', or 'all'")
        if value <= 0:
            parser.error("--embedding-dim integer value must be > 0")
        args.embedding_dim = str(value)
    else:
        args.embedding_dim = embedding_dim_arg

    args.topk_list = parse_topk_list(args.topk)
    return args


def make_query_vector(vectors: np.ndarray, query_rows: list[int], pooling: str) -> np.ndarray:
    query_vectors = vectors[np.array(query_rows, dtype=np.int64)]
    if pooling == "mean":
        query_vec = query_vectors.mean(axis=0, dtype=np.float32)
    elif pooling == "max":
        query_vec = query_vectors.max(axis=0)
    elif pooling == "last":
        query_vec = query_vectors[-1]
    else:
        raise ValueError(f"Unsupported query pooling method: {pooling}")
    query_vec = np.asarray(query_vec, dtype=np.float32)
    norm = float(np.linalg.norm(query_vec))
    if norm <= 1e-12:
        raise ValueError("zero-norm query vector")
    return (query_vec / norm).astype(np.float32, copy=False)


def search_topk_excluding_rows(
    *,
    index: Any,
    item_ids: list[str],
    query_vec: np.ndarray,
    excluded_rows: set[int],
    top_k: int,
) -> list[dict[str, Any]]:
    if top_k <= 0:
        return []
    ntotal = int(index.ntotal)
    if ntotal <= 0:
        return []

    target_k = min(top_k, ntotal)
    search_k = min(ntotal, max(target_k + len(excluded_rows) + 16, target_k))
    neighbors: list[dict[str, Any]] = []
    seen_rows: set[int] = set()

    while True:
        scores, row_ids = index.search(query_vec.reshape(1, -1), search_k)
        for row_id, score in zip(row_ids[0].tolist(), scores[0].tolist()):
            if row_id < 0:
                continue
            if row_id in excluded_rows:
                continue
            if row_id in seen_rows:
                continue
            seen_rows.add(row_id)
            neighbors.append(
                {
                    "rank": len(neighbors) + 1,
                    "item_id": item_ids[row_id],
                    "score": float(score),
                }
            )
            if len(neighbors) >= target_k:
                return neighbors

        if search_k >= ntotal:
            break
        search_k = min(ntotal, search_k * 2)

    return neighbors


def build_query_recency_weights(
    num_queries: int,
    recency_weighting: str,
    recency_alpha: float,
) -> list[float]:
    if num_queries <= 0:
        raise ValueError("num_queries must be > 0")
    if recency_weighting == "none":
        raw_weights = [1.0] * num_queries
    elif recency_weighting == "linear":
        # Query order is assumed oldest -> newest; newer queries get larger weights.
        raw_weights = [float(i + 1) for i in range(num_queries)]
    elif recency_weighting == "exp":
        raw_weights = [
            math.exp(-recency_alpha * float((num_queries - 1) - i)) for i in range(num_queries)
        ]
    else:
        raise ValueError(f"Unsupported recency weighting: {recency_weighting}")

    total = float(sum(raw_weights))
    if total <= 1e-12:
        raise ValueError("invalid recency weights")
    return [float(w / total) for w in raw_weights]


def make_single_query_vector(vectors: np.ndarray, query_row: int) -> np.ndarray:
    query_vec = np.asarray(vectors[query_row], dtype=np.float32)
    norm = float(np.linalg.norm(query_vec))
    if norm <= 1e-12:
        raise ValueError("zero-norm single query vector")
    return (query_vec / norm).astype(np.float32, copy=False)


def _sorted_predictions_from_score_map(score_by_item_id: dict[str, float]) -> list[dict[str, Any]]:
    merged = sorted(score_by_item_id.items(), key=lambda x: (-x[1], x[0]))
    return [
        {
            "rank": idx + 1,
            "item_id": item_id,
            "score": float(score),
        }
        for idx, (item_id, score) in enumerate(merged)
    ]


def merge_predictions_max_score(
    per_query_predictions: list[list[dict[str, Any]]],
    query_weights: list[float],
) -> list[dict[str, Any]]:
    if len(per_query_predictions) != len(query_weights):
        raise ValueError("per_query_predictions and query_weights length mismatch")
    score_by_item_id: dict[str, float] = {}
    for preds, query_weight in zip(per_query_predictions, query_weights):
        for pred in preds:
            item_id = normalize_text(pred.get("item_id"))
            if not item_id:
                continue
            score = float(query_weight) * float(pred["score"])
            prev = score_by_item_id.get(item_id)
            if prev is None or score > prev:
                score_by_item_id[item_id] = score

    return _sorted_predictions_from_score_map(score_by_item_id)


def merge_predictions_rrf(
    per_query_predictions: list[list[dict[str, Any]]],
    query_weights: list[float],
    rrf_k: int,
) -> list[dict[str, Any]]:
    if rrf_k <= 0:
        raise ValueError("rrf_k must be > 0")
    if len(per_query_predictions) != len(query_weights):
        raise ValueError("per_query_predictions and query_weights length mismatch")
    score_by_item_id: dict[str, float] = {}
    for preds, query_weight in zip(per_query_predictions, query_weights):
        for pred in preds:
            item_id = normalize_text(pred.get("item_id"))
            if not item_id:
                continue
            rank = int(pred["rank"])
            if rank <= 0:
                continue
            score = float(query_weight) * (1.0 / float(rrf_k + rank))
            score_by_item_id[item_id] = score_by_item_id.get(item_id, 0.0) + score

    return _sorted_predictions_from_score_map(score_by_item_id)


def find_rank(predictions: list[dict[str, Any]], target_item_id: str) -> int | None:
    for pred in predictions:
        if pred["item_id"] == target_item_id:
            return int(pred["rank"])
    return None


def metric_value(rank: int | None, k: int) -> tuple[float, float, float]:
    if rank is None or rank > k:
        return 0.0, 0.0, 0.0
    recall = 1.0
    mrr = 1.0 / float(rank)
    ndcg = 1.0 / math.log2(float(rank) + 1.0)
    return recall, mrr, ndcg


def run_eval_once(
    *,
    args: argparse.Namespace,
    eval_input: Path,
    eval_input_sha256: str,
    embedding_dir: Path,
    item_ids_input: Path,
    embeddings_input: Path,
    item_ids: list[str],
    item_id_to_row: dict[str, int],
    eval_run_id: str,
    predictions_output: Path,
    report_output: Path,
    info_output: Path,
    requested_embedding_dim: int | None,
) -> dict[str, Any]:
    ensure_parent_dir(predictions_output)
    ensure_parent_dir(report_output)
    ensure_parent_dir(info_output)

    embeddings = load_embeddings(embeddings_input)
    if embeddings.shape[0] != len(item_ids):
        raise ValueError(
            "Row mismatch between embeddings and item_ids: "
            f"embeddings_rows={embeddings.shape[0]}, item_ids_rows={len(item_ids)}"
        )

    index, vectors = build_faiss_index(
        embeddings=embeddings,
        index_type=args.index_type,
        hnsw_m=args.hnsw_m,
        hnsw_ef_search=args.hnsw_ef_search,
        hnsw_ef_construction=args.hnsw_ef_construction,
        normalize=True,
    )

    ks = args.topk_list
    max_k = max(ks)
    sum_hit = {k: 0.0 for k in ks}
    sum_mrr = {k: 0.0 for k in ks}
    sum_ndcg = {k: 0.0 for k in ks}

    input_rows_total = 0
    parse_error_rows = 0
    non_object_rows = 0
    rows_missing_user_id = 0
    rows_missing_target_item_id = 0
    rows_invalid_query_list = 0

    dropped_target_not_in_index = 0
    dropped_query_contains_target = 0
    dropped_query_item_not_in_index = 0
    dropped_query_empty_after_clean = 0
    dropped_zero_norm_query = 0

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

            if args.query_history_n > 0:
                effective_query_item_ids = cleaned_query_item_ids[-args.query_history_n :]
            else:
                effective_query_item_ids = cleaned_query_item_ids

            if not effective_query_item_ids:
                dropped_query_empty_after_clean += 1
                continue
            if target_item_id in set(effective_query_item_ids):
                dropped_query_contains_target += 1
                continue

            target_row = item_id_to_row.get(target_item_id)
            if target_row is None:
                dropped_target_not_in_index += 1
                continue

            query_rows: list[int] = []
            missing_query_item = False
            for query_item_id in effective_query_item_ids:
                query_row = item_id_to_row.get(query_item_id)
                if query_row is None:
                    missing_query_item = True
                    break
                query_rows.append(query_row)
            if missing_query_item:
                dropped_query_item_not_in_index += 1
                continue
            if not query_rows:
                dropped_query_empty_after_clean += 1
                continue

            excluded_rows = set(query_rows)
            try:
                if args.query_retrieval_mode == "pooling":
                    query_vec = make_query_vector(vectors, query_rows, args.query_pooling)
                    predictions = search_topk_excluding_rows(
                        index=index,
                        item_ids=item_ids,
                        query_vec=query_vec,
                        excluded_rows=excluded_rows,
                        top_k=max_k,
                    )
                else:
                    query_weights = build_query_recency_weights(
                        num_queries=len(query_rows),
                        recency_weighting=args.recency_weighting,
                        recency_alpha=args.recency_alpha,
                    )
                    per_query_predictions: list[list[dict[str, Any]]] = []
                    for query_row in query_rows:
                        single_query_vec = make_single_query_vector(vectors, query_row)
                        per_query_predictions.append(
                            search_topk_excluding_rows(
                                index=index,
                                item_ids=item_ids,
                                query_vec=single_query_vec,
                                excluded_rows=excluded_rows,
                                top_k=args.per_query_topk,
                            )
                        )
                    if args.merge_fusion == "max":
                        predictions = merge_predictions_max_score(
                            per_query_predictions=per_query_predictions,
                            query_weights=query_weights,
                        )[:max_k]
                    else:
                        predictions = merge_predictions_rrf(
                            per_query_predictions=per_query_predictions,
                            query_weights=query_weights,
                            rrf_k=args.rrf_k,
                        )[:max_k]
            except ValueError:
                dropped_zero_norm_query += 1
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
                        "query_item_ids": effective_query_item_ids,
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

    embedding_identity = infer_embedding_identity(embedding_dir)
    eval_input_resolved = eval_input.resolve()
    generated_at = datetime.now(timezone.utc).isoformat()

    report = {
        "generated_at_utc": generated_at,
        "config": {
            "seed": args.seed,
            "eval_run_id": eval_run_id,
            "eval_input": str(eval_input_resolved),
            "eval_input_sha256": eval_input_sha256,
            "embedding_dir": str(embedding_dir.resolve()),
            "item_ids_input": str(item_ids_input.resolve()),
            "embeddings_input": str(embeddings_input.resolve()),
            "requested_embedding_dim": requested_embedding_dim,
            "output_root": str(Path(args.output_root).resolve()),
            "predictions_output": str(predictions_output.resolve()),
            "report_output": str(report_output.resolve()),
            "info_output": str(info_output.resolve()),
            "topk": ks,
            "query_history_n": args.query_history_n,
            "query_pooling": args.query_pooling,
            "query_retrieval_mode": args.query_retrieval_mode,
            "per_query_topk": args.per_query_topk,
            "merge_fusion": args.merge_fusion,
            "rrf_k": args.rrf_k,
            "recency_weighting": args.recency_weighting,
            "recency_alpha": args.recency_alpha,
            "max_query": args.max_query,
            "index_type": args.index_type,
            "hnsw_m": args.hnsw_m,
            "hnsw_ef_search": args.hnsw_ef_search,
            "hnsw_ef_construction": args.hnsw_ef_construction,
        },
        "embedding_identity": embedding_identity,
        "index_stats": {
            "num_items": int(len(item_ids)),
            "embedding_dim": int(vectors.shape[1]),
            "faiss_ntotal": int(index.ntotal),
        },
        "input_stats": {
            "rows_total": input_rows_total,
            "parse_error_rows": parse_error_rows,
            "non_object_rows": non_object_rows,
            "rows_missing_user_id": rows_missing_user_id,
            "rows_missing_target_item_id": rows_missing_target_item_id,
            "rows_invalid_query_item_ids": rows_invalid_query_list,
        },
        "filter_stats": {
            "dropped_target_not_in_index": dropped_target_not_in_index,
            "dropped_query_contains_target": dropped_query_contains_target,
            "dropped_query_item_not_in_index": dropped_query_item_not_in_index,
            "dropped_query_empty_after_clean": dropped_query_empty_after_clean,
            "dropped_zero_norm_query": dropped_zero_norm_query,
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
        "eval_run_id": eval_run_id,
        "eval_input": {
            "path": str(eval_input_resolved),
            "sha256": eval_input_sha256,
        },
        "embedding": {
            "dir": str(embedding_dir.resolve()),
            "item_ids_path": str(item_ids_input.resolve()),
            "embeddings_path": str(embeddings_input.resolve()),
            "requested_embedding_dim": requested_embedding_dim,
            "resolved_embedding_dim": int(vectors.shape[1]),
            **embedding_identity,
        },
        "retrieval": {
            "topk": ks,
            "query_history_n": args.query_history_n,
            "query_pooling": args.query_pooling,
            "query_retrieval_mode": args.query_retrieval_mode,
            "per_query_topk": args.per_query_topk,
            "merge_fusion": args.merge_fusion,
            "rrf_k": args.rrf_k,
            "recency_weighting": args.recency_weighting,
            "recency_alpha": args.recency_alpha,
            "max_query": args.max_query,
            "index_type": args.index_type,
            "hnsw_m": args.hnsw_m,
            "hnsw_ef_search": args.hnsw_ef_search,
            "hnsw_ef_construction": args.hnsw_ef_construction,
            "seed": args.seed,
        },
        "outputs": {
            "run_output_dir": str(predictions_output.parent.resolve()),
            "predictions_output": str(predictions_output.resolve()),
            "report_output": str(report_output.resolve()),
            "info_output": str(info_output.resolve()),
        },
    }
    info_output.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    eval_input = Path(args.eval_input)
    embedding_dir = Path(args.embedding_dir)
    item_ids_input = embedding_dir / "item_ids.jsonl"
    if not item_ids_input.exists():
        raise FileNotFoundError(f"Missing item_ids.jsonl in embedding dir: {item_ids_input}")
    embedding_files = resolve_embedding_files(embedding_dir, args.embedding_dim)

    eval_run_id = args.eval_run_id.strip() or datetime.now().strftime("%Y%m%d%H%M%S")
    run_output_dir = Path(args.output_root) / eval_run_id
    run_output_dir.mkdir(parents=True, exist_ok=True)

    item_ids, item_id_to_row = load_item_ids(item_ids_input)
    eval_sha = file_sha256(eval_input)
    run_reports: list[dict[str, Any]] = []

    for dim, embeddings_input in embedding_files:
        if len(embedding_files) == 1:
            dim_output_dir = run_output_dir
        else:
            dim_label = "legacy" if dim is None else str(dim)
            dim_output_dir = run_output_dir / f"dim_{dim_label}"
        predictions_output = dim_output_dir / "predictions.jsonl"
        report_output = dim_output_dir / "run_eval_report.json"
        info_output = dim_output_dir / "info.json"

        report = run_eval_once(
            args=args,
            eval_input=eval_input,
            eval_input_sha256=eval_sha,
            embedding_dir=embedding_dir,
            item_ids_input=item_ids_input,
            embeddings_input=embeddings_input,
            item_ids=item_ids,
            item_id_to_row=item_id_to_row,
            eval_run_id=eval_run_id,
            predictions_output=predictions_output,
            report_output=report_output,
            info_output=info_output,
            requested_embedding_dim=dim,
        )
        run_reports.append(
            {
                "requested_embedding_dim": dim,
                "resolved_embedding_dim": report["index_stats"]["embedding_dim"],
                "embeddings_input": str(embeddings_input.resolve()),
                "predictions_output": str(predictions_output.resolve()),
                "report_output": str(report_output.resolve()),
                "info_output": str(info_output.resolve()),
                "valid_eval_rows": report["eval_stats"]["valid_eval_rows"],
                "metrics": report["metrics"],
            }
        )

    if len(run_reports) > 1:
        summary_report_output = run_output_dir / "run_eval_report.json"
        summary_info_output = run_output_dir / "info.json"
        generated_at = datetime.now(timezone.utc).isoformat()
        summary_report = {
            "generated_at_utc": generated_at,
            "config": {
                "seed": args.seed,
                "eval_run_id": eval_run_id,
                "eval_input": str(eval_input.resolve()),
                "eval_input_sha256": eval_sha,
                "embedding_dir": str(embedding_dir.resolve()),
                "item_ids_input": str(item_ids_input.resolve()),
                "embedding_dim": args.embedding_dim,
                "output_root": str(Path(args.output_root).resolve()),
                "run_output_dir": str(run_output_dir.resolve()),
            },
            "runs": run_reports,
        }
        summary_report_output.write_text(
            json.dumps(summary_report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        summary_info = {
            "generated_at_utc": generated_at,
            "eval_run_id": eval_run_id,
            "eval_input": {
                "path": str(eval_input.resolve()),
                "sha256": eval_sha,
            },
            "embedding": {
                "dir": str(embedding_dir.resolve()),
                "item_ids_path": str(item_ids_input.resolve()),
                "embedding_dim": args.embedding_dim,
            },
            "outputs": {
                "run_output_dir": str(run_output_dir.resolve()),
                "report_output": str(summary_report_output.resolve()),
                "info_output": str(summary_info_output.resolve()),
                "runs": run_reports,
            },
        }
        summary_info_output.write_text(
            json.dumps(summary_info, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
