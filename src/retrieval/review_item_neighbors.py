#!/usr/bin/env python3
"""Build ANN index from item embeddings and print query item + top-k neighbor texts."""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path
from typing import Any

from ann_utils import (
    build_faiss_index,
    fetch_items_by_ids,
    load_embeddings,
    load_item_ids,
    search_topk_by_item_id,
)

EMBEDDING_DIM_FILE_RE = re.compile(r"^item_embeddings_(\d+)\.npy$")


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def parse_text_fields(value: str) -> list[str]:
    fields = [x.strip() for x in value.split(",")]
    fields = [x for x in fields if x]
    if not fields:
        raise ValueError("--text-fields must contain at least one field.")
    return fields


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Review nearest neighbors for one item id by FAISS ANN search.",
    )
    parser.add_argument(
        "--run-output-dir",
        default=None,
        help="Embedding run dir containing item_embeddings_<dim>.npy and item_ids.jsonl.",
    )
    parser.add_argument(
        "--embeddings-path",
        default=None,
        help="Path to one embedding file. Optional when --run-output-dir is provided.",
    )
    parser.add_argument(
        "--embedding-dim",
        default="max",
        help=(
            "Which embedding dim file to use when --run-output-dir is provided and "
            "--embeddings-path is omitted. Use integer or 'max'."
        ),
    )
    parser.add_argument(
        "--item-ids-path",
        default=None,
        help="Path to item_ids.jsonl. Optional when --run-output-dir is provided.",
    )
    parser.add_argument(
        "--items-input",
        default="data/processed/items.jsonl",
        help="Path to data/processed/items.jsonl.",
    )
    parser.add_argument(
        "--query-item-id",
        default=None,
        help="Query item_id. If omitted (or with --random-query), script samples one item_id.",
    )
    parser.add_argument(
        "--random-query",
        action="store_true",
        help="Sample query item_id from item_ids.jsonl.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used by --random-query (or when --query-item-id is omitted).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of nearest neighbors to print (excluding query item).",
    )
    parser.add_argument(
        "--index-type",
        choices=["hnsw", "flat"],
        default="hnsw",
        help="FAISS index type.",
    )
    parser.add_argument("--hnsw-m", type=int, default=32, help="HNSW M.")
    parser.add_argument("--hnsw-ef-search", type=int, default=128, help="HNSW efSearch.")
    parser.add_argument("--hnsw-ef-construction", type=int, default=200, help="HNSW efConstruction.")
    parser.add_argument(
        "--text-fields",
        default="title,author,categories",
        help="Comma-separated fields for review text output.",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable L2 normalization before indexing/search.",
    )
    args = parser.parse_args()
    if args.top_k <= 0:
        parser.error("--top-k must be > 0.")
    if args.random_query and args.query_item_id:
        parser.error("--random-query and --query-item-id cannot be used together.")
    embedding_dim_arg = normalize_text(args.embedding_dim).lower()
    if embedding_dim_arg != "max":
        try:
            value = int(embedding_dim_arg)
        except ValueError:
            parser.error("--embedding-dim must be an integer or 'max'.")
        if value <= 0:
            parser.error("--embedding-dim integer value must be > 0.")
        args.embedding_dim = str(value)
    else:
        args.embedding_dim = "max"
    return args


def collect_embedding_dim_files(run_dir: Path) -> dict[int, Path]:
    files: dict[int, Path] = {}
    for path in run_dir.iterdir():
        if not path.is_file():
            continue
        match = EMBEDDING_DIM_FILE_RE.fullmatch(path.name)
        if not match:
            continue
        files[int(match.group(1))] = path
    return dict(sorted(files.items(), key=lambda x: x[0]))


def resolve_embedding_path(run_dir: Path, embedding_dim: str) -> Path:
    dim_files = collect_embedding_dim_files(run_dir)
    if embedding_dim == "max":
        if dim_files:
            max_dim = max(dim_files.keys())
            return dim_files[max_dim]
    else:
        requested_dim = int(embedding_dim)
        path = dim_files.get(requested_dim)
        if path is not None:
            return path
        available_dims = sorted(dim_files.keys())
        raise FileNotFoundError(
            f"Requested --embedding-dim={requested_dim} not found in {run_dir}. "
            f"Available dims: {available_dims}"
        )
    legacy_path = run_dir / "item_embeddings.npy"
    if legacy_path.exists():
        return legacy_path
    raise FileNotFoundError(
        f"No embedding file found in {run_dir}. "
        "Expected item_embeddings_<dim>.npy (or legacy item_embeddings.npy)."
    )


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    run_dir = Path(args.run_output_dir) if args.run_output_dir else None
    embeddings_path = Path(args.embeddings_path) if args.embeddings_path else None
    item_ids_path = Path(args.item_ids_path) if args.item_ids_path else None

    if run_dir is not None:
        if embeddings_path is None:
            embeddings_path = resolve_embedding_path(run_dir, args.embedding_dim)
        if item_ids_path is None:
            item_ids_path = run_dir / "item_ids.jsonl"

    if embeddings_path is None or item_ids_path is None:
        raise ValueError(
            "Please provide --run-output-dir, or provide both --embeddings-path and --item-ids-path."
        )
    return embeddings_path, item_ids_path


def format_item_text(item: dict[str, Any], fields: list[str]) -> str:
    parts: list[str] = []
    for field in fields:
        parts.append(f"{field}={normalize_text(item.get(field, ''))}")
    return " | ".join(parts)


def resolve_query_item_id(
    query_item_id: str | None,
    random_query: bool,
    item_ids: list[str],
    seed: int,
) -> str:
    if not item_ids:
        raise ValueError("item_ids is empty, cannot choose query item.")
    if random_query or not query_item_id:
        rng = random.Random(seed)
        return item_ids[rng.randrange(len(item_ids))]
    return query_item_id


def main() -> None:
    args = parse_args()
    embeddings_path, item_ids_path = resolve_paths(args)
    items_input = Path(args.items_input)
    text_fields = parse_text_fields(args.text_fields)

    print(f"[review] embeddings={embeddings_path}")
    print(f"[review] item_ids={item_ids_path}")
    print(f"[review] items={items_input}")

    embeddings = load_embeddings(embeddings_path)
    item_ids, item_id_to_row = load_item_ids(item_ids_path)
    if embeddings.shape[0] != len(item_ids):
        raise ValueError(
            f"Row mismatch: embeddings rows={embeddings.shape[0]} != item_ids count={len(item_ids)}"
        )
    print(f"[review] rows={embeddings.shape[0]} dim={embeddings.shape[1]}")
    query_item_id = resolve_query_item_id(
        query_item_id=args.query_item_id,
        random_query=bool(args.random_query),
        item_ids=item_ids,
        seed=int(args.seed),
    )
    if args.random_query or not args.query_item_id:
        print(f"[review] sampled_query_item_id={query_item_id} seed={args.seed}")

    index, indexed_vectors = build_faiss_index(
        embeddings=embeddings,
        index_type=args.index_type,
        hnsw_m=args.hnsw_m,
        hnsw_ef_search=args.hnsw_ef_search,
        hnsw_ef_construction=args.hnsw_ef_construction,
        normalize=not args.no_normalize,
    )
    print(f"[review] index_type={args.index_type} ntotal={index.ntotal}")

    neighbors = search_topk_by_item_id(
        index=index,
        vectors=indexed_vectors,
        item_ids=item_ids,
        item_id_to_row=item_id_to_row,
        query_item_id=query_item_id,
        top_k=args.top_k,
    )
    selected_ids = [query_item_id] + [x.item_id for x in neighbors]
    selected_items = fetch_items_by_ids(items_input, set(selected_ids))

    print()
    print(f"[query] item_id={query_item_id}")
    query_item = selected_items.get(query_item_id)
    if query_item is None:
        print("text=<NOT_FOUND_IN_ITEMS_JSONL>")
    else:
        print(f"text={format_item_text(query_item, text_fields)}")

    print()
    print(f"[neighbors] top_k={args.top_k}")
    for n in neighbors:
        item = selected_items.get(n.item_id)
        text = "<NOT_FOUND_IN_ITEMS_JSONL>" if item is None else format_item_text(item, text_fields)
        print(f"{n.rank}. item_id={n.item_id} score={n.score:.6f}")
        print(f"   text={text}")


if __name__ == "__main__":
    main()
