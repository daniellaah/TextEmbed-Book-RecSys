#!/usr/bin/env python3
"""Generate item embeddings from items.jsonl and experiment config."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from numpy.lib.format import open_memmap
from transformers import AutoModel, AutoTokenizer


class SafeFormatDict(dict[str, str]):
    def __missing__(self, key: str) -> str:
        return ""


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def sanitize_model_name_for_path(model_name: str) -> str:
    return model_name.replace("/", "__")


def canonical_json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def compute_config_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json_dumps(value).encode("utf-8")).hexdigest()


def resolve_device(requested: str, allow_fallback: bool) -> str:
    requested = requested.lower()
    if requested == "mps":
        if torch.backends.mps.is_available():
            return "mps"
        reason = (
            "Requested device 'mps' is unavailable. "
            f"torch.backends.mps.is_built()={torch.backends.mps.is_built()}, "
            f"torch.backends.mps.is_available()={torch.backends.mps.is_available()}."
        )
        if allow_fallback:
            print(f"[embed] warning: {reason} Falling back to cpu.")
            return "cpu"
        raise RuntimeError(reason)
    if requested == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        reason = "Requested device 'cuda' is unavailable."
        if allow_fallback:
            print(f"[embed] warning: {reason} Falling back to cpu.")
            return "cpu"
        raise RuntimeError(reason)
    if requested == "cpu":
        return "cpu"
    raise ValueError(f"Unsupported device '{requested}'. Expected one of: mps/cuda/cpu.")


def is_valid_local_model_dir(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    has_config = (path / "config.json").exists()
    has_weights = (path / "pytorch_model.bin").exists() or (path / "model.safetensors").exists()
    has_tokenizer = (
        (path / "tokenizer.json").exists()
        or (path / "tokenizer_config.json").exists()
        or (path / "sentencepiece.bpe.model").exists()
    )
    return has_config and has_weights and has_tokenizer


def resolve_local_model_ref(model_name: str) -> str:
    """Resolve to a local model directory only; never trigger remote download."""
    candidate = Path(model_name).expanduser()
    if candidate.is_absolute() or str(model_name).startswith("."):
        if is_valid_local_model_dir(candidate):
            return str(candidate)
        raise FileNotFoundError(
            "model.name points to local path but required files are missing. "
            f"path={candidate}"
        )

    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    repo_cache_dir = cache_root / f"models--{model_name.replace('/', '--')}"
    snapshots_dir = repo_cache_dir / "snapshots"
    if snapshots_dir.exists():
        snapshots = sorted(
            [p for p in snapshots_dir.iterdir() if p.is_dir()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for snapshot in snapshots:
            if is_valid_local_model_dir(snapshot):
                return str(snapshot)

    download_hint = (
        f"Local model not found for '{model_name}'.\n"
        "Please pre-download the model, then re-run.\n"
        "Example download command:\n"
        "  .venv/bin/python - <<'PY'\n"
        "from transformers import AutoTokenizer, AutoModel\n"
        f"AutoTokenizer.from_pretrained('{model_name}')\n"
        f"AutoModel.from_pretrained('{model_name}')\n"
        "PY"
    )
    raise FileNotFoundError(download_hint)


def parse_experiment_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Experiment config must be a YAML object.")
    return cfg


def validate_experiment_config(cfg: dict[str, Any]) -> None:
    if not normalize_text(cfg.get("experiment_id")):
        raise ValueError("Missing 'experiment_id' in config.")

    model_cfg = cfg.get("model")
    if not isinstance(model_cfg, dict):
        raise ValueError("Missing 'model' section in config.")
    if not normalize_text(model_cfg.get("name")):
        raise ValueError("Missing 'model.name' in config.")
    if int(model_cfg.get("max_length", 0)) <= 0:
        raise ValueError("'model.max_length' must be > 0.")
    if int(model_cfg.get("batch_size", 0)) <= 0:
        raise ValueError("'model.batch_size' must be > 0.")

    text_views_cfg = cfg.get("text_views", {})
    views = text_views_cfg.get("views")
    if not isinstance(views, list) or not views:
        raise ValueError("'text_views.views' must be a non-empty list.")
    view_ids: set[str] = set()
    for view in views:
        if not isinstance(view, dict):
            raise ValueError("Each view in 'text_views.views' must be an object.")
        view_id = normalize_text(view.get("view_id"))
        if not view_id:
            raise ValueError("Every view must contain non-empty 'view_id'.")
        if view_id in view_ids:
            raise ValueError(f"Duplicate view_id detected: {view_id}")
        view_ids.add(view_id)
        if not isinstance(view.get("fields"), list):
            raise ValueError(f"View '{view_id}' must contain list field 'fields'.")
        if not normalize_text(view.get("template")):
            raise ValueError(f"View '{view_id}' must contain non-empty 'template'.")

    fusion_cfg = cfg.get("fusion")
    if not isinstance(fusion_cfg, dict):
        raise ValueError("Missing 'fusion' section in config.")
    method = normalize_text(fusion_cfg.get("method"))
    if method not in {"identity", "weighted_mean"}:
        raise ValueError("fusion.method must be one of: identity, weighted_mean")

    input_views = fusion_cfg.get("input_views")
    if not isinstance(input_views, list) or not input_views:
        raise ValueError("fusion.input_views must be a non-empty list.")
    for view_id in input_views:
        if view_id not in view_ids:
            raise ValueError(f"fusion.input_views includes undefined view_id: {view_id}")

    if method == "identity" and len(input_views) != 1:
        raise ValueError("fusion.method=identity requires exactly one fusion.input_views entry.")

    if method == "weighted_mean":
        weights = fusion_cfg.get("weights")
        if not isinstance(weights, dict):
            raise ValueError("fusion.weights must be provided for weighted_mean.")
        for view_id in input_views:
            if view_id not in weights:
                raise ValueError(f"fusion.weights missing weight for view_id: {view_id}")
            _ = float(weights[view_id])  # validate cast


def adapt_text_for_model(model_name: str, text: str) -> str:
    """Apply lightweight model-specific text formatting."""
    lowered = model_name.lower()
    if lowered.startswith("intfloat/e5") or "/e5-" in lowered or lowered.endswith("/e5"):
        if text.lower().startswith(("query: ", "passage: ")):
            return text
        return f"passage: {text}"
    return text


def render_view_text(item: dict[str, Any], view_cfg: dict[str, Any]) -> str:
    template = str(view_cfg["template"])
    field_values: dict[str, str] = {}
    for field in view_cfg.get("fields", []):
        field_values[str(field)] = normalize_text(item.get(field, ""))
    rendered = template.format_map(SafeFormatDict(field_values))
    return normalize_text(rendered)


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def encode_text_batch(
    texts: list[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: str,
    max_length: int,
    normalize_embeddings: bool,
) -> tuple[torch.Tensor, int, float]:
    start = time.perf_counter()
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    token_count = int(encoded["attention_mask"].sum().item())
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.no_grad():
        outputs = model(**encoded)
        pooled = mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
        if normalize_embeddings:
            pooled = F.normalize(pooled, p=2, dim=1)
    elapsed = time.perf_counter() - start
    return pooled.detach().cpu().to(torch.float32), token_count, elapsed


def fuse_batch_embeddings(
    view_embeddings: dict[str, torch.Tensor],
    fusion_cfg: dict[str, Any],
) -> torch.Tensor:
    method = fusion_cfg["method"]
    input_views = fusion_cfg["input_views"]
    if method == "identity":
        fused = view_embeddings[input_views[0]].clone()
    else:
        weights: dict[str, float] = {k: float(v) for k, v in fusion_cfg["weights"].items()}
        fused = torch.zeros_like(view_embeddings[input_views[0]])
        for view_id in input_views:
            fused += view_embeddings[view_id] * weights[view_id]
    if bool(fusion_cfg.get("normalization", False)):
        fused = F.normalize(fused, p=2, dim=1)
    return fused


def iter_items(items_path: Path, max_items: int | None) -> Iterator[dict[str, Any]]:
    with items_path.open("r", encoding="utf-8") as f:
        yielded = 0
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                raise ValueError(f"Empty line detected at {items_path}:{lineno}")
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at {items_path}:{lineno}: {e}") from e
            if not isinstance(raw, dict):
                raise ValueError(f"Expected object JSON at {items_path}:{lineno}")
            item_id = normalize_text(raw.get("item_id"))
            if not item_id:
                raise ValueError(f"Missing item_id at {items_path}:{lineno}")
            yielded += 1
            yield raw
            if max_items is not None and yielded >= max_items:
                break


def count_items(items_path: Path, max_items: int | None) -> int:
    count = 0
    for _ in iter_items(items_path, max_items=max_items):
        count += 1
    return count


def batched_items(items_path: Path, batch_size: int, max_items: int | None) -> Iterator[list[dict[str, Any]]]:
    batch: list[dict[str, Any]] = []
    for item in iter_items(items_path, max_items=max_items):
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate item embeddings from experiment config.")
    parser.add_argument(
        "--items-input",
        default="data/processed/items.jsonl",
        help="Input items jsonl path.",
    )
    parser.add_argument(
        "--experiment-config",
        required=True,
        help="Path to experiment yaml config.",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/embeddings",
        help="Root output directory for embedding artifacts.",
    )
    parser.add_argument(
        "--runs-root",
        default="outputs/runs",
        help="Root output directory for run config snapshots.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional run id. If omitted, auto-generated.",
    )
    parser.add_argument(
        "--device",
        default="mps",
        help="Requested device: mps/cuda/cpu.",
    )
    parser.add_argument(
        "--allow-device-fallback",
        action="store_true",
        help="If set, fallback to cpu when requested device is unavailable.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--save-view-embeddings",
        action="store_true",
        help="If set, save each input view embedding matrix alongside fused output.",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Optional cap for processed item rows (for smoke run/debug).",
    )
    args = parser.parse_args()
    if args.max_items is not None and args.max_items <= 0:
        parser.error("--max-items must be > 0 when provided.")
    return args


def main() -> None:
    args = parse_args()
    items_input = Path(args.items_input)
    config_path = Path(args.experiment_config)
    output_root = Path(args.output_root)
    runs_root = Path(args.runs_root)

    exp_cfg = parse_experiment_config(config_path)
    validate_experiment_config(exp_cfg)

    model_cfg: dict[str, Any] = exp_cfg["model"]
    model_name = str(model_cfg["name"])
    max_length = int(model_cfg["max_length"])
    batch_size = int(model_cfg["batch_size"])
    normalize_embeddings = bool(model_cfg.get("normalize_embeddings", True))

    fusion_cfg: dict[str, Any] = exp_cfg["fusion"]
    view_cfgs: list[dict[str, Any]] = exp_cfg["text_views"]["views"]
    view_by_id = {str(v["view_id"]): v for v in view_cfgs}
    input_views: list[str] = [str(v) for v in fusion_cfg["input_views"]]
    views_to_encode = input_views if fusion_cfg["method"] == "weighted_mean" else [input_views[0]]

    requested_device = str(args.device)
    resolved_device = resolve_device(requested_device, allow_fallback=args.allow_device_fallback)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    run_id = args.run_id or f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    model_dir = sanitize_model_name_for_path(model_name)
    experiment_id = str(exp_cfg["experiment_id"])

    run_output_dir = output_root / model_dir / experiment_id / run_id
    embedding_path = run_output_dir / "item_embeddings.npy"
    item_ids_path = run_output_dir / "item_ids.jsonl"

    view_embedding_paths: dict[str, Path] = {
        view_id: run_output_dir / f"item_embeddings__{view_id}.npy" for view_id in views_to_encode
    }

    ensure_parent_dir(embedding_path)
    ensure_parent_dir(item_ids_path)

    print(f"[embed] run_id={run_id}")
    local_model_ref = resolve_local_model_ref(model_name)
    print(
        f"[embed] loading model={model_name} -> local={local_model_ref} "
        f"on device={resolved_device} (requested={requested_device})"
    )
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            local_model_ref,
            local_files_only=True,
        )
        model = AutoModel.from_pretrained(
            local_model_ref,
            local_files_only=True,
        )
    except FileNotFoundError:
        raise
    except Exception as e:
        hint = (
            "Failed to load model/tokenizer. "
            "This pipeline is local-only. Please ensure model files exist in local cache/path. "
            "If error mentions protobuf, install protobuf in the runtime."
        )
        raise RuntimeError(f"{hint} Root cause: {type(e).__name__}: {e}") from e
    model.eval()
    model.to(resolved_device)

    print("[embed] counting items...")
    total_items = count_items(items_input, args.max_items)
    if total_items == 0:
        raise ValueError("No items found to encode.")
    print(f"[embed] total_items={total_items}")

    embedding_memmap: np.memmap | None = None
    view_memmaps: dict[str, np.memmap] = {}
    embedding_dim: int | None = None
    offset = 0
    total_tokens_processed = 0
    total_encode_seconds = 0.0
    progress_bar_width = 30

    with item_ids_path.open("w", encoding="utf-8") as id_f:
        for batch in batched_items(items_input, batch_size=batch_size, max_items=args.max_items):
            batch_ids = [normalize_text(item["item_id"]) for item in batch]
            batch_view_embeddings: dict[str, torch.Tensor] = {}
            batch_tokens = 0
            batch_encode_seconds = 0.0

            for view_id in views_to_encode:
                view_cfg = view_by_id[view_id]
                texts = [
                    adapt_text_for_model(model_name, render_view_text(item, view_cfg))
                    for item in batch
                ]
                encoded_batch, token_count, elapsed_seconds = encode_text_batch(
                    texts=texts,
                    tokenizer=tokenizer,
                    model=model,
                    device=resolved_device,
                    max_length=max_length,
                    normalize_embeddings=normalize_embeddings,
                )
                batch_view_embeddings[view_id] = encoded_batch
                total_tokens_processed += token_count
                total_encode_seconds += elapsed_seconds
                batch_tokens += token_count
                batch_encode_seconds += elapsed_seconds

            fused = fuse_batch_embeddings(batch_view_embeddings, fusion_cfg=fusion_cfg)
            fused_np = fused.numpy()

            if embedding_memmap is None:
                embedding_dim = int(fused_np.shape[1])
                embedding_memmap = open_memmap(
                    embedding_path,
                    mode="w+",
                    dtype=np.float32,
                    shape=(total_items, embedding_dim),
                )
                if args.save_view_embeddings:
                    for view_id, emb in batch_view_embeddings.items():
                        view_memmaps[view_id] = open_memmap(
                            view_embedding_paths[view_id],
                            mode="w+",
                            dtype=np.float32,
                            shape=(total_items, int(emb.shape[1])),
                        )

            batch_size_actual = fused_np.shape[0]
            if embedding_memmap is None:
                raise RuntimeError("Embedding memmap was not initialized.")
            embedding_memmap[offset : offset + batch_size_actual] = fused_np
            if args.save_view_embeddings:
                for view_id, emb in batch_view_embeddings.items():
                    view_memmaps[view_id][offset : offset + batch_size_actual] = emb.numpy()

            for item_id in batch_ids:
                id_f.write(json.dumps({"item_id": item_id}, ensure_ascii=False) + "\n")

            offset += batch_size_actual
            instant_tps = batch_tokens / max(batch_encode_seconds, 1e-9)
            ratio = offset / total_items
            filled = int(progress_bar_width * ratio)
            if filled >= progress_bar_width:
                bar = "=" * progress_bar_width
            else:
                bar = "=" * filled + ">" + "." * (progress_bar_width - filled - 1)
            print(
                "\r"
                f"[embed] [{bar}] {offset}/{total_items} "
                f"{instant_tps:.2f} tokens/s",
                end="",
                flush=True,
            )

    print()

    if offset != total_items:
        raise RuntimeError(f"Processed rows mismatch: offset={offset}, total_items={total_items}")

    if embedding_memmap is None or embedding_dim is None:
        raise RuntimeError("No embeddings generated.")
    embedding_memmap.flush()
    for mm in view_memmaps.values():
        mm.flush()

    config_hash_payload = {
        "experiment_config": exp_cfg,
        "items_input": str(items_input),
        "seed": args.seed,
        "requested_device": requested_device,
        "resolved_device": resolved_device,
        "save_view_embeddings": args.save_view_embeddings,
        "local_model_ref": local_model_ref,
        "allow_device_fallback": args.allow_device_fallback,
    }
    config_hash = compute_config_hash(config_hash_payload)

    run_snapshot_path = runs_root / run_id / "config.json"
    ensure_parent_dir(run_snapshot_path)
    run_snapshot = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "experiment_id": experiment_id,
        "experiment_config_path": str(config_path),
        "items_input": str(items_input),
        "model": {
            "name": model_name,
            "local_model_ref": local_model_ref,
            "max_length": max_length,
            "batch_size": batch_size,
            "normalize_embeddings": normalize_embeddings,
        },
        "device": {
            "requested": requested_device,
            "resolved": resolved_device,
            "allow_fallback": args.allow_device_fallback,
        },
        "seed": args.seed,
        "fusion": fusion_cfg,
        "views_to_encode": views_to_encode,
        "output": {
            "embedding_path": str(embedding_path),
            "item_ids_path": str(item_ids_path),
            "view_embedding_paths": (
                {k: str(v) for k, v in view_embedding_paths.items()} if args.save_view_embeddings else {}
            ),
            "rows": total_items,
            "dim": embedding_dim,
            "tokens_processed": total_tokens_processed,
            "encode_seconds": total_encode_seconds,
            "tokens_per_second": (
                total_tokens_processed / total_encode_seconds if total_encode_seconds > 0 else 0.0
            ),
        },
        "config_hash": config_hash,
    }
    run_snapshot_path.write_text(
        json.dumps(run_snapshot, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        "[embed] done "
        f"rows={total_items} dim={embedding_dim} "
        f"tokens={total_tokens_processed}"
    )
    print(f"[embed] item_embeddings={embedding_path}")
    print(f"[embed] item_ids={item_ids_path}")
    print(f"[embed] config_snapshot={run_snapshot_path}")


if __name__ == "__main__":
    main()
