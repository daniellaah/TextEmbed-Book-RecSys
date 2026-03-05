#!/usr/bin/env python3
"""Generate item embeddings from items.jsonl and experiment config."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from numpy.lib.format import open_memmap
from transformers import AutoModel, AutoTokenizer


MODEL_REPO_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")


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
    has_single_weights = (path / "pytorch_model.bin").exists() or (path / "model.safetensors").exists()
    has_sharded_safetensors = (
        (path / "model.safetensors.index.json").exists()
        and any(path.glob("model-*.safetensors"))
    )
    has_sharded_pytorch = (
        (path / "pytorch_model.bin.index.json").exists()
        and any(path.glob("pytorch_model-*.bin"))
    )
    has_weights = has_single_weights or has_sharded_safetensors or has_sharded_pytorch
    has_tokenizer = (
        (path / "tokenizer.json").exists()
        or (path / "tokenizer_config.json").exists()
        or (path / "sentencepiece.bpe.model").exists()
    )
    return has_config and has_weights and has_tokenizer


def get_hf_cache_root() -> Path:
    return Path.home() / ".cache" / "huggingface" / "hub"


def validate_model_name(model_name: str) -> None:
    if model_name.startswith(("/", "./", "../", "~")):
        raise ValueError(
            "model.name must be a Hugging Face repo id in '<namespace>/<model>' format; "
            "path-like values are not allowed."
        )
    if not MODEL_REPO_ID_RE.fullmatch(model_name):
        raise ValueError(
            "model.name must match '^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$'."
        )


def resolve_local_model_ref(model_name: str) -> str:
    """Resolve to a local model directory only; never trigger remote download."""
    validate_model_name(model_name)
    namespace, model = model_name.split("/", 1)
    cache_root = get_hf_cache_root()
    repo_cache_dir = cache_root / f"models--{namespace}--{model}"
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
        f"Local model snapshot not found for repo id '{model_name}'.\n"
        f"Expected cache root: {cache_root}\n"
        f"Expected snapshots dir: {snapshots_dir}\n"
        "Please pre-download this model to local Hugging Face cache, then re-run.\n"
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
    model_name = normalize_text(model_cfg.get("name"))
    if not model_name:
        raise ValueError("Missing 'model.name' in config.")
    validate_model_name(model_name)
    embedding_dims_raw = model_cfg.get("embedding_dim")
    if not isinstance(embedding_dims_raw, list) or not embedding_dims_raw:
        raise ValueError("'model.embedding_dim' must be a non-empty list of positive integers.")
    embedding_dims: list[int] = []
    for idx, value in enumerate(embedding_dims_raw):
        dim = int(value)
        if dim <= 0:
            raise ValueError(f"'model.embedding_dim[{idx}]' must be > 0.")
        embedding_dims.append(dim)
    if sorted(set(embedding_dims)) != embedding_dims:
        raise ValueError("'model.embedding_dim' must be strictly increasing with no duplicates.")
    if int(model_cfg.get("max_length", 0)) <= 0:
        raise ValueError("'model.max_length' must be > 0.")
    trust_remote_code = model_cfg.get("trust_remote_code", False)
    if not isinstance(trust_remote_code, bool):
        raise ValueError("'model.trust_remote_code' must be a boolean when provided.")
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


def synchronize_device_for_timing(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
        return
    if device == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def apply_embedding_dim(pooled: torch.Tensor, embedding_dim: int) -> torch.Tensor:
    current_dim = int(pooled.shape[1])
    if current_dim < embedding_dim:
        raise ValueError(
            f"Configured model.embedding_dim={embedding_dim} exceeds model output dim={current_dim}."
        )
    if current_dim == embedding_dim:
        return pooled
    return pooled[:, :embedding_dim]


def encode_text_batch(
    texts: list[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: str,
    max_length: int,
    max_embedding_dim: int,
) -> tuple[torch.Tensor, int, float]:
    synchronize_device_for_timing(device)
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
        pooled = apply_embedding_dim(pooled, embedding_dim=max_embedding_dim)
    synchronize_device_for_timing(device)
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
        "--batch-size",
        type=int,
        default=64,
        help="Encoding batch size.",
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
    if args.batch_size <= 0:
        parser.error("--batch-size must be > 0.")
    return args


def generate_run_id_local() -> str:
    now_local = datetime.now()
    # Include milliseconds to reduce collision probability in model/<run_id> layout.
    return now_local.strftime("%Y%m%d%H%M%S%f")[:-3]


def format_eta(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    total = int(seconds)
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def main() -> None:
    args = parse_args()
    items_input = Path(args.items_input)
    config_path = Path(args.experiment_config)
    output_root = Path(args.output_root)

    exp_cfg = parse_experiment_config(config_path)
    validate_experiment_config(exp_cfg)

    model_cfg: dict[str, Any] = exp_cfg["model"]
    model_name = str(model_cfg["name"])
    embedding_dims_config = [int(x) for x in model_cfg["embedding_dim"]]
    max_embedding_dim_config = max(embedding_dims_config)
    max_length = int(model_cfg["max_length"])
    trust_remote_code = bool(model_cfg.get("trust_remote_code", False))
    batch_size = int(args.batch_size)
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

    run_id = generate_run_id_local()
    model_dir = sanitize_model_name_for_path(model_name)
    experiment_id = str(exp_cfg["experiment_id"])

    run_output_dir = output_root / model_dir / run_id
    embedding_paths: dict[int, Path] = {
        dim: run_output_dir / f"item_embeddings_{dim}.npy" for dim in embedding_dims_config
    }
    item_ids_path = run_output_dir / "item_ids.jsonl"

    view_embedding_paths: dict[int, dict[str, Path]] = {
        dim: {
            view_id: run_output_dir / f"item_embeddings__{view_id}_{dim}.npy"
            for view_id in views_to_encode
        }
        for dim in embedding_dims_config
    }

    for path in embedding_paths.values():
        ensure_parent_dir(path)
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
            trust_remote_code=trust_remote_code,
        )
        model = AutoModel.from_pretrained(
            local_model_ref,
            local_files_only=True,
            trust_remote_code=trust_remote_code,
        )
    except FileNotFoundError:
        raise
    except Exception as e:
        hint = (
            "Failed to load model/tokenizer. "
            "This pipeline is local-only. Please ensure model files exist in ~/.cache/huggingface/hub. "
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

    embedding_memmaps: dict[int, np.memmap] = {}
    view_memmaps: dict[int, dict[str, np.memmap]] = {}
    offset = 0
    total_tokens_processed = 0
    total_encode_seconds = 0.0
    progress_bar_width = 30
    progress_last_print = 0.0
    progress_print_interval_seconds = 0.5

    with item_ids_path.open("w", encoding="utf-8") as id_f:
        for batch in batched_items(items_input, batch_size=batch_size, max_items=args.max_items):
            batch_ids = [normalize_text(item["item_id"]) for item in batch]
            batch_view_embeddings_max_dim: dict[str, torch.Tensor] = {}
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
                    max_embedding_dim=max_embedding_dim_config,
                )
                batch_view_embeddings_max_dim[view_id] = encoded_batch
                total_tokens_processed += token_count
                total_encode_seconds += elapsed_seconds
                batch_tokens += token_count
                batch_encode_seconds += elapsed_seconds

            batch_embeddings_by_dim: dict[int, np.ndarray] = {}
            for dim in embedding_dims_config:
                per_view_dim_embeddings: dict[str, torch.Tensor] = {}
                for view_id in views_to_encode:
                    emb = apply_embedding_dim(batch_view_embeddings_max_dim[view_id], embedding_dim=dim)
                    if normalize_embeddings:
                        emb = F.normalize(emb, p=2, dim=1)
                    per_view_dim_embeddings[view_id] = emb
                fused_dim = fuse_batch_embeddings(per_view_dim_embeddings, fusion_cfg=fusion_cfg)
                fused_dim_np = fused_dim.numpy()
                if int(fused_dim_np.shape[1]) != dim:
                    raise ValueError(
                        f"Output dim mismatch for dim={dim}, got fused dim={int(fused_dim_np.shape[1])}."
                    )
                batch_embeddings_by_dim[dim] = fused_dim_np

                if dim not in embedding_memmaps:
                    embedding_memmaps[dim] = open_memmap(
                        embedding_paths[dim],
                        mode="w+",
                        dtype=np.float32,
                        shape=(total_items, dim),
                    )
                if args.save_view_embeddings and dim not in view_memmaps:
                    view_memmaps[dim] = {}
                    for view_id in views_to_encode:
                        view_memmaps[dim][view_id] = open_memmap(
                            view_embedding_paths[dim][view_id],
                            mode="w+",
                            dtype=np.float32,
                            shape=(total_items, dim),
                        )
                if args.save_view_embeddings:
                    for view_id, emb in per_view_dim_embeddings.items():
                        view_memmaps[dim][view_id][offset : offset + emb.shape[0]] = emb.numpy()

            first_dim = embedding_dims_config[0]
            batch_size_actual = int(batch_embeddings_by_dim[first_dim].shape[0])
            for dim in embedding_dims_config:
                embedding_memmaps[dim][offset : offset + batch_size_actual] = batch_embeddings_by_dim[dim]

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
            now = time.perf_counter()
            should_print = offset == total_items or (now - progress_last_print) >= progress_print_interval_seconds
            if should_print:
                remaining_items = max(total_items - offset, 0)
                items_per_second = batch_size_actual / max(batch_encode_seconds, 1e-9)
                eta_seconds = remaining_items / max(items_per_second, 1e-9)
                print(
                    "\r"
                    f"[embed] [{bar}] {offset}/{total_items} "
                    f"{instant_tps:.2f} token/s "
                    f"ETA {format_eta(eta_seconds)}",
                    end="",
                    flush=True,
                )
                progress_last_print = now

    print()

    if offset != total_items:
        raise RuntimeError(f"Processed rows mismatch: offset={offset}, total_items={total_items}")

    if not embedding_memmaps:
        raise RuntimeError("No embeddings generated.")
    for mm in embedding_memmaps.values():
        mm.flush()
    for per_dim_mm in view_memmaps.values():
        for mm in per_dim_mm.values():
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
        "batch_size": batch_size,
        "embedding_dim": embedding_dims_config,
        "trust_remote_code": trust_remote_code,
    }
    config_hash = compute_config_hash(config_hash_payload)

    run_snapshot_path = run_output_dir / "config.json"
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
            "embedding_dim": embedding_dims_config,
            "trust_remote_code": trust_remote_code,
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
            "embedding_paths": {str(dim): str(path) for dim, path in embedding_paths.items()},
            "item_ids_path": str(item_ids_path),
            "view_embedding_paths": (
                (
                    {
                        str(dim): {view_id: str(path) for view_id, path in per_dim_paths.items()}
                        for dim, per_dim_paths in view_embedding_paths.items()
                    }
                )
                if args.save_view_embeddings
                else {}
            ),
            "rows": total_items,
            "dims": embedding_dims_config,
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
        f"rows={total_items} dims={embedding_dims_config} "
        f"tokens={total_tokens_processed}"
    )
    for dim in embedding_dims_config:
        print(f"[embed] item_embeddings_{dim}={embedding_paths[dim]}")
    print(f"[embed] item_ids={item_ids_path}")
    print(f"[embed] config_snapshot={run_snapshot_path}")


if __name__ == "__main__":
    main()
