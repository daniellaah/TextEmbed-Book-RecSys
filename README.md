# TextEmbed-Book-RecSys

Offline evaluation pipeline for embedding-based **semantic retrieval** on Amazon Reviews 2023 (Books).

## Overview

This repository provides an end-to-end offline workflow:

1. Build cleaned item metadata (`items.jsonl`)
2. Build cleaned interactions (`interactions.jsonl`)
3. Build eval queries (`eval.jsonl`)
4. Generate item embeddings from experiment config
5. Evaluate retrieval metrics (Recall/NDCG/MRR)

Primary metrics:

- Recall@10, Recall@50
- NDCG@10, NDCG@50
- MRR@10, MRR@50

Detailed protocol lives in [`docs/dev_guide.md`](docs/dev_guide.md).

## Repository Layout

```text
configs/
  experiments/
    tac/                      # TAC-style single-view experiments
    other/                    # concat / weighted multi-view experiments
docs/
  dev_guide.md
src/
  data/
    build_items.py
    build_interactions.py
    build_eval.py
    build_items_subset_from_eval.py
  embedding/
    generate_item_embeddings.py
  retrieval/
    ann_utils.py
    review_item_neighbors.py
  eval/
    run_eval.py
tests/
```

## Requirements

- Python 3.10
- `uv`
- Local Hugging Face model cache (embedding is local-only by design)

## Environment Setup

```bash
uv python install 3.10
uv venv --python 3.10

# Install runtime dependencies (no pyproject.toml required)
UV_CACHE_DIR=.uv-cache uv pip install --python .venv/bin/python \
  numpy torch transformers faiss-cpu pyyaml
```

Run all scripts via `uv`:

```bash
UV_CACHE_DIR=.uv-cache uv run python <script>.py ...
```

## Data

Expected raw files are **uncompressed** JSONL:

- `data/raw/meta_Books.jsonl`
- `data/raw/Books.jsonl`

Download example:

```bash
mkdir -p data/raw
curl -L --fail "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/meta_categories/meta_Books.jsonl" -o data/raw/meta_Books.jsonl
curl -L --fail "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/review_categories/Books.jsonl" -o data/raw/Books.jsonl
```

## Quickstart

### 1) Build items

```bash
UV_CACHE_DIR=.uv-cache uv run python src/data/build_items.py \
  --input data/raw/meta_Books.jsonl \
  --output data/processed/items.jsonl \
  --report reports/data_profile/build_items_report.json
```

### 2) Build interactions

```bash
UV_CACHE_DIR=.uv-cache uv run python src/data/build_interactions.py \
  --books-input data/raw/Books.jsonl \
  --items-input data/processed/items.jsonl \
  --output data/processed/interactions.jsonl \
  --report reports/data_profile/build_interactions_report.json \
  --seed 42
```

### 3) Build eval set

```bash
UV_CACHE_DIR=.uv-cache uv run python src/data/build_eval.py \
  --interactions-input data/processed/interactions.jsonl \
  --queries-output data/processed/eval.jsonl \
  --report-output reports/data_profile/build_eval_report.json \
  --rating-threshold 4.0 \
  --min-user-pos 1 \
  --min-item-pos 1 \
  --query-history-n 1 \
  --seed 42
```

### 4) (Optional) Build smaller items subset from eval

```bash
UV_CACHE_DIR=.uv-cache uv run python src/data/build_items_subset_from_eval.py \
  --eval-input data/processed/eval.jsonl \
  --items-input data/processed/items.jsonl \
  --output data/processed/items_eval_subset.jsonl \
  --report reports/data_profile/build_items_subset_from_eval_report.json
```

Use subset for faster embedding iteration by passing it to `--items-input` in step 5.

### 5) Generate embeddings

```bash
UV_CACHE_DIR=.uv-cache uv run python src/embedding/generate_item_embeddings.py \
  --experiment-config configs/experiments/tac/exp_bge_tac.yaml \
  --items-input data/processed/items.jsonl \
  --device mps \
  --allow-device-fallback \
  --batch-size 64
```

### 6) Run retrieval evaluation

```bash
UV_CACHE_DIR=.uv-cache uv run python src/eval/run_eval.py \
  --eval-input data/processed/eval.jsonl \
  --embedding-dir outputs/embeddings/BAAI__bge-m3/exp_bge_tac/<run_id> \
  --output-root outputs/eval \
  --topk 10,50 \
  --index-type hnsw
```

## Experiment Configs

Config files are under:

- `configs/experiments/tac/*.yaml`
- `configs/experiments/other/*.yaml`

### Config schema

```yaml
experiment_id: exp_bge_weighted_v1

model:
  name: BAAI/bge-m3
  embedding_dim: 1024
  max_length: 512
  normalize_embeddings: true
  # trust_remote_code: false

text_views:
  views:
    - view_id: view_title
      fields: [title, subtitle, author]
      template: |
        Title: {title}
        Subtitle: {subtitle}
        Author: {author}

fusion:
  method: weighted_mean   # identity | weighted_mean
  input_views: [view_title]
  weights:
    view_title: 1.0
  normalization: true     # controls post-fusion normalization only
```

### Config field reference

| Field | Type | Required | Description |
|---|---|---|---|
| `experiment_id` | string | yes | Unique experiment identifier, used in output path. |
| `model.name` | string | yes | Hugging Face repo id (`namespace/model`). |
| `model.embedding_dim` | int | yes | Final embedding dimension (must be <= model output dim). |
| `model.max_length` | int | yes | Token truncation length. |
| `model.normalize_embeddings` | bool | yes | Normalize each model output embedding before fusion. |
| `model.trust_remote_code` | bool | no | Needed for models that require custom code. |
| `text_views.views[].view_id` | string | yes | View identifier. Must be unique. |
| `text_views.views[].fields` | list[string] | yes | Item fields used to fill template. |
| `text_views.views[].template` | string | yes | Render template for the view. |
| `fusion.method` | enum | yes | `identity` or `weighted_mean`. |
| `fusion.input_views` | list[string] | yes | View ids to fuse. Must exist in `text_views.views`. |
| `fusion.weights` | map | conditional | Required when `fusion.method=weighted_mean`. |
| `fusion.normalization` | bool | yes | Whether to normalize after fusion. |

Notes:

- `identity` requires exactly one `fusion.input_views` entry.
- Canonical `fusion.method` values are strictly: `identity`, `weighted_mean`.

## CLI Reference

### `src/data/build_items.py`

| Arg | Default | Description |
|---|---|---|
| `--input` | `data/raw/meta_Books.jsonl` | Raw metadata input JSONL. |
| `--output` | `data/processed/items.jsonl` | Cleaned items output JSONL. |
| `--report` | `reports/data_profile/build_items_report.json` | Build report output path. |
| `--tmp-db` | `data/processed/.tmp_build_items.sqlite3` | Temporary sqlite for dedup / merge logic. |

### `src/data/build_interactions.py`

| Arg | Default | Description |
|---|---|---|
| `--books-input` | `data/raw/Books.jsonl` | Raw interactions input JSONL. |
| `--items-input` | `data/processed/items.jsonl` | Filter interactions to valid item ids from this file. |
| `--output` | `data/processed/interactions.jsonl` | Cleaned interactions output JSONL. |
| `--report` | `reports/data_profile/build_interactions_report.json` | Build report output path. |
| `--seed` | `42` | Protocol metadata seed (reserved for deterministic config tracking). |

### `src/data/build_eval.py`

| Arg | Default | Description |
|---|---|---|
| `--interactions-input` | `data/processed/interactions.jsonl` | Input interactions. |
| `--queries-output` | `data/processed/eval.jsonl` | Output eval query set. |
| `--report-output` | `reports/data_profile/build_eval_report.json` | Build report output path. |
| `--rating-threshold` | `4.0` | Positive sample rule: `rating >= threshold`. |
| `--min-user-pos` | `1` | Minimum positives per user after filtering. |
| `--min-item-pos` | `1` | Minimum positives per item after filtering. |
| `--query-history-n` | `1` | Number of historical positives used as query context. |
| `--seed` | `42` | Protocol metadata seed. |

### `src/data/build_items_subset_from_eval.py`

| Arg | Default | Description |
|---|---|---|
| `--eval-input` | `data/processed/eval.jsonl` | Eval set containing query/target ids. |
| `--items-input` | `data/processed/items.jsonl` | Full items file. |
| `--output` | `data/processed/items_eval_subset.jsonl` | Reduced items file. |
| `--report` | `reports/data_profile/build_items_subset_from_eval_report.json` | Build report output path. |

### `src/embedding/generate_item_embeddings.py`

| Arg | Default | Description |
|---|---|---|
| `--experiment-config` | required | Experiment YAML path. |
| `--items-input` | `data/processed/items.jsonl` | Items input used for rendering views. |
| `--output-root` | `outputs/embeddings` | Embedding artifact root. |
| `--runs-root` | `outputs/runs` | Run snapshot root (`config.json`). |
| `--device` | `mps` | Requested device (`mps`/`cuda`/`cpu`). |
| `--allow-device-fallback` | false | Fallback to CPU if requested device unavailable. |
| `--seed` | `42` | RNG seed. |
| `--batch-size` | `64` | Encoding batch size. |
| `--save-view-embeddings` | false | Also save per-view embeddings (`item_embeddings__<view>.npy`). |
| `--max-items` | none | Debug cap on number of items to encode. |

Important behavior:

- Local-only model loading (`local_files_only=True`); model must already exist in `~/.cache/huggingface/hub`.
- Output path: `outputs/embeddings/<model_dir>/<experiment_id>/<run_id>/...`
  - `model_dir` is `model.name` with `/` replaced by `__`.

### `src/eval/run_eval.py`

| Arg | Default | Description |
|---|---|---|
| `--eval-input` | `data/processed/eval.jsonl` | Eval query set input. |
| `--embedding-dir` | required | Embedding run dir containing `item_embeddings.npy` and `item_ids.jsonl`. |
| `--output-root` | `outputs/eval` | Output root; writes into `<output_root>/<eval_run_id>/`. |
| `--eval-run-id` | timestamp | Optional eval run id (`YYYYMMDDHHMMSS` if omitted). |
| `--max-query` | `0` | `0` = all valid queries; `>0` = first N valid queries. |
| `--topk` | `10,50` | Comma-separated K list. |
| `--index-type` | `flat` | `flat` or `hnsw`. |
| `--hnsw-m` | `32` | HNSW M (when `index-type=hnsw`). |
| `--hnsw-ef-search` | `64` | HNSW efSearch (when `index-type=hnsw`). |
| `--hnsw-ef-construction` | `200` | HNSW efConstruction (when `index-type=hnsw`). |
| `--seed` | `42` | RNG seed for deterministic metadata/runtime behavior. |

### `src/retrieval/review_item_neighbors.py`

| Arg | Default | Description |
|---|---|---|
| `--run-output-dir` | none | Embedding run dir (auto resolves embeddings + ids paths). |
| `--embeddings-path` | none | Explicit path to `item_embeddings.npy` (if not using run dir). |
| `--item-ids-path` | none | Explicit path to `item_ids.jsonl` (if not using run dir). |
| `--items-input` | `data/processed/items.jsonl` | Item metadata for readable output. |
| `--query-item-id` | none | Query item id. |
| `--random-query` | false | Sample query item id randomly. |
| `--seed` | `42` | RNG seed for random query. |
| `--top-k` | `5` | Number of neighbors shown. |
| `--index-type` | `hnsw` | `hnsw` or `flat`. |
| `--hnsw-m` | `32` | HNSW M. |
| `--hnsw-ef-search` | `128` | HNSW efSearch. |
| `--hnsw-ef-construction` | `200` | HNSW efConstruction. |
| `--text-fields` | `title,author,categories` | Fields shown in text summary. |
| `--no-normalize` | false | Disable L2 normalization before search. |

## Outputs

### Data stage

- `data/processed/items.jsonl`
- `data/processed/interactions.jsonl`
- `data/processed/eval.jsonl`
- `reports/data_profile/*.json`

### Embedding stage

- `outputs/embeddings/<model_dir>/<experiment_id>/<run_id>/item_embeddings.npy`
- `outputs/embeddings/<model_dir>/<experiment_id>/<run_id>/item_ids.jsonl`
- Optional: `item_embeddings__<view_id>.npy` when `--save-view-embeddings` is enabled.
- `outputs/runs/<run_id>/config.json` (run snapshot + config hash)

### Eval stage

- `outputs/eval/<eval_run_id>/predictions.jsonl`
- `outputs/eval/<eval_run_id>/run_eval_report.json`
- `outputs/eval/<eval_run_id>/info.json`

## Run Tests

```bash
UV_CACHE_DIR=.uv-cache uv run python -m unittest \
  tests/test_build_interactions.py \
  tests/test_build_eval.py \
  tests/test_build_items_subset_from_eval.py \
  tests/test_generate_item_embeddings.py \
  tests/test_ann_utils.py \
  tests/test_run_eval.py
```

## Troubleshooting

### Model not found in cache

This pipeline does not auto-download model files during embedding. Pre-download first:

```bash
UV_CACHE_DIR=.uv-cache uv run python - <<'PY'
from transformers import AutoTokenizer, AutoModel
AutoTokenizer.from_pretrained("BAAI/bge-m3")
AutoModel.from_pretrained("BAAI/bge-m3")
PY
```

### Device issues

- Add `--allow-device-fallback` to fallback from unavailable `mps/cuda` to CPU.

### Memory / speed issues

- Reduce `--batch-size`.
- Use `--max-items` for embedding smoke tests.
- Use `--max-query` for eval smoke tests.
- Prefer `--index-type hnsw` for large-scale eval speed.
