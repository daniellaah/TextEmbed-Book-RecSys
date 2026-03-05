# TextEmbed-Book-RecSys

Offline evaluation pipeline for text-embedding-based book recommendation on Amazon Reviews 2023 (Books).

Current implemented stages:

- Data preprocessing (`items.jsonl`, `interactions.jsonl`, `eval.jsonl`)
- Item embedding generation (local-only model loading)
- ANN neighbor inspection tool
- Offline retrieval evaluation (`run_eval.py`)
- Unit tests for data and embedding core logic

## Table Of Contents

- [Project Scope](#project-scope)
- [Repository Layout](#repository-layout)
- [Environment Setup](#environment-setup)
- [Data Preparation](#data-preparation)
- [Pipeline Quickstart](#pipeline-quickstart)
- [Experiment Configs](#experiment-configs)
- [Output Artifacts](#output-artifacts)
- [Run Tests](#run-tests)
- [Troubleshooting](#troubleshooting)

## Project Scope

Primary task:

- Item-to-Item semantic retrieval

Primary metrics:

- Recall@10, Recall@50
- NDCG@10, NDCG@50
- MRR@10, MRR@50

For detailed protocol and conventions, see:

- `docs/dev_guide.md`

## Repository Layout

```text
configs/experiments/tac/       # TAC embedding experiment configs
docs/dev_guide.md              # development protocol
reports/data_profile/          # data build reports
src/data/                      # build_items / build_interactions / build_eval
src/embedding/                 # embedding generator
src/retrieval/                 # ANN utilities and neighbor review tool
src/eval/                      # retrieval evaluation runner
tests/                         # unit tests
```

## Environment Setup

Python and package manager:

- Python 3.10
- uv

Setup:

```bash
uv python install 3.10
uv venv --python 3.10
uv sync --extra dev
export UV_CACHE_DIR=.uv-cache
```

Run commands with `uv`:

```bash
UV_CACHE_DIR=.uv-cache uv run python <script>.py ...
```

## Data Preparation

This repo expects uncompressed JSONL inputs:

- `data/raw/meta_Books.jsonl`
- `data/raw/Books.jsonl`

Example download:

```bash
mkdir -p data/raw
curl -L --fail "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/meta_categories/meta_Books.jsonl" -o data/raw/meta_Books.jsonl
curl -L --fail "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/review_categories/Books.jsonl" -o data/raw/Books.jsonl
```

## Pipeline Quickstart

### 1) Build items

```bash
UV_CACHE_DIR=.uv-cache uv run python src/data/build_items.py \
  --input data/raw/meta_Books.jsonl \
  --output data/processed/items.jsonl \
  --report reports/data_profile/build_items_report.json
```

### 2) Build cleaned interactions

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

### 3.5) Build smaller items subset from eval (optional, for faster embedding iteration)

```bash
UV_CACHE_DIR=.uv-cache uv run python src/data/build_items_subset_from_eval.py \
  --eval-input data/processed/eval.jsonl \
  --items-input data/processed/items.jsonl \
  --output data/processed/items_eval_subset.jsonl \
  --report reports/data_profile/build_items_subset_from_eval_report.json
```

Use `data/processed/items_eval_subset.jsonl` as `--items-input` in embedding runs for fast dev iteration.

### 4) Generate item embeddings

```bash
UV_CACHE_DIR=.uv-cache uv run python src/embedding/generate_item_embeddings.py \
  --experiment-config configs/experiments/tac/exp_bge_tac.yaml \
  --items-input data/processed/items.jsonl \
  --device mps \
  --allow-device-fallback \
  --batch-size 64
```

Important constraints:

- `model.name` in config must be Hugging Face repo-id format (`namespace/model`).
- Model loading is local-only.
- Model files must already exist under `~/.cache/huggingface/hub`.

### 5) Review ANN neighbors (optional)

```bash
UV_CACHE_DIR=.uv-cache uv run python src/retrieval/review_item_neighbors.py \
  --run-output-dir outputs/embeddings/BAAI__bge-m3/exp_bge_tac/<run_id> \
  --items-input data/processed/items.jsonl \
  --top-k 5 \
  --index-type hnsw
```

### 6) Run retrieval evaluation

```bash
UV_CACHE_DIR=.uv-cache uv run python src/eval/run_eval.py \
  --eval-input data/processed/eval.jsonl \
  --embedding-dir outputs/embeddings/BAAI__bge-m3/exp_bge_tac/<run_id> \
  --output-root outputs/eval \
  --max-query 200 \
  --topk 10,50 \
  --index-type hnsw
```

Evaluation CLI notes:

- `--embedding-dir` points to one embedding run directory and must contain:
  - `item_embeddings.npy`
  - `item_ids.jsonl`
- `--eval-run-id` is optional; if omitted it uses local timestamp `YYYYMMDDHHMMSS`.
- `--max-query` is optional:
  - `0` means evaluate all valid queries
  - `>0` means evaluate only first N valid queries (recommended for smoke tests)
- `--index-type`:
  - `hnsw` recommended for full eval on large datasets
  - `flat` is exact but much slower at current scale

Output layout per run:

- `outputs/eval/<eval_run_id>/predictions.jsonl`
  - per-query topK retrieval result with `target_rank`
- `outputs/eval/<eval_run_id>/run_eval_report.json`
  - aggregate metrics and filtering stats
- `outputs/eval/<eval_run_id>/info.json`
  - run metadata (eval input hash, embedding source, retrieval params, output paths)

Example full eval command (all valid queries):

```bash
UV_CACHE_DIR=.uv-cache uv run python src/eval/run_eval.py \
  --eval-input data/processed/eval.jsonl \
  --embedding-dir outputs/embeddings/BAAI__bge-m3/exp_bge_tac/<run_id> \
  --output-root outputs/eval \
  --topk 10,50 \
  --index-type hnsw
```

## Experiment Configs

Current TAC configs in `configs/experiments/tac/`:

- `exp_bge_tac.yaml`
- `exp_qwen3_0_6b_tac.yaml`
- `exp_qwen3_4b_tac.yaml`
- `exp_qwen3_8b_tac.yaml`
- `exp_nvidia_llama_embed_nemotron_8b_tac.yaml`
- `exp_multilingual_e5_large_instruct_tac.yaml`

Common schema:

```yaml
experiment_id: exp_xxx
model:
  name: BAAI/bge-m3
  embedding_dim: 1024
  max_length: 512
  normalize_embeddings: true
text_views:
  views:
    - view_id: view_tac
      fields: [title, author, categories]
      template: |
        Title: {title}
        Author: {author}
        Categories: {categories}
fusion:
  method: identity
  input_views: [view_tac]
  normalization: false
```

## Output Artifacts

Main outputs:

- `data/processed/items.jsonl`
- `data/processed/interactions.jsonl`
- `data/processed/eval.jsonl`
- `reports/data_profile/build_items_report.json`
- `reports/data_profile/build_interactions_report.json`
- `reports/data_profile/build_eval_report.json`
- `reports/data_profile/build_items_subset_from_eval_report.json`
- `outputs/embeddings/<model>/<experiment_id>/<run_id>/item_embeddings.npy`
- `outputs/embeddings/<model>/<experiment_id>/<run_id>/item_ids.jsonl`
- `outputs/eval/<eval_run_id>/predictions.jsonl`
- `outputs/eval/<eval_run_id>/run_eval_report.json`
- `outputs/eval/<eval_run_id>/info.json`
- `outputs/runs/<run_id>/config.json`

Notes:

- `data/` and `outputs/` are git-ignored by default.
- `run_id` is generated automatically by the embedding script.

## Run Tests

Run all core unit tests:

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

Model not found:

- Error usually means required files are missing from local Hugging Face cache.
- Pre-download once, then re-run:

```bash
UV_CACHE_DIR=.uv-cache uv run python - <<'PY'
from transformers import AutoTokenizer, AutoModel
AutoTokenizer.from_pretrained("BAAI/bge-m3")
AutoModel.from_pretrained("BAAI/bge-m3")
PY
```

Device issues:

- Use `--allow-device-fallback` to fallback from unavailable `mps/cuda` to `cpu`.

Memory issues:

- Reduce `--batch-size`.
- Use `--max-items` for smoke runs.
