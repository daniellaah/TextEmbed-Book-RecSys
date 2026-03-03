# TextEmbed-Book-RecSys

## Data Source Format

This project uses Amazon Reviews 2023 Books raw files in **uncompressed** `.jsonl` format:

- `data/raw/meta_Books.jsonl`
- `data/raw/Books.jsonl`

Do not use `.jsonl.gz` paths in this repo workflow.

## Download (Example)

```bash
mkdir -p data/raw
curl -L --fail "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/meta_categories/meta_Books.jsonl" -o data/raw/meta_Books.jsonl
curl -L --fail "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/review_categories/Books.jsonl" -o data/raw/Books.jsonl
```
