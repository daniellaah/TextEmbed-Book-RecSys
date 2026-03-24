from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts/eval/plot_baseline_vs_embedding.py"


class PlotBaselineVsEmbeddingTests(unittest.TestCase):
    def test_generates_plots_and_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            embedding_dir = tmp_path / "embedding_eval"
            baseline_root = tmp_path / "baselines"
            output_dir = tmp_path / "plots"
            embedding_dir.mkdir(parents=True)
            (baseline_root / "global_popular" / "20260324000101").mkdir(parents=True)
            (baseline_root / "random" / "20260324000102").mkdir(parents=True)

            self._write_json(
                embedding_dir / "run_eval_report.json",
                {
                    "config": {
                        "eval_input": "/tmp/eval.jsonl",
                        "topk": [10, 50, 100],
                        "max_query": 0,
                    },
                    "eval_stats": {"valid_eval_rows": 123},
                    "embedding_identity": {"model_name_guess": "Qwen/Qwen3-Embedding-8B"},
                    "index_stats": {"embedding_dim": 1024},
                    "metrics": {
                        "@10": {"recall": 0.1, "mrr": 0.05, "ndcg": 0.06},
                        "@50": {"recall": 0.2, "mrr": 0.06, "ndcg": 0.09},
                        "@100": {"recall": 0.3, "mrr": 0.07, "ndcg": 0.11},
                    },
                },
            )
            self._write_json(
                baseline_root / "global_popular" / "20260324000101" / "report.json",
                {
                    "baseline": "global_popular",
                    "config": {
                        "eval_input": "/tmp/eval.jsonl",
                        "topk": [10, 50, 100],
                        "max_query": 0,
                    },
                    "eval_stats": {"valid_eval_rows": 123},
                    "metrics": {
                        "@10": {"recall": 0.01, "mrr": 0.005, "ndcg": 0.006},
                        "@50": {"recall": 0.02, "mrr": 0.006, "ndcg": 0.009},
                        "@100": {"recall": 0.03, "mrr": 0.007, "ndcg": 0.011},
                    },
                },
            )
            self._write_json(
                baseline_root / "random" / "20260324000102" / "report.json",
                {
                    "baseline": "random",
                    "config": {
                        "eval_input": "/tmp/eval.jsonl",
                        "topk": [10, 50, 100],
                        "max_query": 0,
                    },
                    "eval_stats": {"valid_eval_rows": 123},
                    "metrics": {
                        "@10": {"recall": 0.0, "mrr": 0.0, "ndcg": 0.0},
                        "@50": {"recall": 0.0, "mrr": 0.0, "ndcg": 0.0},
                        "@100": {"recall": 0.0, "mrr": 0.0, "ndcg": 0.0},
                    },
                },
            )

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "--embedding-eval-dir",
                    str(embedding_dir),
                    "--baseline-root",
                    str(baseline_root),
                    "--baselines",
                    "global_popular,random",
                    "--output-dir",
                    str(output_dir),
                ],
                check=True,
                cwd=str(REPO_ROOT),
            )

            self.assertTrue((output_dir / "results.csv").exists())
            self.assertTrue((output_dir / "summary.json").exists())
            self.assertTrue((output_dir / "recall_comparison.png").exists())
            self.assertTrue((output_dir / "mrr_comparison.png").exists())
            self.assertTrue((output_dir / "ndcg_comparison.png").exists())

            with (output_dir / "results.csv").open() as f:
                rows = list(csv.DictReader(f))
            system_names = {row["system_name"] for row in rows}
            self.assertEqual(system_names, {"Qwen/Qwen3-Embedding-8B (1024)", "global_popular", "random"})

    def test_rejects_incompatible_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            embedding_dir = tmp_path / "embedding_eval"
            baseline_root = tmp_path / "baselines"
            output_dir = tmp_path / "plots"
            embedding_dir.mkdir(parents=True)
            (baseline_root / "global_popular" / "20260324000101").mkdir(parents=True)

            self._write_json(
                embedding_dir / "run_eval_report.json",
                {
                    "config": {
                        "eval_input": "/tmp/eval_a.jsonl",
                        "topk": [10, 50, 100],
                        "max_query": 0,
                    },
                    "eval_stats": {"valid_eval_rows": 123},
                    "embedding_identity": {"model_name_guess": "Qwen/Qwen3-Embedding-8B"},
                    "index_stats": {"embedding_dim": 1024},
                    "metrics": {
                        "@10": {"recall": 0.1, "mrr": 0.05, "ndcg": 0.06},
                        "@50": {"recall": 0.2, "mrr": 0.06, "ndcg": 0.09},
                        "@100": {"recall": 0.3, "mrr": 0.07, "ndcg": 0.11},
                    },
                },
            )
            self._write_json(
                baseline_root / "global_popular" / "20260324000101" / "report.json",
                {
                    "baseline": "global_popular",
                    "config": {
                        "eval_input": "/tmp/eval_b.jsonl",
                        "topk": [10, 50, 100],
                        "max_query": 0,
                    },
                    "eval_stats": {"valid_eval_rows": 123},
                    "metrics": {
                        "@10": {"recall": 0.01, "mrr": 0.005, "ndcg": 0.006},
                        "@50": {"recall": 0.02, "mrr": 0.006, "ndcg": 0.009},
                        "@100": {"recall": 0.03, "mrr": 0.007, "ndcg": 0.011},
                    },
                },
            )

            with self.assertRaises(subprocess.CalledProcessError):
                subprocess.run(
                    [
                        sys.executable,
                        str(SCRIPT_PATH),
                        "--embedding-eval-dir",
                        str(embedding_dir),
                        "--baseline-root",
                        str(baseline_root),
                        "--baselines",
                        "global_popular",
                        "--output-dir",
                        str(output_dir),
                    ],
                    check=True,
                    cwd=str(REPO_ROOT),
                )

    def _write_json(self, path: Path, payload: dict) -> None:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
