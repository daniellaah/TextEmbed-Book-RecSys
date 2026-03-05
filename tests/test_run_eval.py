from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.eval.run_eval import build_query_recency_weights, merge_predictions_rrf


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "src/eval/run_eval.py"


class RunEvalTests(unittest.TestCase):
    def test_build_query_recency_weights(self) -> None:
        weights_none = build_query_recency_weights(3, "none", 1.0)
        self.assertAlmostEqual(sum(weights_none), 1.0)
        self.assertEqual(weights_none, [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])

        weights_linear = build_query_recency_weights(3, "linear", 1.0)
        self.assertAlmostEqual(sum(weights_linear), 1.0)
        self.assertTrue(weights_linear[0] < weights_linear[1] < weights_linear[2])

        weights_exp = build_query_recency_weights(3, "exp", 1.0)
        self.assertAlmostEqual(sum(weights_exp), 1.0)
        self.assertTrue(weights_exp[0] < weights_exp[1] < weights_exp[2])

    def test_merge_predictions_rrf_respects_recency_weights(self) -> None:
        per_query_predictions = [
            [
                {"rank": 1, "item_id": "A", "score": 0.9},
                {"rank": 2, "item_id": "B", "score": 0.8},
            ],
            [
                {"rank": 1, "item_id": "B", "score": 0.95},
                {"rank": 2, "item_id": "C", "score": 0.85},
            ],
        ]
        no_recency = merge_predictions_rrf(
            per_query_predictions=per_query_predictions,
            query_weights=build_query_recency_weights(2, "none", 1.0),
            rrf_k=1,
        )
        linear_recency = merge_predictions_rrf(
            per_query_predictions=per_query_predictions,
            query_weights=build_query_recency_weights(2, "linear", 1.0),
            rrf_k=1,
        )
        self.assertEqual(no_recency[0]["item_id"], "B")
        self.assertEqual(no_recency[1]["item_id"], "A")
        self.assertEqual(linear_recency[0]["item_id"], "B")
        self.assertEqual(linear_recency[1]["item_id"], "C")

    def test_run_eval_stable_and_no_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            embedding_dir = tmp_path / "emb" / "BAAI__bge-m3" / "20260304000000"
            embedding_dir.mkdir(parents=True, exist_ok=True)
            item_ids_path = embedding_dir / "item_ids.jsonl"
            embeddings_path = embedding_dir / "item_embeddings.npy"

            eval_input_path = tmp_path / "eval.jsonl"
            output_root = tmp_path / "outputs" / "eval"

            item_ids = ["A", "B", "C", "D"]
            with item_ids_path.open("w", encoding="utf-8") as f:
                for item_id in item_ids:
                    f.write(json.dumps({"item_id": item_id}, ensure_ascii=False) + "\n")

            embeddings = np.array(
                [
                    [1.0, 0.0],
                    [0.95, 0.05],
                    [0.9, 0.1],
                    [-1.0, 0.0],
                ],
                dtype=np.float32,
            )
            np.save(embeddings_path, embeddings)
            (embedding_dir / "config.json").write_text(
                json.dumps(
                    {
                        "run_id": "20260304000000",
                        "experiment_id": "exp_bge_tac",
                        "model": {"name": "BAAI/bge-m3"},
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            eval_rows = [
                {
                    "user_id": "U1",
                    "query_item_ids": ["A"],
                    "target_item_id": "B",
                }
            ]
            with eval_input_path.open("w", encoding="utf-8") as f:
                for row in eval_rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

            run1 = self._run_script(
                eval_input_path,
                embedding_dir,
                output_root,
                eval_run_id="run_1",
            )
            run2 = self._run_script(
                eval_input_path,
                embedding_dir,
                output_root,
                eval_run_id="run_2",
            )

            self.assertEqual(
                run1["predictions"].read_text(encoding="utf-8"),
                run2["predictions"].read_text(encoding="utf-8"),
            )

            pred_rows = [json.loads(x) for x in run1["predictions"].read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(pred_rows), 1)
            self.assertEqual(pred_rows[0]["user_id"], "U1")
            self.assertEqual(pred_rows[0]["target_item_id"], "B")
            self.assertEqual(pred_rows[0]["target_rank"], 1)

            predicted_item_ids = [x["item_id"] for x in pred_rows[0]["predictions"]]
            self.assertNotIn("A", predicted_item_ids)
            self.assertEqual(predicted_item_ids[:2], ["B", "C"])

            report = json.loads(run1["report"].read_text(encoding="utf-8"))
            self.assertEqual(report["eval_stats"]["valid_eval_rows"], 1)
            self.assertAlmostEqual(report["metrics"]["@1"]["recall"], 1.0)
            self.assertAlmostEqual(report["metrics"]["@1"]["mrr"], 1.0)
            self.assertAlmostEqual(report["metrics"]["@1"]["ndcg"], 1.0)

            info = json.loads(run1["info"].read_text(encoding="utf-8"))
            self.assertEqual(info["eval_run_id"], "run_1")
            self.assertEqual(info["embedding"]["model_name_guess"], "BAAI/bge-m3")
            self.assertEqual(info["embedding"]["experiment_id"], "exp_bge_tac")

    def test_drops_rows_with_missing_query_item(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            embedding_dir = tmp_path / "emb" / "M" / "E" / "R"
            embedding_dir.mkdir(parents=True, exist_ok=True)
            item_ids_path = embedding_dir / "item_ids.jsonl"
            embeddings_path = embedding_dir / "item_embeddings.npy"
            eval_input_path = tmp_path / "eval.jsonl"
            output_root = tmp_path / "outputs" / "eval"

            with item_ids_path.open("w", encoding="utf-8") as f:
                f.write('{"item_id":"A"}\n')
                f.write('{"item_id":"B"}\n')
            np.save(
                embeddings_path,
                np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            )

            with eval_input_path.open("w", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "user_id": "U1",
                            "query_item_ids": ["A", "Z"],
                            "target_item_id": "B",
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            run = self._run_script(
                eval_input_path,
                embedding_dir,
                output_root,
                eval_run_id="run_x",
            )

            self.assertEqual(run["predictions"].read_text(encoding="utf-8"), "")
            report = json.loads(run["report"].read_text(encoding="utf-8"))
            self.assertEqual(report["filter_stats"]["dropped_query_item_not_in_index"], 1)
            self.assertEqual(report["eval_stats"]["valid_eval_rows"], 0)

    def test_max_query_limits_number_of_evaluated_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            embedding_dir = tmp_path / "emb" / "BAAI__bge-m3" / "20260304000000"
            embedding_dir.mkdir(parents=True, exist_ok=True)
            item_ids_path = embedding_dir / "item_ids.jsonl"
            embeddings_path = embedding_dir / "item_embeddings.npy"
            eval_input_path = tmp_path / "eval.jsonl"
            output_root = tmp_path / "outputs" / "eval"

            with item_ids_path.open("w", encoding="utf-8") as f:
                for item_id in ["A", "B", "C", "D"]:
                    f.write(json.dumps({"item_id": item_id}, ensure_ascii=False) + "\n")
            np.save(
                embeddings_path,
                np.array(
                    [
                        [1.0, 0.0],
                        [0.9, 0.1],
                        [0.8, 0.2],
                        [-1.0, 0.0],
                    ],
                    dtype=np.float32,
                ),
            )

            with eval_input_path.open("w", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {"user_id": "U1", "query_item_ids": ["A"], "target_item_id": "B"},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                f.write(
                    json.dumps(
                        {"user_id": "U2", "query_item_ids": ["C"], "target_item_id": "B"},
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            run = self._run_script(
                eval_input=eval_input_path,
                embedding_dir=embedding_dir,
                output_root=output_root,
                eval_run_id="run_limit",
                max_query=1,
            )
            pred_rows = run["predictions"].read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(pred_rows), 1)
            report = json.loads(run["report"].read_text(encoding="utf-8"))
            self.assertEqual(report["eval_stats"]["valid_eval_rows"], 1)
            self.assertEqual(report["config"]["max_query"], 1)

    def test_query_pooling_max_changes_ranking(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            embedding_dir = tmp_path / "emb" / "BAAI__bge-m3" / "20260304000000"
            embedding_dir.mkdir(parents=True, exist_ok=True)
            item_ids_path = embedding_dir / "item_ids.jsonl"
            embeddings_path = embedding_dir / "item_embeddings.npy"
            eval_input_path = tmp_path / "eval.jsonl"
            output_root = tmp_path / "outputs" / "eval"

            with item_ids_path.open("w", encoding="utf-8") as f:
                for item_id in ["A", "B", "C", "D"]:
                    f.write(json.dumps({"item_id": item_id}, ensure_ascii=False) + "\n")

            # A + C query history is crafted so mean pooling favors B, max pooling favors D.
            np.save(
                embeddings_path,
                np.array(
                    [
                        [1.0, -0.2],  # A (query)
                        [0.99, 0.1],  # B
                        [0.2, 1.0],  # C (query)
                        [0.4, 0.9],  # D (target)
                    ],
                    dtype=np.float32,
                ),
            )

            with eval_input_path.open("w", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {"user_id": "U1", "query_item_ids": ["A", "C"], "target_item_id": "D"},
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            run_mean = self._run_script(
                eval_input=eval_input_path,
                embedding_dir=embedding_dir,
                output_root=output_root,
                eval_run_id="run_mean",
                query_pooling="mean",
            )
            run_max = self._run_script(
                eval_input=eval_input_path,
                embedding_dir=embedding_dir,
                output_root=output_root,
                eval_run_id="run_max",
                query_pooling="max",
            )

            mean_pred = json.loads(run_mean["predictions"].read_text(encoding="utf-8").strip())
            max_pred = json.loads(run_max["predictions"].read_text(encoding="utf-8").strip())
            self.assertEqual(mean_pred["target_rank"], 2)
            self.assertEqual(max_pred["target_rank"], 1)

            mean_report = json.loads(run_mean["report"].read_text(encoding="utf-8"))
            max_report = json.loads(run_max["report"].read_text(encoding="utf-8"))
            self.assertEqual(mean_report["config"]["query_pooling"], "mean")
            self.assertEqual(max_report["config"]["query_pooling"], "max")

    def test_query_pooling_last_uses_last_query_item(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            embedding_dir = tmp_path / "emb" / "BAAI__bge-m3" / "20260304000000"
            embedding_dir.mkdir(parents=True, exist_ok=True)
            item_ids_path = embedding_dir / "item_ids.jsonl"
            embeddings_path = embedding_dir / "item_embeddings.npy"
            eval_input_path = tmp_path / "eval.jsonl"
            output_root = tmp_path / "outputs" / "eval"

            with item_ids_path.open("w", encoding="utf-8") as f:
                for item_id in ["A", "B", "C", "D"]:
                    f.write(json.dumps({"item_id": item_id}, ensure_ascii=False) + "\n")

            np.save(
                embeddings_path,
                np.array(
                    [
                        [1.0, -0.2],  # A (query, not last)
                        [0.99, 0.1],  # B
                        [0.2, 1.0],  # C (query, last)
                        [0.4, 0.9],  # D (target)
                    ],
                    dtype=np.float32,
                ),
            )

            with eval_input_path.open("w", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {"user_id": "U1", "query_item_ids": ["A", "C"], "target_item_id": "D"},
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            run_last = self._run_script(
                eval_input=eval_input_path,
                embedding_dir=embedding_dir,
                output_root=output_root,
                eval_run_id="run_last",
                query_pooling="last",
            )

            last_pred = json.loads(run_last["predictions"].read_text(encoding="utf-8").strip())
            self.assertEqual(last_pred["target_rank"], 1)

            last_report = json.loads(run_last["report"].read_text(encoding="utf-8"))
            self.assertEqual(last_report["config"]["query_pooling"], "last")

    def test_query_retrieval_mode_merging_dedups_and_reranks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            embedding_dir = tmp_path / "emb" / "BAAI__bge-m3" / "20260304000000"
            embedding_dir.mkdir(parents=True, exist_ok=True)
            item_ids_path = embedding_dir / "item_ids.jsonl"
            embeddings_path = embedding_dir / "item_embeddings.npy"
            eval_input_path = tmp_path / "eval.jsonl"
            output_root = tmp_path / "outputs" / "eval"

            with item_ids_path.open("w", encoding="utf-8") as f:
                for item_id in ["A", "B", "C", "D", "E"]:
                    f.write(json.dumps({"item_id": item_id}, ensure_ascii=False) + "\n")

            np.save(
                embeddings_path,
                np.array(
                    [
                        [1.0, 0.0],  # A (query)
                        [0.8, 0.2],  # B
                        [0.0, 1.0],  # C (query)
                        [0.05, 0.99],  # D (target)
                        [0.7, 0.7],  # E (can appear in both query retrieval lists)
                    ],
                    dtype=np.float32,
                ),
            )

            with eval_input_path.open("w", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {"user_id": "U1", "query_item_ids": ["A", "C"], "target_item_id": "D"},
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            run = self._run_script(
                eval_input=eval_input_path,
                embedding_dir=embedding_dir,
                output_root=output_root,
                eval_run_id="run_merging",
                query_retrieval_mode="merging",
                per_query_topk=2,
            )
            pred_row = json.loads(run["predictions"].read_text(encoding="utf-8").strip())
            self.assertEqual(pred_row["target_rank"], 1)

            pred_item_ids = [x["item_id"] for x in pred_row["predictions"]]
            self.assertEqual(len(pred_item_ids), len(set(pred_item_ids)))

            report = json.loads(run["report"].read_text(encoding="utf-8"))
            self.assertEqual(report["config"]["query_retrieval_mode"], "merging")
            self.assertEqual(report["config"]["per_query_topk"], 2)

    def test_query_retrieval_mode_merging_rrf_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            embedding_dir = tmp_path / "emb" / "BAAI__bge-m3" / "20260304000000"
            embedding_dir.mkdir(parents=True, exist_ok=True)
            item_ids_path = embedding_dir / "item_ids.jsonl"
            embeddings_path = embedding_dir / "item_embeddings.npy"
            eval_input_path = tmp_path / "eval.jsonl"
            output_root = tmp_path / "outputs" / "eval"

            with item_ids_path.open("w", encoding="utf-8") as f:
                for item_id in ["A", "B", "C", "D"]:
                    f.write(json.dumps({"item_id": item_id}, ensure_ascii=False) + "\n")

            np.save(
                embeddings_path,
                np.array(
                    [
                        [1.0, 0.0],  # A (query)
                        [0.9, 0.1],  # B
                        [0.0, 1.0],  # C (query)
                        [0.1, 0.95],  # D (target)
                    ],
                    dtype=np.float32,
                ),
            )

            with eval_input_path.open("w", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {"user_id": "U1", "query_item_ids": ["A", "C"], "target_item_id": "D"},
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            run = self._run_script(
                eval_input=eval_input_path,
                embedding_dir=embedding_dir,
                output_root=output_root,
                eval_run_id="run_merging_rrf",
                query_retrieval_mode="merging",
                per_query_topk=2,
                merge_fusion="rrf",
                rrf_k=30,
                recency_weighting="exp",
                recency_alpha=1.5,
            )

            report = json.loads(run["report"].read_text(encoding="utf-8"))
            self.assertEqual(report["config"]["query_retrieval_mode"], "merging")
            self.assertEqual(report["config"]["merge_fusion"], "rrf")
            self.assertEqual(report["config"]["rrf_k"], 30)
            self.assertEqual(report["config"]["recency_weighting"], "exp")
            self.assertAlmostEqual(report["config"]["recency_alpha"], 1.5)

            info = json.loads(run["info"].read_text(encoding="utf-8"))
            self.assertEqual(info["retrieval"]["merge_fusion"], "rrf")
            self.assertEqual(info["retrieval"]["rrf_k"], 30)
            self.assertEqual(info["retrieval"]["recency_weighting"], "exp")
            self.assertAlmostEqual(info["retrieval"]["recency_alpha"], 1.5)

    def test_query_history_n_uses_last_n_items(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            embedding_dir = tmp_path / "emb" / "BAAI__bge-m3" / "20260304000000"
            embedding_dir.mkdir(parents=True, exist_ok=True)
            item_ids_path = embedding_dir / "item_ids.jsonl"
            embeddings_path = embedding_dir / "item_embeddings.npy"
            eval_input_path = tmp_path / "eval.jsonl"
            output_root = tmp_path / "outputs" / "eval"

            with item_ids_path.open("w", encoding="utf-8") as f:
                for item_id in ["A", "B", "C", "D", "E"]:
                    f.write(json.dumps({"item_id": item_id}, ensure_ascii=False) + "\n")

            np.save(
                embeddings_path,
                np.array(
                    [
                        [1.0, 0.0],  # A (old query)
                        [0.8, 0.2],  # B (query)
                        [0.0, 1.0],  # C (latest query)
                        [0.45, 0.89],  # D (target)
                        [0.95, 0.1],  # E (competitor)
                    ],
                    dtype=np.float32,
                ),
            )

            with eval_input_path.open("w", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {"user_id": "U1", "query_item_ids": ["A", "B", "C"], "target_item_id": "D"},
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            run_all = self._run_script(
                eval_input=eval_input_path,
                embedding_dir=embedding_dir,
                output_root=output_root,
                eval_run_id="run_history_all",
                query_pooling="mean",
                query_history_n=0,
            )
            run_last2 = self._run_script(
                eval_input=eval_input_path,
                embedding_dir=embedding_dir,
                output_root=output_root,
                eval_run_id="run_history_last2",
                query_pooling="mean",
                query_history_n=2,
            )

            pred_all = json.loads(run_all["predictions"].read_text(encoding="utf-8").strip())
            pred_last2 = json.loads(run_last2["predictions"].read_text(encoding="utf-8").strip())

            self.assertEqual(pred_all["target_rank"], 2)
            self.assertEqual(pred_last2["target_rank"], 1)
            self.assertEqual(pred_last2["query_item_ids"], ["B", "C"])

            report_last2 = json.loads(run_last2["report"].read_text(encoding="utf-8"))
            self.assertEqual(report_last2["config"]["query_history_n"], 2)

    def test_embedding_dim_selects_requested_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            embedding_dir = tmp_path / "emb" / "BAAI__bge-m3" / "20260304000000"
            embedding_dir.mkdir(parents=True, exist_ok=True)
            item_ids_path = embedding_dir / "item_ids.jsonl"
            eval_input_path = tmp_path / "eval.jsonl"
            output_root = tmp_path / "outputs" / "eval"

            with item_ids_path.open("w", encoding="utf-8") as f:
                for item_id in ["A", "B", "C"]:
                    f.write(json.dumps({"item_id": item_id}, ensure_ascii=False) + "\n")
            np.save(
                embedding_dir / "item_embeddings_2.npy",
                np.array(
                    [
                        [1.0, 0.0],
                        [0.9, 0.1],
                        [0.0, 1.0],
                    ],
                    dtype=np.float32,
                ),
            )
            np.save(
                embedding_dir / "item_embeddings_3.npy",
                np.array(
                    [
                        [1.0, 0.0, 0.0],
                        [0.9, 0.1, 0.0],
                        [0.0, 1.0, 0.0],
                    ],
                    dtype=np.float32,
                ),
            )

            with eval_input_path.open("w", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {"user_id": "U1", "query_item_ids": ["A"], "target_item_id": "B"},
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            run = self._run_script(
                eval_input=eval_input_path,
                embedding_dir=embedding_dir,
                output_root=output_root,
                eval_run_id="run_dim_2",
                embedding_dim="2",
            )
            report = json.loads(run["report"].read_text(encoding="utf-8"))
            info = json.loads(run["info"].read_text(encoding="utf-8"))
            self.assertEqual(report["config"]["requested_embedding_dim"], 2)
            self.assertEqual(report["index_stats"]["embedding_dim"], 2)
            self.assertEqual(info["embedding"]["requested_embedding_dim"], 2)
            self.assertEqual(info["embedding"]["resolved_embedding_dim"], 2)

    def test_embedding_dim_all_writes_per_dim_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            embedding_dir = tmp_path / "emb" / "BAAI__bge-m3" / "20260304000000"
            embedding_dir.mkdir(parents=True, exist_ok=True)
            item_ids_path = embedding_dir / "item_ids.jsonl"
            eval_input_path = tmp_path / "eval.jsonl"
            output_root = tmp_path / "outputs" / "eval"
            eval_run_id = "run_dim_all"

            with item_ids_path.open("w", encoding="utf-8") as f:
                for item_id in ["A", "B", "C"]:
                    f.write(json.dumps({"item_id": item_id}, ensure_ascii=False) + "\n")
            np.save(
                embedding_dir / "item_embeddings_2.npy",
                np.array(
                    [
                        [1.0, 0.0],
                        [0.9, 0.1],
                        [0.0, 1.0],
                    ],
                    dtype=np.float32,
                ),
            )
            np.save(
                embedding_dir / "item_embeddings_3.npy",
                np.array(
                    [
                        [1.0, 0.0, 0.0],
                        [0.9, 0.1, 0.0],
                        [0.0, 1.0, 0.0],
                    ],
                    dtype=np.float32,
                ),
            )

            with eval_input_path.open("w", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {"user_id": "U1", "query_item_ids": ["A"], "target_item_id": "B"},
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            cmd = [
                sys.executable,
                str(SCRIPT_PATH),
                "--eval-input",
                str(eval_input_path),
                "--embedding-dir",
                str(embedding_dir),
                "--embedding-dim",
                "all",
                "--output-root",
                str(output_root),
                "--eval-run-id",
                eval_run_id,
                "--topk",
                "1,2",
                "--index-type",
                "flat",
                "--seed",
                "42",
            ]
            subprocess.run(
                cmd,
                cwd=REPO_ROOT,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            run_dir = output_root / eval_run_id
            summary_report = json.loads((run_dir / "run_eval_report.json").read_text(encoding="utf-8"))
            self.assertEqual(summary_report["config"]["embedding_dim"], "all")
            self.assertEqual(len(summary_report["runs"]), 2)

            dim2_report = json.loads((run_dir / "dim_2" / "run_eval_report.json").read_text(encoding="utf-8"))
            dim3_report = json.loads((run_dir / "dim_3" / "run_eval_report.json").read_text(encoding="utf-8"))
            self.assertEqual(dim2_report["index_stats"]["embedding_dim"], 2)
            self.assertEqual(dim3_report["index_stats"]["embedding_dim"], 3)
            self.assertTrue((run_dir / "dim_2" / "predictions.jsonl").exists())
            self.assertTrue((run_dir / "dim_3" / "predictions.jsonl").exists())

    def _run_script(
        self,
        eval_input: Path,
        embedding_dir: Path,
        output_root: Path,
        eval_run_id: str,
        max_query: int = 0,
        query_history_n: int = 0,
        query_pooling: str = "mean",
        query_retrieval_mode: str = "pooling",
        per_query_topk: int = 20,
        merge_fusion: str = "max",
        rrf_k: int = 60,
        recency_weighting: str = "none",
        recency_alpha: float = 1.0,
        embedding_dim: str | None = None,
    ) -> dict[str, Path]:
        cmd = [
            sys.executable,
            str(SCRIPT_PATH),
            "--eval-input",
            str(eval_input),
            "--embedding-dir",
            str(embedding_dir),
            "--output-root",
            str(output_root),
            "--eval-run-id",
            str(eval_run_id),
            "--topk",
            "1,2",
            "--query-history-n",
            str(query_history_n),
            "--query-pooling",
            query_pooling,
            "--query-retrieval-mode",
            query_retrieval_mode,
            "--per-query-topk",
            str(per_query_topk),
            "--merge-fusion",
            merge_fusion,
            "--rrf-k",
            str(rrf_k),
            "--recency-weighting",
            recency_weighting,
            "--recency-alpha",
            str(recency_alpha),
            "--index-type",
            "flat",
            "--seed",
            "42",
        ]
        if embedding_dim is not None:
            cmd.extend(["--embedding-dim", embedding_dim])
        if max_query > 0:
            cmd.extend(["--max-query", str(max_query)])
        subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        run_dir = output_root / eval_run_id
        return {
            "run_dir": run_dir,
            "predictions": run_dir / "predictions.jsonl",
            "report": run_dir / "run_eval_report.json",
            "info": run_dir / "info.json",
        }


if __name__ == "__main__":
    unittest.main()
