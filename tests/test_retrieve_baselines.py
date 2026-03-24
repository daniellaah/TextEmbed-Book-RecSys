from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts/baselines/retrieve_baselines.py"


class RetrieveBaselinesTests(unittest.TestCase):
    def test_random_is_stable_for_same_seed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            items_input, interactions_input, eval_input = self._write_basic_fixture(tmp_path)
            output_root = tmp_path / "outputs"

            run1 = self._run_script(
                baseline="random",
                items_input=items_input,
                interactions_input=interactions_input,
                eval_input=eval_input,
                output_root=output_root,
                run_id="run_1",
                seed=42,
            )
            run2 = self._run_script(
                baseline="random",
                items_input=items_input,
                interactions_input=interactions_input,
                eval_input=eval_input,
                output_root=output_root,
                run_id="run_2",
                seed=42,
            )

            self.assertEqual(
                run1["predictions"].read_text(encoding="utf-8"),
                run2["predictions"].read_text(encoding="utf-8"),
            )

    def test_global_popular_orders_by_interaction_count_and_excludes_query_items(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            items_input = tmp_path / "items.jsonl"
            interactions_input = tmp_path / "interactions.jsonl"
            eval_input = tmp_path / "eval.jsonl"
            output_root = tmp_path / "outputs"

            self._write_jsonl(
                items_input,
                [
                    {"item_id": "A", "categories": "cat1"},
                    {"item_id": "B", "categories": "cat1"},
                    {"item_id": "C", "categories": "cat1"},
                    {"item_id": "D", "categories": "cat2"},
                ],
            )
            self._write_jsonl(
                interactions_input,
                [
                    {"user_id": "U1", "item_id": "A", "rating": 5, "timestamp": 1},
                    {"user_id": "U2", "item_id": "A", "rating": 5, "timestamp": 2},
                    {"user_id": "U3", "item_id": "A", "rating": 5, "timestamp": 3},
                    {"user_id": "U4", "item_id": "B", "rating": 5, "timestamp": 4},
                    {"user_id": "U5", "item_id": "B", "rating": 5, "timestamp": 5},
                    {"user_id": "U6", "item_id": "C", "rating": 5, "timestamp": 6},
                    {"user_id": "U7", "item_id": "D", "rating": 5, "timestamp": 7},
                ],
            )
            self._write_jsonl(
                eval_input,
                [{"user_id": "Q1", "query_item_ids": ["A"], "target_item_id": "D"}],
            )

            run = self._run_script(
                baseline="global_popular",
                items_input=items_input,
                interactions_input=interactions_input,
                eval_input=eval_input,
                output_root=output_root,
                run_id="popular",
                topk="3",
            )

            pred_row = json.loads(run["predictions"].read_text(encoding="utf-8").strip())
            predicted_item_ids = [x["item_id"] for x in pred_row["predictions"]]
            self.assertEqual(predicted_item_ids, ["B", "C", "D"])
            self.assertNotIn("A", predicted_item_ids)

    def test_category_popular_restricts_candidates_to_same_category(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            items_input = tmp_path / "items.jsonl"
            interactions_input = tmp_path / "interactions.jsonl"
            eval_input = tmp_path / "eval.jsonl"
            output_root = tmp_path / "outputs"

            self._write_jsonl(
                items_input,
                [
                    {"item_id": "A", "categories": "cat1"},
                    {"item_id": "B", "categories": "cat2"},
                    {"item_id": "C", "categories": "cat1"},
                    {"item_id": "D", "categories": "cat1"},
                    {"item_id": "E", "categories": "cat2"},
                ],
            )
            self._write_jsonl(
                interactions_input,
                [
                    {"user_id": "U1", "item_id": "C", "rating": 5, "timestamp": 1},
                    {"user_id": "U2", "item_id": "C", "rating": 5, "timestamp": 2},
                    {"user_id": "U3", "item_id": "D", "rating": 5, "timestamp": 3},
                    {"user_id": "U4", "item_id": "E", "rating": 5, "timestamp": 4},
                ],
            )
            self._write_jsonl(
                eval_input,
                [{"user_id": "Q1", "query_item_ids": ["A"], "target_item_id": "D"}],
            )

            run = self._run_script(
                baseline="category_popular",
                items_input=items_input,
                interactions_input=interactions_input,
                eval_input=eval_input,
                output_root=output_root,
                run_id="category_popular",
                topk="2",
            )
            pred_row = json.loads(run["predictions"].read_text(encoding="utf-8").strip())
            predicted_item_ids = [x["item_id"] for x in pred_row["predictions"]]
            self.assertEqual(predicted_item_ids, ["C", "D"])

            report = json.loads(run["report"].read_text(encoding="utf-8"))
            self.assertEqual(report["baseline_stats"]["category_rows_with_matched_category_pool"], 1)
            self.assertEqual(report["baseline_stats"]["category_fallback_missing_category"], 0)

    def test_category_random_restricts_candidates_to_same_category(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            items_input = tmp_path / "items.jsonl"
            interactions_input = tmp_path / "interactions.jsonl"
            eval_input = tmp_path / "eval.jsonl"
            output_root = tmp_path / "outputs"

            self._write_jsonl(
                items_input,
                [
                    {"item_id": "A", "categories": "cat1"},
                    {"item_id": "B", "categories": "cat2"},
                    {"item_id": "C", "categories": "cat1"},
                    {"item_id": "D", "categories": "cat1"},
                    {"item_id": "E", "categories": "cat2"},
                ],
            )
            self._write_jsonl(
                interactions_input,
                [{"user_id": "U1", "item_id": "C", "rating": 5, "timestamp": 1}],
            )
            self._write_jsonl(
                eval_input,
                [{"user_id": "Q1", "query_item_ids": ["A"], "target_item_id": "D"}],
            )

            run = self._run_script(
                baseline="category_random",
                items_input=items_input,
                interactions_input=interactions_input,
                eval_input=eval_input,
                output_root=output_root,
                run_id="category_random",
                topk="2",
                seed=123,
            )
            pred_row = json.loads(run["predictions"].read_text(encoding="utf-8").strip())
            predicted_item_ids = {x["item_id"] for x in pred_row["predictions"]}
            self.assertEqual(predicted_item_ids, {"C", "D"})

    def test_category_uses_last_query_item_category(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            items_input = tmp_path / "items.jsonl"
            interactions_input = tmp_path / "interactions.jsonl"
            eval_input = tmp_path / "eval.jsonl"
            output_root = tmp_path / "outputs"

            self._write_jsonl(
                items_input,
                [
                    {"item_id": "A", "categories": "cat1"},
                    {"item_id": "B", "categories": "cat2"},
                    {"item_id": "C", "categories": "cat1"},
                    {"item_id": "D", "categories": "cat2"},
                    {"item_id": "E", "categories": "cat2"},
                ],
            )
            self._write_jsonl(
                interactions_input,
                [
                    {"user_id": "U1", "item_id": "C", "rating": 5, "timestamp": 1},
                    {"user_id": "U2", "item_id": "D", "rating": 5, "timestamp": 2},
                    {"user_id": "U3", "item_id": "E", "rating": 5, "timestamp": 3},
                ],
            )
            self._write_jsonl(
                eval_input,
                [{"user_id": "Q1", "query_item_ids": ["A", "B"], "target_item_id": "D"}],
            )

            run = self._run_script(
                baseline="category_popular",
                items_input=items_input,
                interactions_input=interactions_input,
                eval_input=eval_input,
                output_root=output_root,
                run_id="last_query_category",
                topk="2",
            )

            pred_row = json.loads(run["predictions"].read_text(encoding="utf-8").strip())
            predicted_item_ids = [x["item_id"] for x in pred_row["predictions"]]
            self.assertEqual(predicted_item_ids, ["D", "E"])
            self.assertNotIn("C", predicted_item_ids)

    def test_missing_category_falls_back_to_global_popular(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            items_input = tmp_path / "items.jsonl"
            interactions_input = tmp_path / "interactions.jsonl"
            eval_input = tmp_path / "eval.jsonl"
            output_root = tmp_path / "outputs"

            self._write_jsonl(
                items_input,
                [
                    {"item_id": "A", "categories": ""},
                    {"item_id": "B", "categories": "cat1"},
                    {"item_id": "C", "categories": "cat2"},
                ],
            )
            self._write_jsonl(
                interactions_input,
                [
                    {"user_id": "U1", "item_id": "B", "rating": 5, "timestamp": 1},
                    {"user_id": "U2", "item_id": "B", "rating": 5, "timestamp": 2},
                    {"user_id": "U3", "item_id": "C", "rating": 5, "timestamp": 3},
                ],
            )
            self._write_jsonl(
                eval_input,
                [{"user_id": "Q1", "query_item_ids": ["A"], "target_item_id": "C"}],
            )

            run = self._run_script(
                baseline="category_popular",
                items_input=items_input,
                interactions_input=interactions_input,
                eval_input=eval_input,
                output_root=output_root,
                run_id="fallback",
                topk="2",
            )

            pred_row = json.loads(run["predictions"].read_text(encoding="utf-8").strip())
            predicted_item_ids = [x["item_id"] for x in pred_row["predictions"]]
            self.assertEqual(predicted_item_ids, ["B", "C"])

            report = json.loads(run["report"].read_text(encoding="utf-8"))
            self.assertEqual(report["baseline_stats"]["category_fallback_missing_category"], 1)

    def test_tfidf_uses_text_overlap_and_excludes_query_items(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            items_input = tmp_path / "items.jsonl"
            interactions_input = tmp_path / "interactions.jsonl"
            eval_input = tmp_path / "eval.jsonl"
            output_root = tmp_path / "outputs"

            self._write_jsonl(
                items_input,
                [
                    {"item_id": "A", "title": "deep learning with python", "author": "alpha", "categories": "ml"},
                    {"item_id": "B", "title": "deep learning handbook", "author": "beta", "categories": "ml"},
                    {"item_id": "C", "title": "roman history atlas", "author": "gamma", "categories": "history"},
                ],
            )
            self._write_jsonl(
                interactions_input,
                [{"user_id": "U1", "item_id": "B", "rating": 5, "timestamp": 1}],
            )
            self._write_jsonl(
                eval_input,
                [{"user_id": "Q1", "query_item_ids": ["A"], "target_item_id": "B"}],
            )

            run = self._run_script(
                baseline="tfidf",
                items_input=items_input,
                interactions_input=interactions_input,
                eval_input=eval_input,
                output_root=output_root,
                run_id="tfidf",
                topk="2",
            )

            pred_row = json.loads(run["predictions"].read_text(encoding="utf-8").strip())
            predicted_item_ids = [x["item_id"] for x in pred_row["predictions"]]
            self.assertEqual(predicted_item_ids[0], "B")
            self.assertNotIn("A", predicted_item_ids)

            report = json.loads(run["report"].read_text(encoding="utf-8"))
            self.assertEqual(report["baseline_stats"]["tfidf"]["text_fields"], ["title", "author", "categories"])

    def test_max_query_and_report_metrics_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            items_input, interactions_input, eval_input = self._write_basic_fixture(tmp_path)
            output_root = tmp_path / "outputs"

            self._write_jsonl(
                eval_input,
                [
                    {"user_id": "Q1", "query_item_ids": ["A"], "target_item_id": "D"},
                    {"user_id": "Q2", "query_item_ids": ["B"], "target_item_id": "C"},
                ],
            )

            run = self._run_script(
                baseline="global_popular",
                items_input=items_input,
                interactions_input=interactions_input,
                eval_input=eval_input,
                output_root=output_root,
                run_id="max_query",
                topk="1,2",
                max_query=1,
            )

            pred_lines = run["predictions"].read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(pred_lines), 1)

            report = json.loads(run["report"].read_text(encoding="utf-8"))
            self.assertEqual(report["eval_stats"]["valid_eval_rows"], 1)
            self.assertIn("@1", report["metrics"])
            self.assertIn("@2", report["metrics"])
            self.assertIn("recall", report["metrics"]["@1"])
            self.assertIn("mrr", report["metrics"]["@1"])
            self.assertIn("ndcg", report["metrics"]["@1"])

    def _write_basic_fixture(self, tmp_path: Path) -> tuple[Path, Path, Path]:
        items_input = tmp_path / "items.jsonl"
        interactions_input = tmp_path / "interactions.jsonl"
        eval_input = tmp_path / "eval.jsonl"

        self._write_jsonl(
            items_input,
            [
                {"item_id": "A", "categories": "cat1"},
                {"item_id": "B", "categories": "cat1"},
                {"item_id": "C", "categories": "cat2"},
                {"item_id": "D", "categories": "cat2"},
            ],
        )
        self._write_jsonl(
            interactions_input,
            [
                {"user_id": "U1", "item_id": "B", "rating": 5, "timestamp": 1},
                {"user_id": "U2", "item_id": "B", "rating": 5, "timestamp": 2},
                {"user_id": "U3", "item_id": "C", "rating": 5, "timestamp": 3},
                {"user_id": "U4", "item_id": "D", "rating": 5, "timestamp": 4},
            ],
        )
        self._write_jsonl(
            eval_input,
            [{"user_id": "Q1", "query_item_ids": ["A"], "target_item_id": "D"}],
        )
        return items_input, interactions_input, eval_input

    def _write_jsonl(self, path: Path, rows: list[dict[str, object]]) -> None:
        path.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
            encoding="utf-8",
        )

    def _run_script(
        self,
        *,
        baseline: str,
        items_input: Path,
        interactions_input: Path,
        eval_input: Path,
        output_root: Path,
        run_id: str,
        topk: str = "10,50",
        seed: int = 42,
        max_query: int = 0,
    ) -> dict[str, Path]:
        cmd = [
            sys.executable,
            str(SCRIPT_PATH),
            "--baseline",
            baseline,
            "--items-input",
            str(items_input),
            "--interactions-input",
            str(interactions_input),
            "--eval-input",
            str(eval_input),
            "--output-root",
            str(output_root),
            "--topk",
            topk,
            "--max-query",
            str(max_query),
            "--seed",
            str(seed),
            "--run-id",
            run_id,
        ]
        subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        run_dir = output_root / baseline / run_id
        return {
            "run_dir": run_dir,
            "predictions": run_dir / "predictions.jsonl",
            "report": run_dir / "report.json",
            "info": run_dir / "info.json",
        }


if __name__ == "__main__":
    unittest.main()
