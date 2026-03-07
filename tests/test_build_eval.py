from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts/data/build_eval.py"


class BuildEvalTests(unittest.TestCase):
    def test_stable_last_target_and_query_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            interactions_path = tmp_path / "interactions.jsonl"
            out1 = tmp_path / "eval_1.jsonl"
            out2 = tmp_path / "eval_2.jsonl"
            report1 = tmp_path / "report_1.json"
            report2 = tmp_path / "report_2.json"

            rows = [
                # U1 positives: B1(10), B2(20), B3(20), B4(30)
                {"user_id": "U1", "item_id": "B1", "rating": 5, "timestamp": 10},
                {"user_id": "U1", "item_id": "B2", "rating": 4, "timestamp": 20},
                {"user_id": "U1", "item_id": "B3", "rating": 4, "timestamp": 20},
                {"user_id": "U1", "item_id": "B4", "rating": 5, "timestamp": 30},
                # below threshold, should be ignored
                {"user_id": "U1", "item_id": "BX", "rating": 2, "timestamp": 40},
                # U2 only one positive -> dropped in eval construction (no history)
                {"user_id": "U2", "item_id": "C1", "rating": 5, "timestamp": 11},
            ]
            interactions_path.write_text(
                "\n".join(json.dumps(r) for r in rows) + "\n",
                encoding="utf-8",
            )

            self._run_script(interactions_path, out1, report1, query_history_n=2)
            self._run_script(interactions_path, out2, report2, query_history_n=2)

            self.assertEqual(
                out1.read_text(encoding="utf-8"), out2.read_text(encoding="utf-8")
            )

            eval_rows = [json.loads(line) for line in out1.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(eval_rows), 1)
            self.assertEqual(eval_rows[0]["user_id"], "U1")
            self.assertEqual(eval_rows[0]["target_item_id"], "B4")
            self.assertEqual(eval_rows[0]["query_item_ids"], ["B2", "B3"])

            report = json.loads(report1.read_text(encoding="utf-8"))
            self.assertEqual(report["input_stats"]["rows_below_rating_threshold"], 1)
            self.assertEqual(report["eval_stats"]["users_dropped_due_to_no_history"], 1)
            self.assertEqual(report["eval_stats"]["eval_query_rows_written"], 1)

    def test_kcore_filter_optional(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            interactions_path = tmp_path / "interactions.jsonl"
            output = tmp_path / "eval.jsonl"
            report = tmp_path / "report.json"

            rows = [
                {"user_id": "U1", "item_id": "I1", "rating": 5, "timestamp": 1},
                {"user_id": "U1", "item_id": "I2", "rating": 5, "timestamp": 2},
                {"user_id": "U2", "item_id": "I1", "rating": 5, "timestamp": 3},
                {"user_id": "U2", "item_id": "I3", "rating": 5, "timestamp": 4},
            ]
            interactions_path.write_text(
                "\n".join(json.dumps(r) for r in rows) + "\n",
                encoding="utf-8",
            )

            self._run_script(
                interactions_path,
                output,
                report,
                query_history_n=1,
                min_user_pos=2,
                min_item_pos=2,
            )
            self.assertEqual(output.read_text(encoding="utf-8"), "")

            report_obj = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(report_obj["kcore"]["enabled"], 1)
            self.assertGreaterEqual(report_obj["kcore"]["iterations"], 1)
            self.assertEqual(report_obj["positive_stats"]["samples_post_filter"], 0)

    def test_default_report_output_follows_queries_output_stem(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            interactions_path = tmp_path / "interactions.jsonl"
            queries_output = tmp_path / "eval_u6_i5_q5.jsonl"
            expected_report = tmp_path / "reports" / "build_eval_report_eval_u6_i5_q5.json"

            rows = [
                {"user_id": "U1", "item_id": "I1", "rating": 5, "timestamp": 1},
                {"user_id": "U1", "item_id": "I2", "rating": 5, "timestamp": 2},
            ]
            interactions_path.write_text(
                "\n".join(json.dumps(r) for r in rows) + "\n",
                encoding="utf-8",
            )

            cmd = [
                sys.executable,
                str(SCRIPT_PATH),
                "--interactions-input",
                str(interactions_path),
                "--queries-output",
                str(queries_output),
                "--query-history-n",
                "1",
            ]
            subprocess.run(
                cmd,
                cwd=tmp_path,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            self.assertTrue(expected_report.exists())
            report = json.loads(expected_report.read_text(encoding="utf-8"))
            self.assertEqual(
                report["config"]["report_output"],
                str(Path("reports/build_eval_report_eval_u6_i5_q5.json")),
            )

    def _run_script(
        self,
        interactions_path: Path,
        queries_output: Path,
        report_output: Path,
        query_history_n: int,
        min_user_pos: int = 1,
        min_item_pos: int = 1,
    ) -> None:
        cmd = [
            sys.executable,
            str(SCRIPT_PATH),
            "--interactions-input",
            str(interactions_path),
            "--queries-output",
            str(queries_output),
            "--report-output",
            str(report_output),
            "--rating-threshold",
            "4.0",
            "--min-user-pos",
            str(min_user_pos),
            "--min-item-pos",
            str(min_item_pos),
            "--query-history-n",
            str(query_history_n),
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


if __name__ == "__main__":
    unittest.main()
