from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts/data/build_items_subset_from_eval.py"


class BuildItemsSubsetFromEvalTests(unittest.TestCase):
    def test_build_subset_from_eval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            eval_input = tmp_path / "eval.jsonl"
            items_input = tmp_path / "items.jsonl"
            output_path = tmp_path / "items_subset.jsonl"
            report_path = tmp_path / "report.json"

            eval_lines = [
                json.dumps(
                    {
                        "user_id": "U1",
                        "query_item_ids": ["A", "B"],
                        "target_item_id": "C",
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "user_id": "U2",
                        "query_item_ids": ["B", "", None],
                        "target_item_id": "D",
                    },
                    ensure_ascii=False,
                ),
                "1",  # non-object
                "{bad json",  # parse error
                json.dumps(
                    {
                        "user_id": "U3",
                        "query_item_ids": ["E"],
                        "target_item_id": "",
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "user_id": "U4",
                        "query_item_ids": "A",
                        "target_item_id": "B",
                    },
                    ensure_ascii=False,
                ),
            ]
            eval_input.write_text("\n".join(eval_lines) + "\n", encoding="utf-8")

            items_rows = [
                {"item_id": "A", "title": "A"},
                {"item_id": "B", "title": "B"},
                {"item_id": "C", "title": "C"},
                {"item_id": "D", "title": "D"},
                {"item_id": "B", "title": "B-dup"},
                {"item_id": "Z", "title": "Z"},
            ]
            items_input.write_text(
                "\n".join(json.dumps(x, ensure_ascii=False) for x in items_rows) + "\n",
                encoding="utf-8",
            )

            cmd = [
                sys.executable,
                str(SCRIPT_PATH),
                "--eval-input",
                str(eval_input),
                "--items-input",
                str(items_input),
                "--output",
                str(output_path),
                "--report",
                str(report_path),
            ]
            subprocess.run(
                cmd,
                cwd=REPO_ROOT,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            out_rows = [json.loads(x) for x in output_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([x["item_id"] for x in out_rows], ["A", "B", "C", "D"])

            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["eval_stats"]["rows_total"], 6)
            self.assertEqual(report["eval_stats"]["rows_valid"], 2)
            self.assertEqual(report["eval_stats"]["wanted_unique_item_ids"], 4)
            self.assertEqual(report["eval_stats"]["parse_error_rows"], 1)
            self.assertEqual(report["eval_stats"]["non_object_rows"], 1)
            self.assertEqual(report["eval_stats"]["rows_missing_target_item_id"], 1)
            self.assertEqual(report["eval_stats"]["rows_invalid_query_item_ids"], 1)

            self.assertEqual(report["items_scan_stats"]["rows_total"], 6)
            self.assertEqual(report["items_scan_stats"]["rows_valid_with_item_id"], 6)
            self.assertEqual(report["items_scan_stats"]["duplicate_item_id_rows"], 1)

            self.assertEqual(report["output_stats"]["rows_written"], 4)
            self.assertEqual(report["output_stats"]["missing_item_ids_from_items"], 0)
            self.assertAlmostEqual(report["output_stats"]["coverage_over_wanted_item_ids"], 1.0)

    def test_default_output_follows_eval_input_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            eval_input = tmp_path / "data" / "processed" / "eval_u6_i5_q5.jsonl"
            items_input = tmp_path / "data" / "processed" / "items.jsonl"
            expected_output = tmp_path / "data" / "processed" / "items_subset_eval_u6_i5_q5.jsonl"
            expected_report = (
                tmp_path
                / "reports"
                / "build_items_subset_report_from_eval_u6_i5_q5.json"
            )

            eval_input.parent.mkdir(parents=True, exist_ok=True)
            eval_input.write_text(
                json.dumps(
                    {
                        "user_id": "U1",
                        "query_item_ids": ["A"],
                        "target_item_id": "B",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            items_input.write_text(
                "\n".join(
                    [
                        json.dumps({"item_id": "A", "title": "A"}, ensure_ascii=False),
                        json.dumps({"item_id": "B", "title": "B"}, ensure_ascii=False),
                        json.dumps({"item_id": "C", "title": "C"}, ensure_ascii=False),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            cmd = [
                sys.executable,
                str(SCRIPT_PATH),
                "--eval-input",
                str(Path("data/processed/eval_u6_i5_q5.jsonl")),
                "--items-input",
                str(Path("data/processed/items.jsonl")),
            ]
            subprocess.run(
                cmd,
                cwd=tmp_path,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            self.assertTrue(expected_output.exists())
            self.assertTrue(expected_report.exists())
            out_rows = [json.loads(x) for x in expected_output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([x["item_id"] for x in out_rows], ["A", "B"])


if __name__ == "__main__":
    unittest.main()
