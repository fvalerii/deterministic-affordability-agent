"""Tests for the Buy or Wait? request orchestrator."""

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "code"))

from domain import OUTPUT_COLUMNS  # noqa: E402
from tools.data import load_dataset  # noqa: E402
from tools.explanation import deterministic_explanation  # noqa: E402
from tools.orchestrator import BuyOrWaitOrchestrator  # noqa: E402
from tools.validation import validate_output_row  # noqa: E402

DATASET = load_dataset(REPO_ROOT / "dataset")


class TestOrchestrator(unittest.TestCase):
    def setUp(self):
        self.orchestrator = BuyOrWaitOrchestrator(
            DATASET, explain=deterministic_explanation
        )

    def test_request_01_is_affordable_now_full_payment(self):
        decision = self.orchestrator.recommend_id("request_01")
        self.assertEqual(decision.recommended_payment_method, "full_payment")
        self.assertEqual(decision.affordability_status, "affordable_now")
        self.assertEqual(decision.payment_plan, "2024-03-03:25256")
        self.assertEqual(decision.earliest_date_for_full_payment.isoformat(), "2024-03-03")
        self.assertEqual(decision.spending_changes_needed, "none")
        self.assertIn("education", decision.decision_explanation)
        row = decision.to_output_row()
        self.assertEqual(
            validate_output_row(row, request=DATASET.sample_requests["request_01"]),
            (),
        )

    def test_run_writes_required_columns_in_order(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "output.csv"
            report = Path(folder) / "usage_report.md"
            written = self.orchestrator.run(
                output_path=output,
                request_ids=("request_01", "request_09"),
                usage_report_path=report,
            )
            self.assertEqual(written, output)
            with output.open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(reader.fieldnames, list(OUTPUT_COLUMNS))
                rows = list(reader)
            self.assertEqual(
                [row["request_id"] for row in rows], ["request_01", "request_09"]
            )
            self.assertTrue(report.is_file())
            self.assertIn("Anthropic", report.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
