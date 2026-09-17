"""Tests for the submission token-usage report."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "code"))

from tools.usage import UsageMeter, pending_full_dataset_report  # noqa: E402


class _FakeResponse:
    def __init__(self, input_tokens: int, output_tokens: int) -> None:
        self.usage = SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        )


class TestUsageReport(unittest.TestCase):
    def test_report_lists_required_overall_and_per_model_fields(self):
        meter = UsageMeter()
        meter.begin_run(split="eval", output_artifact="output.csv", request_count=250)
        meter.record(
            _FakeResponse(1000, 40), model="claude-sonnet-4-6", purpose="explanation"
        )
        meter.record(
            _FakeResponse(8000, 20), model="claude-sonnet-4-6", purpose="vision"
        )
        meter.record(
            _FakeResponse(500, 30),
            model="claude-3-5-sonnet-20241022",
            purpose="explanation",
        )
        meter.requests_processed = 2
        text = meter.render_markdown()
        for needle in (
            "Provider:",
            "Model names:",
            "Model calls:",
            "Input tokens:",
            "Output tokens:",
            "Total tokens:",
            "Average tokens per request:",
            "Estimated total cost (USD):",
            "Estimated cost per request (USD):",
            "## Per-model totals",
            "## Overall totals",
            "`claude-sonnet-4-6`",
            "`claude-3-5-sonnet-20241022`",
            "output.csv",
        ):
            self.assertIn(needle, text)
        self.assertNotIn("sk-ant", text)
        self.assertEqual(meter.calls, 3)
        self.assertEqual(meter.input_tokens, 9500)
        self.assertEqual(meter.output_tokens, 90)
        self.assertAlmostEqual(meter.average_tokens_per_request, 9590 / 2)
        # Same published $3/$15 per MTok for both Sonnet IDs.
        expected_cost = (9500 / 1_000_000) * 3.0 + (90 / 1_000_000) * 15.0
        self.assertAlmostEqual(meter.estimated_total_cost_usd, expected_cost)

    def test_write_rejects_api_keys(self):
        meter = UsageMeter()
        meter.begin_run(split="eval", output_artifact="output.csv")
        meter.record(_FakeResponse(1, 1), model="claude-sonnet-4-6")
        original = meter.render_markdown

        def poisoned() -> str:
            return original() + "\nsk-ant-api03-not-a-real-key\n"

        meter.render_markdown = poisoned  # type: ignore[method-assign]
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                meter.write(Path(folder) / "usage_report.md")

    def test_placeholder_has_required_headings(self):
        text = pending_full_dataset_report()
        self.assertIn("## Overall totals", text)
        self.assertIn("## Per-model totals", text)
        self.assertIn("awaiting the final full-dataset run", text)
        self.assertNotIn("sk-ant", text)


if __name__ == "__main__":
    unittest.main()
