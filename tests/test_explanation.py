"""Tests for Claude 3.5 decision_explanation synthesis.

These tests mock the Anthropic client. They never call the network.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "code"))

from domain import (  # noqa: E402
    AffordabilityStatus,
    CandidatePlan,
    Payment,
    RecommendedPaymentMethod,
    to_money,
)
from tools import explanation  # noqa: E402


def _plan() -> CandidatePlan:
    return CandidatePlan(
        request_id="request_01",
        method=RecommendedPaymentMethod.FULL_PAYMENT,
        affordability_status=AffordabilityStatus.AFFORDABLE_NOW,
        payments=(Payment(date=date(2024, 3, 3), amount=to_money("25256")),),
        total_payable=to_money("25256"),
        completes_request=True,
        completes_by_deadline=True,
    )


PROFILE = {
    "financial_priorities": ["emergency_savings", "debt_reduction"],
    "home_currency": "ZAR",
}


class _ToolUse:
    def __init__(self, payload: dict) -> None:
        self.type = "tool_use"
        self.name = explanation.TOOL_NAME
        self.input = payload


class _Response:
    def __init__(self, payload: dict) -> None:
        self.content = [_ToolUse(payload)]
        self.usage = type("Usage", (), {"input_tokens": 11, "output_tokens": 7})()


class _FakeMessages:
    def __init__(self, responses: list[_Response]) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError("unexpected extra Anthropic call")
        return self._responses.pop(0)


class _FakeClient:
    def __init__(self, texts: list[str]) -> None:
        self.messages = _FakeMessages(
            [_Response({"decision_explanation": text}) for text in texts]
        )


class TestExplanationSchemaAndPrompt(unittest.TestCase):
    def test_system_prompt_requires_priorities_and_forbids_hallucinated_numbers(self):
        self.assertIn("precise financial text synthesizer", explanation.SYSTEM_PROMPT)
        self.assertIn("financial_priorities", explanation.SYSTEM_PROMPT)
        self.assertIn("To maintain your priority of [Priority]", explanation.SYSTEM_PROMPT)
        self.assertIn("Do NOT recalculate or hallucinate", explanation.SYSTEM_PROMPT)

    def test_fallback_cites_priority_and_copies_plan_values(self):
        text = explanation.deterministic_explanation(_plan(), PROFILE)
        self.assertIn("emergency_savings", text)
        self.assertIn("full_payment", text)
        self.assertIn("2024-03-03:25256", text)
        self.assertIn("affordable_now", text)


class TestExplanationGenerator(unittest.TestCase):
    def test_forced_tool_choice_and_exact_system_prompt(self):
        client = _FakeClient(
            [
                "To maintain your priority of emergency_savings, the recommendation is "
                "full_payment (affordable_now) using payment_plan 2024-03-03:25256."
            ]
        )
        generator = explanation.ExplanationGenerator(client=client)
        text = generator.generate(_plan(), PROFILE)
        self.assertIn("emergency_savings", text)
        call = client.messages.calls[0]
        self.assertEqual(call["system"], explanation.SYSTEM_PROMPT)
        self.assertEqual(call["model"], explanation.DEFAULT_MODEL)
        self.assertEqual(call["tools"], explanation.EXPLANATION_TOOLS)
        self.assertEqual(call["tool_choice"], explanation.TOOL_CHOICE)

    def test_invented_number_retries_once_then_accepts_valid_draft(self):
        client = _FakeClient(
            [
                "To maintain your priority of emergency_savings, pay 999999 today.",
                "To maintain your priority of emergency_savings, the recommendation is "
                "full_payment (affordable_now) using payment_plan 2024-03-03:25256.",
            ]
        )
        generator = explanation.ExplanationGenerator(client=client)
        text = generator.generate(_plan(), PROFILE)
        self.assertEqual(len(client.messages.calls), 2)
        self.assertIn("25256", text)
        self.assertNotIn("999999", text)

    def test_two_invalid_drafts_fall_back_to_deterministic_text(self):
        client = _FakeClient(
            [
                "Pay 999999 today.",
                "Pay 888888 tomorrow.",
            ]
        )
        generator = explanation.ExplanationGenerator(client=client)
        text = generator.generate(_plan(), PROFILE)
        self.assertEqual(len(client.messages.calls), 2)
        self.assertIn("emergency_savings", text)
        self.assertIn("2024-03-03:25256", text)

    def test_sha256_cache_skips_a_second_api_call(self):
        with tempfile.TemporaryDirectory() as folder:
            cache_path = Path(folder) / "explanations.json"
            client = _FakeClient(
                [
                    "To maintain your priority of emergency_savings, the recommendation is "
                    "full_payment (affordable_now) using payment_plan 2024-03-03:25256."
                ]
            )
            generator = explanation.ExplanationGenerator(
                client=client, cache_path=cache_path
            )
            first = generator.generate(_plan(), PROFILE)
            second = generator.generate(_plan(), PROFILE)
            reloaded = explanation.ExplanationGenerator(
                client=_FakeClient([]), cache_path=cache_path
            )
            third = reloaded.generate(_plan(), PROFILE)
        self.assertEqual(len(client.messages.calls), 1)
        self.assertEqual(first, second)
        self.assertEqual(second, third)

    def test_public_generate_explanation_uses_fallback_without_api_key(self):
        previous = os.environ.pop("ANTHROPIC_API_KEY", None)
        explanation._GENERATOR = None
        try:
            text = explanation.generate_explanation(_plan(), PROFILE)
        finally:
            explanation._GENERATOR = None
            if previous is not None:
                os.environ["ANTHROPIC_API_KEY"] = previous
        self.assertIn("emergency_savings", text)
        self.assertIn("full_payment", text)


if __name__ == "__main__":
    unittest.main()
