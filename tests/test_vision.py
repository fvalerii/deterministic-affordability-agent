"""Tests for Claude 3.5 receipt extraction via forced Anthropic tool use.

These tests mock the Anthropic client. They never call the network and they
never read ``ANTHROPIC_API_KEY``.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "code"))

from pydantic import ValidationError  # noqa: E402

from domain import Currency, to_money  # noqa: E402
from tools import evidence, vision  # noqa: E402
from tools.data import load_dataset  # noqa: E402

DATASET = load_dataset(REPO_ROOT / "dataset")
BLANK_EVENT = "event_253"
BLANK_IMAGE = "image_01"


class _ToolUse:
    def __init__(self, payload: dict, *, name: str = vision.TOOL_NAME) -> None:
        self.type = "tool_use"
        self.name = name
        self.input = payload


class _TextBlock:
    type = "text"
    text = "ignored"


class _Response:
    def __init__(self, *blocks: object) -> None:
        self.content = list(blocks)


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
    def __init__(self, responses: list[_Response]) -> None:
        self.messages = _FakeMessages(responses)


class TestReceiptExtractionSchema(unittest.TestCase):
    def test_positive_amount_is_accepted(self):
        parsed = vision.ReceiptExtraction(extracted_amount=4365000.0)
        self.assertEqual(parsed.extracted_amount, 4365000.0)

    def test_zero_and_negative_amounts_are_rejected(self):
        with self.assertRaises(ValidationError):
            vision.ReceiptExtraction(extracted_amount=0)
        with self.assertRaises(ValidationError):
            vision.ReceiptExtraction(extracted_amount=-1.5)

    def test_tool_schema_requires_extracted_amount(self):
        schema = vision.receipt_tool_schema()
        self.assertEqual(schema["type"], "object")
        self.assertIn("extracted_amount", schema["required"])
        self.assertFalse(schema["additionalProperties"])
        field = schema["properties"]["extracted_amount"]
        self.assertEqual(field["description"], "The final total payable amount")


class TestClaudeImageAmountExtractor(unittest.TestCase):
    def setUp(self):
        self.event = DATASET.event(BLANK_EVENT)
        self.image_path = DATASET.image_path(BLANK_IMAGE)
        self.context = evidence.extraction_context(DATASET, self.event)

    def _extractor(self, *payloads: dict, cache_path: Path | None = None):
        responses = [_Response(_ToolUse(payload)) for payload in payloads]
        client = _FakeClient(responses)
        extractor = vision.ImageAmountExtractor(client=client, cache_path=cache_path)
        return extractor, client

    def test_forced_tool_choice_and_system_prompt(self):
        extractor, client = self._extractor({"extracted_amount": 4365000.0})
        result = extractor.extract_amount(
            image_path=self.image_path, event=self.event, context=self.context
        )
        self.assertEqual(result.amount, to_money("4365000"))
        self.assertEqual(result.currency, Currency.IDR)
        self.assertEqual(result.content_hash, vision.sha256_digest(self.image_path))
        call = client.messages.calls[0]
        self.assertEqual(call["system"], vision.SYSTEM_PROMPT)
        self.assertEqual(call["model"], vision.DEFAULT_MODEL)
        self.assertEqual(call["tools"], vision.RECEIPT_TOOLS)
        self.assertEqual(call["tool_choice"], vision.TOOL_CHOICE)
        self.assertEqual(call["tool_choice"]["type"], "tool")
        self.assertEqual(call["tool_choice"]["name"], vision.TOOL_NAME)

    def test_sha256_cache_skips_a_second_api_call(self):
        with tempfile.TemporaryDirectory() as folder:
            cache_path = Path(folder) / "vision_cache.json"
            extractor, client = self._extractor(
                {"extracted_amount": 4365000.0}, cache_path=cache_path
            )
            first = extractor.extract_amount(
                image_path=self.image_path, event=self.event, context=self.context
            )
            second = extractor.extract_amount(
                image_path=self.image_path, event=self.event, context=self.context
            )
            reloaded = vision.ImageAmountExtractor(
                client=_FakeClient([]), cache_path=cache_path
            )
            third = reloaded.extract_amount(
                image_path=self.image_path, event=self.event, context=self.context
            )
        self.assertEqual(client.messages.calls.__len__(), 1)
        self.assertEqual(extractor.api_calls, 1)
        self.assertEqual(first.amount, second.amount)
        self.assertEqual(second.amount, third.amount)
        self.assertEqual(reloaded.api_calls, 0)

    def test_edited_bytes_miss_the_cache(self):
        extractor, client = self._extractor(
            {"extracted_amount": 4365000.0},
            {"extracted_amount": 12.5},
        )
        extractor.extract_amount(
            image_path=self.image_path, event=self.event, context=self.context
        )
        with tempfile.TemporaryDirectory() as folder:
            fake = Path(folder) / "image_01.png"
            fake.write_bytes(b"not-the-original-document")
            edited = extractor.extract_amount(
                image_path=fake, event=self.event, context=self.context
            )
        self.assertEqual(len(client.messages.calls), 2)
        self.assertEqual(edited.amount, to_money("12.5"))
        self.assertNotEqual(edited.content_hash, vision.sha256_digest(self.image_path))

    def test_validation_failure_retries_once(self):
        extractor, client = self._extractor(
            {"extracted_amount": 0},
            {"extracted_amount": 4365000.0},
        )
        result = extractor.extract_amount(
            image_path=self.image_path, event=self.event, context=self.context
        )
        self.assertEqual(result.amount, to_money("4365000"))
        self.assertEqual(len(client.messages.calls), 2)
        retry_text = client.messages.calls[1]["messages"][0]["content"][1]["text"]
        self.assertIn("schema validation", retry_text)

    def test_second_validation_failure_is_refused(self):
        extractor, client = self._extractor(
            {"extracted_amount": 0},
            {"extracted_amount": -4},
        )
        with self.assertRaises(evidence.UnresolvedAmountError):
            extractor.extract_amount(
                image_path=self.image_path, event=self.event, context=self.context
            )
        self.assertEqual(len(client.messages.calls), 2)
        self.assertEqual(extractor.cached_keys, ())

    def test_missing_tool_use_retries_then_refuses(self):
        client = _FakeClient([_Response(_TextBlock()), _Response(_TextBlock())])
        extractor = vision.ImageAmountExtractor(client=client)
        with self.assertRaises(evidence.UnresolvedAmountError):
            extractor.extract_amount(
                image_path=self.image_path, event=self.event, context=self.context
            )
        self.assertEqual(len(client.messages.calls), 2)

    def test_wraps_with_existing_sha256_file_cache(self):
        client = _FakeClient([_Response(_ToolUse({"extracted_amount": 4365000.0}))])
        with tempfile.TemporaryDirectory() as folder:
            cached = evidence.CachedImageAmountExtractor(
                cache_path=Path(folder) / "cache.json",
                delegate=vision.ImageAmountExtractor(client=client),
            )
            first = evidence.extract_image_amount(
                DATASET,
                image_id=BLANK_IMAGE,
                event_id=BLANK_EVENT,
                extractor=cached,
            )
            reloaded = evidence.CachedImageAmountExtractor(
                cache_path=Path(folder) / "cache.json",
                delegate=vision.ImageAmountExtractor(client=_FakeClient([])),
            )
            second = evidence.extract_image_amount(
                DATASET,
                image_id=BLANK_IMAGE,
                event_id=BLANK_EVENT,
                extractor=reloaded,
            )
        self.assertEqual(client.messages.calls.__len__(), 1)
        self.assertEqual(first.amount, second.amount)


if __name__ == "__main__":
    unittest.main()
