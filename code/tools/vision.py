"""Claude 3.5 Sonnet receipt OCR via forced Anthropic tool use.

This module is the multimodal adapter behind :class:`tools.evidence.ImageAmountExtractor`.
Money math never happens here. The model is only allowed to emit a single positive
``extracted_amount`` that matches :class:`ReceiptExtraction`. SHA-256 of the PNG
bytes is the cache key so an edited file cannot reuse a stale total.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

_CODE_DIR = Path(__file__).resolve().parent.parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from domain import ExtractedAmount, EventRecord, to_money  # noqa: E402
from tools.evidence import UnresolvedAmountError  # noqa: E402
from tools.usage import METER  # noqa: E402

SYSTEM_PROMPT = (
    "You are a highly precise OCR and data extraction system. Analyze the "
    "receipt image and extract ONLY the final total payable amount. Ignore "
    "subtotals, taxes, and dates."
)

DEFAULT_MODEL = os.environ.get("ANTHROPIC_VISION_MODEL", "claude-sonnet-4-6")
TOOL_NAME = "receipt_extraction"
MAX_TOKENS = 256
MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


class ReceiptExtraction(BaseModel):
    """The only structured object Claude is allowed to return for a receipt."""

    extracted_amount: float = Field(
        ..., gt=0, description="The final total payable amount"
    )


def receipt_tool_schema() -> dict[str, Any]:
    """JSON Schema Anthropic receives for the forced ``receipt_extraction`` tool."""

    schema = ReceiptExtraction.model_json_schema()
    return {
        "type": "object",
        "properties": schema["properties"],
        "required": schema["required"],
        "additionalProperties": False,
    }


RECEIPT_TOOLS: list[dict[str, Any]] = [
    {
        "name": TOOL_NAME,
        "description": (
            "Report the final total payable amount shown on the receipt. "
            "Do not report subtotals, tax lines, cash tendered, or change."
        ),
        "input_schema": receipt_tool_schema(),
    }
]

TOOL_CHOICE: dict[str, str] = {"type": "tool", "name": TOOL_NAME}


def sha256_digest(image_path: Path) -> str:
    """Hex SHA-256 of the image file bytes. The cache key is the document, not its id."""

    return hashlib.sha256(image_path.read_bytes()).hexdigest()


def _media_type(image_path: Path) -> str:
    return MEDIA_TYPES.get(image_path.suffix.lower(), "image/png")


def _parse_tool_use(response: Any) -> ReceiptExtraction:
    """Require a tool_use block whose input validates as :class:`ReceiptExtraction`."""

    content = getattr(response, "content", None)
    if content is None and isinstance(response, dict):
        content = response.get("content", [])
    if not content:
        raise ValueError("Claude returned no content blocks")

    for block in content:
        block_type = getattr(block, "type", None)
        if block_type is None and isinstance(block, dict):
            block_type = block.get("type")
        if block_type != "tool_use":
            continue
        payload = getattr(block, "input", None)
        if payload is None and isinstance(block, dict):
            payload = block.get("input")
        if payload is None:
            raise ValueError("tool_use block is missing input")
        return ReceiptExtraction.model_validate(payload)

    raise ValueError("Claude did not call receipt_extraction")


class ImageAmountExtractor:
    """Reads one positive total from a receipt image using Claude 3.5 Sonnet.

    The Anthropic ``tools`` array plus ``tool_choice`` force a parseable JSON
    object that matches :class:`ReceiptExtraction`. Hits on the SHA-256 cache
    skip the API. A validation failure is retried once and never cached.
    """

    name = "claude-3-5"

    def __init__(
        self,
        *,
        client: Any | None = None,
        model: str = DEFAULT_MODEL,
        cache_path: Path | None = None,
        max_retries: int = 1,
    ) -> None:
        self.client = client
        self.model = model
        self.cache_path = cache_path
        self.max_retries = max_retries
        self._cache: dict[str, float] = {}
        self.api_calls = 0
        if cache_path is not None and cache_path.is_file():
            raw = json.loads(cache_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                self._cache = {str(key): float(value) for key, value in raw.items()}

    def extract_amount(
        self, *, image_path: Path, event: EventRecord, context: str
    ) -> ExtractedAmount:
        if not image_path.is_file():
            raise UnresolvedAmountError(
                f"{event.event_id}: image file {image_path} is missing; a blank "
                "amount is never zero"
            )

        digest = sha256_digest(image_path)
        cached_amount = self._cache.get(digest)
        if cached_amount is not None:
            METER.record_local_cache_hit(purpose="vision")
            return self._to_extracted(
                image_path=image_path,
                event=event,
                amount=cached_amount,
                digest=digest,
                evidence_text="sha256-cache",
            )

        extraction = self._extract_with_retry(image_path=image_path, context=context)
        self._remember(digest, extraction.extracted_amount)
        return self._to_extracted(
            image_path=image_path,
            event=event,
            amount=extraction.extracted_amount,
            digest=digest,
            evidence_text=f"extracted_amount={extraction.extracted_amount}",
        )

    def _extract_with_retry(
        self, *, image_path: Path, context: str
    ) -> ReceiptExtraction:
        attempts = self.max_retries + 1
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                response = self._call_model(
                    image_path=image_path,
                    context=context,
                    retry=attempt > 0,
                )
                return _parse_tool_use(response)
            except (ValidationError, ValueError, TypeError) as exc:
                last_error = exc
                continue
        raise UnresolvedAmountError(
            f"{image_path.name}: Claude 3.5 did not return a valid "
            f"ReceiptExtraction after {attempts} attempt(s): {last_error}"
        ) from last_error

    def _call_model(self, *, image_path: Path, context: str, retry: bool) -> Any:
        client = self.client if self.client is not None else self._build_client()
        image_bytes = image_path.read_bytes()
        user_text = context.strip() or (
            "Extract ONLY the final total payable amount from this receipt."
        )
        if retry:
            user_text += (
                " Previous extraction failed schema validation. Return a single "
                "positive extracted_amount via the receipt_extraction tool."
            )
        self.api_calls += 1
        response = client.messages.create(
            model=self.model,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=RECEIPT_TOOLS,
            tool_choice=TOOL_CHOICE,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": _media_type(image_path),
                                "data": base64.b64encode(image_bytes).decode("ascii"),
                            },
                        },
                        {"type": "text", "text": user_text},
                    ],
                }
            ],
        )
        METER.record(response, model=self.model, purpose="vision")
        return response

    def _build_client(self) -> Any:
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise UnresolvedAmountError(
                "the anthropic package is not installed; pip install anthropic"
            ) from exc
        from tools.env import anthropic_api_key

        api_key = anthropic_api_key()
        if not api_key:
            raise UnresolvedAmountError(
                "ANTHROPIC_API_KEY is not set; refusing to guess a receipt total"
            )
        return Anthropic(api_key=api_key)

    def _remember(self, digest: str, amount: float) -> None:
        self._cache[digest] = amount
        if self.cache_path is None:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps(self._cache, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    @staticmethod
    def _to_extracted(
        *,
        image_path: Path,
        event: EventRecord,
        amount: float,
        digest: str,
        evidence_text: str,
    ) -> ExtractedAmount:
        return ExtractedAmount(
            image_id=image_path.stem,
            event_id=event.event_id,
            amount=to_money(str(amount)),
            currency=event.currency,
            confidence=1.0,
            evidence_text=evidence_text,
            content_hash=digest,
        )

    @property
    def cached_keys(self) -> tuple[str, ...]:
        return tuple(sorted(self._cache))
