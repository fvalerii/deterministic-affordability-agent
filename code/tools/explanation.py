"""Claude 3.5 Sonnet synthesizer for ``decision_explanation``.

Money math never happens here. The model is only allowed to rewrite a
:class:`~domain.CandidatePlan` into one or two sentences, and it must cite the
user's ``financial_priorities``. Numbers, dates, and spending cuts are copied
from the plan; they are never recalculated.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

_CODE_DIR = Path(__file__).resolve().parent.parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from domain import CandidatePlan  # noqa: E402
from tools.usage import METER  # noqa: E402

SYSTEM_PROMPT = (
    "You are a precise financial text synthesizer. Write a concise 1-2 sentence "
    "decision_explanation summarizing the provided CandidatePlan.\n"
    "CRITICAL RULE 1: You MUST explicitly cite the user's financial_priorities "
    "from the provided profile in your explanation (e.g., \"To maintain your "
    "priority of [Priority]...\").\n"
    "CRITICAL RULE 2: Do NOT recalculate or hallucinate any numbers, dates, or "
    "spending cuts. Use EXACTLY the numerical and categorical values provided "
    "in the CandidatePlan."
)

DEFAULT_MODEL = os.environ.get("ANTHROPIC_EXPLANATION_MODEL", "claude-sonnet-4-6")
TOOL_NAME = "submit_explanation"
MAX_TOKENS = 300
DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
EVENT_RE = re.compile(r"event_\d+", re.IGNORECASE)
NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")

EXPLANATION_TOOLS: list[dict[str, Any]] = [
    {
        "name": TOOL_NAME,
        "description": "Submit the 1-2 sentence decision_explanation.",
        "input_schema": {
            "type": "object",
            "properties": {
                "decision_explanation": {
                    "type": "string",
                    "minLength": 1,
                    "description": "Concise 1-2 sentence decision_explanation",
                }
            },
            "required": ["decision_explanation"],
            "additionalProperties": False,
        },
    }
]

TOOL_CHOICE: dict[str, str] = {"type": "tool", "name": TOOL_NAME}


class ExplanationDraft(BaseModel):
    decision_explanation: str = Field(..., min_length=1)


def profile_priorities(profile: dict) -> tuple[str, ...]:
    raw = profile.get("financial_priorities", ())
    if raw is None:
        return ()
    if isinstance(raw, str):
        return tuple(part.strip() for part in raw.split("|") if part.strip())
    return tuple(str(item) for item in raw if str(item).strip())


def plan_payload(plan: CandidatePlan) -> dict[str, Any]:
    """The only numerical and categorical values the model may quote."""

    return {
        "request_id": plan.request_id,
        "recommended_payment_method": str(plan.method),
        "affordability_status": str(plan.affordability_status),
        "payment_plan": plan.render_payment_plan(),
        "spending_changes_needed": plan.render_spending_changes(),
        "payments": [
            {"date": payment.date.isoformat(), "amount": format(payment.amount, "f")}
            for payment in plan.payments
        ],
        "total_payable": format(plan.total_payable, "f"),
        "completes_request": plan.completes_request,
        "completes_by_deadline": plan.completes_by_deadline,
        "payment_option_id": plan.payment_option_id,
    }


def deterministic_explanation(plan: CandidatePlan, profile: dict) -> str:
    """Grounded fallback that cites priorities and copies plan values exactly."""

    priorities = profile_priorities(profile)
    if not priorities:
        lead = "To maintain your stated financial priorities"
    elif len(priorities) == 1:
        lead = f"To maintain your priority of {priorities[0]}"
    else:
        lead = (
            "To maintain your priority of "
            + f"{priorities[0]} (also {', '.join(priorities[1:])})"
        )
    method = str(plan.method)
    status = str(plan.affordability_status)
    changes = plan.render_spending_changes()
    schedule = plan.render_payment_plan()
    if method == "not_recommended":
        return (
            f"{lead}, this request is {status}, so the recommendation is "
            f"{method} with payment_plan {schedule} and spending_changes_needed "
            f"{changes}."
        )
    change_clause = "" if changes == "none" else f" after {changes}"
    return (
        f"{lead}, the recommendation is {method} ({status}){change_clause} "
        f"using payment_plan {schedule}."
    )


def explanation_cache_key(plan: CandidatePlan, profile: dict) -> str:
    blob = json.dumps(
        {"plan": plan_payload(plan), "priorities": list(profile_priorities(profile))},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _parse_decimal(token: str) -> Decimal | None:
    try:
        return Decimal(token.strip().replace(",", ""))
    except (InvalidOperation, AttributeError):
        return None


def _allowed_dates_and_amounts(plan: CandidatePlan, profile: dict) -> tuple[set[str], set[Decimal]]:
    dates = {payment.date.isoformat() for payment in plan.payments}
    amounts = {payment.amount for payment in plan.payments}
    safe = profile.get("amount_safe_to_pay")
    parsed = _parse_decimal(str(safe)) if safe is not None else None
    if parsed is not None:
        amounts.add(parsed)
    return dates, amounts


def _validate_explanation(text: str, plan: CandidatePlan, profile: dict) -> str:
    cleaned = " ".join(text.split()).strip()
    if not cleaned:
        raise ValueError("decision_explanation is empty")
    priorities = profile_priorities(profile)
    lowered = cleaned.lower()
    if priorities and not any(priority.lower() in lowered for priority in priorities):
        raise ValueError("decision_explanation does not cite financial_priorities")
    allowed_dates, allowed_amounts = _allowed_dates_and_amounts(plan, profile)
    bad_dates = [when for when in DATE_RE.findall(cleaned) if when not in allowed_dates]
    stripped = DATE_RE.sub(" ", cleaned)
    stripped = EVENT_RE.sub(" ", stripped)
    bad_amounts = []
    for token in NUMBER_RE.findall(stripped):
        parsed = _parse_decimal(token)
        if parsed is None or parsed not in allowed_amounts:
            bad_amounts.append(token)
    if bad_dates or bad_amounts:
        raise ValueError(f"decision_explanation invents values dates={bad_dates} amounts={bad_amounts}")
    return cleaned


def _parse_tool_use(response: Any) -> ExplanationDraft:
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
        return ExplanationDraft.model_validate(payload)
    raise ValueError("Claude did not call submit_explanation")


class ExplanationGenerator:
    """Forced-tool Claude 3.5 writer for ``decision_explanation``."""

    name = "claude-3-5-sonnet"

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
        self.api_calls = 0
        self._cache: dict[str, str] = {}
        if cache_path is not None and cache_path.is_file():
            raw = json.loads(cache_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                self._cache = {str(key): str(value) for key, value in raw.items()}

    def generate(self, plan: CandidatePlan, profile: dict) -> str:
        key = explanation_cache_key(plan, profile)
        cached = self._cache.get(key)
        if cached is not None:
            METER.record_local_cache_hit(purpose="explanation")
            return cached
        if not self._should_call_model():
            text = deterministic_explanation(plan, profile)
            self._remember(key, text)
            return text
        text = self._generate_with_retry(plan, profile)
        self._remember(key, text)
        return text

    def _should_call_model(self) -> bool:
        if self.client is not None:
            return True
        from tools.env import anthropic_api_key

        return bool(anthropic_api_key())

    def _generate_with_retry(self, plan: CandidatePlan, profile: dict) -> str:
        last_error: Exception | None = None
        attempts = self.max_retries + 1
        for attempt in range(attempts):
            try:
                response = self._call_model(plan, profile, retry=attempt > 0)
                draft = _parse_tool_use(response)
                return _validate_explanation(draft.decision_explanation, plan, profile)
            except (ValidationError, ValueError, TypeError) as exc:
                last_error = exc
                continue
        _ = last_error
        return deterministic_explanation(plan, profile)

    def _call_model(self, plan: CandidatePlan, profile: dict, *, retry: bool) -> Any:
        client = self.client if self.client is not None else self._build_client()
        user_payload = {
            "candidate_plan": plan_payload(plan),
            "profile": {
                "financial_priorities": list(profile_priorities(profile)),
                "home_currency": profile.get("home_currency"),
                "amount_safe_to_pay": profile.get("amount_safe_to_pay"),
            },
        }
        user_text = (
            "Write the decision_explanation from this CandidatePlan and profile. "
            "Cite financial_priorities. The only numbers and dates you may use are "
            "amount_safe_to_pay and the payment_plan entries.\n"
            + json.dumps(user_payload, indent=2, sort_keys=True)
        )
        if retry:
            user_text += (
                "\nPrevious draft failed validation. Cite a provided priority and "
                "reuse only the CandidatePlan values above."
            )
        self.api_calls += 1
        response = client.messages.create(
            model=self.model,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=EXPLANATION_TOOLS,
            tool_choice=TOOL_CHOICE,
            messages=[{"role": "user", "content": user_text}],
        )
        METER.record(response, model=self.model, purpose="explanation")
        return response

    def _build_client(self) -> Any:
        from anthropic import Anthropic

        from tools.env import anthropic_api_key

        api_key = anthropic_api_key()
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        return Anthropic(api_key=api_key)

    def _remember(self, key: str, text: str) -> None:
        self._cache[key] = text
        if self.cache_path is None:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps(self._cache, indent=2, sort_keys=True), encoding="utf-8"
        )


_GENERATOR: ExplanationGenerator | None = None


def default_explanation_generator() -> ExplanationGenerator:
    global _GENERATOR
    if _GENERATOR is None:
        _GENERATOR = ExplanationGenerator()
    return _GENERATOR


def generate_explanation(plan: CandidatePlan, profile: dict) -> str:
    """Write a 1-2 sentence ``decision_explanation`` for a ranked plan."""

    return default_explanation_generator().generate(plan, profile)
