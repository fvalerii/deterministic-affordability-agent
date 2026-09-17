"""Token and cost accounting for Anthropic calls in this submission.

``evaluation/usage_report.md`` is overwritten by the run that writes
``output.csv``. Calibration and hold-out runs write a separate file so they
cannot be mistaken for the full-dataset report. API keys are never recorded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


def prices_usd_per_million(model: str) -> tuple[float, float]:
    """Return ``(input, output)`` USD per million tokens for a Claude model id."""

    name = (model or "").lower()
    if "haiku-4-5" in name or "haiku-4.5" in name:
        return 1.0, 5.0
    if "haiku-3-5" in name or "haiku-3.5" in name:
        return 0.80, 4.0
    if "opus-5" in name:
        return 5.0, 25.0
    if "opus-4" in name:
        return 15.0, 75.0
    if "sonnet-5" in name and "sonnet-4" not in name:
        return 3.0, 15.0
    # Claude 3.5 Sonnet, Sonnet 4.5, Sonnet 4.6
    return 3.0, 15.0


def _cost_usd(input_tokens: int, output_tokens: int, model: str) -> float:
    input_rate, output_rate = prices_usd_per_million(model)
    return (input_tokens / 1_000_000) * input_rate + (output_tokens / 1_000_000) * output_rate


def _usage_fields(response: object) -> tuple[int, int, int, int]:
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage") or {}
        return (
            int(usage.get("input_tokens", 0) or 0),
            int(usage.get("output_tokens", 0) or 0),
            int(usage.get("cache_creation_input_tokens", 0) or 0),
            int(usage.get("cache_read_input_tokens", 0) or 0),
        )
    return (
        int(getattr(usage, "input_tokens", 0) or 0),
        int(getattr(usage, "output_tokens", 0) or 0),
        int(getattr(usage, "cache_creation_input_tokens", 0) or 0),
        int(getattr(usage, "cache_read_input_tokens", 0) or 0),
    )


@dataclass
class ModelBucket:
    provider: str
    model: str
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    purpose_calls: dict[str, int] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def estimated_cost_usd(self) -> float:
        return _cost_usd(self.input_tokens, self.output_tokens, self.model)


@dataclass
class UsageMeter:
    provider: str = "Anthropic"
    requests_processed: int = 0
    local_cache_hits: int = 0
    output_artifact: str = "output.csv"
    split: str = "eval"
    started_at: str = ""
    planned_requests: int = 0
    buckets: dict[str, ModelBucket] = field(default_factory=dict)

    def reset(self) -> None:
        self.requests_processed = 0
        self.local_cache_hits = 0
        self.output_artifact = "output.csv"
        self.split = "eval"
        self.started_at = ""
        self.planned_requests = 0
        self.buckets = {}

    def begin_run(self, *, split: str, output_artifact: str, request_count: int = 0) -> None:
        self.reset()
        self.split = split
        self.output_artifact = output_artifact
        self.planned_requests = request_count
        self.started_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def record(
        self,
        response: object,
        *,
        model: str,
        purpose: str = "unspecified",
        provider: str = "Anthropic",
    ) -> None:
        input_tokens, output_tokens, cache_write, cache_read = _usage_fields(response)
        bucket = self.buckets.setdefault(
            model, ModelBucket(provider=provider, model=model)
        )
        bucket.calls += 1
        bucket.input_tokens += input_tokens
        bucket.output_tokens += output_tokens
        bucket.cache_creation_tokens += cache_write
        bucket.cache_read_tokens += cache_read
        bucket.purpose_calls[purpose] = bucket.purpose_calls.get(purpose, 0) + 1

    def record_local_cache_hit(self, *, purpose: str = "unspecified") -> None:
        self.local_cache_hits += 1
        _ = purpose

    @property
    def calls(self) -> int:
        return sum(bucket.calls for bucket in self.buckets.values())

    @property
    def input_tokens(self) -> int:
        return sum(bucket.input_tokens for bucket in self.buckets.values())

    @property
    def output_tokens(self) -> int:
        return sum(bucket.output_tokens for bucket in self.buckets.values())

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def average_tokens_per_request(self) -> float:
        if self.requests_processed <= 0:
            return 0.0
        return self.total_tokens / self.requests_processed

    @property
    def estimated_total_cost_usd(self) -> float:
        return sum(bucket.estimated_cost_usd for bucket in self.buckets.values())

    @property
    def estimated_cost_per_request_usd(self) -> float:
        if self.requests_processed <= 0:
            return 0.0
        return self.estimated_total_cost_usd / self.requests_processed

    @property
    def model_names(self) -> tuple[str, ...]:
        return tuple(sorted(self.buckets))

    def render_markdown(self) -> str:
        models = self.model_names or ("(none — no model API calls)",)
        lines = [
            "# Token usage report",
            "",
            "This file is the single token-usage artifact required in `code.zip` "
            "as `evaluation/usage_report.md`. It contains **no API keys, credentials, "
            "or environment values**.",
            "",
            "## Run that produced this report",
            "",
            f"- Output artifact: `{self.output_artifact}`",
            f"- Dataset split: `{self.split}`",
            f"- UTC start: {self.started_at or '(not recorded)'}",
            f"- Evaluation requests processed: {self.requests_processed}",
            f"- Local SHA-256 cache hits (no API call): {self.local_cache_hits}",
            "",
            "## Overall totals",
            "",
            f"- Provider: {self.provider}",
            f"- Model names: {', '.join(models)}",
            f"- Model calls: {self.calls}",
            f"- Input tokens: {self.input_tokens}",
            f"- Output tokens: {self.output_tokens}",
            f"- Total tokens: {self.total_tokens}",
            f"- Average tokens per request: {self.average_tokens_per_request:.2f}",
            f"- Estimated total cost (USD): {self.estimated_total_cost_usd:.6f}",
            f"- Estimated cost per request (USD): {self.estimated_cost_per_request_usd:.6f}",
            "",
            "## Per-model totals",
            "",
        ]
        if not self.buckets:
            lines.append("No model API calls were made in this run.")
            lines.append("")
        else:
            lines.append(
                "| provider | model | calls | input tokens | output tokens | "
                "total tokens | purposes | estimated cost (USD) |"
            )
            lines.append("|---|---|---:|---:|---:|---:|---|---:|")
            for model in sorted(self.buckets):
                bucket = self.buckets[model]
                purposes = ", ".join(
                    f"{name}={count}" for name, count in sorted(bucket.purpose_calls.items())
                ) or "unspecified"
                input_rate, output_rate = prices_usd_per_million(model)
                lines.append(
                    f"| {bucket.provider} | `{bucket.model}` | {bucket.calls} | "
                    f"{bucket.input_tokens} | {bucket.output_tokens} | "
                    f"{bucket.total_tokens} | {purposes} | "
                    f"{bucket.estimated_cost_usd:.6f} |"
                )
                lines.append("")
                lines.append(
                    f"Pricing used for `{bucket.model}`: "
                    f"${input_rate:.2f}/M input, ${output_rate:.2f}/M output "
                    "(Anthropic published list prices, global standard routing)."
                )
                lines.append("")
            lines.append(
                f"**Overall** — calls {self.calls}, tokens {self.total_tokens}, "
                f"estimated cost ${self.estimated_total_cost_usd:.6f}."
            )
            lines.append("")
        lines.extend(
            [
                "## Notes",
                "",
                "- Money math is done by deterministic Python tools and is not billed as tokens.",
                "- Vision (receipt OCR) and explanation synthesis are the only Anthropic calls.",
                "- Average tokens per request uses evaluation requests in `output.csv`, "
                "not model-call count.",
                "- Re-running the same images or plans hits an in-process SHA-256 cache and "
                "does not add API tokens.",
                "",
            ]
        )
        return "\n".join(lines)

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        text = self.render_markdown()
        if "sk-ant" in text.lower() or "sk-ant" in text:
            raise ValueError("usage report must not contain API keys")
        path.write_text(text, encoding="utf-8")
        return path


def pending_full_dataset_report() -> str:
    """Placeholder until `python3 code/main.py` writes the real full-dataset figures."""

    return (
        "# Token usage report\n\n"
        "This file is the single token-usage artifact required in `code.zip` "
        "as `evaluation/usage_report.md`. It contains **no API keys, credentials, "
        "or environment values**.\n\n"
        "## Run that produced this report\n\n"
        "- Output artifact: `output.csv`\n"
        "- Dataset split: `eval` (`dataset/requests.csv`, 250 requests)\n"
        "- Status: **awaiting the final full-dataset run**\n"
        "- Evaluation requests processed: 0\n\n"
        "## Overall totals\n\n"
        "- Provider: Anthropic\n"
        "- Model names: (none yet)\n"
        "- Model calls: 0\n"
        "- Input tokens: 0\n"
        "- Output tokens: 0\n"
        "- Total tokens: 0\n"
        "- Average tokens per request: 0.00\n"
        "- Estimated total cost (USD): 0.000000\n"
        "- Estimated cost per request (USD): 0.000000\n\n"
        "## Per-model totals\n\n"
        "No model API calls have been recorded for the full-dataset run yet. "
        "`python3 code/main.py` (default `--split eval`) overwrites this file "
        "when it writes `output.csv`.\n\n"
        "Expected models in that run:\n\n"
        "| provider | model | purpose |\n"
        "|---|---|---|\n"
        "| Anthropic | `claude-sonnet-4-6` | receipt vision (`code/tools/vision.py`) |\n"
        "| Anthropic | `claude-sonnet-4-6` | decision_explanation (`code/tools/explanation.py`) |\n\n"
        "If those IDs differ at run time, both per-model and overall totals will "
        "appear in this file.\n\n"
        "## Notes\n\n"
        "- Money math is done by deterministic Python tools and is not billed as tokens.\n"
        "- Do not paste API keys into this file.\n"
    )


METER = UsageMeter()
