"""Request-level orchestrator: deterministic kernel plus explanation synthesis.

The LLM never computes money. It only writes ``decision_explanation`` from a
validated :class:`~domain.CandidatePlan`. Plan selection is the ranked output of
the existing ledger, forecast, planning, and validation tools.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

_CODE_DIR = Path(__file__).resolve().parent.parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from domain import (  # noqa: E402
    OUTPUT_COLUMNS,
    AffordabilityStatus,
    CandidatePlan,
    FinalDecision,
    RequestRecord,
    format_amount_field,
)
from tools.data import Dataset  # noqa: E402
from tools.evidence import build_adjustments  # noqa: E402
from tools.explanation import generate_explanation  # noqa: E402
from tools.forecast import (  # noqa: E402
    calculate_safe_amount_today,
    find_earliest_safe_full_payment,
)
from tools.ledger import DEFAULT_POLICY, build_financial_state  # noqa: E402
from tools.planning import (  # noqa: E402
    generate_candidates,
    not_recommended_candidate,
    rank_valid_candidates,
)
from tools.usage import METER  # noqa: E402
from tools.validation import validate_output_row, validate_plan_constraints  # noqa: E402


class BuyOrWaitOrchestrator:
    """Runs one request through the financial kernel, then asks Claude to explain."""

    def __init__(self, dataset: Dataset, *, explain=generate_explanation, extractor=None) -> None:
        self.dataset = dataset
        self.explain = explain
        self.extractor = extractor

    def resolve_request(self, request_id: str) -> RequestRecord:
        if request_id in self.dataset.requests:
            return self.dataset.requests[request_id]
        sample = self.dataset.sample_requests.get(request_id)
        if sample is None:
            raise KeyError(f"unknown request_id {request_id!r}")
        return sample

    def recommend(self, request: RequestRecord) -> FinalDecision:
        adjustments, _, _ = build_adjustments(
            self.dataset, request, extractor=self.extractor
        )
        state = build_financial_state(
            self.dataset, request, adjustments=adjustments, policy=DEFAULT_POLICY
        )
        safe = calculate_safe_amount_today(state, request)
        earliest = find_earliest_safe_full_payment(state, request)
        generated = generate_candidates(self.dataset, state, request)

        valid: list[CandidatePlan] = []
        for candidate in generated:
            result = validate_plan_constraints(self.dataset, state, request, candidate)
            if result.valid:
                valid.append(candidate)

        if valid:
            ranked = rank_valid_candidates(request, tuple(valid))
            best = ranked.best
        else:
            best = not_recommended_candidate(
                request, "no candidate survived independent validation"
            )
        assert best is not None
        validation = validate_plan_constraints(self.dataset, state, request, best)
        if not validation.valid:
            best = not_recommended_candidate(
                request, "; ".join(validation.violations) or "validation failed"
            )
            validation = validate_plan_constraints(self.dataset, state, request, best)

        profile = self.dataset.profile(request.user_id).model_dump(mode="json")
        profile["amount_safe_to_pay"] = format_amount_field(safe.amount_safe_to_pay)
        explanation = self.explain(best, profile)
        earliest_date = earliest.earliest_date
        if best.affordability_status is AffordabilityStatus.AFFORDABLE_NOW:
            earliest_date = request.request_date

        return FinalDecision(
            request_id=request.request_id,
            request_date=request.request_date,
            requested_amount=request.requested_amount,
            amount_safe_to_pay=safe.amount_safe_to_pay,
            affordability_status=best.affordability_status,
            recommended_payment_method=best.method,
            payments=best.payments,
            earliest_date_for_full_payment=earliest_date,
            spending_changes=best.spending_changes,
            decision_explanation=explanation,
            validation=validation,
        )

    def recommend_id(self, request_id: str) -> FinalDecision:
        return self.recommend(self.resolve_request(request_id))

    def run(
        self,
        *,
        output_path: Path | None = None,
        request_ids: tuple[str, ...] | None = None,
        usage_report_path: Path | None = None,
    ) -> Path:
        if request_ids is None:
            requests = tuple(self.dataset.iter_requests())
        else:
            requests = tuple(self.resolve_request(request_id) for request_id in request_ids)
        rows: list[dict[str, str]] = []
        for request in requests:
            print(f"processing {request.request_id}", flush=True)
            decision = self.recommend(request)
            row = decision.to_output_row()
            problems = validate_output_row(row, request=request)
            if problems:
                raise ValueError(
                    f"{request.request_id}: output row failed validation: "
                    + "; ".join(problems)
                )
            rows.append(row)
            METER.requests_processed += 1
        path = output_path or self.dataset.paths.output_csv
        write_output_csv(path, rows)
        report_path = usage_report_path or (
            self.dataset.paths.repo_root / "code" / "evaluation" / "usage_report.md"
        )
        METER.write(report_path)
        return path


def write_output_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(OUTPUT_COLUMNS),
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row[column] for column in OUTPUT_COLUMNS})
