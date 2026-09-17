"""Deterministic verification of a candidate plan against every hard rule.

This is the gate the agent cannot talk its way past. It re-derives the facts it
needs from the dataset rather than trusting anything handed to it, and it
re-simulates the plan instead of believing an earlier simulation.

A violation list is returned rather than an exception so a caller can repair a
plan, and every message names the rule it failed.
"""

from __future__ import annotations

import sys
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Mapping

_CODE_DIR = Path(__file__).resolve().parent.parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from domain import (  # noqa: E402
    MAX_SPENDING_CHANGES,
    NO_PLAN,
    NO_SPENDING_CHANGES,
    OUTPUT_COLUMNS,
    AffordabilityStatus,
    CandidatePlan,
    Direction,
    FinancialState,
    PaymentMethod,
    RecommendedPaymentMethod,
    RequestRecord,
    SpendingChange,
    SpendingChangeAction,
    ValidationResult,
    to_money,
)
from tools.data import Dataset  # noqa: E402
from tools.forecast import simulate  # noqa: E402
from tools.planning import (  # noqa: E402
    _user_authorizes_reduce,
    _user_authorizes_stop,
    check_user_preferences,
    installment_months,
    source_row_flexibility,
)

ZERO = Decimal("0.00")

CHECKS = (
    "request_identity",
    "payment_amounts_positive",
    "payments_chronological",
    "payments_within_horizon",
    "method_preference",
    "method_shape",
    "installments_match_supplied_option",
    "partial_payment_rules",
    "spending_changes_permitted",
    "status_consistency",
    "completes_by_deadline",
    "ninety_day_safety",
)


def _validate_spending_changes(
    state: FinancialState,
    dataset: Dataset,
    changes: tuple[SpendingChange, ...],
    violations: list[str],
) -> None:
    profile = dataset.profile(state.user_id)
    if len(changes) > MAX_SPENDING_CHANGES:
        violations.append(f"more than {MAX_SPENDING_CHANGES} spending changes")
    seen: set[str] = set()
    for change in changes:
        if change.event_id in seen:
            violations.append(f"{change.event_id}: targeted by more than one change")
        seen.add(change.event_id)

        commitment = state.recurring_by_event_id(change.event_id)
        if commitment is None:
            violations.append(
                f"{change.event_id}: not a recurring commitment for {state.user_id}; "
                "only recurring expenses may be changed"
            )
            continue
        if commitment.direction is not Direction.DEBIT:
            violations.append(f"{change.event_id}: only expenses may be changed")
            continue
        if profile.is_protected_category(commitment.category):
            violations.append(
                f"{change.event_id}: {commitment.category} is a protected category"
            )
            continue

        if change.action is SpendingChangeAction.STOP:
            row_flex = source_row_flexibility(commitment)
            if not _user_authorizes_stop(commitment, profile):
                violations.append(
                    f"{change.event_id}: row flexibility {row_flex} is not stoppable "
                    f"and {commitment.category} is not on the user's stop list"
                )
        else:
            row_flex = source_row_flexibility(commitment)
            if not _user_authorizes_reduce(commitment, profile):
                violations.append(
                    f"{change.event_id}: row flexibility {row_flex} is not reducible "
                    f"and {commitment.category} is not on the user's reduce list"
                )
            if not profile.may_reduce_category(commitment.category):
                violations.append(
                    f"{change.event_id}: user will not reduce {commitment.category}"
                )
            new_amount = change.new_amount or ZERO
            if new_amount >= commitment.amount:
                violations.append(
                    f"{change.event_id}: reduce_to {new_amount} does not reduce "
                    f"{commitment.amount}"
                )
            floor = commitment.minimum_allowed_amount
            if floor is not None and new_amount < floor:
                violations.append(
                    f"{change.event_id}: reduce_to {new_amount} is below the "
                    f"minimum allowed {floor}"
                )


def _validate_installments(
    dataset: Dataset,
    request: RequestRecord,
    candidate: CandidatePlan,
    violations: list[str],
) -> None:
    option_id = candidate.payment_option_id
    if not option_id:
        violations.append("installments must reference a payment_option_id")
        return
    option = dataset.payment_options.get(option_id)
    if option is None:
        violations.append(f"{option_id}: unknown payment option")
        return
    if option.request_id != request.request_id:
        violations.append(f"{option_id}: belongs to {option.request_id}")
        return
    if option.payment_method is not PaymentMethod.INSTALLMENTS:
        violations.append(f"{option_id}: is not an installment option")
        return
    if candidate.payments != option.schedule:
        violations.append(
            f"{option_id}: plan does not reproduce the supplied schedule "
            f"({option.number_of_payments} x {option.payment_amount} from "
            f"{option.first_payment_date} every {option.payment_frequency_days} days)"
        )
    if candidate.payment_total != option.total_payable_amount:
        violations.append(
            f"{option_id}: plan total {candidate.payment_total} != "
            f"total_payable_amount {option.total_payable_amount}"
        )
    preference = check_user_preferences(
        dataset.profile(request.user_id),
        payment_method=PaymentMethod.INSTALLMENTS,
        installment_months=installment_months(option),
    )
    if not preference.eligible:
        violations.extend(preference.rejection_reasons)


def validate_plan_constraints(
    dataset: Dataset,
    state: FinancialState,
    request: RequestRecord,
    candidate: CandidatePlan,
) -> ValidationResult:
    """Check one candidate against every hard rule and re-simulate it."""
    violations: list[str] = []
    profile = dataset.profile(request.user_id)

    if candidate.request_id != request.request_id:
        violations.append(
            f"candidate is for {candidate.request_id}, not {request.request_id}"
        )
    if state.request_id != request.request_id:
        violations.append(f"financial state is for {state.request_id}")

    for payment in candidate.payments:
        if payment.amount <= ZERO:
            violations.append(f"{payment.date}: payment amount must be positive")
        if payment.date < request.request_date:
            violations.append(f"{payment.date}: payment precedes request_date")
        if payment.date > state.horizon_end:
            violations.append(f"{payment.date}: payment falls outside the 90-day forecast")

    method = candidate.method
    status = candidate.affordability_status

    if method is RecommendedPaymentMethod.NOT_RECOMMENDED:
        if candidate.payments or candidate.spending_changes:
            violations.append("not_recommended must carry no payments or spending changes")
        if status is not AffordabilityStatus.NOT_AFFORDABLE:
            violations.append("not_recommended requires not_affordable")
        return ValidationResult(
            request_id=request.request_id,
            checks_performed=CHECKS,
            violations=tuple(violations),
        )

    if not candidate.payments:
        violations.append(f"{method} requires at least one payment")

    equivalent = {
        RecommendedPaymentMethod.FULL_PAYMENT: PaymentMethod.FULL_PAYMENT,
        RecommendedPaymentMethod.WAIT: PaymentMethod.FULL_PAYMENT,
        RecommendedPaymentMethod.PARTIAL_PAYMENT: PaymentMethod.PARTIAL_PAYMENT,
        RecommendedPaymentMethod.INSTALLMENTS: PaymentMethod.INSTALLMENTS,
    }[method]
    preference = check_user_preferences(profile, payment_method=equivalent)
    if not preference.eligible and method is not RecommendedPaymentMethod.INSTALLMENTS:
        violations.extend(preference.rejection_reasons)

    if method is RecommendedPaymentMethod.INSTALLMENTS:
        _validate_installments(dataset, request, candidate, violations)
    elif method is RecommendedPaymentMethod.PARTIAL_PAYMENT:
        if not request.allows_partial_payment:
            violations.append("this request does not allow partial payment")
        if len(candidate.payments) != 2:
            violations.append("partial_payment requires exactly two payments")
        else:
            first, second = candidate.payments
            if first.date != request.request_date:
                violations.append("the first partial payment must fall on request_date")
            if to_money(first.amount + second.amount) != request.requested_amount:
                violations.append("partial payments must sum to requested_amount")
            if not ZERO < first.amount < request.requested_amount:
                violations.append(
                    "partial_payment requires 0 < amount_safe_to_pay < requested_amount"
                )
            if second.date > request.desired_completion_date:
                violations.append(
                    "the second partial payment falls after desired_completion_date"
                )
        if status is not AffordabilityStatus.AFFORDABLE_WITH_PLAN:
            violations.append("partial_payment requires affordable_with_plan")
    else:
        if len(candidate.payments) != 1:
            violations.append(f"{method} requires exactly one payment")
        elif candidate.payments[0].amount != request.requested_amount:
            violations.append(
                f"{method} must pay the full requested_amount "
                f"{request.requested_amount}, not {candidate.payments[0].amount}"
            )

    if method is RecommendedPaymentMethod.WAIT:
        if candidate.payments and candidate.payments[0].date <= request.request_date:
            violations.append("wait requires a payment date after request_date")
        if status is not AffordabilityStatus.AFFORDABLE_LATER:
            violations.append("wait requires affordable_later")
    if method is RecommendedPaymentMethod.FULL_PAYMENT:
        expected = (
            AffordabilityStatus.AFFORDABLE_WITH_PLAN
            if candidate.spending_changes
            else AffordabilityStatus.AFFORDABLE_NOW
        )
        if status is not expected:
            violations.append(
                f"full_payment with {len(candidate.spending_changes)} spending change(s) "
                f"requires {expected}, not {status}"
            )
    if method is RecommendedPaymentMethod.INSTALLMENTS and (
        status is not AffordabilityStatus.AFFORDABLE_WITH_PLAN
    ):
        violations.append("installments require affordable_with_plan")

    _validate_spending_changes(state, dataset, candidate.spending_changes, violations)

    if candidate.payments and candidate.payments[-1].date > request.desired_completion_date:
        violations.append(
            f"plan completes on {candidate.payments[-1].date}, after "
            f"desired_completion_date {request.desired_completion_date}"
        )

    expected_total = (
        candidate.total_payable
        if method is RecommendedPaymentMethod.INSTALLMENTS
        else request.requested_amount
    )
    result = simulate(
        state,
        payments=candidate.payments,
        spending_changes=candidate.spending_changes,
        requested_amount=expected_total,
    )
    if not result.safe:
        violations.append(
            f"90-day safety check failed: {result.violations[0].reason}"
            if result.violations
            else "90-day safety check failed"
        )
    if not result.all_payments_completed:
        violations.append("the plan does not complete the request within the forecast")

    return ValidationResult(
        request_id=request.request_id,
        checks_performed=CHECKS,
        violations=tuple(violations),
        simulation=result,
    )


# --------------------------------------------------------------------------- #
# Output-row validation
# --------------------------------------------------------------------------- #


def validate_output_row(
    row: Mapping[str, str], request: RequestRecord
) -> tuple[str, ...]:
    """Structural check of one rendered output row against the required contract."""
    violations: list[str] = []
    missing = [column for column in OUTPUT_COLUMNS if column not in row]
    if missing:
        return (f"missing columns: {', '.join(missing)}",)
    extra = [column for column in row if column not in OUTPUT_COLUMNS]
    if extra:
        violations.append(f"unexpected columns: {', '.join(extra)}")
    if row["request_id"] != request.request_id:
        violations.append(f"row is for {row['request_id']}, not {request.request_id}")

    try:
        safe = Decimal(row["amount_safe_to_pay"])
    except Exception:  # noqa: BLE001 - any unparsable amount is a contract breach
        violations.append(f"amount_safe_to_pay {row['amount_safe_to_pay']!r} is not a number")
        safe = None
    if safe is not None and not ZERO <= safe <= request.requested_amount:
        violations.append(
            f"amount_safe_to_pay {safe} outside [0, {request.requested_amount}]"
        )

    if row["affordability_status"] not in set(AffordabilityStatus):
        violations.append(f"invalid affordability_status {row['affordability_status']!r}")
    if row["recommended_payment_method"] not in set(RecommendedPaymentMethod):
        violations.append(
            f"invalid recommended_payment_method {row['recommended_payment_method']!r}"
        )

    plan = row["payment_plan"]
    payment_dates: list[date] = []
    if plan != NO_PLAN:
        total = ZERO
        for part in plan.split("|"):
            if part.count(":") != 1:
                violations.append(f"malformed payment_plan entry {part!r}")
                continue
            when, amount = part.split(":")
            try:
                payment_dates.append(date.fromisoformat(when))
                total = to_money(total + Decimal(amount))
            except Exception:  # noqa: BLE001
                violations.append(f"malformed payment_plan entry {part!r}")
        if payment_dates != sorted(payment_dates):
            violations.append("payment_plan is not chronological")

    earliest = row["earliest_date_for_full_payment"]
    if earliest:
        try:
            parsed = date.fromisoformat(earliest)
        except ValueError:
            violations.append(f"earliest_date_for_full_payment {earliest!r} is not a date")
            parsed = None
        if parsed and row["affordability_status"] == AffordabilityStatus.AFFORDABLE_NOW:
            if parsed != request.request_date:
                violations.append(
                    "affordable_now requires earliest_date_for_full_payment == request_date"
                )
    elif row["affordability_status"] == AffordabilityStatus.AFFORDABLE_NOW:
        violations.append("affordable_now requires a non-empty earliest_date_for_full_payment")

    changes = row["spending_changes_needed"]
    if changes != NO_SPENDING_CHANGES:
        parts = changes.split("|")
        if len(parts) > MAX_SPENDING_CHANGES:
            violations.append(f"more than {MAX_SPENDING_CHANGES} spending changes")
        targets: list[str] = []
        for part in parts:
            bits = part.split(":")
            if bits[0] == "stop" and len(bits) == 2:
                targets.append(bits[1])
            elif bits[0] == "reduce_to" and len(bits) == 3:
                targets.append(bits[1])
                try:
                    Decimal(bits[2])
                except Exception:  # noqa: BLE001
                    violations.append(f"malformed reduce_to amount in {part!r}")
            else:
                violations.append(f"malformed spending change {part!r}")
        if len(targets) != len(set(targets)):
            violations.append("the same event is targeted by more than one change")

    if row["recommended_payment_method"] == RecommendedPaymentMethod.NOT_RECOMMENDED:
        if plan != NO_PLAN:
            violations.append("not_recommended requires payment_plan 'none'")
    elif plan == NO_PLAN:
        violations.append(
            f"{row['recommended_payment_method']} requires a non-empty payment_plan"
        )

    if not row["decision_explanation"].strip():
        violations.append("decision_explanation is empty")

    return tuple(violations)
