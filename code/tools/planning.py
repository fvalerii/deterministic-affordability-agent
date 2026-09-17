"""Candidate plan generation, preference filtering and official ranking.

A candidate is only produced when the user's own preferences allow the method,
and every candidate is simulated before it is offered. Ranking follows the
published order exactly:

1. complete the full request by ``desired_completion_date``
2. require no spending changes
3. minimize the total amount paid
4. start payment earlier
5. use fewer payments
6. lowest ``payment_option_id``
"""

from __future__ import annotations

import itertools
import math
import re
import sys
from datetime import date
from decimal import Decimal
from pathlib import Path

_CODE_DIR = Path(__file__).resolve().parent.parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from domain import (  # noqa: E402
    AffordabilityStatus,
    CandidatePlan,
    Direction,
    Flexibility,
    FORECAST_HORIZON_DAYS,
    FinancialState,
    Payment,
    PaymentMethod,
    PaymentOptionRecord,
    PreferenceCheck,
    ProfileRecord,
    RankedCandidates,
    RecommendedPaymentMethod,
    RecurringCommitment,
    RequestRecord,
    SpendingChange,
    SpendingChangeAction,
    to_money,
)
from tools.data import Dataset, load_dataset  # noqa: E402
from tools.forecast import (  # noqa: E402
    calculate_safe_amount_today,
    find_earliest_safe_full_payment,
    simulate,
)
from tools.ledger import DINING_CATEGORIES, DISCRETIONARY_CATEGORIES  # noqa: E402

LIFESTYLE_CATEGORIES = DINING_CATEGORIES | DISCRETIONARY_CATEGORIES

ZERO = Decimal("0.00")
MAX_CHANGE_CANDIDATES = 6  # keeps the subset search bounded and deterministic


# --------------------------------------------------------------------------- #
# Preferences
# --------------------------------------------------------------------------- #


def check_user_preferences(
    profile: ProfileRecord,
    *,
    payment_method: PaymentMethod | str,
    installment_months: int | None = None,
) -> PreferenceCheck:
    """Decide whether a method is eligible for this user."""
    method = PaymentMethod(payment_method)
    reasons: list[str] = []

    if not profile.accepts_method(method):
        reasons.append(
            f"{method} is not in payment_methods_user_will_consider "
            f"({'|'.join(profile.payment_methods_user_will_consider)})"
        )
    if method is PaymentMethod.INSTALLMENTS:
        if profile.max_installment_months is None:
            reasons.append("max_installment_months is blank, so installments are declined")
        elif installment_months is not None and (
            installment_months > profile.max_installment_months
        ):
            reasons.append(
                f"{installment_months} monthly payments exceed the user's limit of "
                f"{profile.max_installment_months}"
            )

    return PreferenceCheck(
        user_id=profile.user_id,
        method=method,
        eligible=not reasons,
        max_installment_months=profile.max_installment_months,
        rejection_reasons=tuple(reasons),
    )


def installment_months(option: PaymentOptionRecord) -> int:
    """Number of months an installment option spans."""
    frequency = option.payment_frequency_days or 0
    if 27 <= frequency <= 32 or option.number_of_payments == 1:
        return option.number_of_payments
    return math.ceil(option.number_of_payments * frequency / 30)


def _option_sort_index(payment_option_id: str | None) -> tuple[int, str]:
    if not payment_option_id:
        return (10**9, "")
    match = re.search(r"(\d+)$", payment_option_id)
    return (int(match.group(1)) if match else 10**9, payment_option_id)


# --------------------------------------------------------------------------- #
# Spending changes
# --------------------------------------------------------------------------- #


def source_row_flexibility(commitment: RecurringCommitment) -> Flexibility:
    """Flexibility on the representative source row, not the inferred override.

    Lifestyle commitments are rewritten to ``reducible_or_stoppable`` in the
    ledger so the forecast can treat dining as variable. A spending-change
    recommendation must still honour the CSV row: ``fixed`` stays untouchable
    unless the user listed that category.
    """
    event = load_dataset().events.get(commitment.representative_event_id)
    if event is None:
        return commitment.flexibility
    return event.flexibility


def _user_authorizes_stop(commitment: RecurringCommitment, profile: ProfileRecord) -> bool:
    """True when a STOP is allowed by row flexibility or the user's stop list."""
    if profile.is_protected_category(commitment.category):
        return False
    return source_row_flexibility(commitment).can_stop or profile.may_stop_category(
        commitment.category
    )


def _user_authorizes_reduce(commitment: RecurringCommitment, profile: ProfileRecord) -> bool:
    """True when a REDUCE is allowed by row flexibility or the user's reduce list."""
    if profile.is_protected_category(commitment.category):
        return False
    return source_row_flexibility(commitment).can_reduce or profile.may_reduce_category(
        commitment.category
    )


def _change_for(
    commitment: RecurringCommitment, profile: ProfileRecord
) -> SpendingChange | None:
    """The largest permitted change to one commitment, or None if untouchable.

    A cut is allowed only when the source row is explicitly stoppable /
    reducible, or the user listed that category. ``fixed`` dining, shopping,
    or entertainment is never cancelled just because a plan needs a larger
    hole filled. Protected categories stay untouchable.
    """
    if commitment.direction is not Direction.DEBIT:
        return None
    if profile.is_protected_category(commitment.category):
        return None
    if _user_authorizes_stop(commitment, profile):
        return SpendingChange(
            action=SpendingChangeAction.STOP, event_id=commitment.representative_event_id
        )
    if _user_authorizes_reduce(commitment, profile) and profile.may_reduce_category(
        commitment.category
    ):
        floor = (
            commitment.minimum_allowed_amount
            if commitment.minimum_allowed_amount is not None
            else ZERO
        )
        if floor < commitment.amount:
            return SpendingChange(
                action=SpendingChangeAction.REDUCE_TO,
                event_id=commitment.representative_event_id,
                new_amount=floor,
            )
    return None


def _monthly_saving(commitment: RecurringCommitment, change: SpendingChange) -> Decimal:
    per_occurrence = (
        commitment.amount
        if change.action is SpendingChangeAction.STOP
        else to_money(commitment.amount - (change.new_amount or ZERO))
    )
    return to_money(per_occurrence * Decimal(30) / Decimal(commitment.frequency_days))


def list_permitted_spending_changes(
    state: FinancialState, profile: ProfileRecord
) -> tuple[SpendingChange, ...]:
    """Every change the user has authorized, richest saving first."""
    scored: list[tuple[int, Decimal, str, SpendingChange]] = []
    for commitment in state.recurring:
        change = _change_for(commitment, profile)
        if change is None:
            continue
        user_asked = int(
            not (
                profile.may_stop_category(commitment.category)
                or profile.may_reduce_category(commitment.category)
            )
        )
        scored.append(
            (
                user_asked,
                _monthly_saving(commitment, change),
                commitment.representative_event_id,
                change,
            )
        )
    scored.sort(key=lambda item: (item[0], -item[1], item[2]))
    return tuple(change for _, _, _, change in scored)


def _find_change_set(
    state: FinancialState,
    request: RequestRecord,
    profile: ProfileRecord,
    payments: tuple[Payment, ...],
) -> tuple[SpendingChange, ...] | None:
    """Smallest authorized change set (at most three) that makes a plan safe."""
    permitted = list_permitted_spending_changes(state, profile)[:MAX_CHANGE_CANDIDATES]
    for size in (1, 2, 3):
        for combination in itertools.combinations(permitted, size):
            result = simulate(
                state,
                payments=payments,
                spending_changes=combination,
                requested_amount=request.requested_amount,
            )
            if result.safe:
                return combination
    return None


# --------------------------------------------------------------------------- #
# Candidate construction
# --------------------------------------------------------------------------- #


def build_full_payment_candidate(
    state: FinancialState,
    request: RequestRecord,
    *,
    payment_date: date,
    spending_changes: tuple[SpendingChange, ...] = (),
    payment_option_id: str | None = None,
) -> CandidatePlan | None:
    """A single full payment on ``payment_date``, if the simulation allows it.

    A plan that lands after ``desired_completion_date`` is not offered at all,
    because the request must be completed by its deadline to be recommendable.
    """
    if payment_date > request.desired_completion_date:
        return None
    payments = (Payment(date=payment_date, amount=request.requested_amount),)
    result = simulate(
        state,
        payments=payments,
        spending_changes=spending_changes,
        requested_amount=request.requested_amount,
    )
    if not result.safe or not result.all_payments_completed:
        return None

    waiting = payment_date > request.request_date
    if waiting:
        status = AffordabilityStatus.AFFORDABLE_LATER
        method = RecommendedPaymentMethod.WAIT
    elif spending_changes:
        status = AffordabilityStatus.AFFORDABLE_WITH_PLAN
        method = RecommendedPaymentMethod.FULL_PAYMENT
    else:
        status = AffordabilityStatus.AFFORDABLE_NOW
        method = RecommendedPaymentMethod.FULL_PAYMENT

    return CandidatePlan(
        request_id=request.request_id,
        method=method,
        affordability_status=status,
        payments=payments,
        spending_changes=spending_changes,
        payment_option_id=payment_option_id,
        total_payable=request.requested_amount,
        completes_request=True,
        completes_by_deadline=payment_date <= request.desired_completion_date,
        notes=(f"minimum projected balance {result.minimum_projected_balance}",),
    )


def build_partial_payment_candidate(
    state: FinancialState,
    request: RequestRecord,
    *,
    amount_today: Decimal,
    second_payment_date: date,
) -> CandidatePlan | None:
    """Exactly two payments: the safe amount today, the remainder on the safe date."""
    if not request.allows_partial_payment:
        return None
    if not ZERO < amount_today < request.requested_amount:
        return None
    if second_payment_date <= request.request_date:
        return None
    if second_payment_date > request.desired_completion_date:
        return None

    remainder = to_money(request.requested_amount - amount_today)
    payments = (
        Payment(date=request.request_date, amount=amount_today),
        Payment(date=second_payment_date, amount=remainder),
    )
    result = simulate(
        state, payments=payments, requested_amount=request.requested_amount
    )
    if not result.safe or not result.all_payments_completed:
        return None

    return CandidatePlan(
        request_id=request.request_id,
        method=RecommendedPaymentMethod.PARTIAL_PAYMENT,
        affordability_status=AffordabilityStatus.AFFORDABLE_WITH_PLAN,
        payments=payments,
        total_payable=request.requested_amount,
        completes_request=True,
        completes_by_deadline=True,
        notes=(f"minimum projected balance {result.minimum_projected_balance}",),
    )


def build_installment_candidate(
    state: FinancialState,
    request: RequestRecord,
    option: PaymentOptionRecord,
    profile: ProfileRecord,
) -> CandidatePlan | None:
    """An installment plan that reproduces a supplied option exactly."""
    if option.payment_method is not PaymentMethod.INSTALLMENTS:
        return None
    if option.last_payment_date > request.desired_completion_date:
        return None
    preference = check_user_preferences(
        profile,
        payment_method=PaymentMethod.INSTALLMENTS,
        installment_months=installment_months(option),
    )
    if not preference.eligible:
        return None

    payments = option.schedule
    result = simulate(
        state, payments=payments, requested_amount=option.total_payable_amount
    )
    if not result.safe or not result.all_payments_completed:
        return None

    return CandidatePlan(
        request_id=request.request_id,
        method=RecommendedPaymentMethod.INSTALLMENTS,
        affordability_status=AffordabilityStatus.AFFORDABLE_WITH_PLAN,
        payments=payments,
        payment_option_id=option.payment_option_id,
        total_payable=option.total_payable_amount,
        completes_request=True,
        completes_by_deadline=option.last_payment_date <= request.desired_completion_date,
        notes=(
            f"{option.number_of_payments} payments of {option.payment_amount}",
            f"minimum projected balance {result.minimum_projected_balance}",
        ),
    )


def not_recommended_candidate(request: RequestRecord, reason: str) -> CandidatePlan:
    return CandidatePlan(
        request_id=request.request_id,
        method=RecommendedPaymentMethod.NOT_RECOMMENDED,
        affordability_status=AffordabilityStatus.NOT_AFFORDABLE,
        total_payable=ZERO,
        completes_request=False,
        completes_by_deadline=False,
        notes=(reason,),
    )


def generate_candidates(
    dataset: Dataset,
    state: FinancialState,
    request: RequestRecord,
) -> tuple[CandidatePlan, ...]:
    """Every safe, preference-eligible way to satisfy this request."""
    profile = dataset.profile(request.user_id)
    safe_today = calculate_safe_amount_today(state, request)
    earliest = find_earliest_safe_full_payment(state, request)
    candidates: list[CandidatePlan] = []

    accepts_full = check_user_preferences(
        profile, payment_method=PaymentMethod.FULL_PAYMENT
    ).eligible

    if accepts_full:
        today = build_full_payment_candidate(
            state, request, payment_date=request.request_date
        )
        if today is not None:
            candidates.append(today)
        else:
            changes = _find_change_set(
                state,
                request,
                profile,
                (Payment(date=request.request_date, amount=request.requested_amount),),
            )
            if changes:
                with_changes = build_full_payment_candidate(
                    state,
                    request,
                    payment_date=request.request_date,
                    spending_changes=changes,
                )
                if with_changes is not None:
                    candidates.append(with_changes)

        if earliest.earliest_date and earliest.earliest_date > request.request_date:
            later = build_full_payment_candidate(
                state, request, payment_date=earliest.earliest_date
            )
            if later is not None:
                candidates.append(later)

    if check_user_preferences(
        profile, payment_method=PaymentMethod.PARTIAL_PAYMENT
    ).eligible and earliest.earliest_date:
        partial = build_partial_payment_candidate(
            state,
            request,
            amount_today=safe_today.amount_safe_to_pay,
            second_payment_date=earliest.earliest_date,
        )
        if partial is not None:
            candidates.append(partial)

    for option in dataset.installment_options(request.request_id):
        candidate = build_installment_candidate(state, request, option, profile)
        if candidate is not None:
            candidates.append(candidate)

    if not candidates:
        # Every request needs exactly one output row, so the ladder has to end
        # somewhere. Nothing was safe, which is itself the recommendation.
        candidates.append(
            not_recommended_candidate(
                request,
                f"no safe payment found within {FORECAST_HORIZON_DAYS} days; "
                f"safe today {safe_today.amount_safe_to_pay}, "
                f"earliest safe full payment {earliest.earliest_date or 'none'}",
            )
        )

    return tuple(candidates)


# --------------------------------------------------------------------------- #
# Ranking
# --------------------------------------------------------------------------- #


def _rank_key(candidate: CandidatePlan) -> tuple:
    return (
        0 if candidate.completes_by_deadline else 1,
        len(candidate.spending_changes),
        candidate.total_payable,
        candidate.first_payment_date or date.max,
        len(candidate.payments),
        _option_sort_index(candidate.payment_option_id),
    )


def rank_valid_candidates(
    request: RequestRecord, candidates: tuple[CandidatePlan, ...]
) -> RankedCandidates:
    """Order candidates by the six official tie-breakers, in order."""
    ordered = tuple(sorted(candidates, key=_rank_key))
    notes = tuple(
        f"{index + 1}. {candidate.method}"
        + (f" via {candidate.payment_option_id}" if candidate.payment_option_id else "")
        + f" total={candidate.total_payable}"
        + f" start={candidate.first_payment_date}"
        + f" payments={len(candidate.payments)}"
        + f" changes={len(candidate.spending_changes)}"
        + f" by_deadline={candidate.completes_by_deadline}"
        for index, candidate in enumerate(ordered)
    )
    return RankedCandidates(request_id=request.request_id, ordered=ordered, ranking_notes=notes)
