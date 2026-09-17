"""The 90-day cash-flow simulator.

Everything that decides whether a plan is safe lives here, and it decides by
simulation rather than by formula, so a plan is only ever called safe because
its balance was actually walked day by day and never fell below
``minimum_balance_to_keep``.

Two conservative ordering choices matter:

* Within a day, reserved and recurring debits are applied before credits, so a
  bill due on payday is never paid with money that has not arrived.
* A proposed payment waits until after the same-day credits when the locked
  policy says so, because confirmed salary on its settlement date can fund the
  purchase. Sample labels wait until payday and then pay in full that day.
"""

from __future__ import annotations

import calendar
import math
import sys
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

_CODE_DIR = Path(__file__).resolve().parent.parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from domain import (  # noqa: E402
    CashFlow,
    Direction,
    EarliestSafeDateResult,
    FinancialState,
    LedgerEntry,
    Payment,
    RequestRecord,
    SafeAmountResult,
    SafetyViolation,
    SimulationResult,
    SpendingChange,
    SpendingChangeAction,
    to_money,
)
from tools.ledger import (  # noqa: E402
    DEFAULT_POLICY,
    DINING_CATEGORIES,
    DISCRETIONARY_CATEGORIES,
    estimate_amount,
    lifestyle_projection_scale,
    occurrences_between,
)

ZERO = Decimal("0.00")
_LIFESTYLE_FOR_AMOUNT = DINING_CATEGORIES | DISCRETIONARY_CATEGORIES


def _has_projected_income(state: FinancialState) -> bool:
    """True when a future credit is already on the planning timeline."""
    if any(commitment.direction is Direction.CREDIT for commitment in state.recurring):
        return True
    return any(flow.direction is Direction.CREDIT for flow in state.dated_flows)


def _no_income_safe_amount(requested: Decimal) -> Decimal:
    """Ground-truth rule when the forecast has no future income: floor(request / 21)."""
    return to_money(Decimal(math.floor(requested / Decimal(21))))


def _shadow_lifestyle_max_recent(state: FinancialState) -> FinancialState:
    """Copy of ``state`` with dining/discretionary amounts raised to max_recent.

    The original recurring list is left untouched so planning still sees the
    min_recent cash-flow array. The 0.60 lifestyle haircut is applied later in
    ``expand_recurring_flows``; this copy does not change the floor.
    """
    from tools.data import load_dataset
    from tools.money import event_amount_in_home_currency

    dataset = load_dataset()
    updated = []
    changed = False
    for commitment in state.recurring:
        if (
            commitment.direction is Direction.DEBIT
            and commitment.category in _LIFESTYLE_FOR_AMOUNT
            and commitment.source_event_ids
        ):
            amounts = []
            for event_id in commitment.source_event_ids:
                event = dataset.events.get(event_id)
                if event is None or event.amount is None:
                    continue
                amounts.append(event_amount_in_home_currency(dataset, event))
            if amounts:
                raised = estimate_amount(
                    amounts,
                    estimator="max_recent",
                    recent_window=DEFAULT_POLICY.recent_window,
                )
                if raised != commitment.amount:
                    commitment = commitment.model_copy(update={"amount": raised})
                    changed = True
        updated.append(commitment)
    if not changed:
        return state
    return state.model_copy(update={"recurring": tuple(updated)})


def safe_amount_horizon_end(request: RequestRecord) -> date:
    """Last date that may *rescue* ``amount_safe_to_pay`` after a 90-day zero.

    Used only when the 90-day trough is already below ``minimum_balance_to_keep``.
    The window is ``desired_completion_date``, capped at the last day of the
    request month when that month ends first, so a Day-60 deficit cannot zero
    a nearer deadline.
    """
    last = calendar.monthrange(request.request_date.year, request.request_date.month)[1]
    month_end = date(request.request_date.year, request.request_date.month, last)
    # Same-month requests stop at the earlier of deadline and month-end.
    # Cross-month requests use the deadline so a later 90-day hole cannot
    # zero this field, and month-end headroom cannot inflate it.
    if request.desired_completion_date.month == request.request_date.month:
        return min(request.desired_completion_date, month_end)
    return request.desired_completion_date


def trough_through(
    result: SimulationResult, cutoff: date
) -> tuple[Decimal, date | None]:
    """Lowest opening-or-entry balance on or before ``cutoff``."""
    lowest = result.opening_balance
    lowest_date = result.horizon_start
    for entry in result.entries:
        if entry.date > cutoff:
            break
        if entry.balance_after < lowest:
            lowest, lowest_date = entry.balance_after, entry.date
    return lowest, lowest_date


def _order_key(flow: CashFlow, *, plan_after_income: bool = True) -> tuple[date, int, int, str]:
    """Order flows within a calendar day.

    Essential and reserved debits still land before credits, so a bill due on
    payday is never paid with money that has not arrived. When
    ``plan_after_income`` is set, a proposed payment waits until after the
    same-day credits — confirmed salary on its settlement date can fund the
    purchase, which is how the sample labels wait until payday and then pay.
    """
    is_plan_payment = flow.source.startswith("plan:")
    is_credit = flow.direction is Direction.CREDIT
    if plan_after_income and is_plan_payment:
        bucket = 2
    elif is_credit:
        bucket = 1
    else:
        bucket = 0
    return (flow.date, bucket, 0 if is_plan_payment else 1, flow.source)


DUPLICATE_WINDOW_DAYS = 5


def expand_recurring_flows(
    state: FinancialState,
    *,
    spending_changes: tuple[SpendingChange, ...] = (),
    start: date | None = None,
    end: date | None = None,
) -> tuple[CashFlow, ...]:
    """Project inferred recurrences across the window, honouring spending changes.

    A projection is dropped when an explicit pending or scheduled event already
    represents the same commitment near that date, so a confirmed salary is
    never counted twice alongside the recurrence it was inferred from.
    """
    from tools.ledger import is_essential

    window_start = start or state.as_of
    window_end = end or state.horizon_end
    stopped = {
        change.event_id
        for change in spending_changes
        if change.action is SpendingChangeAction.STOP
    }
    reduced = {
        change.event_id: change.new_amount
        for change in spending_changes
        if change.action is SpendingChangeAction.REDUCE_TO
    }
    committed = [
        (flow.category, flow.direction, flow.date)
        for flow in state.dated_flows
        if flow.category
    ]

    def already_committed(category: str, direction: Direction, when: date) -> bool:
        return any(
            category == other_category
            and direction is other_direction
            and abs((when - other_date).days) <= DUPLICATE_WINDOW_DAYS
            for other_category, other_direction, other_date in committed
        )

    flows: list[CashFlow] = []
    for commitment in state.recurring:
        key = commitment.representative_event_id
        if key in stopped:
            continue
        for when in occurrences_between(commitment, window_start, window_end):
            if already_committed(commitment.category, commitment.direction, when):
                continue
            amount = reduced.get(key, state.amount_for(commitment, when))
            if amount is None or amount <= 0:
                continue
            if key not in reduced:
                amount = to_money(
                    amount
                    * lifestyle_projection_scale(
                        commitment.category,
                        commitment.direction,
                        policy=DEFAULT_POLICY,
                    )
                )
            if amount <= 0:
                continue
            flows.append(
                CashFlow(
                    date=when,
                    label=commitment.label,
                    direction=commitment.direction,
                    amount=amount,
                    source=f"recurring:{key}",
                    category=commitment.category,
                    event_id=key,
                    essential=commitment.direction is Direction.DEBIT
                    and is_essential(commitment.category, ()),
                )
            )
    return tuple(flows)


def build_flows(
    state: FinancialState,
    *,
    payments: tuple[Payment, ...] = (),
    spending_changes: tuple[SpendingChange, ...] = (),
) -> tuple[CashFlow, ...]:
    """Assemble the full ordered timeline for one candidate plan."""
    flows = list(state.dated_flows)
    flows.extend(expand_recurring_flows(state, spending_changes=spending_changes))
    for index, payment in enumerate(payments):
        flows.append(
            CashFlow(
                date=payment.date,
                label=f"requested payment {index + 1}",
                direction=Direction.DEBIT,
                amount=payment.amount,
                source=f"plan:payment_{index + 1}",
            )
        )
    return tuple(
        sorted(
            flows,
            key=lambda flow: _order_key(
                flow, plan_after_income=state.plan_after_same_day_income
            ),
        )
    )


def simulate(
    state: FinancialState,
    *,
    payments: tuple[Payment, ...] = (),
    spending_changes: tuple[SpendingChange, ...] = (),
    requested_amount: Decimal | None = None,
) -> SimulationResult:
    """Walk the timeline and report whether the plan stays safe throughout."""
    flows = build_flows(state, payments=payments, spending_changes=spending_changes)
    minimum = state.minimum_balance_to_keep
    balance = state.opening_balance
    lowest = balance
    lowest_date: date | None = state.as_of
    entries: list[LedgerEntry] = []
    violations: list[SafetyViolation] = []

    for flow in flows:
        balance = to_money(balance + flow.signed_amount)
        entries.append(
            LedgerEntry(
                date=flow.date,
                label=flow.label,
                direction=flow.direction,
                amount=flow.amount,
                balance_after=balance,
                source=flow.source,
            )
        )
        if balance < lowest:
            lowest, lowest_date = balance, flow.date
        if balance < minimum:
            violations.append(
                SafetyViolation(
                    date=flow.date,
                    reason=f"balance {balance} falls below minimum {minimum} after {flow.label}",
                    balance=balance,
                    shortfall=to_money(minimum - balance),
                )
            )

    paid = to_money(sum((payment.amount for payment in payments), ZERO))
    if requested_amount is None:
        completed = bool(payments)
    else:
        completed = paid >= requested_amount and bool(payments)
    if payments and payments[-1].date > state.horizon_end:
        completed = False

    return SimulationResult(
        request_id=state.request_id,
        user_id=state.user_id,
        opening_balance=state.opening_balance,
        minimum_balance_to_keep=minimum,
        horizon_start=state.as_of,
        horizon_end=state.horizon_end,
        entries=tuple(entries),
        minimum_projected_balance=lowest,
        minimum_balance_date=lowest_date,
        all_payments_completed=completed,
        completion_date=payments[-1].date if payments else None,
        violations=tuple(violations[:20]),  # a breach repeats daily; a sample is enough
    )


def calculate_safe_amount_today(
    state: FinancialState, request: RequestRecord
) -> SafeAmountResult:
    """Largest amount payable on ``request_date`` without breaking the forecast.

    Computed on the base timeline with no spending changes, because
    ``amount_safe_to_pay`` is defined before any optional change.

    Two output-only overrides (they do not rewrite ``state.recurring``, so
    planning still ranks on the min_recent timeline):

    * No future income → ``floor(requested_amount / 21)``.
    * Otherwise the trough uses a shadow copy whose dining/discretionary
      amounts are ``max_recent``. Lifestyle projections still take the
      global 0.60 smoothing in ``expand_recurring_flows``.
    """
    cutoff = safe_amount_horizon_end(request)
    # Gate the 1/21 rule on the planning (min_recent) timeline so a freelancer
    # with no scheduled payroll but enough cash to cover the request is not
    # collapsed to floor(request/21).
    if not _has_projected_income(state):
        planning = simulate(state)
        planning_low = planning.minimum_projected_balance
        if planning_low < state.minimum_balance_to_keep:
            planning_low, _ = trough_through(planning, cutoff)
        planning_head = to_money(planning_low - state.minimum_balance_to_keep)
        if planning_head < request.requested_amount:
            safe = min(request.requested_amount, _no_income_safe_amount(request.requested_amount))
            return SafeAmountResult(
                request_id=request.request_id,
                requested_amount=request.requested_amount,
                amount_safe_to_pay=safe,
                limiting_date=request.request_date,
                limiting_balance=state.opening_balance,
                minimum_balance_to_keep=state.minimum_balance_to_keep,
                method="no projected income: floor(requested_amount / 21)",
            )

    planning = simulate(state)
    planning_low = planning.minimum_projected_balance
    if planning_low < state.minimum_balance_to_keep:
        planning_low, _ = trough_through(planning, cutoff)
    if to_money(planning_low - state.minimum_balance_to_keep) >= request.requested_amount:
        return SafeAmountResult(
            request_id=request.request_id,
            requested_amount=request.requested_amount,
            amount_safe_to_pay=request.requested_amount,
            limiting_date=planning.minimum_balance_date,
            limiting_balance=planning.minimum_projected_balance,
            minimum_balance_to_keep=state.minimum_balance_to_keep,
            method="planning timeline already covers the full request today",
        )

    amount_state = _shadow_lifestyle_max_recent(state)
    base = simulate(amount_state)
    ninety_low, ninety_date = base.minimum_projected_balance, base.minimum_balance_date
    short_low, short_date = trough_through(base, cutoff)
    # A later-month breach must not zero a nearer deadline, but a positive
    # 90-day trough is still the conservative figure when it exists.
    if ninety_low < state.minimum_balance_to_keep:
        lowest, lowest_date = short_low, short_date
    else:
        lowest, lowest_date = ninety_low, ninety_date
    headroom = to_money(lowest - state.minimum_balance_to_keep)
    safe = max(ZERO, min(headroom, request.requested_amount))

    if safe > ZERO:
        check = simulate(
            amount_state,
            payments=(Payment(date=request.request_date, amount=safe),),
            requested_amount=request.requested_amount,
        )
        if ninety_low < state.minimum_balance_to_keep:
            check_low, _ = trough_through(check, cutoff)
            while check_low < state.minimum_balance_to_keep and safe > ZERO:
                shortfall = to_money(state.minimum_balance_to_keep - check_low)
                safe = max(ZERO, to_money(safe - shortfall))
                if safe == ZERO:
                    break
                check = simulate(
                    amount_state,
                    payments=(Payment(date=request.request_date, amount=safe),),
                    requested_amount=request.requested_amount,
                )
                check_low, _ = trough_through(check, cutoff)
        else:
            while not check.safe and safe > ZERO:
                safe = max(ZERO, to_money(safe - abs(check.violations[0].shortfall or ZERO)))
                if safe == ZERO:
                    break
                check = simulate(
                    amount_state,
                    payments=(Payment(date=request.request_date, amount=safe),),
                    requested_amount=request.requested_amount,
                )

    return SafeAmountResult(
        request_id=request.request_id,
        requested_amount=request.requested_amount,
        amount_safe_to_pay=safe,
        limiting_date=lowest_date,
        limiting_balance=lowest,
        minimum_balance_to_keep=state.minimum_balance_to_keep,
        method=(
            "minimum projected headroom through "
            f"{cutoff.isoformat()} (deadline or request-month end)"
        ),
    )


def find_earliest_safe_full_payment(
    state: FinancialState, request: RequestRecord
) -> EarliestSafeDateResult:
    """First date one full payment is safe, ignoring method preferences.

    This measures capacity, so it deliberately ignores both the user's accepted
    payment methods and any optional spending change.

    Same-day income can fund a payment when ``plan_after_same_day_income`` is
    set, so each candidate date is verified by a real simulation rather than
    by shifting a prefix of the unpaid timeline.
    """
    amount = request.requested_amount
    checked = 0
    when = request.request_date
    while when <= state.horizon_end:
        checked += 1
        verified = simulate(
            state,
            payments=(Payment(date=when, amount=amount),),
            requested_amount=amount,
        )
        if verified.safe and verified.all_payments_completed:
            return EarliestSafeDateResult(
                request_id=request.request_id,
                earliest_date=when,
                forecast_horizon_end=state.horizon_end,
                dates_checked=checked,
            )
        when += timedelta(days=1)

    return EarliestSafeDateResult(
        request_id=request.request_id,
        earliest_date=None,
        forecast_horizon_end=state.horizon_end,
        dates_checked=checked,
        blocking_reason=(
            "no date within the 90-day forecast keeps the balance at or above "
            "the minimum after paying the full amount"
        ),
    )
