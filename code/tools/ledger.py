"""Financial-state reconstruction: lifecycle resolution and recurrence.

Three jobs, all deterministic:

* Decide which event rows may touch the forecast at all (lifecycle resolution).
* Reserve pending debits without counting pending credits.
* Infer recurring commitments from history, and only from history.

Evidence never reaches this module as free text. It arrives as
:class:`ForecastAdjustment` records, so untrusted content can change an amount
or a date but can never change a rule.
"""

from __future__ import annotations

import calendar
import re
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

_CODE_DIR = Path(__file__).resolve().parent.parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from domain import (  # noqa: E402
    FORECAST_HORIZON_DAYS,
    CashFlow,
    CashState,
    Direction,
    EssentialSpendingBaseline,
    EventRecord,
    EventStatus,
    Flexibility,
    FinancialState,
    ForecastAdjustment,
    LifecycleResolution,
    RecurringAmountOverride,
    RecurringCommitment,
    RequestRecord,
    to_money,
)
from tools.data import Dataset  # noqa: E402
from tools.money import event_amount_in_home_currency  # noqa: E402

ZERO = Decimal("0.00")

# Categories treated as essential when forecasting variable spending. Anything
# the user has protected is also essential, which is handled per profile.
ESSENTIAL_CATEGORIES = frozenset(
    {
        "rent",
        "housing",
        "utilities",
        "groceries",
        "transport",
        "healthcare",
        "insurance",
        "education",
        "debt_repayment",
        "family_support",
    }
)

MONTHLY_GAP_RANGE = range(27, 33)

# Variable spending: amounts bounce, so the estimator is a separate knob from
# fixed bills (rent, debt) whose last amount is usually the next amount.
VARIABLE_DEBIT_CATEGORIES = frozenset(
    {
        "groceries",
        "transport",
        "dining",
        "shopping",
        "entertainment",
        "healthcare",
        "utilities",
    }
)

# Dining is variable and frequent; shopping/entertainment are lumpier.
# Split so request_06 can pick up weekly meals without also forecasting
# a monthly shop that would make a single streaming stop insufficient.
DINING_CATEGORIES = frozenset({"dining"})
DISCRETIONARY_CATEGORIES = frozenset(
    {
        "shopping",
        "entertainment",
    }
)

# Fixed-cadence subscriptions. Separate from dining/shopping so a user who
# should stop one streaming plan is not also forced to forecast shopping.
SUBSCRIPTION_CATEGORIES = frozenset(
    {
        "streaming",
        "cloud_storage",
        "delivery_membership",
        "music_subscription",
        "gym",
    }
)


@dataclass(frozen=True, slots=True)
class RecurrencePolicy:
    """Locked knobs for recurrence, income projection, and spending estimates.

    Defaults project lifestyle spend with ``min_recent`` so dining,
    shopping, and entertainment are not treated as zero, then haircut
    those projections unless the series is a rigid weekly standing charge.
    """

    lookback_days: int = 180
    min_occurrences: int = 2
    recent_window: int = 6
    debit_estimator: str = "max_recent"
    variable_debit_estimator: str = "min_recent"
    credit_estimator: str = "min_recent"
    # Multiplier on groceries and transport only; the essential-spending knob.
    variable_scale: str = "1.00"
    project_income: bool = True
    income_min_occurrences: int = 2
    anchor_income_on_scheduled: bool = True
    drop_final_income: bool = True
    filter_off_cadence: bool = True
    project_dining: bool = True
    project_discretionary: bool = True
    project_subscriptions: bool = True
    # Weekly gig / platform payouts are history, not confirmed future income.
    project_irregular_income: bool = False
    # Confirmed salary on a payment date can fund that payment (sample labels
    # wait until payday, then pay in full the same day).
    plan_after_same_day_income: bool = True
    # Dining / shopping / entertainment are not standing bills. Replay them
    # weekly only when coverage is almost every week *and* amounts barely
    # move. Otherwise stretch the cadence and/or haircut the projected amount
    # so skipped meals are not invented into a 90-day crunch.
    lifestyle_weekly_density: float = 0.90
    lifestyle_rigid_cv: float = 0.12
    lifestyle_sparse_min_frequency_days: int = 14
    lifestyle_smoothing: str = "0.60"


# Phase-4 lock + amount-gap patch (dining/discretionary on, median).
DEFAULT_POLICY = RecurrencePolicy()


def add_months(anchor: date, count: int) -> date:
    """Step a date by whole months, clamping to the end of a shorter month."""
    total = anchor.month - 1 + count
    year = anchor.year + total // 12
    month = total % 12 + 1
    return date(year, month, min(anchor.day, calendar.monthrange(year, month)[1]))


# --------------------------------------------------------------------------- #
# Lifecycle resolution
# --------------------------------------------------------------------------- #


def resolve_event_lifecycle(
    dataset: Dataset,
    *,
    user_id: str,
    as_of_date: date,
    event_ids: tuple[str, ...] | None = None,
) -> LifecycleResolution:
    """Decide which of a user's events may contribute to the cash forecast.

    A linked event does not by itself mean the earlier row is void, so the link
    is only honoured when the earlier row actually failed or was cancelled.
    """
    events = (
        tuple(dataset.event(event_id) for event_id in event_ids)
        if event_ids is not None
        else dataset.events_by_user.get(user_id, ())
    )
    counted: list[str] = []
    excluded: list[str] = []
    reasons: dict[str, str] = {}

    superseded: dict[str, str] = {}
    for event in events:
        link = event.linked_event_id
        if not link or link not in dataset.events:
            continue
        earlier = dataset.events[link]
        if earlier.status in {EventStatus.CANCELLED, EventStatus.FAILED}:
            superseded[earlier.event_id] = event.event_id

    for event in events:
        reason: str | None = None
        if event.status is EventStatus.CANCELLED:
            reason = "cancelled"
        elif event.status is EventStatus.FAILED:
            reason = "failed"
        elif event.status is EventStatus.UNREALIZED:
            reason = "unrealized valuation is not cash"
        elif not event.is_cash:
            reason = "non-cash record"
        elif event.direction is Direction.CREDIT and event.status is EventStatus.PENDING:
            reason = "pending credit is not counted until it settles"
        elif event.event_id in superseded:
            reason = f"superseded by {superseded[event.event_id]}"

        if reason is None:
            counted.append(event.event_id)
        else:
            excluded.append(event.event_id)
            reasons[event.event_id] = reason

    return LifecycleResolution(
        user_id=user_id,
        as_of_date=as_of_date,
        counted_event_ids=tuple(counted),
        excluded_event_ids=tuple(excluded),
        exclusion_reasons=reasons,
    )


# --------------------------------------------------------------------------- #
# Recurrence inference
# --------------------------------------------------------------------------- #


ESTIMATORS = ("max_recent", "min_recent", "mean_recent", "last", "median")


def estimate_amount(
    amounts: list[Decimal], *, estimator: str, recent_window: int
) -> Decimal:
    """Summarize an observed amount series into the amount to project forward.

    The default policy overstates outflows and understates inflows, which is the
    financially safer interpretation. It is a parameter rather than a constant
    because it is the main knob prompt/policy calibration needs to turn.
    """
    recent = amounts[-recent_window:] if recent_window > 0 else list(amounts)
    if estimator == "max_recent":
        return max(recent)
    if estimator == "min_recent":
        return min(recent)
    if estimator == "mean_recent":
        return to_money(sum(recent, ZERO) / len(recent))
    if estimator == "last":
        return recent[-1]
    if estimator == "median":
        return to_money(statistics.median(sorted(recent)))
    raise ValueError(f"unknown estimator {estimator!r}; expected one of {ESTIMATORS}")


def _is_regular(gaps: list[int], median_gap: float, tolerance: int) -> bool:
    if not gaps:
        return False
    within = sum(1 for gap in gaps if abs(gap - median_gap) <= tolerance)
    return within * 2 >= len(gaps)  # a majority of intervals must match


def _coefficient_of_variation(amounts: list[Decimal]) -> float:
    if len(amounts) < 2:
        return 0.0
    values = [float(amount) for amount in amounts]
    mean = statistics.fmean(values)
    if mean <= 0:
        return 1.0
    return statistics.pstdev(values) / mean


def lifestyle_weekly_density(series: list[EventRecord]) -> float:
    """Observed events per week of the spanned history."""
    if len(series) < 2:
        return 0.0
    span = (series[-1].effective_date - series[0].effective_date).days
    weeks = max(span / 7.0, 1.0)
    return len(series) / weeks


def is_rigid_weekly_lifestyle(
    series: list[EventRecord],
    amounts: list[Decimal],
    *,
    median_gap: float,
    policy: RecurrencePolicy | None = None,
) -> bool:
    """True only for a near-constant, almost-every-week lifestyle charge."""
    chosen = policy or DEFAULT_POLICY
    if median_gap > 9:
        return False
    if lifestyle_weekly_density(series) < chosen.lifestyle_weekly_density:
        return False
    return _coefficient_of_variation(amounts) <= chosen.lifestyle_rigid_cv


def soften_variable_lifestyle_cadence(
    series: list[EventRecord],
    amounts: list[Decimal],
    *,
    median_gap: float,
    frequency_days: int,
    last_date: date,
    policy: RecurrencePolicy | None = None,
) -> tuple[int, date]:
    """Lower the replay rate of sporadic dining / shopping / entertainment.

    A series that merely *averages* weekly is not a standing bill. Unless the
    density and amount-stability tests both pass, stretch a sub-fortnight
    cadence out to ``lifestyle_sparse_min_frequency_days`` so the forecast
    does not invent a debit every seven days.
    """
    chosen = policy or DEFAULT_POLICY
    if is_rigid_weekly_lifestyle(
        series, amounts, median_gap=median_gap, policy=chosen
    ):
        return frequency_days, last_date + timedelta(days=frequency_days)
    if (
        frequency_days <= 9
        and lifestyle_weekly_density(series) < chosen.lifestyle_weekly_density
    ):
        frequency_days = max(frequency_days, chosen.lifestyle_sparse_min_frequency_days)
    return frequency_days, last_date + timedelta(days=frequency_days)


def lifestyle_projection_scale(
    category: str,
    direction: Direction,
    *,
    policy: RecurrencePolicy | None = None,
) -> Decimal:
    """Haircut applied when projecting a variable lifestyle debit."""
    chosen = policy or DEFAULT_POLICY
    if direction is not Direction.DEBIT:
        return Decimal("1")
    if category not in DINING_CATEGORIES | DISCRETIONARY_CATEGORIES:
        return Decimal("1")
    return Decimal(chosen.lifestyle_smoothing)


# An income row whose description marks the stream as finished. Projecting a
# salary past its final payroll invents income the user will not receive.
FINAL_INCOME_RE = re.compile(
    r"\b(final|last|closing|terakhir|penghabisan)\b", re.IGNORECASE
)


def _on_cadence(events: list[EventRecord], median_gap: float, tolerance: int) -> list[EventRecord]:
    """Drop payments squeezed between the regular ones.

    A promotion arrears payment five days after payday, or a quarterly bonus a
    week after it, is a one-off. Left in the series it drags the projected
    monthly amount down to the size of the one-off, which is how a 4,365,000
    salary came to be forecast as 1,964,250.
    """
    if len(events) < 3:
        return events
    kept = [events[0]]
    for event in events[1:]:
        gap = (event.effective_date - kept[-1].effective_date).days
        if gap >= median_gap - tolerance:
            kept.append(event)
    return kept if len(kept) >= 2 else events


def infer_recurring_commitments(
    dataset: Dataset,
    *,
    user_id: str,
    as_of_date: date,
    lookback_days: int | None = None,
    min_occurrences: int | None = None,
    recent_window: int | None = None,
    debit_estimator: str | None = None,
    credit_estimator: str | None = None,
    project_income: bool | None = None,
    counted_event_ids: frozenset[str] | None = None,
    income_min_occurrences: int | None = None,
    anchor_income_on_scheduled: bool | None = None,
    drop_final_income: bool | None = None,
    filter_off_cadence: bool | None = None,
    policy: RecurrencePolicy | None = None,
) -> tuple[RecurringCommitment, ...]:
    """Infer recurrences that the user's own history supports.

    Amounts are converted to home currency first, so a salary paid in USD and a
    rent paid in INR sit on the same timeline.

    Income is treated differently from spending in three ways, because the rule
    is to count confirmed income and never invent the rest. A scheduled payroll
    row is confirmed, so it anchors both the cadence and the amount; two
    occurrences are enough when one of them is that confirmed row; and a stream
    whose last payment is described as final is not projected at all.
    """
    chosen = policy or DEFAULT_POLICY
    lookback_days = chosen.lookback_days if lookback_days is None else lookback_days
    min_occurrences = chosen.min_occurrences if min_occurrences is None else min_occurrences
    recent_window = chosen.recent_window if recent_window is None else recent_window
    debit_estimator = chosen.debit_estimator if debit_estimator is None else debit_estimator
    credit_estimator = chosen.credit_estimator if credit_estimator is None else credit_estimator
    project_income = chosen.project_income if project_income is None else project_income
    income_min_occurrences = (
        chosen.income_min_occurrences
        if income_min_occurrences is None
        else income_min_occurrences
    )
    anchor_income_on_scheduled = (
        chosen.anchor_income_on_scheduled
        if anchor_income_on_scheduled is None
        else anchor_income_on_scheduled
    )
    drop_final_income = (
        chosen.drop_final_income if drop_final_income is None else drop_final_income
    )
    filter_off_cadence = (
        chosen.filter_off_cadence if filter_off_cadence is None else filter_off_cadence
    )

    profile = dataset.profile(user_id)
    window_start = as_of_date - timedelta(days=lookback_days)
    groups: dict[tuple[str, str, Direction], list[EventRecord]] = defaultdict(list)
    scheduled_income: dict[tuple[str, str, Direction], list[EventRecord]] = defaultdict(list)

    for event in dataset.events_by_user.get(user_id, ()):
        if not event.is_cash or event.amount is None:
            continue
        key = (event.event_type.value, event.category, event.direction)

        if (
            event.direction is Direction.CREDIT
            and event.status is EventStatus.SCHEDULED
            and event.effective_date >= as_of_date
        ):
            scheduled_income[key].append(event)
            continue

        if counted_event_ids is not None and event.event_id not in counted_event_ids:
            continue
        if event.status is not EventStatus.SETTLED:
            continue
        if not window_start <= event.effective_date <= as_of_date:
            continue
        if event.direction is Direction.CREDIT and not project_income:
            continue
        if event.direction is Direction.DEBIT:
            if not chosen.project_dining and event.category in DINING_CATEGORIES:
                continue
            if (
                not chosen.project_discretionary
                and event.category in DISCRETIONARY_CATEGORIES
            ):
                continue
            if (
                not chosen.project_subscriptions
                and event.category in SUBSCRIPTION_CATEGORIES
            ):
                continue
        groups[(event.event_type.value, event.category, event.direction)].append(event)

    commitments: list[RecurringCommitment] = []
    keys = sorted(
        set(groups) | (set(scheduled_income) if project_income else set()),
        key=lambda key: (key[1], key[0]),
    )
    for key in keys:
        event_type, category, direction = key
        events = sorted(groups.get(key, ()), key=lambda e: (e.effective_date, e.event_id))
        confirmed = sorted(
            scheduled_income.get(key, ()) if anchor_income_on_scheduled else (),
            key=lambda e: (e.effective_date, e.event_id),
        )
        series = events + confirmed

        if direction is Direction.CREDIT:
            if not project_income:
                continue
            if drop_final_income and events and FINAL_INCOME_RE.search(events[-1].description):
                continue  # the stream has ended; projecting it would invent income
            required = income_min_occurrences if confirmed else min_occurrences
        else:
            required = min_occurrences
        if len(series) < max(2, required):
            continue

        gaps = [
            (series[index + 1].effective_date - series[index].effective_date).days
            for index in range(len(series) - 1)
        ]
        median_gap = statistics.median(gaps)
        if median_gap < 1:
            continue
        monthly = int(median_gap) in MONTHLY_GAP_RANGE
        tolerance = 3 if monthly else max(1, int(round(median_gap * 0.25)))

        if filter_off_cadence:
            series = _on_cadence(series, median_gap, tolerance)
            gaps = [
                (series[index + 1].effective_date - series[index].effective_date).days
                for index in range(len(series) - 1)
            ]
            if not gaps:
                continue
            median_gap = statistics.median(gaps)
            monthly = int(median_gap) in MONTHLY_GAP_RANGE
            tolerance = 3 if monthly else max(1, int(round(median_gap * 0.25)))

        if not _is_regular(gaps, median_gap, tolerance):
            continue

        if (
            direction is Direction.CREDIT
            and not chosen.project_irregular_income
            and not confirmed
            and not monthly
        ):
            continue

        if confirmed:
            # A confirmed future payroll states the amount; no need to estimate it.
            amount = to_money(event_amount_in_home_currency(dataset, confirmed[-1]))
        else:
            amounts = [
                to_money(event_amount_in_home_currency(dataset, event)) for event in series
            ]
            if direction is Direction.CREDIT:
                estimator = credit_estimator
            elif category in VARIABLE_DEBIT_CATEGORIES:
                estimator = chosen.variable_debit_estimator
            else:
                estimator = debit_estimator
            amount = estimate_amount(
                amounts,
                estimator=estimator,
                recent_window=recent_window,
            )
        if category in {"groceries", "transport"} and chosen.variable_scale != "1.00":
            amount = to_money(amount * Decimal(chosen.variable_scale))
        if amount <= 0:
            continue
        last = series[-1]
        events = series
        minimums = [e.minimum_allowed_amount for e in events if e.minimum_allowed_amount]
        flexibility = last.flexibility
        if category in DINING_CATEGORIES | DISCRETIONARY_CATEGORIES:
            flexibility = Flexibility.REDUCIBLE_OR_STOPPABLE
        frequency_days = 30 if monthly else int(median_gap)
        if monthly:
            next_occurrence = add_months(last.effective_date, 1)
        else:
            next_occurrence = last.effective_date + timedelta(days=frequency_days)
        if (
            direction is Direction.DEBIT
            and category in DINING_CATEGORIES | DISCRETIONARY_CATEGORIES
            and not monthly
        ):
            series_amounts = [
                to_money(event_amount_in_home_currency(dataset, event))
                for event in events
                if event.amount is not None
            ]
            frequency_days, next_occurrence = soften_variable_lifestyle_cadence(
                events,
                series_amounts,
                median_gap=median_gap,
                frequency_days=frequency_days,
                last_date=last.effective_date,
                policy=chosen,
            )
        commitments.append(
            RecurringCommitment(
                user_id=user_id,
                label=f"{category} ({event_type})",
                category=category,
                direction=direction,
                amount=amount,
                frequency_days=frequency_days,
                next_occurrence=next_occurrence,
                flexibility=flexibility,
                minimum_allowed_amount=min(minimums) if minimums else None,
                representative_event_id=last.event_id,
                source_event_ids=tuple(event.event_id for event in events),
                occurrences_observed=len(events),
            )
        )
    return tuple(commitments)


def occurrences_between(
    commitment: RecurringCommitment, start: date, end: date
) -> tuple[date, ...]:
    """Dates on which a commitment falls inside ``[start, end]``."""
    monthly = commitment.frequency_days == 30
    anchor = commitment.next_occurrence
    dates: list[date] = []
    if monthly:
        index = 0
        while True:
            when = add_months(anchor, index)
            if when > end:
                break
            if when >= start:
                dates.append(when)
            index += 1
            if index > 2 * FORECAST_HORIZON_DAYS:
                break
    else:
        when = anchor
        while when <= end:
            if when >= start:
                dates.append(when)
            when += timedelta(days=commitment.frequency_days)
    return tuple(dates)


def is_essential(category: str, protected: tuple[str, ...]) -> bool:
    return category in ESSENTIAL_CATEGORIES or category in protected


def calculate_essential_spending_baseline(
    dataset: Dataset,
    *,
    user_id: str,
    as_of_date: date,
    commitments: tuple[RecurringCommitment, ...],
    horizon_days: int = FORECAST_HORIZON_DAYS,
) -> EssentialSpendingBaseline:
    """Summarize forecast essential outflow per 30 days."""
    profile = dataset.profile(user_id)
    per_category: dict[str, Decimal] = defaultdict(lambda: ZERO)
    sources: list[str] = []
    for commitment in commitments:
        if commitment.direction is not Direction.DEBIT:
            continue
        if not is_essential(commitment.category, profile.expense_categories_to_protect):
            continue
        monthly_multiple = Decimal(30) / Decimal(commitment.frequency_days)
        per_category[commitment.category] += to_money(commitment.amount * monthly_multiple)
        sources.extend(commitment.source_event_ids)
    return EssentialSpendingBaseline(
        user_id=user_id,
        as_of_date=as_of_date,
        horizon_days=horizon_days,
        per_category_monthly={k: to_money(v) for k, v in sorted(per_category.items())},
        total_monthly=to_money(sum(per_category.values(), ZERO)),
        source_event_ids=tuple(dict.fromkeys(sources)),
        method="conservative recurrence of essential and protected categories",
    )


# --------------------------------------------------------------------------- #
# Dated flows already committed on the calendar
# --------------------------------------------------------------------------- #


def committed_flows(
    dataset: Dataset,
    *,
    user_id: str,
    start: date,
    end: date,
    lifecycle: LifecycleResolution,
    resolved_amounts: dict[str, Decimal] | None = None,
) -> tuple[CashFlow, ...]:
    """Pending and scheduled events that fall inside the forecast window.

    Pending debits are reserved. Pending credits were already dropped by
    lifecycle resolution, so nothing here can count unsettled money in.
    """
    counted = frozenset(lifecycle.counted_event_ids)
    overrides = resolved_amounts or {}
    flows: list[CashFlow] = []
    for event in dataset.events_by_user.get(user_id, ()):
        if event.event_id not in counted or event.status is EventStatus.SETTLED:
            continue
        if not start <= event.effective_date <= end:
            continue
        state = event.cash_state
        if state not in {CashState.RESERVED_DEBIT, CashState.CONFIRMED_CREDIT}:
            continue
        amount = overrides.get(event.event_id, event.amount)
        if amount is None:
            continue  # blank amount awaiting evidence; surfaced as a state note
        flows.append(
            CashFlow(
                date=event.effective_date,
                label=event.description,
                direction=event.direction,
                amount=event_amount_in_home_currency(dataset, event, amount=amount),
                source=f"event:{event.event_id}",
                category=event.category,
                event_id=event.event_id,
                essential=is_essential(
                    event.category, dataset.profile(user_id).expense_categories_to_protect
                ),
            )
        )
    return tuple(sorted(flows, key=lambda flow: (flow.date, flow.source)))


# --------------------------------------------------------------------------- #
# State assembly
# --------------------------------------------------------------------------- #


def build_financial_state(
    dataset: Dataset,
    request: RequestRecord,
    *,
    adjustments: tuple[ForecastAdjustment, ...] = (),
    horizon_days: int = FORECAST_HORIZON_DAYS,
    lookback_days: int | None = None,
    min_occurrences: int | None = None,
    recent_window: int | None = None,
    debit_estimator: str | None = None,
    credit_estimator: str | None = None,
    project_income: bool | None = None,
    policy: RecurrencePolicy | None = None,
) -> FinancialState:
    """Reconstruct one user's cash position as of a request date."""
    chosen = policy or DEFAULT_POLICY
    profile = dataset.profile(request.user_id)
    horizon_end = request.request_date + timedelta(days=horizon_days)

    lifecycle = resolve_event_lifecycle(
        dataset, user_id=request.user_id, as_of_date=request.request_date
    )
    counted = frozenset(lifecycle.counted_event_ids)

    resolved_amounts: dict[str, Decimal] = {}
    cancelled_by_evidence: set[str] = set()
    shifted: dict[str, date] = {}
    amount_rules: list[tuple[str, Decimal, date | None, str]] = []
    excluded_categories: set[str] = set()
    extra_flows: list[CashFlow] = []
    applied: list[ForecastAdjustment] = []

    for adjustment in adjustments:
        applied.append(adjustment)
        if adjustment.kind == "set_event_amount" and adjustment.target_event_id:
            if adjustment.amount is None:
                raise ValueError(f"{adjustment.source_id}: set_event_amount needs an amount")
            resolved_amounts[adjustment.target_event_id] = adjustment.amount
        elif adjustment.kind == "cancel_event" and adjustment.target_event_id:
            cancelled_by_evidence.add(adjustment.target_event_id)
        elif adjustment.kind == "shift_event_date" and adjustment.target_event_id:
            if adjustment.new_date is None:
                raise ValueError(f"{adjustment.source_id}: shift_event_date needs a date")
            shifted[adjustment.target_event_id] = adjustment.new_date
        elif adjustment.kind == "set_recurring_amount" and adjustment.target_category:
            if adjustment.amount is None:
                raise ValueError(
                    f"{adjustment.source_id}: set_recurring_amount needs an amount"
                )
            amount_rules.append(
                (
                    adjustment.target_category,
                    adjustment.amount,
                    adjustment.effective_date,
                    adjustment.source_id,
                )
            )
        elif adjustment.kind == "exclude_recurring" and adjustment.target_category:
            excluded_categories.add(adjustment.target_category)
        elif adjustment.kind in {"add_confirmed_credit", "add_scheduled_debit"}:
            if adjustment.amount is None or adjustment.effective_date is None:
                raise ValueError(
                    f"{adjustment.source_id}: {adjustment.kind} needs an amount and date"
                )
            credit = adjustment.kind == "add_confirmed_credit"
            extra_flows.append(
                CashFlow(
                    date=adjustment.effective_date,
                    label=adjustment.reason or adjustment.kind,
                    direction=Direction.CREDIT if credit else Direction.DEBIT,
                    amount=adjustment.amount,
                    source=f"evidence:{adjustment.source_id}",
                    category=adjustment.target_category or "",
                )
            )

    if cancelled_by_evidence:
        counted = frozenset(counted - cancelled_by_evidence)
        lifecycle = LifecycleResolution(
            user_id=lifecycle.user_id,
            as_of_date=lifecycle.as_of_date,
            counted_event_ids=tuple(
                e for e in lifecycle.counted_event_ids if e not in cancelled_by_evidence
            ),
            excluded_event_ids=lifecycle.excluded_event_ids
            + tuple(sorted(cancelled_by_evidence)),
            exclusion_reasons={
                **lifecycle.exclusion_reasons,
                **{e: "cancelled by evidence" for e in sorted(cancelled_by_evidence)},
            },
        )

    commitments = infer_recurring_commitments(
        dataset,
        user_id=request.user_id,
        as_of_date=request.request_date,
        lookback_days=lookback_days,
        min_occurrences=min_occurrences,
        recent_window=recent_window,
        debit_estimator=debit_estimator,
        credit_estimator=credit_estimator,
        project_income=project_income,
        counted_event_ids=counted,
        policy=policy,
    )
    if excluded_categories:
        commitments = tuple(
            commitment
            for commitment in commitments
            if commitment.category not in excluded_categories
        )
    overrides = tuple(
        RecurringAmountOverride(
            event_id=commitment.representative_event_id,
            category=commitment.category,
            amount=amount,
            source_id=source_id,
            effective_date=effective_date,
        )
        for category, amount, effective_date, source_id in amount_rules
        for commitment in commitments
        if commitment.category == category
    )

    flows = list(
        committed_flows(
            dataset,
            user_id=request.user_id,
            start=request.request_date,
            end=horizon_end,
            lifecycle=lifecycle,
            resolved_amounts=resolved_amounts,
        )
    )
    if shifted:
        flows = [
            flow.model_copy(update={"date": shifted[flow.event_id]})
            if flow.event_id in shifted
            else flow
            for flow in flows
        ]
    flows.extend(extra_flows)

    notes: list[str] = []
    unresolved = [
        event.event_id
        for event in dataset.events_by_user.get(request.user_id, ())
        if event.amount_needs_resolution
        and event.event_id in counted
        and event.event_id not in resolved_amounts
    ]
    if unresolved:
        notes.append(
            "blank amounts awaiting image evidence: " + ", ".join(sorted(unresolved))
        )
    if not any(c.direction is Direction.CREDIT for c in commitments) and not any(
        flow.direction is Direction.CREDIT for flow in flows
    ):
        notes.append("no income is projected in the forecast window")

    return FinancialState(
        user_id=request.user_id,
        request_id=request.request_id,
        as_of=request.request_date,
        horizon_end=horizon_end,
        home_currency=profile.home_currency,
        opening_balance=profile.current_available_balance,
        minimum_balance_to_keep=profile.minimum_balance_to_keep,
        dated_flows=tuple(sorted(flows, key=lambda flow: (flow.date, flow.source))),
        recurring=commitments,
        recurring_overrides=overrides,
        lifecycle=lifecycle,
        baseline=calculate_essential_spending_baseline(
            dataset,
            user_id=request.user_id,
            as_of_date=request.request_date,
            commitments=commitments,
            horizon_days=horizon_days,
        ),
        adjustments_applied=tuple(applied),
        notes=tuple(notes),
        plan_after_same_day_income=chosen.plan_after_same_day_income,
    )
