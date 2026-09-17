"""Test whether financial_priorities sets the shadow-trough horizon.

Does not change ledger or forecast logic. Hold-out labels are never loaded.

Hypothesis: emergency_savings or retirement_investment → 90-day trough;
otherwise trough only through next payday, falling back to 30 days.
No 1.1x floor buffer. Lifestyle spend uses a global 0.60 haircut on
max_recent (the pre-category-rule amount path).

    .venv/bin/python code/evaluation/priority_horizon.py
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from domain import Direction, to_money  # noqa: E402
from tools.data import load_dataset, load_sample_labels  # noqa: E402
from tools.evidence import build_adjustments  # noqa: E402
from tools.forecast import (  # noqa: E402
    calculate_safe_amount_today,
    simulate,
    trough_through,
)
from tools.ledger import (  # noqa: E402
    DEFAULT_POLICY,
    DINING_CATEGORIES,
    DISCRETIONARY_CATEGORIES,
    build_financial_state,
    estimate_amount,
)
from tools.money import event_amount_in_home_currency  # noqa: E402

ZERO = Decimal("0.00")
GLOBAL_LIFESTYLE_SCALE = Decimal("0.60")
LIFESTYLE = DINING_CATEGORIES | DISCRETIONARY_CATEGORIES
LONG_HORIZON_PRIORITIES = frozenset({"emergency_savings", "retirement_investment"})
MISS_IDS = (
    "request_02",
    "request_03",
    "request_04",
    "request_06",
    "request_07",
    "request_08",
)
ABS_TOL = Decimal("0.05")


def _close(left: Decimal, right: Decimal) -> bool:
    return abs(left - right) <= ABS_TOL


def _capped(headroom: Decimal, requested: Decimal) -> Decimal:
    return max(ZERO, min(requested, to_money(headroom)))


def _next_income_date(state) -> date | None:
    dates = [
        commitment.next_occurrence
        for commitment in state.recurring
        if commitment.direction is Direction.CREDIT
    ]
    dates.extend(
        flow.date for flow in state.dated_flows if flow.direction is Direction.CREDIT
    )
    future = [when for when in dates if when >= state.as_of]
    return min(future) if future else None


def _lifestyle_max_recent(dataset, state):
    """Old shadow: dining/shopping/entertainment at max_recent, floor unchanged."""
    updated = []
    changed = False
    for commitment in state.recurring:
        if (
            commitment.direction is Direction.DEBIT
            and commitment.category in LIFESTYLE
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


def _force_global_lifestyle_smoothing(dataset, state):
    """Identity: expand_recurring_flows already applies the global 0.60 haircut."""
    return state


def _headroom_capped(state, cutoff: date, requested: Decimal) -> tuple[Decimal, date | None, Decimal]:
    result = simulate(state)
    lowest, lowest_date = trough_through(result, cutoff)
    head = to_money(lowest - state.minimum_balance_to_keep)
    return lowest, lowest_date, _capped(head, requested)


def _uses_long_horizon(profile) -> bool:
    return any(name in LONG_HORIZON_PRIORITIES for name in profile.financial_priorities)


def main() -> int:
    dataset = load_dataset()
    labels = {lab.request_id: lab for lab in load_sample_labels("calibration")}

    print("# financial_priorities as shadow-trough horizon")
    print()
    print("Diagnostic only. ledger.py and forecast.py are not modified.")
    print("Amount path: lifestyle max_recent × global 0.60. No 1.1× floor.")
    print(
        "Horizon: 90-day if emergency_savings or retirement_investment; "
        "else next payday, else 30 days."
    )
    print()
    print(
        "| request_id | priorities | long_horizon | next_payday | "
        "days_to_payday | cutoff | label | current_engine | "
        "hyp_payday_or_30 | hyp_min(payday,30) | 90d | payday | payday-1 | 30d |"
    )
    print("|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|")

    closed_02 = closed_08 = False
    notes = []

    for rid in MISS_IDS:
        lab = labels[rid]
        request = dataset.sample_requests[rid]
        profile = dataset.profile(request.user_id)
        adjustments, _, _ = build_adjustments(dataset, request)
        state = build_financial_state(
            dataset, request, adjustments=adjustments, policy=DEFAULT_POLICY
        )
        shadow = _force_global_lifestyle_smoothing(
            dataset, _lifestyle_max_recent(dataset, state)
        )
        label = Decimal(lab.amount_safe_to_pay)
        engine = calculate_safe_amount_today(state, request).amount_safe_to_pay
        payday = _next_income_date(state)
        d30 = request.request_date + timedelta(days=30)
        long_horizon = _uses_long_horizon(profile)
        days_to_payday = (payday - request.request_date).days if payday else None

        if long_horizon:
            cutoff_or = cutoff_min = state.horizon_end
        else:
            cutoff_or = payday if payday is not None else d30
            cutoff_min = min(payday, d30) if payday is not None else d30

        _low90, date90, cap90 = _headroom_capped(shadow, state.horizon_end, request.requested_amount)
        cap_pay = cap_pay1 = None
        date_pay = date_pay1 = None
        if payday is not None:
            _lp, date_pay, cap_pay = _headroom_capped(
                shadow, payday, request.requested_amount
            )
            _lp1, date_pay1, cap_pay1 = _headroom_capped(
                shadow, payday - timedelta(days=1), request.requested_amount
            )
        _l30, date30, cap30 = _headroom_capped(shadow, d30, request.requested_amount)
        _lh, date_or, cap_or = _headroom_capped(
            shadow, cutoff_or, request.requested_amount
        )
        _lh2, date_min, cap_min = _headroom_capped(
            shadow, cutoff_min, request.requested_amount
        )

        hit_or = _close(cap_or, label)
        hit_min = _close(cap_min, label)
        if rid == "request_02":
            closed_02 = hit_or or hit_min
        if rid == "request_08":
            closed_08 = hit_or or hit_min

        flags = []
        if hit_or:
            flags.append("HIT payday-or-30")
        if hit_min:
            flags.append("HIT min(payday,30)")
        if _close(cap90, label):
            flags.append("HIT 90d")
        notes.append(
            (
                rid,
                profile.financial_priorities,
                long_horizon,
                payday,
                cutoff_or,
                date_or,
                label,
                cap_or,
                cap_min,
                engine,
                flags,
            )
        )
        print(
            f"| {rid} | {'|'.join(profile.financial_priorities)} | {long_horizon} | "
            f"{payday} | {days_to_payday if days_to_payday is not None else 'n/a'} | "
            f"{cutoff_or} | {label} | {engine} | {cap_or} | {cap_min} | {cap90} | "
            f"{cap_pay if cap_pay is not None else 'n/a'} | "
            f"{cap_pay1 if cap_pay1 is not None else 'n/a'} | {cap30} |"
        )

    print()
    print("## Deltas (label − hypothesis payday-or-30)")
    print()
    print("| request_id | label − hyp | label − engine | trough date (hyp cutoff) | flags |")
    print("|---|---:|---:|---|---|")
    for (
        rid,
        _prio,
        _long,
        _payday,
        cutoff_or,
        date_or,
        label,
        cap_or,
        _cap_min,
        engine,
        flags,
    ) in notes:
        print(
            f"| {rid} | {to_money(label - cap_or)} | {to_money(label - engine)} | "
            f"{date_or} (cutoff {cutoff_or}) | {', '.join(flags) or '—'} |"
        )

    print()
    print("## Does dynamic horizon close request_02 or request_08?")
    print()
    print(
        f"- request_02: {'YES, exact match under this rule' if closed_02 else 'NO, not an exact match'}"
    )
    print(
        f"- request_08: {'YES, exact match under this rule' if closed_08 else 'NO, not an exact match'}"
    )
    print()
    print(
        "If the 90-day trough already falls on or before payday, shortening "
        "the window cannot change the number. That is the case to check on "
        "the two rows that do not list emergency_savings / retirement_investment."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
