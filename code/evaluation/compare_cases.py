"""Compare calibration input features vs amount_safe_to_pay labels.

Does not change ledger or forecast logic. Hold-out labels are never loaded.

    .venv/bin/python code/evaluation/compare_cases.py
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from domain import Direction, EventStatus, to_money  # noqa: E402
from tools.data import load_dataset, load_sample_labels  # noqa: E402
from tools.evidence import build_adjustments  # noqa: E402
from tools.forecast import calculate_safe_amount_today, simulate  # noqa: E402
from tools.ledger import DEFAULT_POLICY, build_financial_state  # noqa: E402

ZERO = Decimal("0.00")
TOL = Decimal("0.05")


def _close(left: Decimal, right: Decimal) -> bool:
    return abs(left - right) <= TOL


def _pending_sum(dataset, user_id: str) -> Decimal:
    total = ZERO
    for event in dataset.events_by_user.get(user_id, ()):
        if (
            event.direction is Direction.DEBIT
            and event.status is EventStatus.PENDING
            and event.amount is not None
        ):
            total += event.amount
    return to_money(total)


def _trough_headroom(state, cutoff: date) -> tuple[Decimal, date | None, Decimal]:
    """Lowest balance on or before cutoff, and headroom above the floor."""
    result = simulate(state)
    lowest = state.opening_balance
    lowest_date = state.as_of
    for entry in result.entries:
        if entry.date > cutoff:
            break
        if entry.balance_after < lowest:
            lowest, lowest_date = entry.balance_after, entry.date
    return lowest, lowest_date, to_money(lowest - state.minimum_balance_to_keep)


def _next_income_date(state) -> date | None:
    dates = [
        commitment.next_occurrence
        for commitment in state.recurring
        if commitment.direction is Direction.CREDIT
    ]
    dates.extend(
        flow.date
        for flow in state.dated_flows
        if flow.direction is Direction.CREDIT
    )
    future = [when for when in dates if when >= state.as_of]
    return min(future) if future else None


def main() -> int:
    dataset = load_dataset()
    labels = load_sample_labels("calibration")

    print(
        "| request_id | requested_amount | label_amount | calculated_amount | "
        "opening_balance | min_protected_balance | net_headroom_today | "
        "trough_before_next_income | is_exact_match | is_label_equal_requested |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---|---|")

    rows: list[dict] = []
    for lab in labels:
        request = dataset.sample_requests[lab.request_id]
        adjustments, _, _ = build_adjustments(dataset, request)
        state = build_financial_state(
            dataset, request, adjustments=adjustments, policy=DEFAULT_POLICY
        )
        calculated = calculate_safe_amount_today(state, request).amount_safe_to_pay
        label = Decimal(lab.amount_safe_to_pay)
        opening = state.opening_balance
        minimum = state.minimum_balance_to_keep
        net = to_money(opening - minimum)
        pending = _pending_sum(dataset, request.user_id)
        next_income = _next_income_date(state)
        if next_income is None:
            trough_bal, trough_date, trough_head = _trough_headroom(
                state, state.horizon_end
            )
        else:
            # Headroom just before the next credit lands.
            trough_bal, trough_date, trough_head = _trough_headroom(
                state, next_income - timedelta(days=1)
            )
        d14_bal, d14_date, d14_head = _trough_headroom(
            state, request.request_date + timedelta(days=14)
        )
        d30_bal, d30_date, d30_head = _trough_headroom(
            state, request.request_date + timedelta(days=30)
        )
        d90_bal, d90_date, d90_head = _trough_headroom(state, state.horizon_end)
        deadline_bal, deadline_date, deadline_head = _trough_headroom(
            state, request.desired_completion_date
        )

        row = {
            "request_id": lab.request_id,
            "requested": request.requested_amount,
            "label": label,
            "calculated": calculated,
            "opening": opening,
            "minimum": minimum,
            "net": net,
            "pending": pending,
            "next_income": next_income,
            "trough_before_income": trough_head,
            "trough_before_income_date": trough_date,
            "trough_before_income_bal": trough_bal,
            "d14": d14_head,
            "d14_date": d14_date,
            "d30": d30_head,
            "d30_date": d30_date,
            "d90": d90_head,
            "d90_date": d90_date,
            "deadline": deadline_head,
            "deadline_date": deadline_date,
            "exact": calculated == label,
            "label_eq_req": label == request.requested_amount,
        }
        rows.append(row)
        print(
            f"| {row['request_id']} | {row['requested']} | {row['label']} | "
            f"{row['calculated']} | {row['opening']} | {row['minimum']} | "
            f"{row['net']} | {row['trough_before_income']} | "
            f"{row['exact']} | {row['label_eq_req']} |"
        )

    print()
    print("## Ratio label_amount / net_headroom_today")
    print()
    print("| request_id | label / net_headroom | label / requested | notes |")
    print("|---|---:|---:|---|")
    ratios: list[Decimal] = []
    for row in rows:
        if row["net"] > 0:
            ratio = (row["label"] / row["net"]).quantize(Decimal("0.0001"))
            ratios.append(ratio)
        else:
            ratio = Decimal("NaN")
        req_ratio = (
            (row["label"] / row["requested"]).quantize(Decimal("0.0001"))
            if row["requested"]
            else Decimal("NaN")
        )
        note = []
        if row["exact"]:
            note.append("exact calc match")
        if row["label_eq_req"]:
            note.append("label = requested")
        print(
            f"| {row['request_id']} | {ratio} | {req_ratio} | "
            f"{'; '.join(note) or '—'} |"
        )
    finite = [r for r in ratios if r == r]
    if finite:
        print()
        print(
            f"ratio range {min(finite)} to {max(finite)}; "
            f"mean {(sum(finite) / len(finite)).quantize(Decimal('0.0001'))}. "
            "No single shared fraction."
        )

    print()
    print("## Failing-row formula checks")
    print()
    print(
        "A hit is |label - candidate| <= 0.05. "
        "`capped_*` is min(requested, max(0, headroom))."
    )
    print()
    print(
        "| request_id | net-pending | capped(net-pending) | "
        "capped 14d | capped 30d | capped 90d | "
        "capped pre-income | capped deadline |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for row in rows:
        if row["exact"]:
            continue
        net_pend = to_money(row["net"] - row["pending"])
        candidates = {
            "net-pending": net_pend,
            "capped(net-pending)": max(ZERO, min(row["requested"], net_pend)),
            "capped 14d": max(ZERO, min(row["requested"], row["d14"])),
            "capped 30d": max(ZERO, min(row["requested"], row["d30"])),
            "capped 90d": max(ZERO, min(row["requested"], row["d90"])),
            "capped pre-income": max(
                ZERO, min(row["requested"], row["trough_before_income"])
            ),
            "capped deadline": max(ZERO, min(row["requested"], row["deadline"])),
        }
        flags = []
        for name in (
            "net-pending",
            "capped(net-pending)",
            "capped 14d",
            "capped 30d",
            "capped 90d",
            "capped pre-income",
            "capped deadline",
        ):
            flags.append("HIT" if _close(candidates[name], row["label"]) else "no")
        print(
            f"| {row['request_id']} | {flags[0]} ({candidates['net-pending']}) | "
            f"{flags[1]} ({candidates['capped(net-pending)']}) | "
            f"{flags[2]} ({candidates['capped 14d']}) | "
            f"{flags[3]} ({candidates['capped 30d']}) | "
            f"{flags[4]} ({candidates['capped 90d']}) | "
            f"{flags[5]} ({candidates['capped pre-income']}) | "
            f"{flags[6]} ({candidates['capped deadline']}) |"
        )

    print()
    print("## Horizon detail on failing rows")
    print()
    for row in rows:
        if row["exact"]:
            continue
        print(
            f"- **{row['request_id']}** pending={row['pending']} "
            f"next_income={row['next_income']} "
            f"pre-income {row['trough_before_income']} on {row['trough_before_income_date']}; "
            f"14d {row['d14']} on {row['d14_date']}; "
            f"30d {row['d30']} on {row['d30_date']}; "
            f"90d {row['d90']} on {row['d90_date']}; "
            f"deadline {row['deadline']} on {row['deadline_date']}; "
            f"label {row['label']} calc {row['calculated']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
