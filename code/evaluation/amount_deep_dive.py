"""Deep-dive the six remaining amount_safe_to_pay misses on calibration.

Does not change ledger or forecast logic. Hold-out labels are never loaded.

    .venv/bin/python code/evaluation/amount_deep_dive.py
"""

from __future__ import annotations

import math
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
    _shadow_lifestyle_max_recent,
    calculate_safe_amount_today,
    simulate,
    trough_through,
)
from tools.ledger import (  # noqa: E402
    DEFAULT_POLICY,
    DINING_CATEGORIES,
    DISCRETIONARY_CATEGORIES,
    SUBSCRIPTION_CATEGORIES,
    VARIABLE_DEBIT_CATEGORIES,
    build_financial_state,
)

ZERO = Decimal("0.00")
ABS_TOL = Decimal("0.05")
REL_TOL = Decimal("0.002")  # 0.2%
MISS_IDS = (
    "request_02",
    "request_03",
    "request_04",
    "request_06",
    "request_07",
    "request_08",
)
LIFESTYLE = DINING_CATEGORIES | DISCRETIONARY_CATEGORIES
FIXED_LIKE = frozenset(
    {
        "rent",
        "housing",
        "insurance",
        "debt_repayment",
        "education",
        "family_support",
    }
) | SUBSCRIPTION_CATEGORIES


def _close(left: Decimal, right: Decimal) -> bool:
    if abs(left - right) <= ABS_TOL:
        return True
    scale = max(abs(right), Decimal("1"))
    return abs(left - right) / scale <= REL_TOL


def _ratio(num: Decimal, den: Decimal) -> str:
    if den == 0:
        return "n/a"
    return str((num / den).quantize(Decimal("0.0001")))


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


def _drop_categories(state, categories: frozenset[str]):
    kept = tuple(
        commitment
        for commitment in state.recurring
        if not (
            commitment.direction is Direction.DEBIT
            and commitment.category in categories
        )
    )
    return state.model_copy(update={"recurring": kept})


def _keep_debit_categories(state, categories: frozenset[str]):
    kept = tuple(
        commitment
        for commitment in state.recurring
        if commitment.direction is Direction.CREDIT
        or commitment.category in categories
    )
    return state.model_copy(update={"recurring": kept})


def _scale_categories(state, categories: frozenset[str], factor: Decimal):
    updated = []
    for commitment in state.recurring:
        if (
            commitment.direction is Direction.DEBIT
            and commitment.category in categories
        ):
            commitment = commitment.model_copy(
                update={"amount": to_money(commitment.amount * factor)}
            )
        updated.append(commitment)
    return state.model_copy(update={"recurring": tuple(updated)})


def _headroom(state, cutoff: date | None = None) -> tuple[Decimal, date | None, Decimal]:
    result = simulate(state)
    if cutoff is None:
        lowest, lowest_date = result.minimum_projected_balance, result.minimum_balance_date
    else:
        lowest, lowest_date = trough_through(result, cutoff)
    return lowest, lowest_date, to_money(lowest - state.minimum_balance_to_keep)


def _slope_until(state, end: date) -> dict:
    """Daily net cash-flow slope from request_date through ``end`` (inclusive)."""
    result = simulate(state)
    start = state.as_of
    days = (end - start).days
    credits = ZERO
    debits = ZERO
    last_balance = state.opening_balance
    end_balance = state.opening_balance
    for entry in result.entries:
        if entry.date < start:
            continue
        if entry.date > end:
            break
        end_balance = entry.balance_after
        last_balance = entry.balance_after
        if entry.direction is Direction.CREDIT:
            credits += entry.amount
        elif entry.direction is Direction.DEBIT:
            debits += entry.amount
    span = Decimal(days) if days > 0 else Decimal("1")
    net = to_money(credits - debits)
    balance_delta = to_money(end_balance - state.opening_balance)
    return {
        "days": days,
        "credits": to_money(credits),
        "debits": to_money(debits),
        "net": net,
        "flow_per_day": to_money(net / span),
        "balance_end": to_money(last_balance),
        "balance_delta": balance_delta,
        "balance_per_day": to_money(balance_delta / span),
    }


def _candidate_table(name: str, value: Decimal, label: Decimal) -> str:
    hit = "HIT" if _close(value, label) else "no"
    err = to_money(label - value)
    return f"{hit} ({value}, Δ{err})"


def main() -> int:
    dataset = load_dataset()
    labels = {lab.request_id: lab for lab in load_sample_labels("calibration")}
    smoothing = Decimal(DEFAULT_POLICY.lifestyle_smoothing)
    unsmooth = (Decimal("1") / smoothing) if smoothing else Decimal("1")

    print("# amount_safe_to_pay deep dive (six remaining misses)")
    print()
    print("Diagnostic only. ledger.py and forecast.py are not modified.")
    print(f"Match: |label-candidate| <= {ABS_TOL} or relative <= {REL_TOL}.")
    print()

    records = []
    print(
        "| request_id | label_amount | current_calculated | delta "
        "(label-calc) | label/opening | label/net_headroom | "
        "label/shadow_trough | days_to_payroll | daily_net_slope |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")

    for rid in MISS_IDS:
        lab = labels[rid]
        request = dataset.sample_requests[rid]
        adjustments, _, _ = build_adjustments(dataset, request)
        state = build_financial_state(
            dataset, request, adjustments=adjustments, policy=DEFAULT_POLICY
        )
        shadow = _shadow_lifestyle_max_recent(state)
        calculated = calculate_safe_amount_today(state, request).amount_safe_to_pay
        label = Decimal(lab.amount_safe_to_pay)
        opening = state.opening_balance
        minimum = state.minimum_balance_to_keep
        net = to_money(opening - minimum)
        shadow_low, shadow_date, shadow_head = _headroom(shadow)
        next_income = _next_income_date(state)
        if next_income is not None and next_income > state.as_of:
            slope = _slope_until(state, next_income - timedelta(days=1))
        elif next_income is not None:
            slope = {
                "days": 0,
                "credits": ZERO,
                "debits": ZERO,
                "net": ZERO,
                "flow_per_day": ZERO,
                "balance_end": opening,
                "balance_delta": ZERO,
                "balance_per_day": ZERO,
            }
        else:
            slope = _slope_until(state, state.as_of + timedelta(days=14))

        rec = {
            "rid": rid,
            "request": request,
            "state": state,
            "shadow": shadow,
            "label": label,
            "calculated": calculated,
            "opening": opening,
            "minimum": minimum,
            "net": net,
            "requested": request.requested_amount,
            "shadow_head": shadow_head,
            "shadow_date": shadow_date,
            "shadow_low": shadow_low,
            "next_income": next_income,
            "slope": slope,
        }
        records.append(rec)
        print(
            f"| {rid} | {label} | {calculated} | {to_money(label - calculated)} | "
            f"{_ratio(label, opening)} | {_ratio(label, net)} | "
            f"{_ratio(label, shadow_head)} | "
            f"{slope['days']} | {slope['flow_per_day']} |"
        )

    print()
    print("## Context per row")
    print()
    for rec in records:
        request = rec["request"]
        slope = rec["slope"]
        print(f"### {rec['rid']}")
        print(
            f"- requested={rec['requested']} opening={rec['opening']} "
            f"min_keep={rec['minimum']} net_headroom={rec['net']}"
        )
        print(
            f"- current_calc={rec['calculated']} label={rec['label']} "
            f"delta={to_money(rec['label'] - rec['calculated'])}"
        )
        print(
            f"- shadow max-recent 90d trough={rec['shadow_low']} on "
            f"{rec['shadow_date']} headroom={rec['shadow_head']}"
        )
        print(
            f"- next payroll/credit={rec['next_income']} "
            f"deadline={request.desired_completion_date}"
        )
        print(
            f"- slope to day before payroll: days={slope['days']} "
            f"credits={slope['credits']} debits={slope['debits']} "
            f"net={slope['net']} flow/day={slope['flow_per_day']} "
            f"balance_end={slope['balance_end']} "
            f"balance_delta/day={slope['balance_per_day']}"
        )
        print()

    print("## A. Shorter lookahead windows")
    print()
    print(
        "Candidate = min(requested, max(0, trough_headroom)). "
        "Planning = current min_recent + 0.60 lifestyle scale. "
        "Shadow = max_recent lifestyle amounts, same scale."
    )
    print()

    window_hits: dict[str, int] = {}
    for rec in records:
        request = rec["request"]
        start = request.request_date
        next_income = rec["next_income"]
        windows = [
            ("7d", start + timedelta(days=7)),
            ("14d", start + timedelta(days=14)),
            ("21d", start + timedelta(days=21)),
            ("30d", start + timedelta(days=30)),
            ("60d", start + timedelta(days=60)),
            ("deadline", request.desired_completion_date),
            ("90d", rec["state"].horizon_end),
        ]
        if next_income is not None:
            windows.append(("payday-1", next_income - timedelta(days=1)))
            windows.append(("payday", next_income))
        month_end_day = (
            date(start.year, start.month % 12 + 1, 1) - timedelta(days=1)
            if start.month < 12
            else date(start.year, 12, 31)
        )
        windows.append(("month_end", month_end_day))

        print(f"### {rec['rid']}  label={rec['label']}")
        print("| window | planning_capped | shadow_capped |")
        print("|---|---|---|")
        for name, cutoff in windows:
            p_low, _, p_head = _headroom(rec["state"], cutoff)
            s_low, _, s_head = _headroom(rec["shadow"], cutoff)
            p_cap = _capped(p_head, rec["requested"])
            s_cap = _capped(s_head, rec["requested"])
            window_hits[f"plan:{name}"] = window_hits.get(f"plan:{name}", 0) + int(
                _close(p_cap, rec["label"])
            )
            window_hits[f"shadow:{name}"] = window_hits.get(f"shadow:{name}", 0) + int(
                _close(s_cap, rec["label"])
            )
            print(
                f"| {name} to {cutoff} | {_candidate_table('p', p_cap, rec['label'])} | "
                f"{_candidate_table('s', s_cap, rec['label'])} |"
            )
        print()

    print("Window HIT counts across the six rows:")
    for name, count in sorted(window_hits.items(), key=lambda item: (-item[1], item[0])):
        if count:
            print(f"- {name}: {count}/6")
    if not any(window_hits.values()):
        print("- none")
    print()

    print("## B. Recurring-category inclusion / exclusion")
    print()
    print(
        "Each variant rebuilds the 90-day capped headroom on a copied state. "
        "`unsmooth` scales lifestyle amounts by 1/0.60 so expand's 0.60 "
        "haircut cancels, approximating an unsmoothed variable drain."
    )
    print()

    variants = [
        ("drop lifestyle (dining+shop+ent)", lambda st: _drop_categories(st, LIFESTYLE)),
        ("drop all variable debits", lambda st: _drop_categories(st, VARIABLE_DEBIT_CATEGORIES)),
        ("fixed-like + income only", lambda st: _keep_debit_categories(st, FIXED_LIKE)),
        ("drop subscriptions", lambda st: _drop_categories(st, SUBSCRIPTION_CATEGORIES)),
        ("unsmooth lifestyle (cancel 0.60)", lambda st: _scale_categories(st, LIFESTYLE, unsmooth)),
        (
            "shadow + unsmooth lifestyle",
            lambda st: _scale_categories(_shadow_lifestyle_max_recent(st), LIFESTYLE, unsmooth),
        ),
        (
            "drop lifestyle on shadow",
            lambda st: _drop_categories(_shadow_lifestyle_max_recent(st), LIFESTYLE),
        ),
        (
            "drop variable on shadow",
            lambda st: _drop_categories(
                _shadow_lifestyle_max_recent(st), VARIABLE_DEBIT_CATEGORIES
            ),
        ),
    ]

    variant_hits: dict[str, int] = {}
    for rec in records:
        print(f"### {rec['rid']}  label={rec['label']}  current={rec['calculated']}")
        print("| variant | 90d capped | payday-1 capped |")
        print("|---|---|---|")
        payday = rec["next_income"]
        for name, builder in variants:
            variant_state = builder(rec["state"])
            _, _, head90 = _headroom(variant_state)
            cap90 = _capped(head90, rec["requested"])
            if payday is not None:
                _, _, head_p = _headroom(variant_state, payday - timedelta(days=1))
                cap_p = _capped(head_p, rec["requested"])
            else:
                cap_p = cap90
            variant_hits[f"{name} / 90d"] = variant_hits.get(f"{name} / 90d", 0) + int(
                _close(cap90, rec["label"])
            )
            variant_hits[f"{name} / payday-1"] = variant_hits.get(
                f"{name} / payday-1", 0
            ) + int(_close(cap_p, rec["label"]))
            print(
                f"| {name} | {_candidate_table('90', cap90, rec['label'])} | "
                f"{_candidate_table('p', cap_p, rec['label'])} |"
            )
        print()

    print("Category-variant HIT counts:")
    for name, count in sorted(variant_hits.items(), key=lambda item: (-item[1], item[0])):
        if count:
            print(f"- {name}: {count}/6")
    if not any(variant_hits.values()):
        print("- none")
    print()

    print("## C. Ceiling / haircut closed forms")
    print()
    haircut_hits: dict[str, int] = {}
    for rec in records:
        requested = rec["requested"]
        calculated = rec["calculated"]
        opening = rec["opening"]
        net = rec["net"]
        label = rec["label"]
        forms = {
            "requested": requested,
            "requested * 0.97": to_money(requested * Decimal("0.97")),
            "requested * 0.95": to_money(requested * Decimal("0.95")),
            "requested * 0.90": to_money(requested * Decimal("0.90")),
            "floor(requested/21)": to_money(Decimal(math.floor(requested / Decimal(21)))),
            "min(requested, net)": _capped(net, requested),
            "net * 0.90": to_money(net * Decimal("0.90")),
            "min(requested, net*0.9)": _capped(net * Decimal("0.90"), requested),
            "min(requested, net*0.55)": _capped(net * Decimal("0.55"), requested),
            "min(requested, net*0.50)": _capped(net * Decimal("0.50"), requested),
            "min(requested, net*0.375)": _capped(net * Decimal("0.375"), requested),
            "calculated * 0.99": to_money(calculated * Decimal("0.99")),
            "calculated * 0.97": to_money(calculated * Decimal("0.97")),
            "calculated * 0.95": to_money(calculated * Decimal("0.95")),
            "calculated * 0.90": to_money(calculated * Decimal("0.90")),
            "shadow_head": rec["shadow_head"],
            "min(requested, shadow)": _capped(rec["shadow_head"], requested),
            "shadow * 0.99": to_money(rec["shadow_head"] * Decimal("0.99")),
            "shadow * 0.97": to_money(rec["shadow_head"] * Decimal("0.97")),
            "shadow * 0.95": to_money(rec["shadow_head"] * Decimal("0.95")),
            "shadow * 0.90": to_money(rec["shadow_head"] * Decimal("0.90")),
            "min(req, shadow*0.99)": _capped(rec["shadow_head"] * Decimal("0.99"), requested),
            "min(req, shadow*0.97)": _capped(rec["shadow_head"] * Decimal("0.97"), requested),
            "min(req, shadow*0.95)": _capped(rec["shadow_head"] * Decimal("0.95"), requested),
            "opening * 0.10": to_money(opening * Decimal("0.10")),
            "opening * 0.2857": to_money(opening * Decimal("0.2857")),
            "requested - 17.10": to_money(requested - Decimal("17.10")),
        }
        print(f"### {rec['rid']}  label={label} calc={calculated}")
        hits = []
        near = []
        for name, value in forms.items():
            haircut_hits[name] = haircut_hits.get(name, 0) + int(_close(value, label))
            err = abs(label - value)
            rel = (err / label) if label else err
            if _close(value, label):
                hits.append(f"{name}={value}")
            elif rel <= Decimal("0.02") or err <= Decimal("50"):
                near.append(f"{name}={value} (err {to_money(label - value)})")
        print(f"- hits: {', '.join(hits) if hits else 'none'}")
        print(f"- near (<=2% or <=50): {', '.join(near) if near else 'none'}")
        print(f"- label/calculated={_ratio(label, calculated)}  calculated-label={to_money(calculated - label)}")
        print()

    print("Haircut HIT counts:")
    for name, count in sorted(haircut_hits.items(), key=lambda item: (-item[1], item[0])):
        if count:
            print(f"- {name}: {count}/6")
    if not any(haircut_hits.values()):
        print("- none")
    print()

    print("## Structural hypotheses")
    print()
    # Derive a short hypothesis list from the hit maps collected above.
    best_windows = [item for item in sorted(window_hits.items(), key=lambda x: -x[1]) if item[1] >= 2]
    best_cats = [item for item in sorted(variant_hits.items(), key=lambda x: -x[1]) if item[1] >= 2]
    best_cuts = [item for item in sorted(haircut_hits.items(), key=lambda x: -x[1]) if item[1] >= 2]
    print("1. Shared shorter window: ", end="")
    if best_windows:
        print(
            ", ".join(f"{name} ({count}/6)" for name, count in best_windows[:6])
            + ". A single 14/30/payday window does not hit all six."
        )
    else:
        print(
            "no 14/21/30/payday/deadline trough matches any of the six labels "
            "on either the planning or shadow timeline."
        )
    print("2. Category filter: ", end="")
    if best_cats:
        print(", ".join(f"{name} ({count}/6)" for name, count in best_cats[:6]) + ".")
    else:
        print(
            "dropping lifestyle, dropping all variable debits, or cancelling "
            "the 0.60 smoothing does not reproduce any label as a 90-day or "
            "pre-payday capped trough."
        )
    print("3. Direct haircut: ", end="")
    if best_cuts:
        print(", ".join(f"{name} ({count}/6)" for name, count in best_cuts[:8]) + ".")
    else:
        print(
            "no shared percentage of requested, opening, net headroom, "
            "current calculation, or shadow trough hits more than one row."
        )
    print(
        "4. Sign of the miss: labels sit *below* the current calculation on "
        "request_02/03/04/07/08 (we are high) and *above* it on request_06 "
        "(we are low). A single extra-lifestyle-cycle subtraction can explain "
        "the five high rows only if the subtracted cycle is not the same "
        "fraction of our number (label/calc ratios differ)."
    )
    print(
        "5. request_06 is structurally different: requested-label=17.10, not "
        "the EUR 19 streaming bill and not a 14/30-day trough. The label "
        "amount is the pre-change safe figure while the plan still pays 620.40 "
        "after stopping streaming."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
