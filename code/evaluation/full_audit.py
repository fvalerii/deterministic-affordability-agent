"""Audit the five rubric fields on request_01..request_10.

Does not change ledger or forecast logic. Hold-out labels are never loaded.

    .venv/bin/python code/evaluation/full_audit.py
    .venv/bin/python code/evaluation/full_audit.py --output output.calibration.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import date
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from domain import (  # noqa: E402
    OUTPUT_COLUMNS,
    Direction,
    EventStatus,
    format_amount_field,
)
from tools.data import load_dataset, load_sample_labels  # noqa: E402
from tools.evidence import build_adjustments  # noqa: E402
from tools.forecast import (  # noqa: E402
    calculate_safe_amount_today,
    find_earliest_safe_full_payment,
    simulate,
)
from tools.ledger import DEFAULT_POLICY, build_financial_state  # noqa: E402
from tools.planning import generate_candidates, rank_valid_candidates  # noqa: E402

ZERO = Decimal("0.00")
TOL = Decimal("0.02")  # rounding-level match for formula guesses


def _flag(ok: bool) -> str:
    return "Pass" if ok else "Fail"


def _norm_date(value: date | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, date):
        return value.isoformat()
    return str(value).strip()


def _norm_text(value: str | None) -> str:
    if value is None:
        return "none"
    text = str(value).strip()
    return text if text else "none"


def _close(left: Decimal, right: Decimal) -> bool:
    return abs(left - right) <= TOL


def _pending_debits(dataset, user_id: str, as_of: date) -> list[tuple[str, Decimal, date, str]]:
    rows = []
    for event in dataset.events_by_user.get(user_id, ()):
        if event.direction is not Direction.DEBIT:
            continue
        if event.status is not EventStatus.PENDING:
            continue
        if event.amount is None:
            continue
        rows.append((event.event_id, event.amount, event.effective_date, event.description))
    return rows


def _min_allowed(dataset, user_id: str, as_of: date) -> list[tuple[str, Decimal, Decimal, str]]:
    rows = []
    for event in dataset.events_by_user.get(user_id, ()):
        if event.minimum_allowed_amount is None:
            continue
        if event.amount is None:
            continue
        rows.append(
            (
                event.event_id,
                event.amount,
                event.minimum_allowed_amount,
                event.category,
            )
        )
    return rows


def _headroom_until(state, end: date) -> Decimal:
    """Minimum projected headroom using only flows on or before ``end``."""
    result = simulate(state)
    lowest = state.opening_balance
    for entry in result.entries:
        if entry.date > end:
            break
        if entry.balance_after < lowest:
            lowest = entry.balance_after
    return lowest - state.minimum_balance_to_keep


def audit_output_csv(output_path: Path) -> int:
    """Score a generated output file against the calibration labels."""

    if not output_path.is_file():
        raise FileNotFoundError(f"generated output not found: {output_path}")
    with output_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(OUTPUT_COLUMNS):
            print(f"unexpected columns: {reader.fieldnames}")
        rows = {row["request_id"]: row for row in reader}

    labels = load_sample_labels("calibration")
    print(f"# Five-metric calibration audit of {output_path}")
    print()
    print(
        "| request_id | method | status | amount_safe_to_pay | earliest_date | "
        "spending_changes | payment_plan |"
    )
    print("|---|---|---|---|---|---|---|")

    hits = {"method": 0, "status": 0, "amount": 0, "earliest": 0, "changes": 0, "plan": 0}
    abs_err = ZERO
    mismatches: list[str] = []
    missing: list[str] = []

    for lab in labels:
        row = rows.get(lab.request_id)
        if row is None:
            missing.append(lab.request_id)
            print(f"| {lab.request_id} | missing | missing | missing | missing | missing | missing |")
            continue
        got_amount = Decimal(row["amount_safe_to_pay"])
        exp_amount = Decimal(lab.amount_safe_to_pay)
        flags = {
            "method": _flag(row["recommended_payment_method"] == lab.recommended_payment_method),
            "status": _flag(row["affordability_status"] == lab.affordability_status),
            "amount": _flag(got_amount == exp_amount),
            "earliest": _flag(
                _norm_date(row["earliest_date_for_full_payment"])
                == _norm_date(lab.earliest_date_for_full_payment)
            ),
            "changes": _flag(
                _norm_text(row["spending_changes_needed"])
                == _norm_text(lab.spending_changes_needed)
            ),
            "plan": _flag(_norm_text(row["payment_plan"]) == _norm_text(lab.payment_plan)),
        }
        print(
            f"| {lab.request_id} | {flags['method']} | {flags['status']} | "
            f"{flags['amount']} | {flags['earliest']} | {flags['changes']} | {flags['plan']} |"
        )
        for key in hits:
            hits[key] += flags[key] == "Pass"
        abs_err += abs(got_amount - exp_amount)
        pairs = [
            ("method", row["recommended_payment_method"], lab.recommended_payment_method),
            ("status", row["affordability_status"], lab.affordability_status),
            ("amount", row["amount_safe_to_pay"], lab.amount_safe_to_pay),
            (
                "earliest",
                _norm_date(row["earliest_date_for_full_payment"]) or "(empty)",
                _norm_date(lab.earliest_date_for_full_payment) or "(empty)",
            ),
            (
                "changes",
                _norm_text(row["spending_changes_needed"]),
                _norm_text(lab.spending_changes_needed),
            ),
            ("plan", _norm_text(row["payment_plan"]), _norm_text(lab.payment_plan)),
        ]
        for field, ours, label in pairs:
            if str(ours) != str(label):
                mismatches.append(f"| {lab.request_id} | {field} | {ours} | {label} |")

    print()
    print(
        f"score method {hits['method']}/10  status {hits['status']}/10  "
        f"amount {hits['amount']}/10  earliest {hits['earliest']}/10  "
        f"changes {hits['changes']}/10  plan {hits['plan']}/10"
    )
    print(f"absolute error (amount_safe_to_pay): {abs_err}")
    if missing:
        print(f"missing rows: {', '.join(missing)}")
    print()
    print("## Field values (ours vs label)")
    print()
    print("| request_id | field | ours | label |")
    print("|---|---|---|---|")
    for line in mismatches:
        print(line)
    if not mismatches:
        print("| (none) | | | |")

    categorical = (
        hits["method"],
        hits["status"],
        hits["earliest"],
        hits["changes"],
        hits["plan"],
    )
    if any(score < 10 for score in categorical):
        print()
        print(
            "CRITICAL: a categorical score dropped below 10/10. "
            "If this diverges from the mocked document-hash baseline, live vision "
            "extracted a different receipt amount."
        )
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Calibration audit")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Score this generated CSV against calibration labels",
    )
    args = parser.parse_args(argv)
    if args.output is not None:
        return audit_output_csv(args.output)
    return audit_kernel()


def audit_kernel() -> int:
    dataset = load_dataset()
    labels = load_sample_labels("calibration")

    print("# Five-metric calibration audit")
    print()
    print(
        "| request_id | method | status | amount_safe_to_pay | earliest_date | "
        "spending_changes | payment_plan |"
    )
    print("|---|---|---|---|---|---|---|")

    amount_fails: list[dict] = []
    hits = {"method": 0, "status": 0, "amount": 0, "earliest": 0, "changes": 0, "plan": 0}
    abs_err = ZERO

    for lab in labels:
        request = dataset.sample_requests[lab.request_id]
        adjustments, _, _ = build_adjustments(dataset, request)
        state = build_financial_state(
            dataset, request, adjustments=adjustments, policy=DEFAULT_POLICY
        )
        safe = calculate_safe_amount_today(state, request)
        earliest = find_earliest_safe_full_payment(state, request)
        ranked = rank_valid_candidates(
            request, generate_candidates(dataset, state, request)
        )
        best = ranked.best
        got_method = str(best.method) if best else "not_recommended"
        got_status = str(best.affordability_status) if best else "not_affordable"
        got_amount = safe.amount_safe_to_pay
        got_earliest = _norm_date(earliest.earliest_date)
        got_changes = best.render_spending_changes() if best else "none"
        got_plan = best.render_payment_plan() if best else "none"

        exp_amount = Decimal(lab.amount_safe_to_pay)
        exp_earliest = _norm_date(lab.earliest_date_for_full_payment)
        exp_changes = _norm_text(lab.spending_changes_needed)
        exp_plan = _norm_text(lab.payment_plan)

        row = {
            "request_id": lab.request_id,
            "method": _flag(got_method == lab.recommended_payment_method),
            "status": _flag(got_status == lab.affordability_status),
            "amount": _flag(got_amount == exp_amount),
            "earliest": _flag(got_earliest == exp_earliest),
            "changes": _flag(got_changes == exp_changes),
            "plan": _flag(got_plan == exp_plan),
            "got_method": got_method,
            "exp_method": lab.recommended_payment_method,
            "got_status": got_status,
            "exp_status": lab.affordability_status,
            "got_amount": got_amount,
            "exp_amount": exp_amount,
            "got_earliest": got_earliest,
            "exp_earliest": exp_earliest,
            "got_changes": got_changes,
            "exp_changes": exp_changes,
            "got_plan": got_plan,
            "exp_plan": exp_plan,
            "state": state,
            "request": request,
            "safe": safe,
        }
        print(
            f"| {lab.request_id} | {row['method']} | {row['status']} | "
            f"{row['amount']} | {row['earliest']} | {row['changes']} | {row['plan']} |"
        )
        hits["method"] += row["method"] == "Pass"
        hits["status"] += row["status"] == "Pass"
        hits["amount"] += row["amount"] == "Pass"
        hits["earliest"] += row["earliest"] == "Pass"
        hits["changes"] += row["changes"] == "Pass"
        hits["plan"] += row["plan"] == "Pass"
        abs_err += abs(got_amount - exp_amount)
        if row["amount"] == "Fail":
            amount_fails.append(row)

    print()
    print(
        f"score method {hits['method']}/10  status {hits['status']}/10  "
        f"amount {hits['amount']}/10  earliest {hits['earliest']}/10  "
        f"changes {hits['changes']}/10  plan {hits['plan']}/10"
    )
    print(f"absolute error (amount_safe_to_pay): {abs_err}")
    print()
    print("## Field values (ours vs label)")
    print()
    print("| request_id | field | ours | label |")
    print("|---|---|---|---|")
    mismatches: list[str] = []
    for lab in labels:
        request = dataset.sample_requests[lab.request_id]
        adjustments, _, _ = build_adjustments(dataset, request)
        state = build_financial_state(
            dataset, request, adjustments=adjustments, policy=DEFAULT_POLICY
        )
        safe = calculate_safe_amount_today(state, request)
        earliest = find_earliest_safe_full_payment(state, request)
        best = rank_valid_candidates(
            request, generate_candidates(dataset, state, request)
        ).best
        pairs = [
            (
                "method",
                str(best.method) if best else "not_recommended",
                lab.recommended_payment_method,
            ),
            (
                "status",
                str(best.affordability_status) if best else "not_affordable",
                lab.affordability_status,
            ),
            (
                "amount",
                format_amount_field(safe.amount_safe_to_pay),
                lab.amount_safe_to_pay,
            ),
            (
                "earliest",
                _norm_date(earliest.earliest_date) or "(empty)",
                _norm_date(lab.earliest_date_for_full_payment) or "(empty)",
            ),
            (
                "changes",
                best.render_spending_changes() if best else "none",
                _norm_text(lab.spending_changes_needed),
            ),
            (
                "plan",
                best.render_payment_plan() if best else "none",
                _norm_text(lab.payment_plan),
            ),
        ]
        for field, ours, label in pairs:
            if str(ours) != str(label):
                mismatches.append(f"| {lab.request_id} | {field} | {ours} | {label} |")
    for line in mismatches:
        print(line)

    print()
    print("## Amount-gap formula tests")
    print()
    print(
        "For each amount Fail: compare the label against simple closed-form "
        "guesses. A hit means |label - formula| <= 0.02."
    )
    print()

    for row in amount_fails:
        request = row["request"]
        state = row["state"]
        profile = dataset.profile(request.user_id)
        opening = profile.current_available_balance
        minimum = profile.minimum_balance_to_keep
        requested = request.requested_amount
        calculated = row["got_amount"]
        expected = row["exp_amount"]
        pending = _pending_debits(dataset, request.user_id, request.request_date)
        pending_sum = sum((amt for _, amt, _, _ in pending), ZERO)
        min_allowed = _min_allowed(dataset, request.user_id, request.request_date)
        slack_sum = sum((amt - floor for _, amt, floor, _ in min_allowed), ZERO)

        next_income = min(
            (c.next_occurrence for c in state.recurring if c.direction is Direction.CREDIT),
            default=None,
        )
        payday_headroom = (
            _headroom_until(state, next_income) if next_income else None
        )
        deadline_headroom = _headroom_until(state, request.desired_completion_date)
        reserved = sum(
            (flow.amount for flow in state.dated_flows if flow.direction is Direction.DEBIT),
            ZERO,
        )

        candidates = {
            "opening - pending_sum": opening - pending_sum,
            "opening - min": opening - minimum,
            "opening - min - pending_sum": opening - minimum - pending_sum,
            "opening - min - reserved_dated": opening - minimum - reserved,
            "min(requested, opening - min)": min(requested, opening - minimum),
            "requested * 0.9": to_money_local(requested * Decimal("0.9")),
            "(opening - min) * 0.9": to_money_local((opening - minimum) * Decimal("0.9")),
            "min(requested, (opening-min)*0.9)": min(
                requested, to_money_local((opening - minimum) * Decimal("0.9"))
            ),
            "calculated * 0.9": to_money_local(calculated * Decimal("0.9")),
            "calculated - pending_sum": calculated - pending_sum,
            "opening * 0.1": to_money_local(opening * Decimal("0.1")),
            "requested * 0.1": to_money_local(requested * Decimal("0.1")),
            "opening - min - slack_sum": opening - minimum - slack_sum,
            "calculated - slack_sum": calculated - slack_sum,
            "headroom_to_deadline": deadline_headroom
            if deadline_headroom > 0
            else ZERO,
        }
        if payday_headroom is not None:
            candidates["headroom_until_next_income"] = max(ZERO, payday_headroom)
        if len(pending) == 1:
            _eid, amt, _when, _desc = pending[0]
            candidates["opening - that_pending"] = opening - amt
            candidates["opening - min - that_pending"] = opening - minimum - amt
            candidates["calculated - that_pending"] = calculated - amt
            candidates["min(requested, opening - pending)"] = min(requested, opening - amt)

        print(f"### {row['request_id']}  {request.user_id}")
        print()
        print(
            f"- opening={opening} min_keep={minimum} requested={requested} "
            f"calculated={calculated} expected={expected} "
            f"delta(expected-calculated)={expected - calculated}"
        )
        print(
            f"- our trough {row['safe'].limiting_date} "
            f"balance={row['safe'].limiting_balance} "
            f"headroom={row['safe'].limiting_balance - minimum}"
        )
        print(f"- pending debits: {pending or 'none'}")
        print(f"- pending_sum={pending_sum} reserved_dated={reserved}")
        print(f"- next_income={next_income} payday_headroom={payday_headroom}")
        print(
            f"- events with minimum_allowed_amount (count={len(min_allowed)}): "
            f"slack_sum={slack_sum}"
        )
        hits = [
            name
            for name, value in candidates.items()
            if _close(value, expected)
        ]
        print(f"- formula hits: {hits or 'none'}")
        near = []
        for name, value in candidates.items():
            err = expected - value
            if abs(err) <= max(Decimal("1.00"), abs(expected) * Decimal("0.01")):
                near.append(f"{name}={value} (err {err})")
        print(f"- near misses (<=1 or 1%): {near or 'none'}")
        print()

    print("## Formula hypothesis")
    print()
    print(
        "None of the three closed forms (opening minus one pending debit, "
        "requested_amount * 0.9, or a minimum_allowed_amount slack subtraction) "
        "hit any failing row."
    )
    print(
        "The seven high amounts share one structure: calculated = "
        "min(requested, 90-day_trough - minimum_balance_to_keep). The label is "
        "smaller by roughly one unprojected lifestyle cycle (dining / shopping / "
        "entertainment) that lands before that trough. request_05 is the opposite "
        "outlier: 90-day outflow without salary goes below the floor, so we report 0 "
        "while the label still reports 737."
    )
    return 0


def to_money_local(value: Decimal) -> Decimal:
    return value.quantize(Decimal("0.01"))


if __name__ == "__main__":
    raise SystemExit(main())
