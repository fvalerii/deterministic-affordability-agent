"""One-time hold-out scorer. This is the only authorized reader of hold-out labels.

    .venv/bin/python code/evaluation/evaluate_holdout.py --output output.holdout.csv

Do not retune tools, prompts, or policies after inspecting these scores.
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

from domain import OUTPUT_COLUMNS  # noqa: E402
from tools.data import load_sample_labels  # noqa: E402

ZERO = Decimal("0.00")


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score hold-out output against quarantined labels")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "output.holdout.csv",
        help="Generated hold-out CSV (default: output.holdout.csv)",
    )
    args = parser.parse_args(argv)
    if not args.output.is_file():
        raise FileNotFoundError(f"generated output not found: {args.output}")

    labels = load_sample_labels("holdout")
    with args.output.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if list(reader.fieldnames or []) != list(OUTPUT_COLUMNS):
            print(f"unexpected columns: {reader.fieldnames}")
        rows = {row["request_id"]: row for row in reader}

    print(f"# Hold-out accuracy vs ground truth ({args.output.name})")
    print()
    print(
        "| request_id | method | status | earliest | changes | plan | amount |"
    )
    print("|---|---|---|---|---|---|---|")

    hits = {
        "method": 0,
        "status": 0,
        "earliest": 0,
        "changes": 0,
        "plan": 0,
        "amount": 0,
    }
    abs_err = ZERO
    categorical_fails: list[tuple[str, str, str, str]] = []
    amount_fails: list[tuple[str, str, str]] = []
    missing: list[str] = []

    for lab in labels:
        row = rows.get(lab.request_id)
        if row is None:
            missing.append(lab.request_id)
            print(
                f"| {lab.request_id} | missing | missing | missing | missing | missing | missing |"
            )
            continue

        got_amount = Decimal(row["amount_safe_to_pay"])
        exp_amount = Decimal(lab.amount_safe_to_pay)
        pairs = [
            ("method", row["recommended_payment_method"], lab.recommended_payment_method),
            ("status", row["affordability_status"], lab.affordability_status),
            (
                "earliest",
                _norm_date(row["earliest_date_for_full_payment"]),
                _norm_date(lab.earliest_date_for_full_payment),
            ),
            (
                "changes",
                _norm_text(row["spending_changes_needed"]),
                _norm_text(lab.spending_changes_needed),
            ),
            ("plan", _norm_text(row["payment_plan"]), _norm_text(lab.payment_plan)),
        ]
        flags = {name: _flag(ours == label) for name, ours, label in pairs}
        flags["amount"] = _flag(got_amount == exp_amount)

        print(
            f"| {lab.request_id} | {flags['method']} | {flags['status']} | "
            f"{flags['earliest']} | {flags['changes']} | {flags['plan']} | "
            f"{flags['amount']} |"
        )
        for key in hits:
            hits[key] += flags[key] == "Pass"
        abs_err += abs(got_amount - exp_amount)

        for name, ours, label in pairs:
            if ours != label:
                categorical_fails.append((lab.request_id, name, ours or "(empty)", label or "(empty)"))
        if got_amount != exp_amount:
            amount_fails.append(
                (lab.request_id, row["amount_safe_to_pay"], lab.amount_safe_to_pay)
            )

    n = len(labels)
    print()
    print("## Final hold-out scores")
    print()
    print(f"- method: {hits['method']}/{n}")
    print(f"- status: {hits['status']}/{n}")
    print(f"- earliest_date_for_full_payment: {hits['earliest']}/{n}")
    print(f"- spending_changes_needed: {hits['changes']}/{n}")
    print(f"- payment_plan: {hits['plan']}/{n}")
    print(f"- amount_safe_to_pay: {hits['amount']}/{n}")
    print(f"- absolute error (amount_safe_to_pay): {abs_err}")
    if missing:
        print(f"- missing rows: {', '.join(missing)}")

    categorical = (
        hits["method"],
        hits["status"],
        hits["earliest"],
        hits["changes"],
        hits["plan"],
    )
    if any(score < n for score in categorical) or missing:
        print()
        print("## Categorical failures (prediction vs ground truth)")
        print()
        print("| request_id | field | prediction | ground_truth |")
        print("|---|---|---|---|")
        for request_id, field, ours, label in categorical_fails:
            print(f"| {request_id} | {field} | {ours} | {label} |")
        for request_id in missing:
            print(f"| {request_id} | (row) | missing | (label present) |")

    if amount_fails:
        print()
        print("## amount_safe_to_pay mismatches")
        print()
        print("| request_id | prediction | ground_truth |")
        print("|---|---|---|")
        for request_id, ours, label in amount_fails:
            print(f"| {request_id} | {ours} | {label} |")

    print()
    print(
        "Hold-out labels were read only by this file. "
        "Do not retune the agent from these scores."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
