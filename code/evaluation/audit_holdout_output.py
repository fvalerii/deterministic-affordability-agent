"""Contract assertions for hold-out *output rows*.

This script never reads hold-out labels. It only checks the generated CSV
against profiles and the eight output columns themselves.

    .venv/bin/python code/evaluation/audit_holdout_output.py --output output.holdout.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from domain import OUTPUT_COLUMNS  # noqa: E402
from tools.data import load_dataset, load_split_manifest  # noqa: E402

DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
EVENT_RE = re.compile(r"event_\d+", re.IGNORECASE)
AMOUNT_RE = re.compile(r"\d+(?:\.\d+)?")
NULLISH = {"", "nan", "none", "null", "n/a"}


def _decimal(token: str) -> Decimal | None:
    text = token.strip().replace(",", "")
    if not text:
        return None
    try:
        return Decimal(text)
    except InvalidOperation:
        return None


def _plan_dates_and_amounts(payment_plan: str) -> tuple[set[str], set[Decimal]]:
    dates: set[str] = set()
    amounts: set[Decimal] = set()
    plan = (payment_plan or "").strip()
    if not plan or plan.lower() == "none":
        return dates, amounts
    for part in plan.split("|"):
        if part.count(":") != 1:
            continue
        when, amount = part.split(":", 1)
        dates.add(when.strip())
        parsed = _decimal(amount)
        if parsed is not None:
            amounts.add(parsed)
    return dates, amounts


def schema_ok(row: dict[str, str]) -> tuple[bool, str]:
    missing = [column for column in OUTPUT_COLUMNS if column not in row]
    if missing:
        return False, f"missing {missing}"
    extra = [column for column in row if column not in OUTPUT_COLUMNS]
    if extra:
        return False, f"extra {extra}"
    if len(row) != 8:
        return False, f"{len(row)} columns"
    for column in OUTPUT_COLUMNS:
        value = row[column]
        if value is None:
            return False, f"{column} is None"
        if column == "earliest_date_for_full_payment":
            continue
        if str(value).strip().lower() in {"nan", "null"}:
            return False, f"{column} is nullish"
        if column != "earliest_date_for_full_payment" and not str(value).strip():
            if column == "decision_explanation":
                return False, "empty explanation"
    explanation = row["decision_explanation"].strip()
    if not explanation or explanation.lower() in NULLISH:
        return False, "empty explanation"
    return True, "ok"


def priority_ok(row: dict[str, str], priorities: tuple[str, ...]) -> tuple[bool, str]:
    text = row["decision_explanation"]
    lowered = text.lower()
    hits = [name for name in priorities if name.lower() in lowered]
    if not priorities:
        return False, "profile has no financial_priorities"
    if not hits:
        return False, "no priority cited"
    return True, ", ".join(hits)


def number_lock_ok(row: dict[str, str]) -> tuple[bool, str]:
    text = row["decision_explanation"]
    allowed_dates, plan_amounts = _plan_dates_and_amounts(row["payment_plan"])
    allowed_amounts = set(plan_amounts)
    safe = _decimal(row["amount_safe_to_pay"])
    if safe is not None:
        allowed_amounts.add(safe)

    dates = DATE_RE.findall(text)
    bad_dates = [when for when in dates if when not in allowed_dates]
    stripped = DATE_RE.sub(" ", text)
    stripped = EVENT_RE.sub(" ", stripped)
    mentioned = AMOUNT_RE.findall(stripped)
    bad_amounts: list[str] = []
    for token in mentioned:
        parsed = _decimal(token)
        if parsed is None:
            bad_amounts.append(token)
            continue
        if parsed not in allowed_amounts:
            bad_amounts.append(token)
    if bad_dates or bad_amounts:
        parts = []
        if bad_dates:
            parts.append("dates " + ", ".join(bad_dates))
        if bad_amounts:
            parts.append("amounts " + ", ".join(bad_amounts))
        return False, "; ".join(parts)
    return True, "ok"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Hold-out output contract audit")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "output.holdout.csv",
        help="Generated hold-out CSV (default: output.holdout.csv)",
    )
    args = parser.parse_args(argv)

    dataset = load_dataset()
    holdout_ids = load_split_manifest(dataset.paths.splits_json).holdout_ids
    with args.output.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    by_id = {row["request_id"]: row for row in rows}
    print("# Hold-out output contract audit (labels not loaded)")
    print()
    print("| request_id | schema | priority | number_lock | notes |")
    print("|---|---|---|---|---|")

    schema_hits = priority_hits = number_hits = 0
    failures = 0
    for request_id in holdout_ids:
        row = by_id.get(request_id)
        if row is None:
            print(f"| {request_id} | Fail | Fail | Fail | missing row |")
            failures += 1
            continue
        request = dataset.sample_requests[request_id]
        priorities = dataset.profile(request.user_id).financial_priorities
        schema, schema_note = schema_ok(row)
        priority, priority_note = priority_ok(row, priorities)
        numbers, number_note = number_lock_ok(row)
        schema_hits += int(schema)
        priority_hits += int(priority)
        number_hits += int(numbers)
        notes = []
        if not schema:
            notes.append(schema_note)
        else:
            notes.append(f"cited {priority_note}" if priority else priority_note)
        if not numbers:
            notes.append(number_note)
        if not (schema and priority and numbers):
            failures += 1
        print(
            f"| {request_id} | {'Pass' if schema else 'Fail'} | "
            f"{'Pass' if priority else 'Fail'} | "
            f"{'Pass' if numbers else 'Fail'} | {'; '.join(notes)} |"
        )

    extra = [rid for rid in by_id if rid not in holdout_ids]
    print()
    print(
        f"score schema {schema_hits}/15  priority {priority_hits}/15  "
        f"number_lock {number_hits}/15"
    )
    print(f"rows in file: {len(rows)}  holdout ids: {len(holdout_ids)}")
    if extra:
        print(f"unexpected request_ids: {', '.join(extra)}")
    return 0 if failures == 0 and not extra and len(rows) == 15 else 1


if __name__ == "__main__":
    raise SystemExit(main())
