"""Temporary Phase-4 amount-gap sweep for request_01..request_10.

Forces lifestyle spending on (dining + discretionary) and searches a milder
variable-debit estimator so amount_safe_to_pay can come down without losing
the 10/10 method score. Does not change ledger or forecast logic.

Hold-out labels are never loaded. Run from the repo root:

    .venv/bin/python code/evaluation/amount_gap.py
"""

from __future__ import annotations

import itertools
import sys
from dataclasses import asdict, replace
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from evaluation.calibrate import evaluate  # noqa: E402
from tools.data import load_dataset, load_sample_labels  # noqa: E402
from tools.evidence import build_adjustments  # noqa: E402
from tools.ledger import DEFAULT_POLICY, RecurrencePolicy  # noqa: E402

# recent_window is last-N occurrences. 3 ~ a quarter of weekly series;
# 6 ~ half a year. The user asked for 3 months vs 6 months on this knob.
RECENT_WINDOWS = (3, 6)
VARIABLE_ESTIMATORS = (
    "mean_recent",
    "max_recent",
    "median",
    "last",
    "min_recent",
)


def _print_delta_table(rows: list[tuple]) -> None:
    print("| request_id | expected_amount | calculated_amount | delta |")
    print("|---|---:|---:|---:|")
    for rid, _gm, _em, _gs, _es, calculated, expected in rows:
        delta = expected - calculated
        print(f"| {rid} | {expected} | {calculated} | {delta} |")


def main() -> int:
    dataset = load_dataset()
    labels = [(lab.request_id, lab) for lab in load_sample_labels("calibration")]
    evidence = {
        rid: build_adjustments(dataset, dataset.sample_requests[rid])
        for rid, _ in labels
    }

    locked = DEFAULT_POLICY
    print("locked baseline (dining/discretionary off)")
    print(asdict(locked))
    baseline = evaluate(dataset, labels, evidence, locked)
    print(
        f"score method {baseline['method']}/10  "
        f"status {baseline['status']}/10  "
        f"amount {baseline['amount']}/10  "
        f"abs_err {baseline['abs_err']}"
    )
    _print_delta_table(baseline["rows"])

    grid = list(itertools.product(RECENT_WINDOWS, VARIABLE_ESTIMATORS))
    print()
    print(
        f"sweep: project_dining=True project_discretionary=True "
        f"x {len(grid)} (recent_window, variable_debit_estimator)"
    )
    results: list[dict] = []
    for recent_window, estimator in grid:
        policy = replace(
            locked,
            project_dining=True,
            project_discretionary=True,
            recent_window=recent_window,
            variable_debit_estimator=estimator,
        )
        result = evaluate(dataset, labels, evidence, policy)
        results.append(result)
        print(
            f"  win={recent_window} var={estimator:11}  "
            f"m={result['method']} s={result['status']} "
            f"a={result['amount']} err={result['abs_err']}"
        )

    keep_method = [r for r in results if r["method"] == 10]
    pool = keep_method if keep_method else results
    pool.sort(key=lambda r: r["score"], reverse=True)
    winner = pool[0]

    print()
    if keep_method:
        print(f"configs that keep method 10/10: {len(keep_method)}")
    else:
        print("NO configuration in this sweep kept method 10/10")
    print("best combined score under the sweep constraint")
    print(asdict(winner["policy"]))
    print(
        f"score method {winner['method']}/10  "
        f"status {winner['status']}/10  "
        f"amount {winner['amount']}/10  "
        f"abs_err {winner['abs_err']}"
    )
    print()
    print("updated delta table")
    _print_delta_table(winner["rows"])
    print()
    print("per-row method/status")
    for rid, gm, em, gs, es, ga, ea in winner["rows"]:
        flags = "".join(
            [
                "M" if gm == em else "m",
                "S" if gs == es else "s",
                "A" if ga == ea else "a",
            ]
        )
        print(
            f"  {rid} [{flags}] method {gm} vs {em}  "
            f"status {gs} vs {es}  safe {ga} vs {ea}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
