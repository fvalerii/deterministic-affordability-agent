"""Calibrate recurrence / income / spending knobs on request_01..request_10 only.

Hold-out labels are never loaded. Run from the repo root:

    .venv/bin/python code/evaluation/calibrate.py
"""

from __future__ import annotations

import itertools
import sys
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from tools.data import load_dataset, load_sample_labels  # noqa: E402
from tools.evidence import build_adjustments  # noqa: E402
from tools.forecast import calculate_safe_amount_today  # noqa: E402
from tools.ledger import RecurrencePolicy, build_financial_state  # noqa: E402
from tools.planning import generate_candidates, rank_valid_candidates  # noqa: E402

ZERO = Decimal("0")


def evaluate(dataset, labels, evidence, policy: RecurrencePolicy) -> dict:
    method_hits = status_hits = amount_hits = 0
    abs_err = Decimal("0")
    rows = []
    for rid, lab in labels:
        request = dataset.sample_requests[rid]
        adjustments, _, _ = evidence[rid]
        state = build_financial_state(dataset, request, adjustments=adjustments, policy=policy)
        safe = calculate_safe_amount_today(state, request).amount_safe_to_pay
        best = rank_valid_candidates(
            request, generate_candidates(dataset, state, request)
        ).best
        method = str(best.method) if best else "not_recommended"
        status = str(best.affordability_status) if best else "not_affordable"
        expected_safe = Decimal(lab.amount_safe_to_pay)
        method_hits += method == lab.recommended_payment_method
        status_hits += status == lab.affordability_status
        amount_hits += safe == expected_safe
        abs_err += abs(safe - expected_safe)
        rows.append((rid, method, lab.recommended_payment_method, status, lab.affordability_status, safe, expected_safe))
    return {
        "method": method_hits,
        "status": status_hits,
        "amount": amount_hits,
        "abs_err": abs_err,
        "score": (method_hits, status_hits, amount_hits, -abs_err),
        "rows": rows,
        "policy": policy,
    }


def main() -> int:
    dataset = load_dataset()
    labels = [(lab.request_id, lab) for lab in load_sample_labels("calibration")]
    evidence = {rid: build_adjustments(dataset, dataset.sample_requests[rid]) for rid, _ in labels}

    baseline = RecurrencePolicy()
    results = [evaluate(dataset, labels, evidence, baseline)]
    print("baseline", asdict(baseline), "m/s/a", results[0]["method"], results[0]["status"], results[0]["amount"])

    # Stage 1: income / spending-structure knobs with a conservative estimator.
    # Stage 2: refine estimators around the structural winner.
    stage1 = list(
        itertools.product(
            (180, 240, 360),  # lookback
            (2, 3),           # min_occurrences
            (True, False),    # project_income
            (True, False),    # drop_final_income
            (True, False),    # project_discretionary
            (True, False),    # project_subscriptions
            (False, True),    # project_irregular_income
            (True, False),    # plan_after_same_day_income
        )
    )
    print(f"stage 1: {len(stage1)} structural configurations")

    seen: set[tuple] = {tuple(asdict(baseline).items())}
    for (
        lookback,
        min_occ,
        project_income,
        drop_final,
        discretionary,
        subscriptions,
        irregular,
        payday,
    ) in stage1:
        policy = RecurrencePolicy(
            lookback_days=lookback,
            min_occurrences=min_occ,
            project_income=project_income,
            drop_final_income=drop_final,
            project_dining=False,
            project_discretionary=discretionary,
            project_subscriptions=subscriptions,
            project_irregular_income=irregular,
            plan_after_same_day_income=payday,
        )
        key = tuple(asdict(policy).items())
        if key in seen:
            continue
        seen.add(key)
        results.append(evaluate(dataset, labels, evidence, policy))

    stage1_best = max(results, key=lambda r: r["score"])
    print(
        "stage 1 best",
        f"m={stage1_best['method']} s={stage1_best['status']} a={stage1_best['amount']}",
        asdict(stage1_best["policy"]),
    )

    base = stage1_best["policy"]
    stage2 = list(
        itertools.product(
            (3, 6),
            ("max_recent", "last", "mean_recent", "median"),
            ("max_recent", "last", "mean_recent", "median"),
            ("min_recent", "last", "mean_recent"),
        )
    )
    print(f"stage 2: {len(stage2)} estimator configurations around the stage-1 winner")
    for recent_window, fixed, variable, credit in stage2:
        policy = RecurrencePolicy(
            lookback_days=base.lookback_days,
            min_occurrences=base.min_occurrences,
            recent_window=recent_window,
            debit_estimator=fixed,
            variable_debit_estimator=variable,
            credit_estimator=credit,
            project_income=base.project_income,
            drop_final_income=base.drop_final_income,
            project_dining=base.project_dining,
            project_discretionary=base.project_discretionary,
            project_subscriptions=base.project_subscriptions,
            project_irregular_income=base.project_irregular_income,
            plan_after_same_day_income=base.plan_after_same_day_income,
        )
        key = tuple(asdict(policy).items())
        if key in seen:
            continue
        seen.add(key)
        results.append(evaluate(dataset, labels, evidence, policy))

    results.sort(key=lambda r: r["score"], reverse=True)
    print("\n=== top 8 ===")
    for result in results[:8]:
        p = result["policy"]
        print(
            f"m={result['method']} s={result['status']} a={result['amount']} "
            f"err={result['abs_err']}  "
            f"lb={p.lookback_days} occ={p.min_occurrences} win={p.recent_window} "
            f"fix={p.debit_estimator} var={p.variable_debit_estimator} "
            f"cred={p.credit_estimator} inc={p.project_income} "
            f"drop_final={p.drop_final_income} disc={p.project_discretionary} "
            f"sub={p.project_subscriptions} payday={p.plan_after_same_day_income}"
        )

    winner = results[0]
    print("\n=== winner detail ===")
    print(asdict(winner["policy"]))
    print(f"method {winner['method']}/10  status {winner['status']}/10  amount {winner['amount']}/10")
    for rid, gm, em, gs, es, ga, ea in winner["rows"]:
        flags = "".join(
            [
                "M" if gm == em else "m",
                "S" if gs == es else "s",
                "A" if ga == ea else "a",
            ]
        )
        print(f"  {rid} [{flags}] method {gm} vs {em}  status {gs} vs {es}  safe {ga} vs {ea}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
