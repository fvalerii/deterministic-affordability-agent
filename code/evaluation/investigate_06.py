"""Deep dive on request_06 spending-change choice (event_476 vs event_556).

Does not change ledger, forecast, or planning logic. Hold-out labels are
never loaded.

    .venv/bin/python code/evaluation/investigate_06.py
"""

from __future__ import annotations

import inspect
import itertools
import sys
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from domain import Direction, Payment, SpendingChange, SpendingChangeAction, to_money  # noqa: E402
from tools.data import load_dataset, load_sample_labels  # noqa: E402
from tools.evidence import build_adjustments  # noqa: E402
from tools.forecast import expand_recurring_flows, simulate  # noqa: E402
from tools.ledger import (  # noqa: E402
    DEFAULT_POLICY,
    DINING_CATEGORIES,
    DISCRETIONARY_CATEGORIES,
    estimate_amount,
    build_financial_state,
)
from tools.planning import (  # noqa: E402
    LIFESTYLE_CATEGORIES,
    MAX_CHANGE_CANDIDATES,
    _change_for,
    _find_change_set,
    _monthly_saving,
    generate_candidates,
    list_permitted_spending_changes,
    rank_valid_candidates,
)

ZERO = Decimal("0.00")
TARGET_IDS = ("event_476", "event_556")
LIFESTYLE = DINING_CATEGORIES | DISCRETIONARY_CATEGORIES


def _hr() -> None:
    print()
    print("=" * 78)
    print()


def _commitment_for(state, event_id):
    for commitment in state.recurring:
        if (
            commitment.representative_event_id == event_id
            or event_id in commitment.source_event_ids
        ):
            return commitment
    return None


def _series_amounts(dataset, commitment):
    amounts = []
    rows = []
    for event_id in commitment.source_event_ids:
        event = dataset.events.get(event_id)
        if event is None or event.amount is None:
            continue
        amounts.append(event.amount)
        rows.append((event.event_id, event.effective_date, event.amount, event.description))
    return amounts, rows


def _print_event_block(dataset, state, event_id: str) -> None:
    event = dataset.events[event_id]
    commitment = _commitment_for(state, event_id)
    print(f"## {event_id}")
    print()
    print(f"category:           {event.category}")
    print(f"description:        {event.description}")
    print(f"event_type:         {event.event_type}")
    print(f"direction:          {event.direction}")
    print(f"row amount:         {event.amount} {event.currency}")
    print(f"event_date:         {event.event_date}")
    print(f"settlement_date:    {event.settlement_date}")
    print(f"status:             {event.status}")
    print(f"flexibility (row):  {event.flexibility}")
    print(f"minimum_allowed:    {event.minimum_allowed_amount}")
    print()
    if commitment is None:
        print("inferred commitment: NONE — this event is not on the planning timeline")
        return
    amounts, rows = _series_amounts(dataset, commitment)
    print("inferred commitment (this is what planning actually cuts):")
    print(f"  label:                    {commitment.label}")
    print(f"  category:                 {commitment.category}")
    print(f"  estimated amount / occ:   {commitment.amount}")
    print(f"  cadence frequency_days:   {commitment.frequency_days}")
    print(f"  next_occurrence:          {commitment.next_occurrence}")
    print(f"  flexibility (inferred):   {commitment.flexibility}")
    print(f"  representative_event_id:  {commitment.representative_event_id}")
    print(f"  occurrences_observed:     {commitment.occurrences_observed}")
    print(f"  source_event_ids:         {' | '.join(commitment.source_event_ids)}")
    if amounts:
        print(f"  series min / max / last:  {min(amounts)} / {max(amounts)} / {amounts[-1]}")
        print(
            "  estimators on series:     "
            f"min_recent={estimate_amount(amounts, estimator='min_recent', recent_window=DEFAULT_POLICY.recent_window)} "
            f"max_recent={estimate_amount(amounts, estimator='max_recent', recent_window=DEFAULT_POLICY.recent_window)} "
            f"mean_recent={estimate_amount(amounts, estimator='mean_recent', recent_window=DEFAULT_POLICY.recent_window)}"
        )
    print()
    print("source series (oldest → newest):")
    for eid, when, amount, desc in rows:
        marker = "  <-- this row" if eid == event_id else ""
        print(f"  {eid}  {when}  {amount:>8}  {desc}{marker}")


def _print_sort_logic() -> None:
    print("## Sorting / prioritization in planning.py")
    print()
    print("list_permitted_spending_changes builds one candidate cut per recurring")
    print("debit, then sorts. Source:")
    print()
    print(inspect.getsource(list_permitted_spending_changes))
    print("_change_for decides STOP vs REDUCE_TO:")
    print()
    print(inspect.getsource(_change_for))
    print("_find_change_set walks combinations in that sorted order, smallest set first:")
    print()
    print(inspect.getsource(_find_change_set))
    print("Sort key is (user_asked, -monthly_saving, event_id):")
    print("  1. user_asked=0 first: categories the profile listed in")
    print("     expense_categories_user_is_willing_to_stop / _reduce.")
    print("     user_asked=1 after: dining/shopping/entertainment inferred as")
    print("     lifestyle even when the source row is 'fixed' and the user did")
    print("     not list the category.")
    print("  2. Then richest monthly saving (amount * 30 / frequency_days).")
    print("  3. Then representative_event_id as a deterministic tie-break.")
    print()
    print(f"LIFESTYLE_CATEGORIES = {sorted(LIFESTYLE_CATEGORIES)}")
    print(f"MAX_CHANGE_CANDIDATES = {MAX_CHANGE_CANDIDATES} (only the first N enter the search)")
    print()
    print("The search is NOT 'smallest cut first'. Size-1 is tried before size-2,")
    print("but inside size-1 the first *successful* combination wins. User-listed")
    print("streaming is therefore tried before inferred dining; if streaming is")
    print("not enough, the next size-1 cut is dining.")


def _sim_line(result, state) -> str:
    floor = state.minimum_balance_to_keep
    first = result.violations[0] if result.violations else None
    if first is None:
        breach = "none"
    else:
        breach = (
            f"{first.date} {first.reason} shortfall={first.shortfall} "
            f"balance={first.balance}"
        )
    return (
        f"safe={result.safe}  min_balance={result.minimum_projected_balance} "
        f"on {result.minimum_balance_date}  "
        f"headroom={to_money(result.minimum_projected_balance - floor)}  "
        f"first_violation={breach}"
    )


def _projected_debits(state, event_id: str):
    commitment = _commitment_for(state, event_id)
    if commitment is None:
        return (), ZERO
    flows = [
        flow
        for flow in expand_recurring_flows(state)
        if flow.event_id == commitment.representative_event_id
        and flow.direction is Direction.DEBIT
    ]
    total = to_money(sum((flow.amount for flow in flows), ZERO))
    return flows, total


def main() -> int:
    dataset = load_dataset()
    labels = {lab.request_id: lab for lab in load_sample_labels("calibration")}
    request = dataset.sample_requests["request_06"]
    label = labels["request_06"]
    profile = dataset.profile(request.user_id)
    adjustments, _, _ = build_adjustments(dataset, request)
    state = build_financial_state(
        dataset, request, adjustments=adjustments, policy=DEFAULT_POLICY
    )
    payment = (Payment(date=request.request_date, amount=request.requested_amount),)

    print("# request_06 spending-change investigation")
    print()
    print(f"request_date:              {request.request_date}")
    print(f"desired_completion_date:   {request.desired_completion_date}")
    print(f"requested_amount:          {request.requested_amount}")
    print(f"label amount_safe_to_pay:  {label.amount_safe_to_pay}")
    print(f"label spending_changes:    {label.spending_changes_needed}")
    print(f"label payment_plan:        {label.payment_plan}")
    print(
        f"requested - label amount:  "
        f"{to_money(request.requested_amount - Decimal(label.amount_safe_to_pay))} "
        "(the 17.10 gap)"
    )
    print(f"opening_balance:           {state.opening_balance}")
    print(f"minimum_balance_to_keep:   {state.minimum_balance_to_keep}")
    print(f"protected categories:      {profile.expense_categories_to_protect}")
    print(f"willing to reduce:         {profile.expense_categories_user_is_willing_to_reduce}")
    print(f"willing to stop:           {profile.expense_categories_user_is_willing_to_stop}")
    print(f"payment methods:           {profile.payment_methods_user_will_consider}")
    print()
    print("payroll evidence (message_04): temporary monthly pay EUR 1037.52")

    _hr()
    _print_event_block(dataset, state, "event_476")
    _hr()
    _print_event_block(dataset, state, "event_556")
    _hr()
    _print_sort_logic()

    _hr()
    print("## Permitted cuts in the exact order planning will try them")
    print()
    scored = []
    for commitment in state.recurring:
        change = _change_for(commitment, profile)
        if change is None:
            continue
        user_listed = profile.may_stop_category(
            commitment.category
        ) or profile.may_reduce_category(commitment.category)
        user_asked = int(not user_listed)
        saving = _monthly_saving(commitment, change)
        scored.append((user_asked, saving, commitment, change, user_listed))
    scored.sort(key=lambda item: (item[0], -item[1], item[2].representative_event_id))

    print(
        "| try# | change | category | user_listed | user_asked | "
        "est_amt | freq_days | monthly_saving | representative |"
    )
    print("|---:|---|---|---|---:|---:|---:|---:|---|")
    for index, (user_asked, saving, commitment, change, user_listed) in enumerate(
        scored, start=1
    ):
        print(
            f"| {index} | {change.render()} | {commitment.category} | "
            f"{user_listed} | {user_asked} | {commitment.amount} | "
            f"{commitment.frequency_days} | {saving} | "
            f"{commitment.representative_event_id} |"
        )
    print()
    permitted = list_permitted_spending_changes(state, profile)
    print(f"list_permitted_spending_changes() → {[c.render() for c in permitted]}")
    print(
        f"search window (first {MAX_CHANGE_CANDIDATES}): "
        f"{[c.render() for c in permitted[:MAX_CHANGE_CANDIDATES]]}"
    )

    _hr()
    print("## 90-day projected outflow that each STOP would remove")
    print()
    for event_id in TARGET_IDS:
        flows, total = _projected_debits(state, event_id)
        print(f"{event_id}: {len(flows)} projected debit(s), total saved if STOP = {total}")
        for flow in flows:
            print(f"  {flow.date}  {flow.amount}  {flow.label}  {flow.source}")
        if not flows:
            print("  (nothing projected — a STOP would save 0 on the timeline)")
        print()

    _hr()
    print("## Planning-loop trace: pay EUR 620.40 on 2026-01-03")
    print()
    baseline = simulate(state, payments=payment, requested_amount=request.requested_amount)
    print(f"0. no spending change:  {_sim_line(baseline, state)}")
    print()

    window = permitted[:MAX_CHANGE_CANDIDATES]
    chosen = None
    step = 1
    for size in (1, 2, 3):
        print(f"-- size {size} combinations, in permitted order --")
        for combination in itertools.combinations(window, size):
            result = simulate(
                state,
                payments=payment,
                spending_changes=combination,
                requested_amount=request.requested_amount,
            )
            names = " + ".join(change.render() for change in combination)
            verdict = "SAFE — this is the first success, search stops here" if result.safe else "not enough"
            print(f"{step}. {names}")
            print(f"   {verdict}")
            print(f"   {_sim_line(result, state)}")
            if result.violations:
                shown = result.violations[:5]
                for violation in shown:
                    print(
                        f"   violation {violation.date}  {violation.reason}  "
                        f"shortfall={violation.shortfall}  balance={violation.balance}"
                    )
                extra = len(result.violations) - len(shown)
                if extra > 0:
                    print(f"   ... {extra} more violation sample(s) truncated")
            print()
            if result.safe and chosen is None:
                chosen = combination
                print("   *** engine returns this combination and does not try later cuts ***")
                print()
                break
            step += 1
        if chosen is not None:
            break

    engine_choice = _find_change_set(state, request, profile, payment)
    print(f"_find_change_set returned: {None if engine_choice is None else [c.render() for c in engine_choice]}")

    ranked = rank_valid_candidates(request, generate_candidates(dataset, state, request))
    best = ranked.best
    print()
    print("## Ranked winner")
    print(f"method:             {best.method if best else None}")
    print(f"status:             {best.affordability_status if best else None}")
    print(f"payment_plan:       {best.render_payment_plan() if best else None}")
    print(f"spending_changes:   {best.render_spending_changes() if best else None}")
    print(f"ranking notes:      {ranked.ranking_notes}")

    _hr()
    print("## Why 476 loses to 556")
    print()
    c476 = _commitment_for(state, "event_476")
    c556 = _commitment_for(state, "event_556")
    _, save476 = _projected_debits(state, "event_476")
    _, save556 = _projected_debits(state, "event_556")
    print("event_476 is the family streaming plan. The profile lists `streaming`")
    print("in expense_categories_user_is_willing_to_stop, so user_asked=0 and it")
    print("is the FIRST size-1 cut tried. Cadence is monthly at EUR 19.00.")
    print(f"Stopping it removes only {save476} across the 90-day forecast.")
    print()
    print("That is not enough. After stop:event_476 the 620.40 payment still")
    print("breaches minimum_balance_to_keep because weekly dining stays on the")
    print("timeline (event_556 is the representative dining row).")
    print()
    print("event_556 is weekly dining, inferred as lifestyle STOP even though")
    print("the source row is 'fixed' and dining is not on the user's stop list")
    print(f"(user_asked=1). Estimated {c556.amount if c556 else '?'} every "
          f"{c556.frequency_days if c556 else '?'} days.")
    print(f"Stopping it removes {save556} across 90 days, which is enough to")
    print("keep the 620.40 payment above the EUR 800 floor. First successful")
    print("size-1 combination wins, so the engine returns stop:event_556")
    print("and never needs a two-cut set.")
    print()
    print("The label instead stops the user-listed streaming plan")
    print("(stop:event_476) and still pays 620.40 today. That only works if")
    print("the generator is not projecting the same weekly dining drain we do,")
    print("or if it is using a shorter / milder lifestyle forecast.")
    print()
    print(
        f"requested {request.requested_amount} − label {label.amount_safe_to_pay} "
        f"= {to_money(request.requested_amount - Decimal(label.amount_safe_to_pay))}."
    )
    print(
        f"That 17.10 is close to, but not equal to, the EUR {c476.amount if c476 else 19} "
        "streaming occurrence; it is not the dining estimate either."
    )
    print("So the amount miss and the change miss are related (lifestyle vs")
    print("streaming) but 17.10 is not itself the streaming bill.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
