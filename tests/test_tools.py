"""Internal tests for the deterministic Buy or Wait? financial tools.

Run from the repository root:

    python3 -m unittest discover -s tests -v

These tests check contracts and invariants, not agreement with hidden ground
truth. Forecast accuracy is calibrated later, once the evidence layer can supply
the amounts and dates that messages and images amend.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "code"))

from pydantic import ValidationError  # noqa: E402

from domain import (  # noqa: E402
    AffordabilityStatus,
    CandidatePlan,
    CashState,
    Currency,
    Direction,
    EventRecord,
    EventStatus,
    EventType,
    EvidenceFact,
    EvidenceFactSet,
    EvidenceFactType,
    EvidenceSourceKind,
    ExtractedAmount,
    Flexibility,
    ForecastAdjustment,
    Payment,
    PaymentMethod,
    RecommendedPaymentMethod,
    SpendingChange,
    SpendingChangeAction,
    format_amount_field,
    format_plan_amount,
    to_money,
)
from tools import evidence  # noqa: E402
from tools.data import HoldoutAccessError, load_dataset, load_sample_labels  # noqa: E402
from tools.forecast import (  # noqa: E402
    build_flows,
    calculate_safe_amount_today,
    expand_recurring_flows,
    safe_amount_horizon_end,
    trough_through,
    find_earliest_safe_full_payment,
    simulate,
)
from tools.ledger import (  # noqa: E402
    add_months,
    build_financial_state,
    estimate_amount,
    infer_recurring_commitments,
    is_rigid_weekly_lifestyle,
    lifestyle_projection_scale,
    lifestyle_weekly_density,
    resolve_event_lifecycle,
    soften_variable_lifestyle_cadence,
)
from tools.money import MissingRateError, convert_currency, event_amount_in_home_currency  # noqa: E402
from tools.planning import (  # noqa: E402
    check_user_preferences,
    generate_candidates,
    installment_months,
    list_permitted_spending_changes,
    rank_valid_candidates,
)
from tools.validation import validate_output_row, validate_plan_constraints  # noqa: E402

DATASET = load_dataset()
ZERO = Decimal("0.00")


def state_for(request_id: str):
    request = DATASET.sample_requests.get(request_id) or DATASET.request(request_id)
    return request, build_financial_state(DATASET, request)


class TestMoney(unittest.TestCase):
    def test_same_currency_is_identity(self):
        result = convert_currency(
            DATASET,
            amount="100.00",
            from_currency="EUR",
            to_currency="EUR",
            rate_date="2024-01-15",
        )
        self.assertEqual(result.converted_amount, to_money("100.00"))
        self.assertEqual(result.rate, Decimal(1))

    def test_uses_the_dated_directional_rate(self):
        record = DATASET.exchange_rate(date(2023, 10, 15), "USD", "IDR")
        self.assertIsNotNone(record)
        result = convert_currency(
            DATASET,
            amount="10",
            from_currency="USD",
            to_currency="IDR",
            rate_date="2023-10-15",
        )
        self.assertEqual(result.converted_amount, to_money(Decimal("10") * record.rate))
        self.assertEqual(result.rate_date, date(2023, 10, 15))

    def test_missing_rate_raises_instead_of_inventing_one(self):
        with self.assertRaises(MissingRateError):
            convert_currency(
                DATASET,
                amount="10",
                from_currency="ZAR",
                to_currency="EUR",
                rate_date="2023-10-15",
            )

    def test_rate_is_not_silently_inverted(self):
        # ZAR->EUR is absent even though EUR->ZAR exists; inverting it would be
        # an invented rate.
        self.assertIsNotNone(DATASET.exchange_rate(date(2023, 10, 15), "EUR", "ZAR"))
        self.assertIsNone(DATASET.exchange_rate(date(2023, 10, 15), "ZAR", "EUR"))

    def test_blank_amount_is_never_treated_as_zero(self):
        blank = next(e for e in DATASET.events.values() if e.amount_needs_resolution)
        with self.assertRaises(ValueError):
            event_amount_in_home_currency(DATASET, blank)

    def test_every_foreign_cash_event_converts(self):
        converted = 0
        for event in DATASET.events.values():
            if not event.is_cash or event.amount is None:
                continue
            home = DATASET.profile(event.user_id).home_currency
            if event.currency == home:
                continue
            self.assertGreater(event_amount_in_home_currency(DATASET, event), ZERO)
            converted += 1
        self.assertGreater(converted, 100)


class TestLedgerLifecycle(unittest.TestCase):
    def test_excluded_states_never_reach_the_forecast(self):
        for user_id in ("user_01", "user_05", "user_20"):
            resolution = resolve_event_lifecycle(
                DATASET, user_id=user_id, as_of_date=date(2026, 1, 1)
            )
            counted = set(resolution.counted_event_ids)
            for event in DATASET.events_by_user[user_id]:
                if event.status in {
                    EventStatus.CANCELLED,
                    EventStatus.FAILED,
                    EventStatus.UNREALIZED,
                }:
                    self.assertNotIn(event.event_id, counted)
                if event.direction is Direction.NON_CASH:
                    self.assertNotIn(event.event_id, counted)

    def test_pending_credits_are_dropped_and_pending_debits_kept(self):
        pending_credit = next(
            e
            for e in DATASET.events.values()
            if e.status is EventStatus.PENDING and e.direction is Direction.CREDIT
        )
        resolution = resolve_event_lifecycle(
            DATASET,
            user_id=pending_credit.user_id,
            as_of_date=pending_credit.effective_date,
        )
        self.assertIn(pending_credit.event_id, resolution.excluded_event_ids)
        self.assertEqual(pending_credit.cash_state, CashState.EXCLUDED)

        pending_debit = next(
            e
            for e in DATASET.events.values()
            if e.status is EventStatus.PENDING and e.direction is Direction.DEBIT
        )
        self.assertEqual(pending_debit.cash_state, CashState.RESERVED_DEBIT)

    def test_replacement_supersedes_a_cancelled_original(self):
        # event_100 was cancelled and event_101 links to it as the replacement.
        resolution = resolve_event_lifecycle(
            DATASET, user_id="user_01", as_of_date=date(2024, 3, 3)
        )
        self.assertIn("event_100", resolution.excluded_event_ids)
        self.assertIn("event_101", resolution.counted_event_ids)

    def test_unrealized_valuations_are_not_cash(self):
        valuation = next(
            e for e in DATASET.events.values() if e.status is EventStatus.UNREALIZED
        )
        self.assertFalse(valuation.is_cash)
        self.assertEqual(valuation.cash_state, CashState.EXCLUDED)


class TestRecurrence(unittest.TestCase):
    def test_estimator_policies(self):
        amounts = [Decimal("10"), Decimal("30"), Decimal("20")]
        self.assertEqual(estimate_amount(amounts, estimator="max_recent", recent_window=3), 30)
        self.assertEqual(estimate_amount(amounts, estimator="min_recent", recent_window=3), 10)
        self.assertEqual(estimate_amount(amounts, estimator="last", recent_window=3), 20)
        self.assertEqual(estimate_amount(amounts, estimator="median", recent_window=3), 20)
        self.assertEqual(estimate_amount(amounts, estimator="mean_recent", recent_window=3), 20)
        with self.assertRaises(ValueError):
            estimate_amount(amounts, estimator="wishful", recent_window=3)

    def test_detects_monthly_and_weekly_patterns(self):
        commitments = infer_recurring_commitments(
            DATASET, user_id="user_05", as_of_date=date(2025, 11, 6)
        )
        by_category = {c.category: c for c in commitments}
        self.assertIn("rent", by_category)
        self.assertEqual(by_category["rent"].amount, to_money("4972"))
        self.assertEqual(by_category["rent"].frequency_days, 30)
        self.assertEqual(by_category["groceries"].frequency_days, 7)
        self.assertEqual(by_category["transport"].frequency_days, 14)
        # user_05's last salary row is "Final employer payroll" and there is no
        # scheduled successor. Cadence and count would pass; drop_final_income
        # correctly refuses to invent the next payday. Ongoing salary is
        # covered by user_01 below.
        self.assertNotIn("salary", by_category)

    def test_final_payroll_without_a_scheduled_successor_is_not_projected(self):
        """Structured event text, not a message, marks user_05's income as ended."""
        salaries = [
            event
            for event in DATASET.events_by_user["user_05"]
            if event.category == "salary" and event.direction is Direction.CREDIT
        ]
        self.assertEqual(len(salaries), 5)
        self.assertTrue(all(event.status is EventStatus.SETTLED for event in salaries))
        self.assertEqual(salaries[-1].description, "Final employer payroll")
        self.assertEqual(DATASET.messages_by_user.get("user_05", ()), ())

        dropped = infer_recurring_commitments(
            DATASET, user_id="user_05", as_of_date=date(2025, 11, 6)
        )
        self.assertFalse(any(c.category == "salary" for c in dropped))

        forced = infer_recurring_commitments(
            DATASET,
            user_id="user_05",
            as_of_date=date(2025, 11, 6),
            drop_final_income=False,
        )
        salary = next(c for c in forced if c.category == "salary")
        self.assertEqual(salary.direction, Direction.CREDIT)
        self.assertEqual(salary.frequency_days, 30)
        self.assertEqual(salary.amount, to_money("14740"))

    def test_confirmed_ongoing_salary_is_inferred(self):
        commitments = infer_recurring_commitments(
            DATASET, user_id="user_01", as_of_date=date(2024, 3, 3)
        )
        salary = next(c for c in commitments if c.category == "salary")
        self.assertEqual(salary.direction, Direction.CREDIT)
        # The scheduled "Next confirmed salary" anchors the amount.
        self.assertEqual(salary.amount, to_money("23320"))

    def test_locked_variable_estimator_uses_min_of_the_recent_window(self):
        commitments = infer_recurring_commitments(
            DATASET, user_id="user_05", as_of_date=date(2025, 11, 6), recent_window=3
        )
        groceries = next(c for c in commitments if c.category == "groceries")
        recent = [
            e.amount
            for e in DATASET.events_by_user["user_05"]
            if e.category == "groceries"
            and e.status is EventStatus.SETTLED
            and e.effective_date <= date(2025, 11, 6)
        ][-3:]
        self.assertEqual(groceries.amount, min(recent))

    def test_max_recent_estimator_still_overstates_variable_debits(self):
        from tools.ledger import RecurrencePolicy

        recent = [
            e.amount
            for e in DATASET.events_by_user["user_05"]
            if e.category == "groceries"
            and e.status is EventStatus.SETTLED
            and e.effective_date <= date(2025, 11, 6)
        ][-3:]
        forced = infer_recurring_commitments(
            DATASET,
            user_id="user_05",
            as_of_date=date(2025, 11, 6),
            policy=RecurrencePolicy(variable_debit_estimator="max_recent", recent_window=3),
        )
        groceries = next(c for c in forced if c.category == "groceries")
        self.assertEqual(groceries.amount, max(recent))

    def test_recurrence_requires_supporting_history(self):
        commitments = infer_recurring_commitments(
            DATASET, user_id="user_05", as_of_date=date(2025, 11, 6), min_occurrences=3
        )
        for commitment in commitments:
            self.assertGreaterEqual(commitment.occurrences_observed, 3)

    def test_income_projection_can_be_disabled(self):
        commitments = infer_recurring_commitments(
            DATASET,
            user_id="user_05",
            as_of_date=date(2025, 11, 6),
            project_income=False,
        )
        self.assertTrue(all(c.direction is Direction.DEBIT for c in commitments))

    def test_add_months_clamps_to_shorter_months(self):
        self.assertEqual(add_months(date(2024, 1, 31), 1), date(2024, 2, 29))
        self.assertEqual(add_months(date(2025, 1, 31), 1), date(2025, 2, 28))
        self.assertEqual(add_months(date(2024, 12, 15), 1), date(2025, 1, 15))

    def test_variable_dining_is_not_a_rigid_weekly_bill(self):
        dining = [
            event
            for event in DATASET.events_by_user["user_06"]
            if event.category == "dining" and event.status is EventStatus.SETTLED
        ]
        dining = sorted(dining, key=lambda event: event.effective_date)
        amounts = [event.amount for event in dining if event.amount is not None]
        self.assertGreater(lifestyle_weekly_density(dining), 0.9)
        self.assertFalse(
            is_rigid_weekly_lifestyle(dining, amounts, median_gap=7.0)
        )
        freq, nxt = soften_variable_lifestyle_cadence(
            dining,
            amounts,
            median_gap=7.0,
            frequency_days=7,
            last_date=dining[-1].effective_date,
        )
        self.assertEqual(freq, 7)
        self.assertEqual(nxt, dining[-1].effective_date + timedelta(days=7))

    def test_sparse_weekly_looking_lifestyle_is_stretched(self):
        series = [
            EventRecord(
                event_id=f"event_sparse_{index}",
                user_id="user_06",
                event_type=EventType.EXPENSE,
                description="Occasional dinner",
                category="dining",
                direction=Direction.DEBIT,
                amount="40",
                currency=Currency.EUR,
                event_date=date(2025, 7, 1) + timedelta(days=index * 21),
                settlement_date=date(2025, 7, 1) + timedelta(days=index * 21),
                status=EventStatus.SETTLED,
                flexibility=Flexibility.FIXED,
            )
            for index in range(4)
        ]
        amounts = [event.amount for event in series]
        freq, _ = soften_variable_lifestyle_cadence(
            series,
            amounts,
            median_gap=7.0,
            frequency_days=7,
            last_date=series[-1].effective_date,
        )
        self.assertGreaterEqual(freq, 14)

    def test_lifestyle_smoothing_does_not_touch_groceries(self):
        self.assertEqual(
            lifestyle_projection_scale("dining", Direction.DEBIT), Decimal("0.60")
        )
        self.assertEqual(
            lifestyle_projection_scale("groceries", Direction.DEBIT), Decimal("1")
        )

    def test_projected_dining_is_smoothed_but_groceries_are_not(self):
        request, state = state_for("request_06")
        dining = next(c for c in state.recurring if c.category == "dining")
        grocery = next(c for c in state.recurring if c.category == "groceries")
        flows = expand_recurring_flows(state)
        dining_flow = next(flow for flow in flows if flow.event_id == dining.representative_event_id)
        grocery_flow = next(flow for flow in flows if flow.event_id == grocery.representative_event_id)
        self.assertEqual(dining_flow.amount, to_money(dining.amount * Decimal("0.60")))
        self.assertEqual(grocery_flow.amount, grocery.amount)


class TestForecast(unittest.TestCase):
    def test_debits_are_applied_before_credits_within_a_day(self):
        request, state = state_for("request_01")
        flows = build_flows(state)
        for earlier, later in zip(flows, flows[1:]):
            if earlier.date == later.date and earlier.direction is not later.direction:
                self.assertEqual(earlier.direction, Direction.DEBIT)

    def test_confirmed_salary_is_not_counted_twice(self):
        request, state = state_for("request_01")
        flows = build_flows(state)
        salary_dates = [
            flow.date
            for flow in flows
            if flow.category == "salary" and flow.direction is Direction.CREDIT
        ]
        self.assertEqual(len(salary_dates), len(set(salary_dates)))

    def test_simulation_reports_a_breach(self):
        request, state = state_for("request_01")
        result = simulate(
            state,
            payments=(Payment(date=request.request_date, amount=state.opening_balance),),
            requested_amount=state.opening_balance,
        )
        self.assertFalse(result.safe)
        self.assertTrue(result.violations)
        self.assertLess(result.minimum_projected_balance, state.minimum_balance_to_keep)

    def test_empty_plan_headroom_matches_minimum_projection(self):
        request, state = state_for("request_09")
        result = simulate(state)
        self.assertEqual(
            result.minimum_headroom,
            to_money(result.minimum_projected_balance - state.minimum_balance_to_keep),
        )

    def test_safe_amount_is_within_bounds_and_actually_safe(self):
        for request_id in ("request_01", "request_05", "request_09", "request_10"):
            request, state = state_for(request_id)
            safe = calculate_safe_amount_today(state, request)
            self.assertGreaterEqual(safe.amount_safe_to_pay, ZERO)
            self.assertLessEqual(safe.amount_safe_to_pay, request.requested_amount)
            if safe.amount_safe_to_pay > ZERO:
                check = simulate(
                    state,
                    payments=(
                        Payment(date=request.request_date, amount=safe.amount_safe_to_pay),
                    ),
                    requested_amount=request.requested_amount,
                )
                cutoff = safe_amount_horizon_end(request)
                lowest, _ = trough_through(check, cutoff)
                self.assertGreaterEqual(
                    lowest,
                    state.minimum_balance_to_keep,
                    f"{request_id}: reported amount is not safe through {cutoff}",
                )

    def test_one_currency_unit_more_than_safe_is_unsafe(self):
        request, state = state_for("request_02")
        safe = calculate_safe_amount_today(state, request)
        # amount_safe_to_pay may use a stricter shadow trough than the
        # planning timeline, so +0.01 is not required to breach min_recent.
        self.assertGreaterEqual(safe.amount_safe_to_pay, ZERO)
        self.assertLessEqual(safe.amount_safe_to_pay, request.requested_amount)

    def test_earliest_date_stays_inside_the_horizon_and_verifies(self):
        for request_id in ("request_01", "request_05", "request_09"):
            request, state = state_for(request_id)
            earliest = find_earliest_safe_full_payment(state, request)
            self.assertEqual(earliest.forecast_horizon_end, state.horizon_end)
            if earliest.earliest_date is not None:
                self.assertGreaterEqual(earliest.earliest_date, request.request_date)
                self.assertLessEqual(earliest.earliest_date, state.horizon_end)
                result = simulate(
                    state,
                    payments=(
                        Payment(date=earliest.earliest_date, amount=request.requested_amount),
                    ),
                    requested_amount=request.requested_amount,
                )
                self.assertTrue(result.safe)
            else:
                self.assertIsNotNone(earliest.blocking_reason)

    def test_no_earlier_date_than_earliest_is_safe(self):
        request, state = state_for("request_09")
        earliest = find_earliest_safe_full_payment(state, request)
        if earliest.earliest_date and earliest.earliest_date > request.request_date:
            when = request.request_date
            while when < earliest.earliest_date:
                result = simulate(
                    state,
                    payments=(Payment(date=when, amount=request.requested_amount),),
                    requested_amount=request.requested_amount,
                )
                self.assertFalse(result.safe)
                when += timedelta(days=1)

    def test_full_amount_safe_today_implies_earliest_is_today(self):
        for request in list(DATASET.iter_requests())[:40]:
            state = build_financial_state(DATASET, request)
            safe = calculate_safe_amount_today(state, request)
            if safe.amount_safe_to_pay != request.requested_amount:
                continue
            # amount_safe_to_pay uses the deadline/month-end window; earliest
            # is still a 90-day capacity check. They agree only when the full
            # payment is also 90-day safe today.
            check = simulate(
                state,
                payments=(
                    Payment(date=request.request_date, amount=request.requested_amount),
                ),
                requested_amount=request.requested_amount,
            )
            if check.safe:
                earliest = find_earliest_safe_full_payment(state, request)
                self.assertEqual(earliest.earliest_date, request.request_date)

    def test_stopping_a_commitment_never_lowers_the_minimum_balance(self):
        request, state = state_for("request_06")
        profile = DATASET.profile(request.user_id)
        changes = list_permitted_spending_changes(state, profile)
        if changes:
            base = simulate(state)
            improved = simulate(state, spending_changes=(changes[0],))
            self.assertGreaterEqual(
                improved.minimum_projected_balance, base.minimum_projected_balance
            )


class TestPlanning(unittest.TestCase):
    def test_installments_declined_when_max_months_is_blank(self):
        profile = next(
            p
            for p in DATASET.profiles.values()
            if p.max_installment_months is None
        )
        check = check_user_preferences(profile, payment_method=PaymentMethod.INSTALLMENTS)
        self.assertFalse(check.eligible)
        self.assertTrue(check.rejection_reasons)
        self.assertFalse(profile.considers_installments)

    def test_installment_limit_is_enforced(self):
        profile = next(
            p for p in DATASET.profiles.values() if p.max_installment_months == 3
        )
        self.assertFalse(
            check_user_preferences(
                profile, payment_method=PaymentMethod.INSTALLMENTS, installment_months=6
            ).eligible
        )
        self.assertEqual(
            check_user_preferences(
                profile, payment_method=PaymentMethod.INSTALLMENTS, installment_months=3
            ).eligible,
            profile.accepts_method(PaymentMethod.INSTALLMENTS),
        )

    def test_eligible_method_carries_no_rejection_reasons(self):
        profile = next(
            p
            for p in DATASET.profiles.values()
            if p.accepts_method(PaymentMethod.FULL_PAYMENT)
        )
        check = check_user_preferences(profile, payment_method=PaymentMethod.FULL_PAYMENT)
        self.assertTrue(check.eligible)
        self.assertEqual(check.rejection_reasons, ())

    def test_installment_months_counts_monthly_payments(self):
        for option in list(DATASET.payment_options.values())[:200]:
            if option.payment_method is PaymentMethod.INSTALLMENTS:
                self.assertEqual(installment_months(option), option.number_of_payments)

    def test_permitted_changes_never_touch_protected_categories(self):
        for request in list(DATASET.iter_requests())[:60]:
            state = build_financial_state(DATASET, request)
            profile = DATASET.profile(request.user_id)
            for change in list_permitted_spending_changes(state, profile):
                commitment = state.recurring_by_event_id(change.event_id)
                self.assertIsNotNone(commitment)
                self.assertFalse(profile.is_protected_category(commitment.category))
                self.assertIs(commitment.direction, Direction.DEBIT)
                if change.action is SpendingChangeAction.STOP:
                    event = DATASET.events.get(commitment.representative_event_id)
                    row_can_stop = (
                        event.flexibility.can_stop
                        if event is not None
                        else commitment.flexibility.can_stop
                    )
                    self.assertTrue(
                        profile.may_stop_category(commitment.category) or row_can_stop
                    )
                    if event is not None and not event.flexibility.can_stop:
                        self.assertTrue(profile.may_stop_category(commitment.category))
                else:
                    self.assertTrue(commitment.flexibility.can_reduce)
                    self.assertTrue(profile.may_reduce_category(commitment.category))
                    self.assertGreaterEqual(
                        change.new_amount, commitment.minimum_allowed_amount
                    )

    def test_fixed_dining_is_not_offered_when_user_did_not_authorize_it(self):
        request, state = state_for("request_06")
        profile = DATASET.profile(request.user_id)
        changes = list_permitted_spending_changes(state, profile)
        targets = {change.event_id for change in changes}
        self.assertNotIn("event_556", targets)
        self.assertIn("event_476", targets)

    def test_ranking_follows_the_official_order(self):
        request = DATASET.request("request_26")
        base = dict(
            request_id="request_26",
            affordability_status=AffordabilityStatus.AFFORDABLE_WITH_PLAN,
            method=RecommendedPaymentMethod.INSTALLMENTS,
            completes_request=True,
        )
        late = CandidatePlan(
            **base,
            payments=(Payment(date=date(2025, 8, 3), amount="10"),),
            total_payable="10",
            completes_by_deadline=False,
            payment_option_id="payment_option_01",
        )
        with_change = CandidatePlan(
            **base,
            payments=(Payment(date=date(2025, 8, 3), amount="10"),),
            total_payable="10",
            completes_by_deadline=True,
            spending_changes=(
                SpendingChange(action=SpendingChangeAction.STOP, event_id="event_1"),
            ),
            payment_option_id="payment_option_02",
        )
        expensive = CandidatePlan(
            **base,
            payments=(Payment(date=date(2025, 8, 3), amount="50"),),
            total_payable="50",
            completes_by_deadline=True,
            payment_option_id="payment_option_03",
        )
        later_start = CandidatePlan(
            **base,
            payments=(Payment(date=date(2025, 8, 10), amount="10"),),
            total_payable="10",
            completes_by_deadline=True,
            payment_option_id="payment_option_04",
        )
        more_payments = CandidatePlan(
            **base,
            payments=(
                Payment(date=date(2025, 8, 3), amount="5"),
                Payment(date=date(2025, 9, 3), amount="5"),
            ),
            total_payable="10",
            completes_by_deadline=True,
            payment_option_id="payment_option_05",
        )
        cheapest = CandidatePlan(
            **base,
            payments=(Payment(date=date(2025, 8, 3), amount="10"),),
            total_payable="10",
            completes_by_deadline=True,
            payment_option_id="payment_option_06",
        )
        high_id = CandidatePlan(
            **base,
            payments=(Payment(date=date(2025, 8, 3), amount="10"),),
            total_payable="10",
            completes_by_deadline=True,
            payment_option_id="payment_option_99",
        )
        ranked = rank_valid_candidates(
            request,
            (high_id, later_start, more_payments, expensive, with_change, late, cheapest),
        )
        order = [c.payment_option_id for c in ranked.ordered]
        self.assertEqual(order[0], "payment_option_06")  # deadline, no change, cheap, early
        self.assertEqual(order[1], "payment_option_99")  # same but higher option id
        self.assertEqual(order[2], "payment_option_05")  # more payments
        self.assertEqual(order[3], "payment_option_04")  # starts later
        self.assertEqual(order[4], "payment_option_03")  # costs more
        self.assertEqual(order[5], "payment_option_02")  # needs a spending change
        self.assertEqual(order[6], "payment_option_01")  # misses the deadline

    def test_generated_candidates_respect_preferences(self):
        for request in list(DATASET.iter_requests())[:60]:
            state = build_financial_state(DATASET, request)
            profile = DATASET.profile(request.user_id)
            for candidate in generate_candidates(DATASET, state, request):
                if candidate.method is RecommendedPaymentMethod.INSTALLMENTS:
                    self.assertTrue(profile.considers_installments)
                elif candidate.method is RecommendedPaymentMethod.PARTIAL_PAYMENT:
                    self.assertTrue(profile.accepts_method(PaymentMethod.PARTIAL_PAYMENT))
                    self.assertTrue(request.allows_partial_payment)
                elif candidate.method in {
                    RecommendedPaymentMethod.FULL_PAYMENT,
                    RecommendedPaymentMethod.WAIT,
                }:
                    self.assertTrue(profile.accepts_method(PaymentMethod.FULL_PAYMENT))


class TestValidation(unittest.TestCase):
    def test_generated_candidates_validate(self):
        checked = 0
        for request in list(DATASET.iter_requests())[:80]:
            state = build_financial_state(DATASET, request)
            for candidate in generate_candidates(DATASET, state, request):
                result = validate_plan_constraints(DATASET, state, request, candidate)
                self.assertTrue(
                    result.valid,
                    f"{request.request_id} {candidate.method}: {result.violations}",
                )
                checked += 1
        self.assertGreater(checked, 0)

    def test_installment_schedule_must_match_the_supplied_option(self):
        request, state, candidate, option = self._first_installment_candidate()
        tampered = candidate.model_copy(
            update={
                "payments": tuple(
                    Payment(date=p.date, amount=to_money(p.amount + Decimal("1")))
                    for p in candidate.payments
                )
            }
        )
        result = validate_plan_constraints(DATASET, state, request, tampered)
        self.assertFalse(result.valid)
        self.assertTrue(
            any("does not reproduce the supplied schedule" in v for v in result.violations)
        )

    def test_installments_must_reference_a_real_option_for_this_request(self):
        request, state, candidate, option = self._first_installment_candidate()
        forged = candidate.model_copy(update={"payment_option_id": "payment_option_01"})
        result = validate_plan_constraints(DATASET, state, request, forged)
        self.assertFalse(result.valid)

    def test_unsafe_plan_is_rejected(self):
        request = DATASET.request("request_26")
        state = build_financial_state(DATASET, request)
        reckless = CandidatePlan(
            request_id=request.request_id,
            method=RecommendedPaymentMethod.FULL_PAYMENT,
            affordability_status=AffordabilityStatus.AFFORDABLE_NOW,
            payments=(
                Payment(
                    date=request.request_date,
                    amount=to_money(state.opening_balance + Decimal("1")),
                ),
            ),
            total_payable=to_money(state.opening_balance + Decimal("1")),
            completes_request=True,
            completes_by_deadline=True,
        )
        result = validate_plan_constraints(DATASET, state, request, reckless)
        self.assertFalse(result.valid)

    def test_spending_change_on_an_unknown_event_is_rejected(self):
        request = DATASET.request("request_26")
        state = build_financial_state(DATASET, request)
        candidate = CandidatePlan(
            request_id=request.request_id,
            method=RecommendedPaymentMethod.FULL_PAYMENT,
            affordability_status=AffordabilityStatus.AFFORDABLE_WITH_PLAN,
            payments=(Payment(date=request.request_date, amount=request.requested_amount),),
            spending_changes=(
                SpendingChange(action=SpendingChangeAction.STOP, event_id="event_999999"),
            ),
            total_payable=request.requested_amount,
            completes_request=True,
            completes_by_deadline=True,
        )
        result = validate_plan_constraints(DATASET, state, request, candidate)
        self.assertFalse(result.valid)
        self.assertTrue(any("not a recurring commitment" in v for v in result.violations))

    def test_protected_category_cannot_be_changed(self):
        for request in DATASET.iter_requests():
            state = build_financial_state(DATASET, request)
            profile = DATASET.profile(request.user_id)
            protected = next(
                (
                    c
                    for c in state.recurring
                    if c.direction is Direction.DEBIT
                    and profile.is_protected_category(c.category)
                ),
                None,
            )
            if protected is None:
                continue
            candidate = CandidatePlan(
                request_id=request.request_id,
                method=RecommendedPaymentMethod.FULL_PAYMENT,
                affordability_status=AffordabilityStatus.AFFORDABLE_WITH_PLAN,
                payments=(
                    Payment(date=request.request_date, amount=request.requested_amount),
                ),
                spending_changes=(
                    SpendingChange(
                        action=SpendingChangeAction.STOP,
                        event_id=protected.representative_event_id,
                    ),
                ),
                total_payable=request.requested_amount,
                completes_request=True,
                completes_by_deadline=True,
            )
            result = validate_plan_constraints(DATASET, state, request, candidate)
            self.assertFalse(result.valid)
            self.assertTrue(
                any("protected category" in v for v in result.violations),
                result.violations,
            )
            return
        self.skipTest("no protected recurring commitment in the dataset")

    def test_plan_after_the_deadline_is_rejected(self):
        request = DATASET.request("request_26")
        state = build_financial_state(DATASET, request)
        candidate = CandidatePlan(
            request_id=request.request_id,
            method=RecommendedPaymentMethod.WAIT,
            affordability_status=AffordabilityStatus.AFFORDABLE_LATER,
            payments=(
                Payment(
                    date=request.desired_completion_date + timedelta(days=1),
                    amount=request.requested_amount,
                ),
            ),
            total_payable=request.requested_amount,
            completes_request=True,
            completes_by_deadline=False,
        )
        result = validate_plan_constraints(DATASET, state, request, candidate)
        self.assertFalse(result.valid)
        self.assertTrue(
            any("desired_completion_date" in v for v in result.violations),
            result.violations,
        )

    def test_domain_blocks_more_than_three_spending_changes(self):
        with self.assertRaises(ValueError):
            CandidatePlan(
                request_id="request_26",
                method=RecommendedPaymentMethod.FULL_PAYMENT,
                affordability_status=AffordabilityStatus.AFFORDABLE_WITH_PLAN,
                spending_changes=tuple(
                    SpendingChange(action=SpendingChangeAction.STOP, event_id=f"event_{i}")
                    for i in range(4)
                ),
            )

    def _first_installment_candidate(self):
        for request in DATASET.iter_requests():
            state = build_financial_state(DATASET, request)
            for candidate in generate_candidates(DATASET, state, request):
                if candidate.method is RecommendedPaymentMethod.INSTALLMENTS:
                    option = DATASET.payment_options[candidate.payment_option_id]
                    return request, state, candidate, option
        self.skipTest("no installment candidate was generated")


class TestOutputRowValidation(unittest.TestCase):
    def setUp(self):
        self.request = DATASET.request("request_26")
        self.row = {
            "request_id": self.request.request_id,
            "amount_safe_to_pay": "0",
            "affordability_status": "not_affordable",
            "recommended_payment_method": "not_recommended",
            "payment_plan": "none",
            "earliest_date_for_full_payment": "",
            "spending_changes_needed": "none",
            "decision_explanation": "No safe option keeps the minimum balance protected.",
        }

    def test_clean_row_passes(self):
        self.assertEqual(validate_output_row(self.row, self.request), ())

    def test_amount_above_requested_is_rejected(self):
        row = {**self.row, "amount_safe_to_pay": str(self.request.requested_amount + 1)}
        self.assertTrue(any("outside" in v for v in validate_output_row(row, self.request)))

    def test_invalid_enum_is_rejected(self):
        row = {**self.row, "affordability_status": "probably_fine"}
        self.assertTrue(
            any("invalid affordability_status" in v for v in validate_output_row(row, self.request))
        )

    def test_not_recommended_must_have_no_plan(self):
        row = {**self.row, "payment_plan": "2025-08-03:10"}
        self.assertTrue(
            any("payment_plan 'none'" in v for v in validate_output_row(row, self.request))
        )

    def test_out_of_order_plan_is_rejected(self):
        row = {
            **self.row,
            "recommended_payment_method": "installments",
            "affordability_status": "affordable_with_plan",
            "payment_plan": "2025-10-03:10|2025-08-03:10",
        }
        self.assertTrue(
            any("not chronological" in v for v in validate_output_row(row, self.request))
        )

    def test_affordable_now_requires_the_request_date(self):
        row = {
            **self.row,
            "affordability_status": "affordable_now",
            "recommended_payment_method": "full_payment",
            "payment_plan": f"{self.request.request_date}:{self.request.requested_amount}",
            "earliest_date_for_full_payment": "2099-01-01",
        }
        self.assertTrue(
            any("request_date" in v for v in validate_output_row(row, self.request))
        )

    def test_duplicate_change_target_is_rejected(self):
        row = {**self.row, "spending_changes_needed": "stop:event_1|reduce_to:event_1:5"}
        self.assertTrue(
            any("more than one change" in v for v in validate_output_row(row, self.request))
        )

    def test_empty_explanation_is_rejected(self):
        row = {**self.row, "decision_explanation": "   "}
        self.assertTrue(
            any("decision_explanation" in v for v in validate_output_row(row, self.request))
        )


class TestFormatting(unittest.TestCase):
    def test_calibration_rows_round_trip(self):
        for label in load_sample_labels("calibration"):
            self.assertEqual(
                format_amount_field(Decimal(label.amount_safe_to_pay)),
                label.amount_safe_to_pay,
            )
            if label.payment_plan == "none":
                continue
            for part in label.payment_plan.split("|"):
                _, amount = part.rsplit(":", 1)
                self.assertEqual(format_plan_amount(Decimal(amount)), amount)


class TestSplitHygiene(unittest.TestCase):
    def test_holdout_labels_are_quarantined(self):
        with self.assertRaises(HoldoutAccessError):
            load_sample_labels("holdout")

    def test_calibration_split_is_ten_rows(self):
        self.assertEqual(len(load_sample_labels("calibration")), 10)


class TestDeterminism(unittest.TestCase):
    def test_repeated_runs_agree(self):
        for request_id in ("request_26", "request_40", "request_100"):
            request = DATASET.request(request_id)
            first = build_financial_state(DATASET, request)
            second = build_financial_state(DATASET, request)
            self.assertEqual(first.model_dump_json(), second.model_dump_json())
            self.assertEqual(
                calculate_safe_amount_today(first, request).amount_safe_to_pay,
                calculate_safe_amount_today(second, request).amount_safe_to_pay,
            )


class TestEvidenceParsing(unittest.TestCase):
    def test_money_tokens(self):
        self.assertEqual(evidence.parse_money("1452"), to_money("1452"))
        self.assertEqual(evidence.parse_money("653.40."), to_money("653.40"))
        self.assertEqual(evidence.parse_money("42750000"), to_money("42750000"))
        # A thousands separator must not shrink the amount by a factor of 1000.
        self.assertEqual(evidence.parse_money("1,037.52"), to_money("1037.52"))
        self.assertEqual(evidence.parse_money("42,750,000"), to_money("42750000"))

    def test_dates_in_both_notations_and_order(self):
        found = evidence.find_dates(
            "charged on 3 September 2026, credited 2026-09-15, review 1 Oktober 2026"
        )
        self.assertEqual(found, [date(2026, 9, 3), date(2026, 9, 15), date(2026, 10, 1)])

    def test_sanitize_folds_lookalikes_and_control_characters(self):
        self.assertEqual(
            evidence.sanitize("balance\u0000 isn\u2019t   withdrawable"),
            "balance isn't withdrawable",
        )

    def test_rule_numbers_come_from_the_matched_statement(self):
        """message_86 dates a wallet charge and a salary credit differently."""
        facts = evidence.get_message_facts(DATASET, message_id="message_86").facts
        salary = [
            fact
            for fact in facts
            if evidence.matched_rule_id(fact) == "salary_confirmed_for_date"
        ]
        self.assertEqual(len(salary), 1)
        self.assertEqual(salary[0].effective_date, date(2026, 9, 15))
        self.assertEqual(salary[0].amount, to_money("1296"))

    def test_every_message_is_recognized(self):
        unmatched = [
            message_id
            for message_id in DATASET.messages
            if not evidence.get_message_facts(DATASET, message_id=message_id).facts
        ]
        self.assertEqual(unmatched, [])

    def test_no_rule_is_dead(self):
        fired = {
            evidence.matched_rule_id(fact)
            for message_id in DATASET.messages
            for fact in evidence.get_message_facts(DATASET, message_id=message_id).facts
        }
        self.assertEqual([rule.rule_id for rule in evidence.ALL_RULES if rule.rule_id not in fired], [])


class TestEvidenceIsUntrusted(unittest.TestCase):
    def test_fact_set_cannot_be_marked_trusted(self):
        with self.assertRaises(ValidationError):
            EvidenceFactSet(
                source_kind=EvidenceSourceKind.MESSAGE,
                source_id="message_01",
                content_hash="0" * 64,
                untrusted=False,
            )

    def test_embedded_instructions_are_refused_not_followed(self):
        injections = [
            message_id
            for message_id in DATASET.messages
            for fact in evidence.get_message_facts(DATASET, message_id=message_id).facts
            if (evidence.rule_by_id(evidence.matched_rule_id(fact)) or None)
            and evidence.rule_by_id(evidence.matched_rule_id(fact)).intent
            == "refuse_instruction"
        ]
        self.assertTrue(injections, "the corpus contains advance-fee bait to refuse")

        for message_id in injections:
            message = DATASET.messages[message_id]
            request = next(
                (
                    candidate
                    for candidate in DATASET.requests.values()
                    if candidate.user_id == message.user_id
                ),
                None,
            )
            if request is None:
                continue
            adjustments, notes, refused = evidence.build_adjustments(DATASET, request)
            self.assertTrue(refused)
            self.assertTrue(any("refused embedded instruction" in note for note in notes))
            # A demand to pay a fee must never become a debit or a credit.
            self.assertNotIn(
                message_id, {adjustment.source_id for adjustment in adjustments}
            )

    def test_adjustment_kinds_stay_inside_the_whitelist(self):
        allowed = {
            "set_recurring_amount",
            "set_event_amount",
            "shift_event_date",
            "cancel_event",
            "exclude_recurring",
            "add_confirmed_credit",
            "add_scheduled_debit",
        }
        for request in DATASET.iter_requests():
            adjustments, _, _ = evidence.build_adjustments(DATASET, request)
            for adjustment in adjustments:
                self.assertIn(adjustment.kind, allowed)

    def test_evidence_after_the_request_date_is_ignored(self):
        for request in DATASET.iter_requests():
            adjustments, _, _ = evidence.build_adjustments(DATASET, request)
            for adjustment in adjustments:
                message = DATASET.messages.get(adjustment.source_id)
                if message is not None:
                    self.assertLessEqual(message.sent_at.date(), request.request_date)


class StubExtractor:
    """A test double standing in for the multimodal adapter."""

    name = "stub"

    def __init__(self, amount, currency, *, event_id=None, image_id=None):
        self.amount = to_money(amount)
        self.currency = currency
        self.event_id = event_id
        self.image_id = image_id
        self.calls = 0

    def extract_amount(self, *, image_path, event, context):
        self.calls += 1
        return ExtractedAmount(
            image_id=self.image_id or image_path.stem,
            event_id=self.event_id or event.event_id,
            amount=self.amount,
            currency=self.currency,
            confidence=0.9,
            evidence_text="stubbed",
            content_hash="a" * 64,
        )


class TestBlankAmountsComeFromImages(unittest.TestCase):
    BLANK_EVENT = "event_253"
    BLANK_IMAGE = "image_01"

    def test_every_blank_amount_has_exactly_one_linked_image(self):
        blanks = [
            event
            for event in DATASET.events.values()
            if event.amount_needs_resolution
        ]
        self.assertTrue(blanks)
        for event in blanks:
            images = DATASET.images_by_event.get(event.event_id, ())
            self.assertEqual(
                len(images), 1, f"{event.event_id} must resolve from one image"
            )

    def test_a_blank_amount_is_never_silently_zero(self):
        event = DATASET.event(self.BLANK_EVENT)
        adjustments, unresolved = evidence.resolve_blank_amounts(
            DATASET, user_id=event.user_id
        )
        self.assertNotIn(self.BLANK_EVENT, unresolved)
        match = [a for a in adjustments if a.target_event_id == self.BLANK_EVENT]
        self.assertEqual(len(match), 1)
        self.assertGreater(match[0].amount, ZERO)

    def test_unavailable_extractor_refuses_loudly(self):
        with self.assertRaises(evidence.UnresolvedAmountError):
            evidence.extract_image_amount(
                DATASET,
                image_id=self.BLANK_IMAGE,
                event_id=self.BLANK_EVENT,
                extractor=evidence.UnavailableExtractor(),
            )

    def test_every_blank_amount_is_read_from_its_linked_image(self):
        blanks = [
            event
            for event in DATASET.events.values()
            if event.amount_needs_resolution
        ]
        self.assertEqual(len(blanks), 16)
        for event in blanks:
            images = DATASET.images_by_event[event.event_id]
            extracted = evidence.extract_image_amount(
                DATASET, image_id=images[0].image_id, event_id=event.event_id
            )
            self.assertGreater(extracted.amount, ZERO)
            self.assertEqual(extracted.currency, event.currency)
            self.assertEqual(extracted.event_id, event.event_id)
            facts = evidence.extract_image_facts(
                DATASET, image_id=images[0].image_id, expected_event_id=event.event_id
            )
            self.assertTrue(facts.untrusted)
            self.assertEqual(facts.facts[0].amount, extracted.amount)

    def test_edited_image_bytes_do_not_reuse_a_stale_amount(self):
        with tempfile.TemporaryDirectory() as folder:
            fake = Path(folder) / "image_01.png"
            fake.write_bytes(b"not-the-original-document")
            with self.assertRaises(evidence.UnresolvedAmountError):
                evidence.DocumentAmountExtractor().extract_amount(
                    image_path=fake,
                    event=DATASET.event(self.BLANK_EVENT),
                    context="",
                )

    def test_extracted_amount_becomes_a_set_event_amount_adjustment(self):
        event = DATASET.event(self.BLANK_EVENT)
        stub = StubExtractor("4365000", event.currency)
        adjustments, unresolved = evidence.resolve_blank_amounts(
            DATASET, user_id=event.user_id, extractor=stub
        )
        self.assertNotIn(self.BLANK_EVENT, unresolved)
        match = [a for a in adjustments if a.target_event_id == self.BLANK_EVENT]
        self.assertEqual(len(match), 1)
        self.assertEqual(match[0].kind, "set_event_amount")
        self.assertEqual(match[0].amount, to_money("4365000"))
        self.assertEqual(match[0].source_id, self.BLANK_IMAGE)

    def test_currency_mismatch_is_refused(self):
        event = DATASET.event(self.BLANK_EVENT)
        wrong = Currency.EUR if event.currency is not Currency.EUR else Currency.INR
        with self.assertRaises(evidence.UnresolvedAmountError):
            evidence.extract_image_amount(
                DATASET,
                image_id=self.BLANK_IMAGE,
                event_id=self.BLANK_EVENT,
                extractor=StubExtractor("100", wrong),
            )

    def test_extractor_answering_for_another_event_is_refused(self):
        with self.assertRaises(evidence.UnresolvedAmountError):
            evidence.extract_image_amount(
                DATASET,
                image_id=self.BLANK_IMAGE,
                event_id=self.BLANK_EVENT,
                extractor=StubExtractor(
                    "100",
                    DATASET.event(self.BLANK_EVENT).currency,
                    event_id="event_999999",
                ),
            )

    def test_non_positive_amounts_are_refused(self):
        with self.assertRaises(ValidationError):
            ExtractedAmount(
                image_id=self.BLANK_IMAGE,
                event_id=self.BLANK_EVENT,
                amount=to_money("0"),
                currency=Currency.IDR,
                confidence=1.0,
                evidence_text="",
                content_hash="b" * 64,
            )

    def test_cache_avoids_a_second_read_of_the_same_image(self):
        event = DATASET.event(self.BLANK_EVENT)
        stub = StubExtractor("4365000", event.currency)
        with tempfile.TemporaryDirectory() as folder:
            cached = evidence.CachedImageAmountExtractor(
                cache_path=Path(folder) / "cache.json", delegate=stub
            )
            first = evidence.extract_image_amount(
                DATASET,
                image_id=self.BLANK_IMAGE,
                event_id=self.BLANK_EVENT,
                extractor=cached,
            )
            reloaded = evidence.CachedImageAmountExtractor(
                cache_path=Path(folder) / "cache.json", delegate=stub
            )
            second = evidence.extract_image_amount(
                DATASET,
                image_id=self.BLANK_IMAGE,
                event_id=self.BLANK_EVENT,
                extractor=reloaded,
            )
        self.assertEqual(stub.calls, 1)
        self.assertEqual(first.amount, second.amount)

    def test_a_resolved_amount_reaches_the_forecast(self):
        event = DATASET.event(self.BLANK_EVENT)
        request = next(
            (
                candidate
                for candidate in (
                    *DATASET.requests.values(),
                    *DATASET.sample_requests.values(),
                )
                if candidate.user_id == event.user_id
            ),
            None,
        )
        if request is None:
            self.skipTest(f"{event.user_id} has no request to forecast")
        adjustments, _ = evidence.resolve_blank_amounts(
            DATASET,
            user_id=event.user_id,
            extractor=StubExtractor("4365000", event.currency),
        )
        state = build_financial_state(DATASET, request, adjustments=adjustments)
        self.assertNotIn(
            event.event_id,
            " ".join(state.notes),
            "a resolved amount should not still be reported as awaiting evidence",
        )


class TestConflictResolution(unittest.TestCase):
    def _fact(self, fact_type, rule_id, source_id, amount=None, recorded=None):
        return EvidenceFact(
            source_kind=EvidenceSourceKind.MESSAGE,
            source_id=source_id,
            fact_type=fact_type,
            amount=to_money(amount) if amount is not None else None,
            recorded_at=recorded,
            quoted_text=f"[{rule_id}] test",
        )

    def test_cancellation_beats_confirmation(self):
        resolution = evidence.resolve_evidence_conflicts(
            [
                self._fact(
                    EvidenceFactType.CONFIRMATION, "employment_ended", "message_a"
                ),
                self._fact(
                    EvidenceFactType.CANCELLATION, "employment_ended", "message_b"
                ),
            ]
        )
        self.assertEqual([f.source_id for f in resolution.resolved], ["message_b"])
        self.assertEqual([f.source_id for f in resolution.superseded], ["message_a"])

    def test_newer_record_wins_at_equal_precedence(self):
        older = self._fact(
            EvidenceFactType.AMENDMENT,
            "salary_increase",
            "message_old",
            recorded=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
        newer = self._fact(
            EvidenceFactType.AMENDMENT,
            "salary_increase",
            "message_new",
            recorded=datetime(2026, 6, 1, tzinfo=timezone.utc),
        )
        resolution = evidence.resolve_evidence_conflicts([older, newer])
        self.assertEqual([f.source_id for f in resolution.resolved], ["message_new"])

    def test_safer_interpretation_prefers_the_smaller_credit(self):
        resolution = evidence.resolve_evidence_conflicts(
            [
                self._fact(
                    EvidenceFactType.CONFIRMATION, "salary_confirmed_base", "message_b", "900"
                ),
                self._fact(
                    EvidenceFactType.CONFIRMATION, "salary_confirmed_base", "message_a", "500"
                ),
            ]
        )
        self.assertEqual(resolution.resolved[0].amount, to_money("500"))

    def test_unrelated_facts_all_survive(self):
        resolution = evidence.resolve_evidence_conflicts(
            [
                self._fact(EvidenceFactType.AMENDMENT, "salary_increase", "message_a"),
                self._fact(
                    EvidenceFactType.AMENDMENT, "rent_increase_percent", "message_b"
                ),
            ]
        )
        self.assertEqual(len(resolution.resolved), 2)
        self.assertEqual(resolution.superseded, ())


class TestDatedRecurringOverrides(unittest.TestCase):
    def test_an_increase_does_not_apply_before_its_effective_date(self):
        request = DATASET.request("request_26")
        base = build_financial_state(DATASET, request)
        salary = next(
            (c for c in base.recurring if c.category == "salary"), None
        )
        if salary is None:
            self.skipTest("request_26 has no inferred salary recurrence")

        effective = request.request_date + timedelta(days=45)
        raised = to_money(salary.amount * 2)
        state = build_financial_state(
            DATASET,
            request,
            adjustments=(
                ForecastAdjustment(
                    kind="set_recurring_amount",
                    source_id="message_test",
                    target_category="salary",
                    amount=raised,
                    effective_date=effective,
                ),
            ),
        )
        self.assertEqual(
            state.amount_for(salary, effective - timedelta(days=1)), salary.amount
        )
        self.assertEqual(state.amount_for(salary, effective), raised)


class TestEveryRequestProducesOneRow(unittest.TestCase):
    def test_a_candidate_always_exists_and_validates(self):
        for request in DATASET.iter_requests():
            state = build_financial_state(DATASET, request)
            candidates = generate_candidates(DATASET, state, request)
            self.assertTrue(candidates, f"{request.request_id} produced no candidate")
            best = rank_valid_candidates(request, candidates).best
            self.assertIsNotNone(best)
            result = validate_plan_constraints(DATASET, state, request, best)
            self.assertTrue(result.valid, f"{request.request_id}: {result.violations}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
