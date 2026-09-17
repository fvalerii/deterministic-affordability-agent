"""Typed domain models for the Buy or Wait? financial agent.

Every model here is frozen and forbids unknown fields, so a tool result can be
passed between tools (and across the LLM boundary) without any component being
able to mutate shared financial state.

Money is always ``Decimal`` quantized to two places. Money crosses the model
boundary to the LLM as a string, never as a float, so binary rounding can never
change a payment amount. Dates are ``datetime.date`` and serialize as
``YYYY-MM-DD``.

Two distinct output formats are required by the challenge and are both derived
from ``dataset/sample_requests.csv``:

* ``amount_safe_to_pay`` uses the minimal representation (``603.3``, ``25256``).
* ``payment_plan`` and ``reduce_to`` amounts use integer-or-two-decimals
  (``25256``, ``620.40``, ``23.50``).
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from enum import StrEnum
from typing import Annotated, Any

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PlainSerializer,
    computed_field,
    model_validator,
)

# --------------------------------------------------------------------------- #
# Challenge constants
# --------------------------------------------------------------------------- #

FORECAST_HORIZON_DAYS = 90
MONEY_QUANTUM = Decimal("0.01")
ZERO = Decimal("0.00")

OUTPUT_COLUMNS: tuple[str, ...] = (
    "request_id",
    "amount_safe_to_pay",
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
    "decision_explanation",
)

NO_PLAN = "none"
NO_SPENDING_CHANGES = "none"
MAX_SPENDING_CHANGES = 3


# --------------------------------------------------------------------------- #
# Scalar parsing and formatting
# --------------------------------------------------------------------------- #


def parse_decimal(value: Any) -> Decimal:
    """Parse a CSV/JSON scalar into an exact Decimal without float artifacts."""
    if isinstance(value, Decimal):
        parsed = value
    elif isinstance(value, bool):
        raise ValueError("boolean is not a numeric amount")
    elif isinstance(value, int):
        parsed = Decimal(value)
    elif isinstance(value, float):
        parsed = Decimal(str(value))
    elif isinstance(value, str):
        text = value.strip().replace(",", "")
        if not text:
            raise ValueError("empty string is not a numeric amount")
        try:
            parsed = Decimal(text)
        except InvalidOperation as exc:
            raise ValueError(f"{value!r} is not a decimal number") from exc
    else:
        raise ValueError(f"unsupported numeric type {type(value).__name__}")
    if not parsed.is_finite():
        raise ValueError(f"{value!r} is not a finite number")
    return parsed


def to_money(value: Any) -> Decimal:
    return parse_decimal(value).quantize(MONEY_QUANTUM, rounding=ROUND_HALF_UP)


def _to_optional_money(value: Any) -> Decimal | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    return to_money(value)


def _to_rate(value: Any) -> Decimal:
    rate = parse_decimal(value)
    if rate <= 0:
        raise ValueError("exchange rate must be positive")
    return rate


def _to_date(value: Any) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, str):
        text = value.strip()
        try:
            return date.fromisoformat(text)
        except ValueError as exc:
            raise ValueError(f"{value!r} is not a YYYY-MM-DD date") from exc
    raise ValueError(f"unsupported date type {type(value).__name__}")


def _to_optional_date(value: Any) -> date | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    return _to_date(value)


def _to_timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        text = value.strip().replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(text)
        except ValueError as exc:
            raise ValueError(f"{value!r} is not an ISO-8601 timestamp") from exc
    raise ValueError(f"unsupported timestamp type {type(value).__name__}")


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "yes", "1"}:
            return True
        if text in {"false", "no", "0"}:
            return False
    raise ValueError(f"{value!r} is not a boolean")


def _to_pipe_list(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(part.strip() for part in value.split("|") if part.strip())
    if isinstance(value, (list, tuple)):
        return tuple(str(part).strip() for part in value if str(part).strip())
    raise ValueError(f"unsupported pipe-delimited type {type(value).__name__}")


def _to_optional_int(value: Any) -> int | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, bool):
        raise ValueError("boolean is not an integer")
    if isinstance(value, int):
        return value
    return int(str(value).strip())


def format_amount_field(amount: Decimal) -> str:
    """Render ``amount_safe_to_pay``: minimal digits, no trailing zeros."""
    normalized = to_money(amount).normalize()
    if normalized == 0:
        return "0"
    return format(normalized, "f")


def format_plan_amount(amount: Decimal) -> str:
    """Render a ``payment_plan`` / ``reduce_to`` amount: integer or 2 decimals."""
    value = to_money(amount)
    if value == value.to_integral_value():
        return format(value.quantize(Decimal("1")), "f")
    return format(value, "f")


Money = Annotated[
    Decimal,
    BeforeValidator(to_money),
    PlainSerializer(format_amount_field, return_type=str, when_used="json"),
]
OptionalMoney = Annotated[
    Decimal | None,
    BeforeValidator(_to_optional_money),
    PlainSerializer(
        lambda v: None if v is None else format_amount_field(v),
        return_type=str | None,
        when_used="json",
    ),
]
Rate = Annotated[
    Decimal,
    BeforeValidator(_to_rate),
    PlainSerializer(lambda v: format(v, "f"), return_type=str, when_used="json"),
]
IsoDate = Annotated[
    date,
    BeforeValidator(_to_date),
    PlainSerializer(lambda v: v.isoformat(), return_type=str, when_used="json"),
]
OptionalIsoDate = Annotated[
    date | None,
    BeforeValidator(_to_optional_date),
    PlainSerializer(
        lambda v: None if v is None else v.isoformat(),
        return_type=str | None,
        when_used="json",
    ),
]
Timestamp = Annotated[
    datetime,
    BeforeValidator(_to_timestamp),
    PlainSerializer(lambda v: v.isoformat(), return_type=str, when_used="json"),
]
PipeList = Annotated[tuple[str, ...], BeforeValidator(_to_pipe_list)]
CsvBool = Annotated[bool, BeforeValidator(_to_bool)]
OptionalInt = Annotated[int | None, BeforeValidator(_to_optional_int)]


# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #


class RequestType(StrEnum):
    PURCHASE = "purchase"
    TRAVEL = "travel"
    EDUCATION = "education"
    FAMILY_TRANSFER = "family_transfer"
    DEBT_REPAYMENT = "debt_repayment"
    INVESTMENT = "investment"
    HOUSING = "housing"
    EMERGENCY_EXPENSE = "emergency_expense"
    OTHER = "other"


class Currency(StrEnum):
    INR = "INR"
    ZAR = "ZAR"
    IDR = "IDR"
    USD = "USD"
    EUR = "EUR"


class EventType(StrEnum):
    EXPENSE = "expense"
    SUBSCRIPTION = "subscription"
    INCOME = "income"
    DEBT_PAYMENT = "debt_payment"
    REFUND = "refund"
    INVESTMENT_PURCHASE = "investment_purchase"
    INVESTMENT_SALE = "investment_sale"
    INVESTMENT_VALUATION = "investment_valuation"


class Direction(StrEnum):
    DEBIT = "debit"
    CREDIT = "credit"
    NON_CASH = "non_cash"


class EventStatus(StrEnum):
    SETTLED = "settled"
    PENDING = "pending"
    SCHEDULED = "scheduled"
    CANCELLED = "cancelled"
    FAILED = "failed"
    UNREALIZED = "unrealized"


class Flexibility(StrEnum):
    FIXED = "fixed"
    REDUCIBLE = "reducible"
    STOPPABLE = "stoppable"
    REDUCIBLE_OR_STOPPABLE = "reducible_or_stoppable"

    @property
    def can_reduce(self) -> bool:
        return self in {Flexibility.REDUCIBLE, Flexibility.REDUCIBLE_OR_STOPPABLE}

    @property
    def can_stop(self) -> bool:
        return self in {Flexibility.STOPPABLE, Flexibility.REDUCIBLE_OR_STOPPABLE}


class PaymentMethod(StrEnum):
    """Methods a user may accept and that payment options may offer."""

    FULL_PAYMENT = "full_payment"
    PARTIAL_PAYMENT = "partial_payment"
    INSTALLMENTS = "installments"


class RecommendedPaymentMethod(StrEnum):
    FULL_PAYMENT = "full_payment"
    PARTIAL_PAYMENT = "partial_payment"
    INSTALLMENTS = "installments"
    WAIT = "wait"
    NOT_RECOMMENDED = "not_recommended"


class AffordabilityStatus(StrEnum):
    AFFORDABLE_NOW = "affordable_now"
    AFFORDABLE_WITH_PLAN = "affordable_with_plan"
    AFFORDABLE_LATER = "affordable_later"
    NOT_AFFORDABLE = "not_affordable"


class SpendingChangeAction(StrEnum):
    STOP = "stop"
    REDUCE_TO = "reduce_to"


class EvidenceSourceKind(StrEnum):
    MESSAGE = "message"
    IMAGE = "image"


class EvidenceFactType(StrEnum):
    AMOUNT = "amount"
    AMENDMENT = "amendment"
    CANCELLATION = "cancellation"
    DELAY = "delay"
    CONFIRMATION = "confirmation"
    UNCERTAIN = "uncertain"


class CashState(StrEnum):
    """How a normalized event affects the forecastable cash position."""

    SETTLED_HISTORY = "settled_history"
    RESERVED_DEBIT = "reserved_debit"
    CONFIRMED_CREDIT = "confirmed_credit"
    EXCLUDED = "excluded"


# --------------------------------------------------------------------------- #
# Base model
# --------------------------------------------------------------------------- #


class Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)


# --------------------------------------------------------------------------- #
# Dataset records
# --------------------------------------------------------------------------- #


class RequestRecord(Frozen):
    """One row of ``requests.csv`` or the input columns of ``sample_requests.csv``."""

    request_id: str
    user_id: str
    request_date: IsoDate
    request_type: RequestType
    requested_amount: Money
    desired_completion_date: IsoDate
    allows_partial_payment: CsvBool
    request_text: str

    @model_validator(mode="after")
    def _check(self) -> RequestRecord:
        if self.requested_amount <= 0:
            raise ValueError(f"{self.request_id}: requested_amount must be positive")
        if self.desired_completion_date < self.request_date:
            raise ValueError(
                f"{self.request_id}: desired_completion_date precedes request_date"
            )
        return self

    @property
    def forecast_end(self) -> date:
        return self.request_date + timedelta(days=FORECAST_HORIZON_DAYS)


class ProfileRecord(Frozen):
    """One row of ``financial_profiles.csv``.

    ``financial_priorities``, ``expense_categories_to_protect``,
    ``expense_categories_user_is_willing_to_reduce``, and
    ``expense_categories_user_is_willing_to_stop`` are pipe-delimited in the
    CSV and parsed into tuples.
    """

    user_id: str
    home_currency: Currency
    current_available_balance: Money
    minimum_balance_to_keep: Money
    financial_priorities: PipeList
    expense_categories_to_protect: PipeList
    expense_categories_user_is_willing_to_reduce: PipeList
    expense_categories_user_is_willing_to_stop: PipeList
    payment_methods_user_will_consider: tuple[PaymentMethod, ...]
    max_installment_months: OptionalInt

    @model_validator(mode="before")
    @classmethod
    def _split_methods(cls, data: Any) -> Any:
        if isinstance(data, dict) and isinstance(
            data.get("payment_methods_user_will_consider"), str
        ):
            data = dict(data)
            data["payment_methods_user_will_consider"] = _to_pipe_list(
                data["payment_methods_user_will_consider"]
            )
        return data

    @model_validator(mode="after")
    def _check(self) -> ProfileRecord:
        if self.minimum_balance_to_keep < 0:
            raise ValueError(f"{self.user_id}: minimum_balance_to_keep is negative")
        if not self.payment_methods_user_will_consider:
            raise ValueError(f"{self.user_id}: no acceptable payment methods")
        if self.max_installment_months is not None and self.max_installment_months <= 0:
            raise ValueError(f"{self.user_id}: max_installment_months must be positive")
        return self

    def accepts_method(self, method: PaymentMethod) -> bool:
        return method in self.payment_methods_user_will_consider

    @property
    def considers_installments(self) -> bool:
        # A blank max_installment_months means installments are off the table
        # regardless of anything else, so both signals must agree.
        return (
            self.accepts_method(PaymentMethod.INSTALLMENTS)
            and self.max_installment_months is not None
        )

    def is_protected_category(self, category: str) -> bool:
        return category in self.expense_categories_to_protect

    def may_reduce_category(self, category: str) -> bool:
        return not self.is_protected_category(category) and (
            category in self.expense_categories_user_is_willing_to_reduce
        )

    def may_stop_category(self, category: str) -> bool:
        return not self.is_protected_category(category) and (
            category in self.expense_categories_user_is_willing_to_stop
        )

    def has_priority(self, name: str) -> bool:
        return name in self.financial_priorities


class EventRecord(Frozen):
    """One row of ``financial_events.csv``.

    ``amount`` is ``None`` when the source row is blank. A blank amount is never
    zero: it must be resolved from the linked image before the event is used.
    """

    event_id: str
    user_id: str
    event_type: EventType
    description: str
    category: str
    direction: Direction
    amount: OptionalMoney
    currency: Currency
    event_date: IsoDate
    settlement_date: OptionalIsoDate
    status: EventStatus
    linked_event_id: str | None = None
    flexibility: Flexibility
    minimum_allowed_amount: OptionalMoney = None

    @model_validator(mode="before")
    @classmethod
    def _blank_link(cls, data: Any) -> Any:
        if isinstance(data, dict):
            link = data.get("linked_event_id")
            if isinstance(link, str) and not link.strip():
                data = dict(data)
                data["linked_event_id"] = None
        return data

    @model_validator(mode="after")
    def _check(self) -> EventRecord:
        if self.amount is not None and self.amount < 0:
            raise ValueError(f"{self.event_id}: amount must not be negative")
        if self.linked_event_id == self.event_id:
            raise ValueError(f"{self.event_id}: linked_event_id points at itself")
        if self.settlement_date is None and self.direction is not Direction.NON_CASH:
            raise ValueError(f"{self.event_id}: cash event has no settlement_date")
        return self

    @property
    def amount_needs_resolution(self) -> bool:
        return self.amount is None

    @property
    def is_cash(self) -> bool:
        return self.direction is not Direction.NON_CASH

    @property
    def cash_state(self) -> CashState:
        """Classify the row's effect on forecastable cash.

        Pending debits are reserved; pending credits are not counted until they
        settle. Failed, cancelled and unrealized rows never move cash.
        """
        if self.status in {
            EventStatus.FAILED,
            EventStatus.CANCELLED,
            EventStatus.UNREALIZED,
        }:
            return CashState.EXCLUDED
        if not self.is_cash:
            return CashState.EXCLUDED
        if self.status is EventStatus.SETTLED:
            return CashState.SETTLED_HISTORY
        if self.direction is Direction.DEBIT:
            return CashState.RESERVED_DEBIT
        return (
            CashState.CONFIRMED_CREDIT
            if self.status is EventStatus.SCHEDULED
            else CashState.EXCLUDED
        )

    @property
    def effective_date(self) -> date:
        return self.settlement_date or self.event_date


class PaymentOptionRecord(Frozen):
    """One row of ``request_payment_options.csv``."""

    payment_option_id: str
    request_id: str
    payment_method: PaymentMethod
    payment_amount: Money
    number_of_payments: int
    first_payment_date: IsoDate
    payment_frequency_days: OptionalInt
    financing_fee: Money
    total_payable_amount: Money

    @model_validator(mode="after")
    def _check(self) -> PaymentOptionRecord:
        if self.number_of_payments < 1:
            raise ValueError(f"{self.payment_option_id}: number_of_payments < 1")
        if self.payment_amount <= 0:
            raise ValueError(f"{self.payment_option_id}: payment_amount must be positive")
        expected = to_money(self.payment_amount * self.number_of_payments)
        if expected != self.total_payable_amount:
            raise ValueError(
                f"{self.payment_option_id}: payment_amount x number_of_payments "
                f"({expected}) != total_payable_amount ({self.total_payable_amount})"
            )
        if self.number_of_payments > 1 and not self.payment_frequency_days:
            raise ValueError(
                f"{self.payment_option_id}: multi-payment option has no frequency"
            )
        if self.payment_method is PaymentMethod.FULL_PAYMENT and self.number_of_payments != 1:
            raise ValueError(
                f"{self.payment_option_id}: full_payment option must have one payment"
            )
        return self

    @property
    def payment_dates(self) -> tuple[date, ...]:
        step = self.payment_frequency_days or 0
        return tuple(
            self.first_payment_date + timedelta(days=step * index)
            for index in range(self.number_of_payments)
        )

    @property
    def schedule(self) -> tuple[Payment, ...]:
        return tuple(
            Payment(date=due, amount=self.payment_amount) for due in self.payment_dates
        )

    @property
    def last_payment_date(self) -> date:
        return self.payment_dates[-1]


class MessageRecord(Frozen):
    """One row of ``messages.csv``. Content is untrusted evidence."""

    message_id: str
    user_id: str
    request_id: str | None = None
    related_event_id: str | None = None
    sent_at: Timestamp
    source_type: str
    message_text: str

    @model_validator(mode="before")
    @classmethod
    def _blank_links(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = dict(data)
            for key in ("request_id", "related_event_id"):
                value = data.get(key)
                if isinstance(value, str) and not value.strip():
                    data[key] = None
        return data


class ImageRecord(Frozen):
    """One row of ``images.csv``. Content is untrusted evidence."""

    image_id: str
    user_id: str
    request_id: str | None = None
    related_event_id: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _blank_links(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = dict(data)
            for key in ("request_id", "related_event_id"):
                value = data.get(key)
                if isinstance(value, str) and not value.strip():
                    data[key] = None
        return data


class ExchangeRateRecord(Frozen):
    """One row of ``exchange_rates.csv``. Rates are directional."""

    rate_date: IsoDate
    from_currency: Currency
    to_currency: Currency
    rate: Rate

    @model_validator(mode="after")
    def _check(self) -> ExchangeRateRecord:
        if self.from_currency == self.to_currency:
            raise ValueError("exchange rate maps a currency to itself")
        return self

    @property
    def key(self) -> tuple[date, Currency, Currency]:
        return (self.rate_date, self.from_currency, self.to_currency)


# --------------------------------------------------------------------------- #
# Evidence contracts
# --------------------------------------------------------------------------- #


class EvidenceFact(Frozen):
    """A single candidate fact extracted from untrusted evidence."""

    source_kind: EvidenceSourceKind
    source_id: str
    fact_type: EvidenceFactType
    subject_event_id: str | None = None
    subject_user_id: str | None = None
    subject_request_id: str | None = None
    amount: OptionalMoney = None
    currency: Currency | None = None
    effective_date: OptionalIsoDate = None
    recorded_at: Timestamp | None = None
    source_type: str | None = None
    quoted_text: str = ""
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)


class EvidenceFactSet(Frozen):
    """Facts read from one evidence source, tagged as untrusted."""

    source_kind: EvidenceSourceKind
    source_id: str
    facts: tuple[EvidenceFact, ...] = ()
    content_hash: str
    untrusted: bool = True

    @model_validator(mode="after")
    def _check(self) -> EvidenceFactSet:
        if not self.untrusted:
            raise ValueError("evidence may never be marked trusted")
        return self


class EvidenceIndex(Frozen):
    """Which evidence exists for a user/request, without reading its content."""

    user_id: str
    request_id: str | None = None
    message_ids: tuple[str, ...] = ()
    image_ids: tuple[str, ...] = ()
    message_ids_by_event: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    image_ids_by_event: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    events_missing_amount: tuple[str, ...] = ()


class ExtractedAmount(Frozen):
    """A resolved amount for an event whose dataset amount was blank."""

    image_id: str
    event_id: str
    amount: Money
    currency: Currency
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_text: str
    content_hash: str

    @model_validator(mode="after")
    def _check(self) -> ExtractedAmount:
        if self.amount <= 0:
            raise ValueError(
                f"{self.image_id}: extracted amount must be positive; a blank "
                "dataset amount is never zero"
            )
        return self


class ConvertedAmount(Frozen):
    """Result of one directional, dated currency conversion."""

    original_amount: Money
    converted_amount: Money
    from_currency: Currency
    to_currency: Currency
    rate: Rate
    rate_date: IsoDate


class ConflictResolution(Frozen):
    """Which candidate facts survived conflict resolution, and why."""

    resolved: tuple[EvidenceFact, ...] = ()
    superseded: tuple[EvidenceFact, ...] = ()
    rule_applied: str
    notes: tuple[str, ...] = ()


class LifecycleResolution(Frozen):
    """Which linked events count toward cash flow after lifecycle resolution."""

    user_id: str
    as_of_date: IsoDate
    counted_event_ids: tuple[str, ...] = ()
    excluded_event_ids: tuple[str, ...] = ()
    exclusion_reasons: dict[str, str] = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Reconstructed financial state
# --------------------------------------------------------------------------- #


class RecurringCommitment(Frozen):
    """A recurrence inferred from repeated history, never assumed."""

    user_id: str
    label: str
    category: str
    direction: Direction
    amount: Money
    frequency_days: int
    next_occurrence: IsoDate
    flexibility: Flexibility
    minimum_allowed_amount: OptionalMoney = None
    representative_event_id: str
    source_event_ids: tuple[str, ...]
    occurrences_observed: int

    @model_validator(mode="after")
    def _check(self) -> RecurringCommitment:
        if self.frequency_days < 1:
            raise ValueError(f"{self.label}: frequency_days must be positive")
        if self.occurrences_observed < 2:
            raise ValueError(
                f"{self.label}: recurrence needs at least two observed occurrences"
            )
        if self.minimum_allowed_amount is not None and (
            self.minimum_allowed_amount > self.amount
        ):
            raise ValueError(f"{self.label}: minimum_allowed_amount exceeds amount")
        return self


class EssentialSpendingBaseline(Frozen):
    """Conservative forecast of essential variable spending."""

    user_id: str
    as_of_date: IsoDate
    horizon_days: int
    per_category_monthly: dict[str, Money] = Field(default_factory=dict)
    total_monthly: Money
    source_event_ids: tuple[str, ...] = ()
    method: str


# --------------------------------------------------------------------------- #
# Plans, simulation and validation
# --------------------------------------------------------------------------- #


class Payment(Frozen):
    date: IsoDate
    amount: Money

    @model_validator(mode="after")
    def _check(self) -> Payment:
        if self.amount <= 0:
            raise ValueError("a payment amount must be positive")
        return self

    def render(self) -> str:
        return f"{self.date.isoformat()}:{format_plan_amount(self.amount)}"


class SpendingChange(Frozen):
    action: SpendingChangeAction
    event_id: str
    new_amount: OptionalMoney = None

    @model_validator(mode="after")
    def _check(self) -> SpendingChange:
        if self.action is SpendingChangeAction.STOP and self.new_amount is not None:
            raise ValueError(f"stop:{self.event_id} must not carry an amount")
        if self.action is SpendingChangeAction.REDUCE_TO:
            if self.new_amount is None:
                raise ValueError(f"reduce_to:{self.event_id} requires a new amount")
            if self.new_amount < 0:
                raise ValueError(f"reduce_to:{self.event_id} amount is negative")
        return self

    def render(self) -> str:
        if self.action is SpendingChangeAction.STOP:
            return f"stop:{self.event_id}"
        return f"reduce_to:{self.event_id}:{format_plan_amount(self.new_amount)}"


class LedgerEntry(Frozen):
    """One dated cash movement in a simulated forecast."""

    date: IsoDate
    label: str
    direction: Direction
    amount: Money
    balance_after: Money
    source: str


class SafetyViolation(Frozen):
    date: OptionalIsoDate = None
    reason: str
    balance: OptionalMoney = None
    shortfall: OptionalMoney = None


class SimulationResult(Frozen):
    """Outcome of simulating a plan across the 90-day forecast."""

    request_id: str
    user_id: str
    opening_balance: Money
    minimum_balance_to_keep: Money
    horizon_start: IsoDate
    horizon_end: IsoDate
    entries: tuple[LedgerEntry, ...] = ()
    minimum_projected_balance: Money
    minimum_balance_date: OptionalIsoDate = None
    all_payments_completed: bool
    completion_date: OptionalIsoDate = None
    violations: tuple[SafetyViolation, ...] = ()

    @computed_field  # type: ignore[prop-decorator]
    @property
    def minimum_headroom(self) -> Money:
        return to_money(self.minimum_projected_balance - self.minimum_balance_to_keep)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def safe(self) -> bool:
        return not self.violations


class SafeAmountResult(Frozen):
    """Largest amount safe to pay on ``request_date``, before spending changes."""

    request_id: str
    requested_amount: Money
    amount_safe_to_pay: Money
    limiting_date: OptionalIsoDate = None
    limiting_balance: OptionalMoney = None
    minimum_balance_to_keep: Money
    method: str

    @model_validator(mode="after")
    def _check(self) -> SafeAmountResult:
        if not ZERO <= self.amount_safe_to_pay <= self.requested_amount:
            raise ValueError(
                f"{self.request_id}: amount_safe_to_pay {self.amount_safe_to_pay} "
                f"outside [0, {self.requested_amount}]"
            )
        return self


class EarliestSafeDateResult(Frozen):
    """First date one safe full payment is possible, ignoring preferences."""

    request_id: str
    earliest_date: OptionalIsoDate = None
    forecast_horizon_end: IsoDate
    dates_checked: int = 0
    blocking_reason: str | None = None

    @model_validator(mode="after")
    def _check(self) -> EarliestSafeDateResult:
        if self.earliest_date is not None and self.earliest_date > self.forecast_horizon_end:
            raise ValueError(
                f"{self.request_id}: earliest_date is beyond the forecast horizon"
            )
        return self


class PreferenceCheck(Frozen):
    user_id: str
    method: PaymentMethod
    eligible: bool
    max_installment_months: OptionalInt = None
    rejection_reasons: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _check(self) -> PreferenceCheck:
        if self.eligible and self.rejection_reasons:
            raise ValueError("an eligible method must not carry rejection reasons")
        if not self.eligible and not self.rejection_reasons:
            raise ValueError("an ineligible method must state a reason")
        return self


class CandidatePlan(Frozen):
    """A proposed way to satisfy a request, before validation and ranking."""

    request_id: str
    method: RecommendedPaymentMethod
    affordability_status: AffordabilityStatus
    payments: tuple[Payment, ...] = ()
    spending_changes: tuple[SpendingChange, ...] = ()
    payment_option_id: str | None = None
    total_payable: Money = ZERO
    completes_request: bool = False
    completes_by_deadline: bool = False
    notes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _check(self) -> CandidatePlan:
        dates = [payment.date for payment in self.payments]
        if dates != sorted(dates):
            raise ValueError(f"{self.request_id}: payments are not chronological")
        if len(self.spending_changes) > MAX_SPENDING_CHANGES:
            raise ValueError(
                f"{self.request_id}: more than {MAX_SPENDING_CHANGES} spending changes"
            )
        targets = [change.event_id for change in self.spending_changes]
        if len(targets) != len(set(targets)):
            raise ValueError(
                f"{self.request_id}: stop and reduce_to target the same event"
            )
        if self.method is RecommendedPaymentMethod.INSTALLMENTS and not self.payment_option_id:
            raise ValueError(
                f"{self.request_id}: installments must reference a payment option"
            )
        if self.method is RecommendedPaymentMethod.NOT_RECOMMENDED and self.payments:
            raise ValueError(f"{self.request_id}: not_recommended must have no payments")
        return self

    @property
    def payment_total(self) -> Decimal:
        return to_money(sum((payment.amount for payment in self.payments), ZERO))

    @property
    def first_payment_date(self) -> date | None:
        return self.payments[0].date if self.payments else None

    @property
    def last_payment_date(self) -> date | None:
        return self.payments[-1].date if self.payments else None

    def render_payment_plan(self) -> str:
        if not self.payments:
            return NO_PLAN
        return "|".join(payment.render() for payment in self.payments)

    def render_spending_changes(self) -> str:
        if not self.spending_changes:
            return NO_SPENDING_CHANGES
        return "|".join(change.render() for change in self.spending_changes)


class ValidationResult(Frozen):
    """Deterministic verdict on one candidate plan."""

    request_id: str
    checks_performed: tuple[str, ...] = ()
    violations: tuple[str, ...] = ()
    simulation: SimulationResult | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def valid(self) -> bool:
        return not self.violations


class RankedCandidates(Frozen):
    """Valid candidates in official preference order."""

    request_id: str
    ordered: tuple[CandidatePlan, ...] = ()
    rejected: tuple[CandidatePlan, ...] = ()
    ranking_notes: tuple[str, ...] = ()

    @property
    def best(self) -> CandidatePlan | None:
        return self.ordered[0] if self.ordered else None


class FinalDecision(Frozen):
    """An accepted, validated recommendation ready to be written to output.csv.

    ``validation`` must already be valid, so an unvalidated plan can never be
    turned into an output row.
    """

    request_id: str
    request_date: IsoDate
    requested_amount: Money
    amount_safe_to_pay: Money
    affordability_status: AffordabilityStatus
    recommended_payment_method: RecommendedPaymentMethod
    payments: tuple[Payment, ...] = ()
    earliest_date_for_full_payment: OptionalIsoDate = None
    spending_changes: tuple[SpendingChange, ...] = ()
    decision_explanation: str
    validation: ValidationResult

    @model_validator(mode="after")
    def _check(self) -> FinalDecision:
        rid = self.request_id
        if not self.validation.valid:
            raise ValueError(f"{rid}: cannot finalize a plan that failed validation")
        if self.validation.request_id != rid:
            raise ValueError(f"{rid}: validation belongs to {self.validation.request_id}")
        if not ZERO <= self.amount_safe_to_pay <= self.requested_amount:
            raise ValueError(
                f"{rid}: amount_safe_to_pay {self.amount_safe_to_pay} outside "
                f"[0, {self.requested_amount}]"
            )
        if not self.decision_explanation.strip():
            raise ValueError(f"{rid}: decision_explanation is empty")

        dates = [payment.date for payment in self.payments]
        if dates != sorted(dates):
            raise ValueError(f"{rid}: payment_plan is not chronological")
        if len(self.spending_changes) > MAX_SPENDING_CHANGES:
            raise ValueError(f"{rid}: more than {MAX_SPENDING_CHANGES} spending changes")
        targets = [change.event_id for change in self.spending_changes]
        if len(targets) != len(set(targets)):
            raise ValueError(f"{rid}: stop and reduce_to target the same event")

        status = self.affordability_status
        method = self.recommended_payment_method

        if status is AffordabilityStatus.AFFORDABLE_NOW:
            if self.earliest_date_for_full_payment != self.request_date:
                raise ValueError(
                    f"{rid}: affordable_now requires earliest_date_for_full_payment "
                    "to equal request_date"
                )
            if method is not RecommendedPaymentMethod.FULL_PAYMENT:
                raise ValueError(f"{rid}: affordable_now requires full_payment")

        if method is RecommendedPaymentMethod.NOT_RECOMMENDED:
            if self.payments or self.spending_changes:
                raise ValueError(
                    f"{rid}: not_recommended must have no payments or spending changes"
                )
        elif not self.payments:
            raise ValueError(f"{rid}: {method} requires at least one payment")

        if method is RecommendedPaymentMethod.PARTIAL_PAYMENT:
            if status is not AffordabilityStatus.AFFORDABLE_WITH_PLAN:
                raise ValueError(f"{rid}: partial_payment requires affordable_with_plan")
            if len(self.payments) != 2:
                raise ValueError(f"{rid}: partial_payment requires exactly two payments")
            if not ZERO < self.amount_safe_to_pay < self.requested_amount:
                raise ValueError(
                    f"{rid}: partial_payment requires 0 < amount_safe_to_pay < "
                    "requested_amount"
                )
            first, second = self.payments
            if first.date != self.request_date:
                raise ValueError(f"{rid}: first partial payment must fall on request_date")
            if first.amount != self.amount_safe_to_pay:
                raise ValueError(
                    f"{rid}: first partial payment must equal amount_safe_to_pay"
                )
            if second.date != self.earliest_date_for_full_payment:
                raise ValueError(
                    f"{rid}: second partial payment must fall on "
                    "earliest_date_for_full_payment"
                )
            if to_money(first.amount + second.amount) != self.requested_amount:
                raise ValueError(f"{rid}: partial payments do not sum to requested_amount")

        if method is RecommendedPaymentMethod.WAIT:
            if len(self.payments) != 1:
                raise ValueError(f"{rid}: wait requires exactly one future payment")
            if self.payments[0].amount != self.requested_amount:
                raise ValueError(f"{rid}: wait must pay the full requested amount")

        return self

    @property
    def payment_plan(self) -> str:
        if not self.payments:
            return NO_PLAN
        return "|".join(payment.render() for payment in self.payments)

    @property
    def spending_changes_needed(self) -> str:
        if not self.spending_changes:
            return NO_SPENDING_CHANGES
        return "|".join(change.render() for change in self.spending_changes)

    def to_output_row(self) -> dict[str, str]:
        """Render exactly the eight required output columns."""
        return {
            "request_id": self.request_id,
            "amount_safe_to_pay": format_amount_field(self.amount_safe_to_pay),
            "affordability_status": str(self.affordability_status),
            "recommended_payment_method": str(self.recommended_payment_method),
            "payment_plan": self.payment_plan,
            "earliest_date_for_full_payment": (
                ""
                if self.earliest_date_for_full_payment is None
                else self.earliest_date_for_full_payment.isoformat()
            ),
            "spending_changes_needed": self.spending_changes_needed,
            "decision_explanation": self.decision_explanation,
        }


# --------------------------------------------------------------------------- #
# Reconstructed state shared between the ledger and the forecast
# --------------------------------------------------------------------------- #


class CashFlow(Frozen):
    """One dated movement of home-currency cash on the forecast timeline."""

    date: IsoDate
    label: str
    direction: Direction
    amount: Money
    source: str
    category: str = ""
    event_id: str | None = None
    essential: bool = False

    @model_validator(mode="after")
    def _check(self) -> CashFlow:
        if self.direction is Direction.NON_CASH:
            raise ValueError(f"{self.source}: a cash flow cannot be non_cash")
        if self.amount < 0:
            raise ValueError(f"{self.source}: cash flow amount must not be negative")
        return self

    @property
    def signed_amount(self) -> Decimal:
        if self.direction is Direction.CREDIT:
            return self.amount
        return -self.amount


class ForecastAdjustment(Frozen):
    """An evidence-derived change to the forecast, always traceable to a source.

    The evidence layer emits these; the forecast applies them. Keeping them as
    data means untrusted message and image content can change amounts and dates
    without being able to change forecasting rules.
    """

    kind: str
    source_id: str
    target_event_id: str | None = None
    target_category: str | None = None
    amount: OptionalMoney = None
    effective_date: OptionalIsoDate = None
    new_date: OptionalIsoDate = None
    reason: str = ""

    @model_validator(mode="after")
    def _check(self) -> ForecastAdjustment:
        allowed = {
            "set_recurring_amount",
            "set_event_amount",
            "shift_event_date",
            "cancel_event",
            "exclude_recurring",
            "add_confirmed_credit",
            "add_scheduled_debit",
        }
        if self.kind not in allowed:
            raise ValueError(f"unknown adjustment kind {self.kind!r}")
        if not self.source_id:
            raise ValueError("an adjustment must name its evidence source")
        return self


class RecurringAmountOverride(Frozen):
    """An evidence-derived amount for a recurrence, from ``effective_date`` on.

    Date scoping matters: a salary that rises on the 15th must keep its old
    amount for occurrences before then, and a salary that has been cut applies
    immediately.
    """

    event_id: str
    category: str
    amount: Money
    source_id: str
    effective_date: OptionalIsoDate = None

    def applies_on(self, when: date) -> bool:
        return self.effective_date is None or when >= self.effective_date


class FinancialState(Frozen):
    """A user's reconstructed cash position as of one request date."""

    user_id: str
    request_id: str
    as_of: IsoDate
    horizon_end: IsoDate
    home_currency: Currency
    opening_balance: Money
    minimum_balance_to_keep: Money
    dated_flows: tuple[CashFlow, ...] = ()
    recurring: tuple[RecurringCommitment, ...] = ()
    recurring_overrides: tuple[RecurringAmountOverride, ...] = ()
    lifecycle: LifecycleResolution
    baseline: EssentialSpendingBaseline
    adjustments_applied: tuple[ForecastAdjustment, ...] = ()
    notes: tuple[str, ...] = ()
    plan_after_same_day_income: bool = True

    def amount_for(self, commitment: RecurringCommitment, when: date) -> Decimal:
        """The amount a commitment carries on ``when`` after evidence overrides."""
        applicable = [
            override
            for override in self.recurring_overrides
            if override.event_id == commitment.representative_event_id
            and override.applies_on(when)
        ]
        if not applicable:
            return commitment.amount
        latest = max(
            applicable, key=lambda o: (o.effective_date or date.min, o.source_id)
        )
        return latest.amount

    def recurring_by_event_id(self, event_id: str) -> RecurringCommitment | None:
        for commitment in self.recurring:
            if commitment.representative_event_id == event_id:
                return commitment
        return None
