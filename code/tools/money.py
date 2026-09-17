"""Currency conversion against the fixed, dated exchange-rate table.

Rates are directional and dated. A foreign-currency cash event is converted
using the row for its settlement date and the stated ``from_currency`` to
``to_currency`` direction. A missing rate is an error, never an inferred or
inverted number, because silently inventing a rate would silently invent money.
"""

from __future__ import annotations

import sys
from datetime import date
from decimal import Decimal
from pathlib import Path

_CODE_DIR = Path(__file__).resolve().parent.parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from domain import ConvertedAmount, Currency, EventRecord, to_money  # noqa: E402
from tools.data import Dataset  # noqa: E402


class MissingRateError(LookupError):
    """Raised when the dataset has no rate for a required conversion."""


def convert_currency(
    dataset: Dataset,
    *,
    amount: Decimal | str,
    from_currency: Currency | str,
    to_currency: Currency | str,
    rate_date: date | str,
) -> ConvertedAmount:
    """Convert one amount using the dated, directional rate for ``rate_date``."""
    source = Currency(from_currency)
    target = Currency(to_currency)
    when = date.fromisoformat(rate_date) if isinstance(rate_date, str) else rate_date
    value = to_money(amount)

    if source == target:
        return ConvertedAmount(
            original_amount=value,
            converted_amount=value,
            from_currency=source,
            to_currency=target,
            rate=Decimal(1),
            rate_date=when,
        )

    record = dataset.exchange_rate(when, source, target)
    if record is None:
        raise MissingRateError(
            f"no {source}->{target} rate on {when.isoformat()}; refusing to infer one"
        )
    return ConvertedAmount(
        original_amount=value,
        converted_amount=to_money(value * record.rate),
        from_currency=source,
        to_currency=target,
        rate=record.rate,
        rate_date=record.rate_date,
    )


def event_amount_in_home_currency(
    dataset: Dataset,
    event: EventRecord,
    *,
    amount: Decimal | None = None,
    home_currency: Currency | None = None,
) -> Decimal:
    """Return an event's cash amount in the user's home currency.

    ``amount`` overrides the row's own value, which is how an amount recovered
    from image evidence is converted.
    """
    resolved = event.amount if amount is None else to_money(amount)
    if resolved is None:
        raise ValueError(
            f"{event.event_id}: amount is blank and must be resolved from evidence "
            "before conversion; a blank amount is never zero"
        )
    target = home_currency or dataset.profile(event.user_id).home_currency
    if event.currency == target:
        return resolved
    if event.settlement_date is None:
        raise MissingRateError(
            f"{event.event_id}: cannot date a conversion without a settlement_date"
        )
    return convert_currency(
        dataset,
        amount=resolved,
        from_currency=event.currency,
        to_currency=target,
        rate_date=event.settlement_date,
    ).converted_amount
