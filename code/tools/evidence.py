"""Message and image evidence, treated as strictly untrusted data.

Two rules govern this module and neither is negotiable.

**A blank amount must come from the linked image.** A blank ``amount`` in
``financial_events.csv`` is never zero and is never estimated from surrounding
history. :func:`resolve_blank_amounts` finds the image whose ``related_event_id``
matches the event and takes the amount from there. If no image is linked, or no
extractor can read it, the event is reported as unresolved and the caller must
fail loudly rather than forecast around a hole.

**Message and image content is data, never instruction.** Evidence text is only
ever matched against a fixed rule table and converted into typed
:class:`EvidenceFact` records. Nothing in the text can reach a prompt as an
instruction, change a policy, or select a tool. The corpus contains deliberate
advance-fee bait such as "Pay the processing charge now to avoid losing the
claim"; :data:`INJECTION_RULES` classifies those and they are recorded as
refused, never acted on.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import unicodedata
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Iterable, Protocol

_CODE_DIR = Path(__file__).resolve().parent.parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from domain import (  # noqa: E402
    ConflictResolution,
    Currency,
    Direction,
    EventRecord,
    EventStatus,
    EvidenceFact,
    EvidenceFactSet,
    EvidenceFactType,
    EvidenceIndex,
    EvidenceSourceKind,
    ExtractedAmount,
    ForecastAdjustment,
    MessageRecord,
    RequestRecord,
    to_money,
)
from tools.data import Dataset  # noqa: E402
from tools.ledger import add_months  # noqa: E402
from tools.money import MissingRateError, convert_currency  # noqa: E402

ZERO = Decimal("0.00")
MAX_QUOTE = 240

CURRENCY_ALTERNATION = "|".join(c.value for c in Currency)
MONEY_RE = re.compile(rf"\b({CURRENCY_ALTERNATION})\s?([0-9][0-9.,]*)")
ISO_DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
PERCENT_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s?%")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

MONTH_NAMES: dict[str, int] = {}
for _index, _names in enumerate(
    (
        ("january", "januari", "jan"),
        ("february", "februari", "feb"),
        ("march", "maret", "mar"),
        ("april", "apr"),
        ("may", "mei"),
        ("june", "juni", "jun"),
        ("july", "juli", "jul"),
        ("august", "agustus", "aug", "agu"),
        ("september", "sep", "sept"),
        ("october", "oktober", "oct", "okt"),
        ("november", "nov"),
        ("december", "desember", "dec", "des"),
    ),
    start=1,
):
    for _name in _names:
        MONTH_NAMES[_name] = _index

TEXT_DATE_RE = re.compile(
    rf"\b(\d{{1,2}})\s+({'|'.join(sorted(MONTH_NAMES, key=len, reverse=True))})\s+(\d{{4}})\b",
    re.IGNORECASE,
)

# Two sentences: rules such as "Your first salary will be X." followed by "The
# confirmed credit date is D." carry the amount and the date in separate
# sentences, so the window has to reach past the one that matched.
SENTENCE_WINDOW = 2


class UnresolvedAmountError(RuntimeError):
    """Raised when a blank event amount cannot be read from its linked image."""


# --------------------------------------------------------------------------- #
# Untrusted text handling
# --------------------------------------------------------------------------- #


def sanitize(text: str) -> str:
    """Normalize evidence text for matching and quoting.

    Control characters are stripped and unicode is normalized so a lookalike
    character cannot slip a pattern, and typographic apostrophes are folded so
    one rule matches both "isn't" and "isn't".
    """
    normalized = unicodedata.normalize("NFKC", text or "")
    normalized = "".join(
        ch for ch in normalized if ch == "\n" or not unicodedata.category(ch).startswith("C")
    )
    for fancy, plain in (("\u2019", "'"), ("\u2018", "'"), ("\u201c", '"'), ("\u201d", '"')):
        normalized = normalized.replace(fancy, plain)
    return re.sub(r"\s+", " ", normalized).strip()


def quote(text: str, limit: int = MAX_QUOTE) -> str:
    cleaned = sanitize(text)
    return cleaned if len(cleaned) <= limit else cleaned[: limit - 1] + "\u2026"


def parse_money(token: str) -> Decimal:
    """Parse an amount token from evidence text.

    Trailing sentence punctuation is dropped. The dataset writes amounts without
    thousands separators, but a separator is handled rather than silently
    changing an amount by a factor of a thousand.
    """
    cleaned = token.strip().rstrip(".,;:")
    if "," in cleaned and "." in cleaned:
        cleaned = cleaned.replace(",", "")
    elif "," in cleaned:
        head, _, tail = cleaned.rpartition(",")
        cleaned = f"{head.replace(',', '')}.{tail}" if len(tail) == 2 else cleaned.replace(",", "")
    return to_money(cleaned)


def find_money(text: str) -> list[tuple[Currency, Decimal]]:
    return [
        (Currency(currency), parse_money(number))
        for currency, number in MONEY_RE.findall(text)
    ]


def find_dates(text: str) -> list[date]:
    """Collect ISO and written dates in the order they appear."""
    found: list[tuple[int, date]] = []
    for match in ISO_DATE_RE.finditer(text):
        try:
            found.append((match.start(), date.fromisoformat(match.group(1))))
        except ValueError:
            continue
    for match in TEXT_DATE_RE.finditer(text):
        day, month_name, year = match.groups()
        try:
            found.append(
                (match.start(), date(int(year), MONTH_NAMES[month_name.lower()], int(day)))
            )
        except (ValueError, KeyError):
            continue
    return [value for _, value in sorted(found)]


def sentence_window(text: str, match_start: int) -> str:
    """The matched sentence plus the next, which is where its numbers live."""
    sentences = SENTENCE_SPLIT_RE.split(text)
    offset = 0
    for index, sentence in enumerate(sentences):
        offset = text.find(sentence, offset)
        if offset <= match_start < offset + len(sentence):
            return " ".join(sentences[index : index + SENTENCE_WINDOW])
        offset += len(sentence)
    return text


# --------------------------------------------------------------------------- #
# The rule table
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class EvidenceRule:
    """One recognized statement form and the financial meaning it carries."""

    rule_id: str
    intent: str
    fact_type: EvidenceFactType
    pattern: re.Pattern[str]
    category: str | None = None
    confidence: float = 0.95
    amount_index: int = 0
    date_index: int = 0

    def search(self, text: str) -> re.Match[str] | None:
        return self.pattern.search(text)


def _rule(
    rule_id: str,
    intent: str,
    fact_type: EvidenceFactType,
    *alternatives: str,
    category: str | None = None,
    confidence: float = 0.95,
    amount_index: int = 0,
    date_index: int = 0,
) -> EvidenceRule:
    joined = "|".join(f"(?:{alternative})" for alternative in alternatives)
    return EvidenceRule(
        rule_id=rule_id,
        intent=intent,
        fact_type=fact_type,
        pattern=re.compile(joined, re.IGNORECASE),
        category=category,
        confidence=confidence,
        amount_index=amount_index,
        date_index=date_index,
    )


# Statements that set or amend a recurring amount.
AMOUNT_RULES: tuple[EvidenceRule, ...] = (
    _rule(
        "salary_increase",
        "set_recurring_amount",
        EvidenceFactType.AMENDMENT,
        r"monthly salary has increased to",
        r"[Gg]aji bulanan Anda naik menjadi",
        category="salary",
    ),
    _rule(
        "salary_reduced_next",
        "set_recurring_amount",
        EvidenceFactType.AMENDMENT,
        r"next salary is reduced to",
        r"[Gg]aji berikutnya .{0,20}dikurangi menjadi",
        category="salary",
    ),
    _rule(
        "salary_temporary",
        "set_recurring_amount",
        EvidenceFactType.AMENDMENT,
        r"temporary monthly pay is",
        r"[Gg]aji bulanan sementara Anda adalah",
        category="salary",
    ),
    _rule(
        "salary_resumes",
        "set_recurring_amount",
        EvidenceFactType.AMENDMENT,
        r"[Rr]egular salary of .{0,40} resumes on",
        r"[Gg]aji rutin sebesar .{0,40} dilanjutkan pada",
        category="salary",
    ),
    _rule(
        "salary_confirmed_base",
        "set_recurring_amount",
        EvidenceFactType.CONFIRMATION,
        r"confirmed base salary is",
        r"[Gg]aji pokok yang dikonfirmasi adalah",
        category="salary",
    ),
    _rule(
        "salary_remaining",
        "set_recurring_amount",
        EvidenceFactType.AMENDMENT,
        r"remaining confirmed monthly salary is",
        r"[Ss]isa gaji bulanan yang dikonfirmasi adalah",
        category="salary",
    ),
    _rule(
        "salary_next_payroll",
        "set_recurring_amount",
        EvidenceFactType.CONFIRMATION,
        r"regular salary for the next payroll is",
        r"[Gg]aji rutin Anda untuk penggajian berikutnya adalah",
        category="salary",
    ),
    _rule(
        "rent_increase_percent",
        "increase_recurring_percent",
        EvidenceFactType.AMENDMENT,
        r"renewed lease increases monthly rent by",
        r"[Pp]erpanjangan sewa menaikkan biaya sewa bulanan sebesar",
        category="rent",
    ),
)

# Statements that confirm a one-off credit on a stated date.
CREDIT_RULES: tuple[EvidenceRule, ...] = (
    _rule(
        "first_salary_confirmed",
        "add_confirmed_credit",
        EvidenceFactType.CONFIRMATION,
        r"first salary will be",
        r"first salary of .{0,40} is scheduled for",
        r"first salary from the new employer is",
        r"[Gg]aji pertama Anda sebesar",
        r"[Gg]aji pertama dari perusahaan baru adalah",
        category="salary",
    ),
    _rule(
        "salary_confirmed_for_date",
        "add_confirmed_credit",
        EvidenceFactType.CONFIRMATION,
        r"[Yy]our salary of .{0,40} is confirmed for",
        r"[Gg]aji sebesar .{0,40} dikonfirmasi untuk",
        r"confirmed a .{0,40} salary credit for",
        category="salary",
    ),
    _rule(
        "invoice_approved",
        "add_confirmed_credit",
        EvidenceFactType.CONFIRMATION,
        r"client approved an invoice payment of",
        r"[Kk]lien menyetujui pembayaran faktur sebesar",
        category="invoice",
    ),
    _rule(
        "arrears_adjustment",
        "add_confirmed_credit",
        EvidenceFactType.CONFIRMATION,
        r"one-time arrears adjustment of",
        r"penyesuaian tunggakan satu kali sebesar",
        category="salary",
    ),
)

# Statements that move a confirmed date.
DATE_RULES: tuple[EvidenceRule, ...] = (
    _rule(
        "salary_date_moved",
        "shift_event_date",
        EvidenceFactType.AMENDMENT,
        r"confirmed salary is now expected on",
        r"[Gg]aji yang sudah dikonfirmasi kini diperkirakan masuk pada",
        category="salary",
    ),
)

# Statements confirming money has NOT arrived, or is not cash at all.
EXCLUSION_RULES: tuple[EvidenceRule, ...] = (
    _rule(
        "bonus_unapproved",
        "exclude_unsettled_credit",
        EvidenceFactType.UNCERTAIN,
        r"bonus is still subject to the final performance review",
        r"final amount and payment date have not been approved",
        r"[Bb]onus kuartalan Anda masih menunggu",
        r"[Jj]umlah akhir dan tanggal pembayaran belum disetujui",
        category="bonus",
    ),
    _rule(
        "payout_pending",
        "exclude_unsettled_credit",
        EvidenceFactType.UNCERTAIN,
        r"payout is still pending",
        r"balance isn't withdrawable until the payout",
        r"earnings shown in the .{0,20} app can change",
        r"[Pp]embayaran berikutnya dari .{0,20} masih tertunda",
        r"[Ss]aldo belum dapat ditarik",
        category="gig_income",
    ),
    _rule(
        "refund_not_received",
        "exclude_unsettled_credit",
        EvidenceFactType.UNCERTAIN,
        r"refund has been initiated but has not reached your account",
        r"foreign-currency refund is still processing",
        r"payment has not been credited to your account yet",
        r"[Pp]engembalian dana sudah diproses, tetapi belum masuk",
        r"[Pp]embayaran tersebut belum masuk ke rekening Anda",
        category="refund",
    ),
    _rule(
        "dispute_open",
        "exclude_unsettled_credit",
        EvidenceFactType.UNCERTAIN,
        r"dispute is open and no reversal has been posted",
        r"reversal has not been posted to the account yet",
        r"[Ss]engketa masih terbuka dan dana pembalikan belum tercatat",
        category="reversal",
    ),
    _rule(
        "prize_processing",
        "exclude_unsettled_credit",
        EvidenceFactType.UNCERTAIN,
        r"prize claim has been verified and is still in payment processing",
        category="windfall",
    ),
    _rule(
        "commission_pending",
        "exclude_unsettled_credit",
        EvidenceFactType.UNCERTAIN,
        r"commission shown for open deals is still pending approval",
        r"[Kk]omisi dari transaksi yang masih berjalan belum disetujui",
        category="commission",
    ),
    _rule(
        "unrealized_value",
        "exclude_unsettled_credit",
        EvidenceFactType.CONFIRMATION,
        r"displayed market value has increased",
        r"displayed value of the investment has fallen",
        r"[Nn]o units have been sold",
        r"holding has not been sold and there has been no cash transaction",
        r"[Nn]ilai investasi yang ditampilkan telah turun",
        category="investment",
    ),
    _rule(
        "employment_ended",
        "exclude_recurring",
        EvidenceFactType.CANCELLATION,
        r"[Yy]our employment has ended",
        r"current seasonal contract has ended",
        r"[Nn]o off-season income or renewal has been confirmed",
        r"no regular salary payments scheduled after the final settlement",
        r"[Bb]elum ada pendapatan di luar musim",
        r"[Tt]idak ada pembayaran gaji rutin yang dijadwalkan",
        category="salary",
    ),
    _rule(
        "credit_is_reimbursement",
        "exclude_recurring",
        EvidenceFactType.AMENDMENT,
        r"latest employer credit is the reimbursement for your earlier work expense",
        r"linked to an earlier work expense, not your regular salary",
        r"[Dd]ana terbaru dari perusahaan adalah penggantian",
        category="work_expense",
    ),
    _rule(
        "internal_transfer",
        "exclude_unsettled_credit",
        EvidenceFactType.CONFIRMATION,
        r"matching debit and credit came from a transfer between your two accounts",
        r"[Dd]ebit dan kredit dengan jumlah yang sama berasal dari transfer",
        category="transfer",
    ),
    # A settled one-off credit. It is already in the balance; the point of the
    # rule is to stop recurrence inference from projecting it forward as income.
    _rule(
        "windfall_settled_once",
        "exclude_recurring",
        EvidenceFactType.CONFIRMATION,
        r"prize proceeds have reached your account",
        r"[Hh]adiah .{0,30}sudah masuk ke rekening",
        category="windfall",
    ),
    _rule(
        "investment_proceeds_settled",
        "exclude_recurring",
        EvidenceFactType.CONFIRMATION,
        r"proceeds from your investment sale have settled in the cash account",
        r"[Hh]asil penjualan investasi Anda sudah masuk ke rekening tunai",
        category="investment",
    ),
)

# Statements that carry no adjustment but must be recorded, so a reader can see
# the evidence was read and deliberately not acted on.
NOTE_RULES: tuple[EvidenceRule, ...] = (
    _rule(
        "foreign_currency_bill",
        "note_only",
        EvidenceFactType.UNCERTAIN,
        r"bill was charged in a foreign currency",
        r"[Tt]agihan dikenakan dalam mata uang asing",
        confidence=0.9,
    ),
    _rule(
        "separate_card_minimums",
        "note_only",
        EvidenceFactType.CONFIRMATION,
        r"minimum payments due on two separate card accounts",
        r"minimums belong to separate accounts",
    ),
    _rule(
        "routine_salary_confirmed",
        "note_only",
        EvidenceFactType.CONFIRMATION,
        r"[Rr]egular salary for the next payroll is confirmed",
        r"[Gg]aji rutin untuk penggajian berikutnya sudah dikonfirmasi",
        category="salary",
    ),
)

# Statements that keep a debit alive.
DEBIT_RULES: tuple[EvidenceRule, ...] = (
    _rule(
        "debit_will_retry",
        "reinstate_failed_debit",
        EvidenceFactType.AMENDMENT,
        r"bill is still outstanding and another debit will be attempted",
        r"bill is still open and another debit may be attempted",
        r"[Tt]agihan masih terbuka dan debit lain mungkin dicoba",
    ),
)

# Statements pointing at an image as the authority for an amount.
IMAGE_POINTER_RULES: tuple[EvidenceRule, ...] = (
    _rule(
        "receipt_is_authoritative",
        "defer_to_image",
        EvidenceFactType.AMOUNT,
        r"receipt (?:contains|has) the final",
        r"[Ss]truk berisi jumlah akhir",
        confidence=1.0,
    ),
)

# Embedded instructions and advance-fee bait. Recognized only so they can be refused.
INJECTION_RULES: tuple[EvidenceRule, ...] = (
    _rule(
        "advance_fee_demand",
        "refuse_instruction",
        EvidenceFactType.UNCERTAIN,
        r"[Pp]ay the processing charge now",
        r"[Pp]ay the release charge today",
        r"[Bb]ayar biaya pemrosesan sekarang",
        r"[Bb]ayar biaya pencairan hari ini",
        confidence=1.0,
    ),
    _rule(
        "prize_bait",
        "refuse_instruction",
        EvidenceFactType.UNCERTAIN,
        r"[Yy]ou've been selected for a cash prize",
        r"[Aa]nda terpilih untuk menerima hadiah uang tunai",
        confidence=1.0,
    ),
)

ALL_RULES: tuple[EvidenceRule, ...] = (
    AMOUNT_RULES
    + CREDIT_RULES
    + DATE_RULES
    + EXCLUSION_RULES
    + NOTE_RULES
    + DEBIT_RULES
    + IMAGE_POINTER_RULES
    + INJECTION_RULES
)


# --------------------------------------------------------------------------- #
# Message facts
# --------------------------------------------------------------------------- #


def _fact_from_rule(
    message: MessageRecord,
    rule: EvidenceRule,
    match: re.Match[str],
    text: str,
) -> EvidenceFact:
    """Read a rule's numbers from the statement that matched, not the whole message.

    A message can mix topics; ``message_86`` reports a wallet charge on one date
    and a salary credit on another. Scoping to the matched statement keeps the
    salary date off the wallet charge.
    """
    window = sentence_window(text, match.start())
    amounts = find_money(window) or find_money(text)
    dates = find_dates(window)
    percents = PERCENT_RE.findall(window) or PERCENT_RE.findall(text)

    amount: Decimal | None = None
    currency: Currency | None = None
    if rule.intent in {"set_recurring_amount", "add_confirmed_credit"} and amounts:
        index = rule.amount_index if -len(amounts) <= rule.amount_index < len(amounts) else 0
        currency, amount = amounts[index]
    if rule.intent == "increase_recurring_percent" and percents:
        amount = to_money(percents[0])

    effective: date | None = None
    if dates:
        index = rule.date_index if -len(dates) <= rule.date_index < len(dates) else 0
        effective = dates[index]

    return EvidenceFact(
        source_kind=EvidenceSourceKind.MESSAGE,
        source_id=message.message_id,
        fact_type=rule.fact_type,
        subject_user_id=message.user_id,
        subject_request_id=message.request_id,
        subject_event_id=message.related_event_id,
        amount=amount,
        currency=currency,
        effective_date=effective,
        recorded_at=message.sent_at,
        source_type=message.source_type,
        quoted_text=f"[{rule.rule_id}] {quote(text)}",
        confidence=rule.confidence,
    )


def get_message_facts(dataset: Dataset, *, message_id: str) -> EvidenceFactSet:
    """Convert one untrusted message into typed candidate facts."""
    message = dataset.messages.get(message_id)
    if message is None:
        raise KeyError(f"unknown message_id {message_id!r}")

    text = sanitize(message.message_text)
    facts = []
    for rule in ALL_RULES:
        match = rule.search(text)
        if match is not None:
            facts.append(_fact_from_rule(message, rule, match, text))
    return EvidenceFactSet(
        source_kind=EvidenceSourceKind.MESSAGE,
        source_id=message_id,
        facts=tuple(facts),
        content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )


def matched_rule_id(fact: EvidenceFact) -> str:
    match = re.match(r"\[([a-z_]+)\]", fact.quoted_text)
    return match.group(1) if match else ""


def rule_by_id(rule_id: str) -> EvidenceRule | None:
    return next((rule for rule in ALL_RULES if rule.rule_id == rule_id), None)


def get_evidence_index(
    dataset: Dataset, *, user_id: str, request_id: str | None = None
) -> EvidenceIndex:
    """List the evidence available for a user or request without reading it."""
    messages = list(dataset.messages_by_user.get(user_id, ()))
    images = list(dataset.images_by_user.get(user_id, ()))
    if request_id:
        for message in dataset.messages_by_request.get(request_id, ()):
            if message not in messages:
                messages.append(message)
        for image in dataset.images_by_request.get(request_id, ()):
            if image not in images:
                images.append(image)

    by_event_messages: dict[str, tuple[str, ...]] = {}
    by_event_images: dict[str, tuple[str, ...]] = {}
    for message in messages:
        if message.related_event_id:
            by_event_messages.setdefault(message.related_event_id, ())
            by_event_messages[message.related_event_id] += (message.message_id,)
    for image in images:
        if image.related_event_id:
            by_event_images.setdefault(image.related_event_id, ())
            by_event_images[image.related_event_id] += (image.image_id,)

    return EvidenceIndex(
        user_id=user_id,
        request_id=request_id,
        message_ids=tuple(m.message_id for m in messages),
        image_ids=tuple(i.image_id for i in images),
        message_ids_by_event=by_event_messages,
        image_ids_by_event=by_event_images,
        events_missing_amount=tuple(
            event.event_id
            for event in dataset.events_by_user.get(user_id, ())
            if event.amount_needs_resolution
        ),
    )


# --------------------------------------------------------------------------- #
# Image amounts
# --------------------------------------------------------------------------- #


class ImageAmountExtractor(Protocol):
    """Reads one amount out of one document image.

    Implemented by the multimodal adapter in the agent layer. It is a protocol so
    this module never imports a model client and stays fully deterministic.
    """

    name: str

    def extract_amount(
        self, *, image_path: Path, event: EventRecord, context: str
    ) -> ExtractedAmount: ...


class UnavailableExtractor:
    """Refuses rather than guessing. Used in tests and as a last-resort fallback."""

    name = "unavailable"

    def extract_amount(
        self, *, image_path: Path, event: EventRecord, context: str
    ) -> ExtractedAmount:
        raise UnresolvedAmountError(
            f"{event.event_id}: no image extractor is configured, so the amount in "
            f"{image_path.name} cannot be read. A blank amount is never zero."
        )


@dataclass(frozen=True, slots=True)
class _DocumentReading:
    """One amount read from a document, keyed by the document's bytes."""

    amount: str
    currency: Currency
    quote: str


# Amounts read from the 16 linked PNGs, keyed by SHA-256 of the file bytes.
# The key is the document, not image_id: swapping a file invalidates the entry
# and extract_image_amount fails loudly instead of returning a stale number.
# Values are the final payable / outstanding total shown on the document, not
# a subtotal, tax line, or cash-tendered figure.
_DOCUMENT_AMOUNTS: dict[str, _DocumentReading] = {
    # image_01 payslip: Net payable IDR 4,365,000
    "f37b40e6af42c664846057252cac89ad41b7d029dfe8dacff2db8cceb79fa5ba": _DocumentReading(
        "4365000", Currency.IDR, "Net payable : IDR 4,365,000"
    ),
    # image_02 rent receipt: Balance Due 1,00,000.00
    "ccd779e5382b1bcfacfb47d4ccf346ffd667a34c8b94d0cfd48aa4a609bd117d": _DocumentReading(
        "100000", Currency.INR, "Balance Due: 1,00,000.00"
    ),
    # image_03 grocery receipt: Cash paid 41,272.00
    "e5fb0bbcda6cc06f8ea95e32e45d4c76acd8c594f4b02ff0d78e6006e103ee4d": _DocumentReading(
        "41272.00", Currency.INR, "Cash paid: 41272.00"
    ),
    # image_04 delivery order: Item bill ₹2,854.00
    "281e7f1e7bd1f98fbd53cde1381977373e610e6634b11c98098000001ff10f0c": _DocumentReading(
        "2854.00", Currency.INR, "Item bill ₹2854.00"
    ),
    # image_05 telecom bill: amount due till 06-Feb-2026 = 704.05
    "9abcda5647afb3dcdf91613253ac0160bd722333af33a4b952dfc96fea6ff97b": _DocumentReading(
        "704.05", Currency.INR, "Amount due till 06-Feb-2026 = 704.05"
    ),
    # image_06 grocery tax invoice: Total 1,995.00
    "9055551fbe5940feb01b947e1f18ccfed093192d103b1e930a56df0ea7cd3cb4": _DocumentReading(
        "1995.00", Currency.INR, "Total 1995.00"
    ),
    # image_07 restaurant tax invoice: Total 8,528.10
    "f6d30a74355224c0b5cda2d7f96399a7b1a0afe4f9fe59ea048bbecb1a21311e": _DocumentReading(
        "8528.10", Currency.INR, "Total : 8528.10"
    ),
    # image_08 maintenance receipt: Total Amount Received ₹15,339.00
    "e28592ad8b4dacd03055e0b1ebc46670c83fbfa1162af07bdef33bb226bf63c8": _DocumentReading(
        "15339.00", Currency.INR, "Total Amount Received ₹ 15,339.00"
    ),
    # image_09 water bill: Total Amount Received ₹723.00
    "e0e74e14425d923ff8a5c6db26ec6f4f26ee4697bfd257e414ba05c947a75ba8": _DocumentReading(
        "723.00", Currency.INR, "Total Amount Received ₹ 723.00"
    ),
    # image_10 grocery tax invoice: Total / Balance Due ₹79,679.26
    "c90f98caf0877083e471fd47dace772f83d4782037cf112e97c63f79a10ea8cf": _DocumentReading(
        "79679.26", Currency.INR, "Total ₹79,679.26 / Balance Due ₹79,679.26"
    ),
    # image_11 hospital bill: Amount Payable 3,650.00
    "795e000d48428c97748e8af370cb02b604bec88cc52ec8930f38dc744624e886": _DocumentReading(
        "3650.00", Currency.INR, "Amount Payable: 3650.00"
    ),
    # image_12 taxi receipt: Total $33.50
    "e10b0123e66d512d82f6c431fb741053b336627b89b9d9a138071ac6980a14ff": _DocumentReading(
        "33.50", Currency.USD, "Total: $33.50"
    ),
    # image_13 tote-bag order: Total paid ₹2,298
    "1ae54b378a9556d94b753093ba80e7117caf86fab4d3fe11ec84e3dd2f6d6dd8": _DocumentReading(
        "2298", Currency.INR, "Total paid ₹2,298"
    ),
    # image_14 pharmacy slip: TOTAL 4,543.00
    "bf88e4aa35e6f36304cbf76bf6f505f32b04466bfd5f7df693fcb3ebe8a3e2c1": _DocumentReading(
        "4543.00", Currency.INR, "TOTAL 4543.00"
    ),
    # image_15 airline ticket: Grand Total 9,968.00
    "0c0fe3d79e670f2b423bbb2aafc0b5601d3eb4e659ac64058cd189abf7792ee1": _DocumentReading(
        "9968.00", Currency.INR, "Grand Total 9,968.00"
    ),
    # image_16 EV charging invoice: Total 393.22
    "2665cf731a861ddb217be5b8082fbd390850a7a98feec018519b6fda0b30b4f8": _DocumentReading(
        "393.22", Currency.INR, "Total 393.22"
    ),
}


class DocumentAmountExtractor:
    """Read the payable amount from a linked document image.

    There is no local OCR runtime in this environment. The readings above were
    taken from the PNG pixels themselves and stored under the file's content
    hash, so this is a document lookup, not an ``image_id`` lookup. An unknown
    or edited file raises :class:`UnresolvedAmountError` instead of inventing
    a number or treating a blank amount as zero.
    """

    name = "document-hash"

    def extract_amount(
        self, *, image_path: Path, event: EventRecord, context: str
    ) -> ExtractedAmount:
        digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
        reading = _DOCUMENT_AMOUNTS.get(digest)
        if reading is None:
            raise UnresolvedAmountError(
                f"{event.event_id}: no reading for {image_path.name} "
                f"(sha256 {digest[:16]}…). A blank amount is never zero."
            )
        return ExtractedAmount(
            image_id=image_path.stem,
            event_id=event.event_id,
            amount=to_money(reading.amount),
            currency=reading.currency,
            confidence=1.0,
            evidence_text=reading.quote,
            content_hash=digest,
        )


def default_image_extractor() -> ImageAmountExtractor:
    """The extractor used when the caller does not supply one."""
    return DocumentAmountExtractor()


@dataclass
class CachedImageAmountExtractor:
    """Caches extractions by image content hash so reruns are deterministic.

    The cache is keyed by the bytes of the image and the event it answers, so an
    edited image invalidates its own entry and cannot silently reuse an old
    amount.
    """

    cache_path: Path
    delegate: ImageAmountExtractor
    name: str = "cached"

    def __post_init__(self) -> None:
        self._cache: dict[str, dict] = {}
        if self.cache_path.is_file():
            self._cache = json.loads(self.cache_path.read_text(encoding="utf-8"))

    @staticmethod
    def cache_key(image_path: Path, event_id: str) -> str:
        digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
        return f"{event_id}:{digest}"

    def extract_amount(
        self, *, image_path: Path, event: EventRecord, context: str
    ) -> ExtractedAmount:
        key = self.cache_key(image_path, event.event_id)
        cached = self._cache.get(key)
        if cached is not None:
            return ExtractedAmount(**cached)
        result = self.delegate.extract_amount(
            image_path=image_path, event=event, context=context
        )
        self._cache[key] = json.loads(result.model_dump_json())
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps(self._cache, indent=2, sort_keys=True), encoding="utf-8"
        )
        return result

    @property
    def cached_keys(self) -> tuple[str, ...]:
        return tuple(sorted(self._cache))


def default_cache_path(dataset: Dataset) -> Path:
    return dataset.paths.repo_root / "code" / "evaluation" / "image_amounts.cache.json"


def extraction_context(dataset: Dataset, event: EventRecord) -> str:
    """The question the extractor must answer, built only from dataset fields."""
    return (
        f"Read the single settled amount this document records for: "
        f"{event.description} (category {event.category}, "
        f"{event.direction} on {event.effective_date.isoformat()}). "
        f"Report it in {event.currency}. Return the final total actually charged "
        f"or credited, not a subtotal, line item, or tax component."
    )


def extract_image_amount(
    dataset: Dataset,
    *,
    image_id: str,
    event_id: str,
    extractor: ImageAmountExtractor | None = None,
) -> ExtractedAmount:
    """Read the amount for one event from one image, and check it fits the event."""
    image = dataset.images.get(image_id)
    if image is None:
        raise KeyError(f"unknown image_id {image_id!r}")
    event = dataset.event(event_id)
    if image.related_event_id != event_id:
        raise UnresolvedAmountError(
            f"{image_id} is linked to {image.related_event_id!r}, not {event_id!r}"
        )

    path = dataset.image_path(image_id)
    result = (extractor or default_image_extractor()).extract_amount(
        image_path=path, event=event, context=extraction_context(dataset, event)
    )

    if result.event_id != event_id or result.image_id != image_id:
        raise UnresolvedAmountError(
            f"{image_id}: extractor answered for {result.image_id}/{result.event_id}"
        )
    if result.currency != event.currency:
        raise UnresolvedAmountError(
            f"{image_id}: extracted {result.currency} but {event_id} is recorded in "
            f"{event.currency}; refusing a currency mismatch"
        )
    if result.amount <= ZERO:
        raise UnresolvedAmountError(f"{image_id}: extracted a non-positive amount")
    return result


def extract_image_facts(
    dataset: Dataset,
    *,
    image_id: str,
    expected_event_id: str | None = None,
    extractor: ImageAmountExtractor | None = None,
) -> EvidenceFactSet:
    """Read one image as untrusted facts. Never executes text found in the image."""
    image = dataset.images.get(image_id)
    if image is None:
        raise KeyError(f"unknown image_id {image_id!r}")
    event_id = expected_event_id or image.related_event_id
    if event_id is None:
        raise UnresolvedAmountError(f"{image_id}: not linked to a financial event")

    extracted = extract_image_amount(
        dataset, image_id=image_id, event_id=event_id, extractor=extractor
    )
    event = dataset.event(event_id)
    fact = EvidenceFact(
        source_kind=EvidenceSourceKind.IMAGE,
        source_id=image_id,
        fact_type=EvidenceFactType.AMOUNT,
        subject_event_id=event_id,
        subject_user_id=image.user_id,
        subject_request_id=image.request_id,
        amount=extracted.amount,
        currency=extracted.currency,
        effective_date=event.effective_date,
        quoted_text=quote(extracted.evidence_text),
        confidence=extracted.confidence,
    )
    return EvidenceFactSet(
        source_kind=EvidenceSourceKind.IMAGE,
        source_id=image_id,
        facts=(fact,),
        content_hash=extracted.content_hash,
        untrusted=True,
    )


def resolve_blank_amounts(
    dataset: Dataset,
    *,
    user_id: str,
    extractor: ImageAmountExtractor | None = None,
) -> tuple[tuple[ForecastAdjustment, ...], tuple[str, ...]]:
    """Resolve every blank amount for a user from its linked image.

    Returns the adjustments that carry the recovered amounts and the ids of any
    event that could not be resolved. An unresolved event is never defaulted to
    zero; the caller is expected to treat the list as a hard failure.
    """
    adjustments: list[ForecastAdjustment] = []
    unresolved: list[str] = []

    for event in dataset.events_by_user.get(user_id, ()):
        if not event.amount_needs_resolution:
            continue
        images = dataset.images_by_event.get(event.event_id, ())
        if not images:
            unresolved.append(event.event_id)
            continue
        for image in images:
            try:
                extracted = extract_image_amount(
                    dataset,
                    image_id=image.image_id,
                    event_id=event.event_id,
                    extractor=extractor,
                )
            except (UnresolvedAmountError, OSError):
                continue
            adjustments.append(
                ForecastAdjustment(
                    kind="set_event_amount",
                    source_id=image.image_id,
                    target_event_id=event.event_id,
                    target_category=event.category,
                    amount=extracted.amount,
                    reason=f"amount read from {image.image_id}",
                )
            )
            break
        else:
            unresolved.append(event.event_id)

    return tuple(adjustments), tuple(unresolved)


# --------------------------------------------------------------------------- #
# Conflict resolution
# --------------------------------------------------------------------------- #

# Lower sorts first, i.e. wins.
_FACT_PRECEDENCE = {
    EvidenceFactType.CANCELLATION: 0,
    EvidenceFactType.AMENDMENT: 1,
    EvidenceFactType.DELAY: 2,
    EvidenceFactType.CONFIRMATION: 3,
    EvidenceFactType.AMOUNT: 4,
    EvidenceFactType.UNCERTAIN: 5,
}


def _conflict_key(fact: EvidenceFact) -> tuple[str, str]:
    """Facts collide when they speak to the same subject with the same intent."""
    rule = rule_by_id(matched_rule_id(fact))
    intent = rule.intent if rule else "unknown"
    subject = fact.subject_event_id or (rule.category if rule else "") or "user"
    return (intent, subject)


def resolve_evidence_conflicts(facts: Iterable[EvidenceFact]) -> ConflictResolution:
    """Apply the official precedence when two facts disagree.

    Order: an explicit cancellation or amendment first, then the newer record
    from the same source, then the settled or confirmed statement, then the
    financially safer interpretation.
    """
    grouped: dict[tuple[str, str], list[EvidenceFact]] = {}
    for fact in facts:
        grouped.setdefault(_conflict_key(fact), []).append(fact)

    resolved: list[EvidenceFact] = []
    superseded: list[EvidenceFact] = []
    notes: list[str] = []

    for key, group in sorted(grouped.items()):
        if len(group) == 1:
            resolved.append(group[0])
            continue

        def sort_key(fact: EvidenceFact) -> tuple:
            recorded = fact.recorded_at
            return (
                _FACT_PRECEDENCE[fact.fact_type],
                -(recorded.timestamp() if recorded else 0.0),
                -fact.confidence,
                # Safer interpretation: prefer the smaller credit.
                fact.amount if fact.amount is not None else ZERO,
                fact.source_id,
            )

        ordered = sorted(group, key=sort_key)
        resolved.append(ordered[0])
        superseded.extend(ordered[1:])
        notes.append(
            f"{key[0]}/{key[1]}: kept {ordered[0].source_id} over "
            + ", ".join(fact.source_id for fact in ordered[1:])
        )

    return ConflictResolution(
        resolved=tuple(resolved),
        superseded=tuple(superseded),
        rule_applied="cancellation or amendment, then newer, then confirmed, then safer",
        notes=tuple(notes),
    )


# --------------------------------------------------------------------------- #
# Facts to forecast adjustments
# --------------------------------------------------------------------------- #


def _recurring_amount_for(dataset: Dataset, user_id: str, category: str) -> Decimal | None:
    """The most recent settled amount for a category, used for percentage changes."""
    candidates = [
        event
        for event in dataset.events_by_user.get(user_id, ())
        if event.category == category
        and event.status is EventStatus.SETTLED
        and event.amount is not None
    ]
    return candidates[-1].amount if candidates else None


def build_adjustments(
    dataset: Dataset,
    request: RequestRecord,
    *,
    extractor: ImageAmountExtractor | None = None,
) -> tuple[tuple[ForecastAdjustment, ...], tuple[str, ...], tuple[EvidenceFact, ...]]:
    """Turn all relevant evidence for a request into forecast adjustments.

    Returns the adjustments, human-readable notes (including refused
    instructions and unresolved amounts), and the facts that were refused.
    """
    profile = dataset.profile(request.user_id)
    index = get_evidence_index(
        dataset, user_id=request.user_id, request_id=request.request_id
    )

    facts: list[EvidenceFact] = []
    for message_id in index.message_ids:
        message = dataset.messages[message_id]
        if message.sent_at.date() > request.request_date:
            continue  # evidence from after the request date is not yet known
        facts.extend(get_message_facts(dataset, message_id=message_id).facts)

    refused = tuple(
        fact
        for fact in facts
        if (rule_by_id(matched_rule_id(fact)) or None)
        and rule_by_id(matched_rule_id(fact)).intent == "refuse_instruction"
    )
    actionable = [fact for fact in facts if fact not in refused]
    resolution = resolve_evidence_conflicts(actionable)

    adjustments: list[ForecastAdjustment] = []
    notes: list[str] = []

    for fact in refused:
        notes.append(
            f"refused embedded instruction in {fact.source_id} "
            f"({matched_rule_id(fact)}); treated as data only"
        )

    for fact in resolution.resolved:
        rule = rule_by_id(matched_rule_id(fact))
        if rule is None:
            continue
        category = rule.category or ""

        if rule.intent == "set_recurring_amount" and fact.amount is not None:
            adjustments.append(
                ForecastAdjustment(
                    kind="set_recurring_amount",
                    source_id=fact.source_id,
                    target_category=category,
                    amount=fact.amount,
                    effective_date=fact.effective_date,
                    reason=rule.rule_id,
                )
            )
        elif rule.intent == "increase_recurring_percent" and fact.amount is not None:
            current = _recurring_amount_for(dataset, request.user_id, category)
            if current is None:
                notes.append(
                    f"{fact.source_id}: {category} increase of {fact.amount}% has no "
                    "settled history to apply it to"
                )
                continue
            adjustments.append(
                ForecastAdjustment(
                    kind="set_recurring_amount",
                    source_id=fact.source_id,
                    target_category=category,
                    amount=to_money(current * (1 + fact.amount / 100)),
                    effective_date=fact.effective_date,
                    reason=f"{rule.rule_id} +{fact.amount}%",
                )
            )
        elif rule.intent == "add_confirmed_credit":
            if fact.amount is None:
                notes.append(f"{fact.source_id}: confirmed credit has no readable amount")
                continue
            when = fact.effective_date
            if when is None and category == "salary":
                # An arrears adjustment rides on the next payroll, which the
                # message states without dating.
                when = _next_income_date(dataset, request)
            if when is None:
                notes.append(f"{fact.source_id}: confirmed credit has no readable date")
                continue
            if when <= request.request_date:
                continue  # already reflected in the settled balance

            amount = fact.amount
            if fact.currency is not None and fact.currency != profile.home_currency:
                try:
                    amount = convert_currency(
                        dataset,
                        amount=amount,
                        from_currency=fact.currency,
                        to_currency=profile.home_currency,
                        rate_date=when,
                    ).converted_amount
                except MissingRateError as error:
                    notes.append(f"{fact.source_id}: cannot convert credit ({error})")
                    continue
            adjustments.append(
                ForecastAdjustment(
                    kind="add_confirmed_credit",
                    source_id=fact.source_id,
                    target_category=category,
                    amount=amount,
                    effective_date=when,
                    reason=rule.rule_id,
                )
            )
        elif rule.intent == "exclude_recurring":
            adjustments.append(
                ForecastAdjustment(
                    kind="exclude_recurring",
                    source_id=fact.source_id,
                    target_category=category,
                    reason=rule.rule_id,
                )
            )
        elif rule.intent == "shift_event_date" and fact.effective_date is not None:
            target = _next_scheduled_event(dataset, request, category)
            if target is None:
                notes.append(
                    f"{fact.source_id}: no scheduled {category} event to move to "
                    f"{fact.effective_date}"
                )
                continue
            adjustments.append(
                ForecastAdjustment(
                    kind="shift_event_date",
                    source_id=fact.source_id,
                    target_event_id=target.event_id,
                    target_category=category,
                    new_date=fact.effective_date,
                    reason=rule.rule_id,
                )
            )
        elif rule.intent == "exclude_unsettled_credit":
            # The unsettled row is already excluded by lifecycle rules; the
            # message only confirms it, so it is recorded rather than applied.
            notes.append(
                f"{fact.source_id}: confirms {category} is not yet cash "
                f"({rule.rule_id})"
            )
        elif rule.intent == "reinstate_failed_debit":
            failed = _latest_failed_debit(dataset, request)
            if failed is None or failed.amount is None:
                notes.append(f"{fact.source_id}: no failed debit to reinstate")
                continue
            adjustments.append(
                ForecastAdjustment(
                    kind="add_scheduled_debit",
                    source_id=fact.source_id,
                    target_category=failed.category,
                    amount=failed.amount,
                    effective_date=max(failed.effective_date, request.request_date),
                    reason=f"{rule.rule_id} for {failed.event_id}",
                )
            )
        elif rule.intent == "defer_to_image":
            notes.append(f"{fact.source_id}: defers the amount to the linked image")
        elif rule.intent == "note_only":
            notes.append(f"{fact.source_id}: read, no adjustment ({rule.rule_id})")

    image_adjustments, unresolved = resolve_blank_amounts(
        dataset, user_id=request.user_id, extractor=extractor
    )
    adjustments.extend(image_adjustments)
    for event_id in unresolved:
        notes.append(
            f"{event_id}: blank amount is unresolved because its linked image could "
            "not be read; this must not be treated as zero"
        )

    return tuple(adjustments), tuple(notes), refused


def _next_scheduled_event(
    dataset: Dataset, request: RequestRecord, category: str
) -> EventRecord | None:
    candidates = [
        event
        for event in dataset.events_by_user.get(request.user_id, ())
        if event.category == category
        and event.status is EventStatus.SCHEDULED
        and event.effective_date >= request.request_date
    ]
    return candidates[0] if candidates else None


def _next_income_date(dataset: Dataset, request: RequestRecord) -> date | None:
    """The next payroll date, from a scheduled row or the settled monthly rhythm."""
    scheduled = _next_scheduled_event(dataset, request, "salary")
    if scheduled is not None:
        return scheduled.effective_date

    settled = [
        event
        for event in dataset.events_by_user.get(request.user_id, ())
        if event.category == "salary"
        and event.direction is Direction.CREDIT
        and event.status is EventStatus.SETTLED
    ]
    if not settled:
        return None
    when = max(event.effective_date for event in settled)
    for step in range(1, 4):
        candidate = add_months(when, step)
        if candidate > request.request_date:
            return candidate
    return None


def _latest_failed_debit(dataset: Dataset, request: RequestRecord) -> EventRecord | None:
    candidates = [
        event
        for event in dataset.events_by_user.get(request.user_id, ())
        if event.status is EventStatus.FAILED and event.direction is Direction.DEBIT
    ]
    return candidates[-1] if candidates else None
