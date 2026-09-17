"""Dataset loading, indexing and integrity checks.

This module is the only place that touches ``dataset/``. Everything it returns
is a validated, frozen model from :mod:`domain`, so downstream tools never see
raw CSV strings.

Two hygiene rules are enforced here rather than left to convention:

* ``sample_requests.csv`` is loaded input-columns-only by default. Its expected
  output columns are labels, and reading them requires an explicit call.
* Hold-out labels can only be read by the quarantined evaluator named in
  ``code/evaluation/splits.json``. Any other caller raises.

Run ``python3 code/tools/data.py`` to print a load summary and integrity report.
"""

from __future__ import annotations

import csv
import hashlib
import inspect
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Iterator, Mapping, Sequence

# Allow both ``python3 code/tools/data.py`` and ``from tools.data import ...``
# by making sure the ``code/`` directory is importable either way.
_CODE_DIR = Path(__file__).resolve().parent.parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from domain import (  # noqa: E402
    OUTPUT_COLUMNS,
    Currency,
    Direction,
    EventRecord,
    ExchangeRateRecord,
    Frozen,
    ImageRecord,
    MessageRecord,
    PaymentMethod,
    PaymentOptionRecord,
    ProfileRecord,
    RequestRecord,
)

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

REQUEST_INPUT_COLUMNS: tuple[str, ...] = (
    "request_id",
    "user_id",
    "request_date",
    "request_type",
    "requested_amount",
    "desired_completion_date",
    "allows_partial_payment",
    "request_text",
)
SAMPLE_LABEL_COLUMNS: tuple[str, ...] = tuple(
    column for column in OUTPUT_COLUMNS if column != "request_id"
)

EXPECTED_HEADERS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "requests.csv": REQUEST_INPUT_COLUMNS,
        "sample_requests.csv": REQUEST_INPUT_COLUMNS + SAMPLE_LABEL_COLUMNS,
        "financial_profiles.csv": (
            "user_id",
            "home_currency",
            "current_available_balance",
            "minimum_balance_to_keep",
            "financial_priorities",
            "expense_categories_to_protect",
            "expense_categories_user_is_willing_to_reduce",
            "expense_categories_user_is_willing_to_stop",
            "payment_methods_user_will_consider",
            "max_installment_months",
        ),
        "financial_events.csv": (
            "event_id",
            "user_id",
            "event_type",
            "description",
            "category",
            "direction",
            "amount",
            "currency",
            "event_date",
            "settlement_date",
            "status",
            "linked_event_id",
            "flexibility",
            "minimum_allowed_amount",
        ),
        "request_payment_options.csv": (
            "payment_option_id",
            "request_id",
            "payment_method",
            "payment_amount",
            "number_of_payments",
            "first_payment_date",
            "payment_frequency_days",
            "financing_fee",
            "total_payable_amount",
        ),
        "exchange_rates.csv": ("rate_date", "from_currency", "to_currency", "rate"),
        "messages.csv": (
            "message_id",
            "user_id",
            "request_id",
            "related_event_id",
            "sent_at",
            "source_type",
            "message_text",
        ),
        "images.csv": ("image_id", "user_id", "request_id", "related_event_id"),
        "output.csv": OUTPUT_COLUMNS,
    }
)


class DatasetError(RuntimeError):
    """Raised when the dataset cannot be loaded as specified."""


class HoldoutAccessError(PermissionError):
    """Raised when an unauthorized caller tries to read hold-out labels."""


@dataclass(frozen=True, slots=True)
class DatasetPaths:
    """Filesystem layout, always resolved relative to the repository root."""

    repo_root: Path
    dataset_dir: Path
    images_dir: Path
    output_csv: Path
    splits_json: Path

    @classmethod
    def resolve(cls, dataset_dir: Path | str | None = None) -> DatasetPaths:
        if dataset_dir is not None:
            resolved = Path(dataset_dir).resolve()
            repo_root = resolved.parent
        else:
            repo_root = find_repo_root()
            resolved = repo_root / "dataset"
        if not resolved.is_dir():
            raise DatasetError(f"dataset directory not found: {resolved}")
        return cls(
            repo_root=repo_root,
            dataset_dir=resolved,
            images_dir=resolved / "media" / "images",
            output_csv=repo_root / "output.csv",
            splits_json=repo_root / "code" / "evaluation" / "splits.json",
        )

    def csv(self, name: str) -> Path:
        path = self.dataset_dir / name
        if not path.is_file():
            raise DatasetError(f"missing dataset file: {path}")
        return path

    def image(self, image_id: str) -> Path:
        return self.images_dir / f"{image_id}.png"


def find_repo_root(start: Path | None = None) -> Path:
    """Walk up from ``start`` to the directory holding AGENTS.md and dataset/."""
    current = (start or Path(__file__)).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "AGENTS.md").is_file() and (candidate / "dataset").is_dir():
            return candidate
    raise DatasetError(f"could not locate repository root from {current}")


# --------------------------------------------------------------------------- #
# CSV reading
# --------------------------------------------------------------------------- #


def _read_rows(path: Path) -> list[dict[str, str]]:
    expected = EXPECTED_HEADERS.get(path.name)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        header = tuple(reader.fieldnames or ())
        if expected is not None and header != expected:
            raise DatasetError(
                f"{path.name}: unexpected header\n  expected: {expected}\n  found:    {header}"
            )
        return [dict(row) for row in reader]


def _build(model: type[Frozen], rows: Iterable[dict[str, str]], source: str) -> list[Any]:
    built: list[Any] = []
    for index, row in enumerate(rows, start=2):  # row 1 is the header
        try:
            built.append(model(**row))
        except Exception as exc:  # noqa: BLE001 - re-raised with file/line context
            raise DatasetError(f"{source} line {index}: {exc}") from exc
    return built


def _project(row: Mapping[str, str], columns: Sequence[str]) -> dict[str, str]:
    return {column: row[column] for column in columns}


# --------------------------------------------------------------------------- #
# Per-file loaders
# --------------------------------------------------------------------------- #


def load_requests(paths: DatasetPaths) -> list[RequestRecord]:
    path = paths.csv("requests.csv")
    return _build(RequestRecord, _read_rows(path), path.name)


def load_sample_request_inputs(paths: DatasetPaths) -> list[RequestRecord]:
    """Load ``sample_requests.csv`` with its label columns stripped."""
    path = paths.csv("sample_requests.csv")
    rows = (_project(row, REQUEST_INPUT_COLUMNS) for row in _read_rows(path))
    return _build(RequestRecord, rows, path.name)


def load_profiles(paths: DatasetPaths) -> list[ProfileRecord]:
    path = paths.csv("financial_profiles.csv")
    return _build(ProfileRecord, _read_rows(path), path.name)


def load_events(paths: DatasetPaths) -> list[EventRecord]:
    path = paths.csv("financial_events.csv")
    return _build(EventRecord, _read_rows(path), path.name)


def load_payment_options(paths: DatasetPaths) -> list[PaymentOptionRecord]:
    path = paths.csv("request_payment_options.csv")
    return _build(PaymentOptionRecord, _read_rows(path), path.name)


def load_exchange_rates(paths: DatasetPaths) -> list[ExchangeRateRecord]:
    path = paths.csv("exchange_rates.csv")
    return _build(ExchangeRateRecord, _read_rows(path), path.name)


def load_messages(paths: DatasetPaths) -> list[MessageRecord]:
    path = paths.csv("messages.csv")
    return _build(MessageRecord, _read_rows(path), path.name)


def load_images(paths: DatasetPaths) -> list[ImageRecord]:
    path = paths.csv("images.csv")
    return _build(ImageRecord, _read_rows(path), path.name)


def load_output_template_ids(paths: DatasetPaths) -> list[str]:
    path = paths.csv("output.csv")
    return [row["request_id"] for row in _read_rows(path)]


# --------------------------------------------------------------------------- #
# Indexed dataset
# --------------------------------------------------------------------------- #


def _group(items: Iterable[Any], key: str) -> Mapping[str, tuple[Any, ...]]:
    grouped: dict[str, list[Any]] = defaultdict(list)
    for item in items:
        value = getattr(item, key)
        if value:
            grouped[value].append(item)
    return MappingProxyType({k: tuple(v) for k, v in grouped.items()})


def _index(items: Iterable[Any], key: str, source: str) -> Mapping[str, Any]:
    indexed: dict[str, Any] = {}
    for item in items:
        value = getattr(item, key)
        if value in indexed:
            raise DatasetError(f"{source}: duplicate {key} {value!r}")
        indexed[value] = item
    return MappingProxyType(indexed)


@dataclass(frozen=True, slots=True)
class Dataset:
    """All participant-facing inputs, validated and indexed for lookup."""

    paths: DatasetPaths
    fingerprint: str
    request_order: tuple[str, ...]
    requests: Mapping[str, RequestRecord]
    sample_requests: Mapping[str, RequestRecord]
    profiles: Mapping[str, ProfileRecord]
    events: Mapping[str, EventRecord]
    events_by_user: Mapping[str, tuple[EventRecord, ...]]
    payment_options: Mapping[str, PaymentOptionRecord]
    payment_options_by_request: Mapping[str, tuple[PaymentOptionRecord, ...]]
    messages: Mapping[str, MessageRecord]
    messages_by_user: Mapping[str, tuple[MessageRecord, ...]]
    messages_by_request: Mapping[str, tuple[MessageRecord, ...]]
    messages_by_event: Mapping[str, tuple[MessageRecord, ...]]
    images: Mapping[str, ImageRecord]
    images_by_user: Mapping[str, tuple[ImageRecord, ...]]
    images_by_request: Mapping[str, tuple[ImageRecord, ...]]
    images_by_event: Mapping[str, tuple[ImageRecord, ...]]
    exchange_rates: Mapping[tuple[date, Currency, Currency], ExchangeRateRecord]
    output_template_ids: tuple[str, ...]

    # -- lookups ---------------------------------------------------------- #

    def request(self, request_id: str) -> RequestRecord:
        try:
            return self.requests[request_id]
        except KeyError:
            raise DatasetError(f"unknown request_id {request_id!r}") from None

    def profile(self, user_id: str) -> ProfileRecord:
        try:
            return self.profiles[user_id]
        except KeyError:
            raise DatasetError(f"no financial profile for user {user_id!r}") from None

    def event(self, event_id: str) -> EventRecord:
        try:
            return self.events[event_id]
        except KeyError:
            raise DatasetError(f"unknown event_id {event_id!r}") from None

    def user_events(
        self,
        user_id: str,
        *,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> tuple[EventRecord, ...]:
        """Events for a user, ordered by effective date then event_id."""
        events = self.events_by_user.get(user_id, ())
        if start_date is not None:
            events = tuple(e for e in events if e.effective_date >= start_date)
        if end_date is not None:
            events = tuple(e for e in events if e.effective_date <= end_date)
        return events

    def request_payment_options(self, request_id: str) -> tuple[PaymentOptionRecord, ...]:
        return self.payment_options_by_request.get(request_id, ())

    def full_payment_option(self, request_id: str) -> PaymentOptionRecord | None:
        for option in self.request_payment_options(request_id):
            if option.payment_method is PaymentMethod.FULL_PAYMENT:
                return option
        return None

    def installment_options(self, request_id: str) -> tuple[PaymentOptionRecord, ...]:
        return tuple(
            option
            for option in self.request_payment_options(request_id)
            if option.payment_method is PaymentMethod.INSTALLMENTS
        )

    def exchange_rate(
        self, rate_date: date, from_currency: Currency, to_currency: Currency
    ) -> ExchangeRateRecord | None:
        return self.exchange_rates.get((rate_date, from_currency, to_currency))

    def image_path(self, image_id: str) -> Path:
        path = self.paths.image(image_id)
        if not path.is_file():
            raise DatasetError(f"image file not found for {image_id!r}: {path}")
        return path

    def iter_requests(self) -> Iterator[RequestRecord]:
        """Evaluation requests in dataset order."""
        for request_id in self.request_order:
            yield self.requests[request_id]


def _fingerprint(paths: DatasetPaths) -> str:
    """Content hash of every input file, for deterministic cache keys."""
    digest = hashlib.sha256()
    for name in sorted(EXPECTED_HEADERS):
        path = paths.dataset_dir / name
        if not path.is_file():
            continue
        digest.update(name.encode("utf-8"))
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    for image in sorted(paths.images_dir.glob("*.png")):
        digest.update(image.name.encode("utf-8"))
        digest.update(hashlib.sha256(image.read_bytes()).digest())
    return digest.hexdigest()


@lru_cache(maxsize=4)
def load_dataset(dataset_dir: Path | str | None = None) -> Dataset:
    """Load, validate and index every participant-facing input file."""
    paths = DatasetPaths.resolve(dataset_dir)

    requests = load_requests(paths)
    sample_requests = load_sample_request_inputs(paths)
    profiles = load_profiles(paths)
    events = sorted(load_events(paths), key=lambda e: (e.effective_date, e.event_id))
    options = load_payment_options(paths)
    messages = load_messages(paths)
    images = load_images(paths)
    rates = load_exchange_rates(paths)

    rate_index: dict[tuple[date, Currency, Currency], ExchangeRateRecord] = {}
    for rate in rates:
        if rate.key in rate_index:
            raise DatasetError(f"exchange_rates.csv: duplicate rate for {rate.key}")
        rate_index[rate.key] = rate

    return Dataset(
        paths=paths,
        fingerprint=_fingerprint(paths),
        request_order=tuple(request.request_id for request in requests),
        requests=_index(requests, "request_id", "requests.csv"),
        sample_requests=_index(sample_requests, "request_id", "sample_requests.csv"),
        profiles=_index(profiles, "user_id", "financial_profiles.csv"),
        events=_index(events, "event_id", "financial_events.csv"),
        events_by_user=_group(events, "user_id"),
        payment_options=_index(options, "payment_option_id", "request_payment_options.csv"),
        payment_options_by_request=_group(options, "request_id"),
        messages=_index(messages, "message_id", "messages.csv"),
        messages_by_user=_group(messages, "user_id"),
        messages_by_request=_group(messages, "request_id"),
        messages_by_event=_group(messages, "related_event_id"),
        images=_index(images, "image_id", "images.csv"),
        images_by_user=_group(images, "user_id"),
        images_by_request=_group(images, "request_id"),
        images_by_event=_group(images, "related_event_id"),
        exchange_rates=MappingProxyType(rate_index),
        output_template_ids=tuple(load_output_template_ids(paths)),
    )


# --------------------------------------------------------------------------- #
# Frozen sample split and label access
# --------------------------------------------------------------------------- #


class SampleLabel(Frozen):
    """The expected output columns of one ``sample_requests.csv`` row."""

    request_id: str
    amount_safe_to_pay: str
    affordability_status: str
    recommended_payment_method: str
    payment_plan: str
    earliest_date_for_full_payment: str
    spending_changes_needed: str
    decision_explanation: str


@dataclass(frozen=True, slots=True)
class SplitManifest:
    schema_version: str
    calibration_ids: tuple[str, ...]
    holdout_ids: tuple[str, ...]
    authorized_holdout_reader: str

    @property
    def all_ids(self) -> tuple[str, ...]:
        return self.calibration_ids + self.holdout_ids


@lru_cache(maxsize=4)
def load_split_manifest(splits_json: Path | str | None = None) -> SplitManifest:
    path = Path(splits_json) if splits_json else DatasetPaths.resolve().splits_json
    if not path.is_file():
        raise DatasetError(f"frozen split manifest not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    manifest = SplitManifest(
        schema_version=payload["schema_version"],
        calibration_ids=tuple(payload["calibration"]["request_ids"]),
        holdout_ids=tuple(payload["holdout"]["request_ids"]),
        authorized_holdout_reader=payload["holdout"]["authorized_reader"],
    )
    overlap = set(manifest.calibration_ids) & set(manifest.holdout_ids)
    if overlap:
        raise DatasetError(f"split manifest overlaps on {sorted(overlap)}")
    return manifest


def _caller_is_authorized(authorized_reader: str) -> bool:
    frame = inspect.stack()[2]
    caller = Path(frame.filename).resolve()
    return caller.as_posix().endswith(authorized_reader)


def load_sample_labels(
    split: str,
    *,
    paths: DatasetPaths | None = None,
) -> tuple[SampleLabel, ...]:
    """Read expected sample outputs for one split.

    ``calibration`` is open: those ten rows are the approved few-shot and
    error-analysis set. ``holdout`` is quarantined and only the evaluator named
    in the frozen manifest may read it.
    """
    resolved = paths or DatasetPaths.resolve()
    manifest = load_split_manifest(resolved.splits_json)

    if split == "calibration":
        wanted = set(manifest.calibration_ids)
    elif split == "holdout":
        if not _caller_is_authorized(manifest.authorized_holdout_reader):
            raise HoldoutAccessError(
                "hold-out labels may only be read by "
                f"{manifest.authorized_holdout_reader}; refusing to leak them into "
                "prompts, caches, traces or tuning artifacts"
            )
        wanted = set(manifest.holdout_ids)
    else:
        raise ValueError(f"unknown split {split!r}; expected 'calibration' or 'holdout'")

    rows = _read_rows(resolved.csv("sample_requests.csv"))
    labels = [
        SampleLabel(**_project(row, OUTPUT_COLUMNS))
        for row in rows
        if row["request_id"] in wanted
    ]
    missing = wanted - {label.request_id for label in labels}
    if missing:
        raise DatasetError(f"split {split!r} references unknown rows: {sorted(missing)}")
    return tuple(labels)


# --------------------------------------------------------------------------- #
# Integrity checks
# --------------------------------------------------------------------------- #


class Finding(Frozen):
    severity: str
    code: str
    detail: str


def check_dataset_integrity(dataset: Dataset) -> tuple[Finding, ...]:
    """Assert every cross-file assumption the deterministic tools rely on."""
    findings: list[Finding] = []

    def error(code: str, detail: str) -> None:
        findings.append(Finding(severity="error", code=code, detail=detail))

    def warn(code: str, detail: str) -> None:
        findings.append(Finding(severity="warning", code=code, detail=detail))

    if tuple(dataset.output_template_ids) != dataset.request_order:
        error("output_template_mismatch", "output.csv ids differ from requests.csv ids")

    for request in dataset.iter_requests():
        if request.user_id not in dataset.profiles:
            error("missing_profile", f"{request.request_id}: no profile for {request.user_id}")
        options = dataset.request_payment_options(request.request_id)
        if len(options) < 2:
            error("too_few_options", f"{request.request_id}: {len(options)} payment options")
        full = [o for o in options if o.payment_method is PaymentMethod.FULL_PAYMENT]
        if len(full) != 1:
            error("full_option_count", f"{request.request_id}: {len(full)} full_payment options")
        elif full[0].payment_amount != request.requested_amount:
            warn(
                "full_option_amount",
                f"{request.request_id}: full_payment option {full[0].payment_amount} "
                f"!= requested_amount {request.requested_amount}",
            )

    for option in dataset.payment_options.values():
        if option.request_id not in dataset.requests and (
            option.request_id not in dataset.sample_requests
        ):
            error("orphan_option", f"{option.payment_option_id}: unknown request")

    for event in dataset.events.values():
        if event.user_id not in dataset.profiles:
            error("orphan_event", f"{event.event_id}: unknown user {event.user_id}")
        if event.linked_event_id and event.linked_event_id not in dataset.events:
            error("dangling_link", f"{event.event_id}: link to {event.linked_event_id}")
        if event.amount_needs_resolution and not dataset.images_by_event.get(event.event_id):
            error(
                "unresolvable_amount",
                f"{event.event_id}: blank amount with no image evidence",
            )
        if event.flexibility.can_reduce and event.minimum_allowed_amount is None:
            warn("no_minimum", f"{event.event_id}: reducible without minimum_allowed_amount")

    for image in dataset.images.values():
        if not dataset.paths.image(image.image_id).is_file():
            error("missing_image_file", f"{image.image_id}: file not found")
        if image.related_event_id and image.related_event_id not in dataset.events:
            error("orphan_image", f"{image.image_id}: unknown event {image.related_event_id}")

    for message in dataset.messages.values():
        if message.user_id not in dataset.profiles:
            error("orphan_message_user", f"{message.message_id}: unknown user")
        if message.related_event_id and message.related_event_id not in dataset.events:
            error("orphan_message_event", f"{message.message_id}: unknown event")

    for event in dataset.events.values():
        if not event.is_cash or event.amount is None:
            continue
        home = dataset.profiles[event.user_id].home_currency
        if event.currency == home or event.settlement_date is None:
            continue
        if dataset.exchange_rate(event.settlement_date, event.currency, home) is None:
            error(
                "missing_rate",
                f"{event.event_id}: no {event.currency}->{home} rate on "
                f"{event.settlement_date}",
            )

    manifest = load_split_manifest(dataset.paths.splits_json)
    for request_id in manifest.all_ids:
        if request_id not in dataset.sample_requests:
            error("split_unknown_row", f"{request_id} is not in sample_requests.csv")
    if len(manifest.all_ids) != len(dataset.sample_requests):
        error(
            "split_incomplete",
            f"split covers {len(manifest.all_ids)} of {len(dataset.sample_requests)} rows",
        )
    if set(manifest.all_ids) & set(dataset.requests):
        error("split_leak", "split ids overlap evaluation request ids")

    return tuple(findings)


# --------------------------------------------------------------------------- #
# CLI self-check
# --------------------------------------------------------------------------- #


def _summarize(dataset: Dataset) -> str:
    manifest = load_split_manifest(dataset.paths.splits_json)
    blank = [e.event_id for e in dataset.events.values() if e.amount_needs_resolution]
    non_cash = [e.event_id for e in dataset.events.values() if not e.is_cash]
    return "\n".join(
        [
            f"repo root          {dataset.paths.repo_root}",
            f"fingerprint        {dataset.fingerprint[:16]}",
            f"requests           {len(dataset.requests)}",
            f"sample requests    {len(dataset.sample_requests)} "
            f"({len(manifest.calibration_ids)} calibration / "
            f"{len(manifest.holdout_ids)} hold-out)",
            f"profiles           {len(dataset.profiles)}",
            f"events             {len(dataset.events)}",
            f"  blank amounts    {len(blank)} (resolved from images)",
            f"  non-cash rows    {len(non_cash)}",
            f"payment options    {len(dataset.payment_options)}",
            f"messages           {len(dataset.messages)}",
            f"images             {len(dataset.images)}",
            f"exchange rates     {len(dataset.exchange_rates)}",
        ]
    )


def main() -> int:
    dataset = load_dataset()
    print(_summarize(dataset))
    findings = check_dataset_integrity(dataset)
    errors = [f for f in findings if f.severity == "error"]
    warnings = [f for f in findings if f.severity == "warning"]
    print(f"\nintegrity: {len(errors)} error(s), {len(warnings)} warning(s)")
    for finding in findings[:40]:
        print(f"  [{finding.severity}] {finding.code}: {finding.detail}")
    if len(findings) > 40:
        print(f"  ... {len(findings) - 40} more")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
