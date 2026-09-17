"""Validate submission artifacts after the full-dataset run.

Checks root ``output.csv``, ``code/evaluation/usage_report.md``, and optional
``code.zip`` against the challenge contract. Never loads hold-out labels.

    .venv/bin/python code/evaluation/validate_submission.py
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from domain import OUTPUT_COLUMNS  # noqa: E402
from tools.data import load_dataset  # noqa: E402
from tools.validation import validate_output_row  # noqa: E402

REQUIRED_USAGE_HEADINGS = (
    "## Overall totals",
    "## Per-model totals",
)
REQUIRED_USAGE_FIELDS = (
    "Provider:",
    "Model names:",
    "Model calls:",
    "Input tokens:",
    "Output tokens:",
    "Total tokens:",
    "Average tokens per request:",
    "Estimated total cost (USD):",
    "Estimated cost per request (USD):",
)
SECRET_MARKERS = ("sk-ant", "sk-ant-api", "ANTHROPIC_API_KEY=")
STATUS_VALUES = {
    "affordable_now",
    "affordable_with_plan",
    "affordable_later",
    "not_affordable",
}
METHOD_VALUES = {
    "full_payment",
    "partial_payment",
    "installments",
    "wait",
    "not_recommended",
}


def _fail(message: str, failures: list[str]) -> None:
    failures.append(message)
    print(f"FAIL  {message}")


def _ok(message: str) -> None:
    print(f"OK    {message}")


def validate_output_csv(path: Path, failures: list[str]) -> None:
    if not path.is_file():
        _fail(f"missing {path}", failures)
        return
    dataset = load_dataset()
    expected_ids = list(dataset.request_order)
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if fieldnames != list(OUTPUT_COLUMNS):
        _fail(f"output.csv columns {fieldnames!r} != {list(OUTPUT_COLUMNS)}", failures)
    else:
        _ok("output.csv column names and order")

    if len(rows) != 250:
        _fail(f"output.csv has {len(rows)} data rows, expected 250", failures)
    else:
        _ok("output.csv has 250 data rows")

    got_ids = [row.get("request_id", "") for row in rows]
    if got_ids != expected_ids:
        missing = [rid for rid in expected_ids if rid not in got_ids]
        extra = [rid for rid in got_ids if rid not in expected_ids]
        if got_ids and got_ids != expected_ids and not missing and not extra:
            _fail("output.csv request_id order does not match dataset/requests.csv", failures)
        else:
            _fail(
                f"output.csv request_ids mismatch (missing {len(missing)}, extra {len(extra)})",
                failures,
            )
    else:
        _ok("output.csv request_ids match dataset/requests.csv in order")

    row_failures = 0
    method_counts: dict[str, int] = {}
    status_counts: dict[str, int] = {}
    for row in rows:
        request_id = row.get("request_id", "")
        request = dataset.requests.get(request_id)
        if request is None:
            row_failures += 1
            print(f"FAIL  {request_id}: not in dataset/requests.csv")
            continue
        problems = list(validate_output_row(row, request=request))
        status = (row.get("affordability_status") or "").strip()
        method = (row.get("recommended_payment_method") or "").strip()
        if status not in STATUS_VALUES:
            problems.append(f"unknown affordability_status {status!r}")
        if method not in METHOD_VALUES:
            problems.append(f"unknown recommended_payment_method {method!r}")
        explanation = (row.get("decision_explanation") or "").strip()
        if not explanation:
            problems.append("empty decision_explanation")
        lowered = " ".join(row.values()).lower()
        if any(marker in lowered for marker in SECRET_MARKERS):
            problems.append("possible secret in output row")
        status_counts[status] = status_counts.get(status, 0) + 1
        method_counts[method] = method_counts.get(method, 0) + 1
        if problems:
            row_failures += 1
            print(f"FAIL  {request_id}: {'; '.join(problems)}")
    if row_failures:
        _fail(f"{row_failures} output row(s) failed contract checks", failures)
    else:
        _ok("every output row passed validate_output_row and enum checks")

    print("      method counts: " + ", ".join(f"{k}={v}" for k, v in sorted(method_counts.items())))
    print("      status counts: " + ", ".join(f"{k}={v}" for k, v in sorted(status_counts.items())))


def validate_usage_report(path: Path, failures: list[str], *, require_nonzero: bool) -> None:
    if not path.is_file():
        _fail(f"missing {path}", failures)
        return
    text = path.read_text(encoding="utf-8")
    lowered = text.lower()
    if any(marker in lowered for marker in SECRET_MARKERS):
        _fail("usage_report.md contains a secret marker", failures)
    else:
        _ok("usage_report.md has no API-key markers")
    if "awaiting the final full-dataset run" in lowered:
        _fail("usage_report.md is still the pre-run placeholder", failures)
    for heading in REQUIRED_USAGE_HEADINGS:
        if heading not in text:
            _fail(f"usage_report.md missing {heading}", failures)
    for field in REQUIRED_USAGE_FIELDS:
        if field not in text:
            _fail(f"usage_report.md missing {field}", failures)
    if "## Overall totals" in text and "## Per-model totals" in text:
        _ok("usage_report.md has overall and per-model sections")

    calls_match = re.search(r"- Model calls:\s*(\d+)", text)
    tokens_match = re.search(r"- Total tokens:\s*(\d+)", text)
    requests_match = re.search(r"- Evaluation requests processed:\s*(\d+)", text)
    if require_nonzero:
        if not calls_match or int(calls_match.group(1)) <= 0:
            _fail("usage_report.md model calls is missing or zero", failures)
        else:
            _ok(f"usage_report.md model calls = {calls_match.group(1)}")
        if not tokens_match or int(tokens_match.group(1)) <= 0:
            _fail("usage_report.md total tokens is missing or zero", failures)
        else:
            _ok(f"usage_report.md total tokens = {tokens_match.group(1)}")
        if not requests_match or int(requests_match.group(1)) != 250:
            _fail(
                "usage_report.md evaluation requests processed is not 250",
                failures,
            )
        else:
            _ok("usage_report.md processed 250 evaluation requests")
        if "output.csv" not in text:
            _fail("usage_report.md does not mention output.csv", failures)
        else:
            _ok("usage_report.md names output.csv as the run artifact")


def validate_code_zip(path: Path, failures: list[str]) -> None:
    if not path.is_file():
        _fail(f"missing {path}", failures)
        return
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if any(name.endswith(".env") or name.endswith(".env.local") for name in names):
            _fail("code.zip contains a .env file", failures)
        else:
            _ok("code.zip does not contain .env")
        blob = "\n".join(names).lower()
        if any(marker in blob for marker in SECRET_MARKERS):
            _fail("code.zip file names look like they contain secrets", failures)
        report_names = [
            name
            for name in names
            if name.replace("\\", "/").endswith("evaluation/usage_report.md")
        ]
        if not report_names:
            _fail("code.zip missing evaluation/usage_report.md", failures)
        else:
            _ok(f"code.zip includes {report_names[0]}")
            report = archive.read(report_names[0]).decode("utf-8")
            if any(marker in report.lower() for marker in SECRET_MARKERS):
                _fail("zipped usage_report.md contains a secret marker", failures)
            if "awaiting the final full-dataset run" in report.lower():
                _fail("zipped usage_report.md is still the placeholder", failures)
        has_main = any(name.replace("\\", "/").endswith("main.py") for name in names)
        has_readme = any(Path(name).name.lower() == "readme.md" for name in names)
        if not has_main:
            _fail("code.zip missing main.py", failures)
        else:
            _ok("code.zip includes main.py")
        if not has_readme:
            _fail("code.zip missing README.md", failures)
        else:
            _ok("code.zip includes README.md")
        print(f"      code.zip entries: {len(names)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate Buy or Wait? submission artifacts")
    parser.add_argument("--output", type=Path, default=ROOT / "output.csv")
    parser.add_argument(
        "--usage-report",
        type=Path,
        default=ROOT / "code" / "evaluation" / "usage_report.md",
    )
    parser.add_argument("--code-zip", type=Path, default=ROOT / "code.zip")
    parser.add_argument(
        "--require-zip",
        action="store_true",
        help="Fail if code.zip is missing (default: skip zip checks when absent)",
    )
    args = parser.parse_args(argv)

    failures: list[str] = []
    print("# Submission artifact validation")
    print()
    validate_output_csv(args.output, failures)
    print()
    validate_usage_report(args.usage_report, failures, require_nonzero=True)
    print()
    if args.code_zip.is_file() or args.require_zip:
        validate_code_zip(args.code_zip, failures)
    else:
        print("SKIP  code.zip not present (pack after the full-dataset run)")

    print()
    if failures:
        print(f"{len(failures)} check(s) failed")
        return 1
    print("all submission artifact checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
