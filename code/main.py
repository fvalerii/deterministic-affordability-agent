"""Buy or Wait? entry point.

Reconstructs each evaluation request with the deterministic financial tools,
then asks Claude 3.5 Sonnet to write ``decision_explanation``. Money amounts,
dates, methods, and spending cuts are never chosen by the model.

    python3 code/main.py

Secrets are read from environment variables. ``load_environment()`` loads the
repository ``.env`` file before any tool that talks to Anthropic is imported.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_CODE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _CODE_DIR.parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))


def load_environment(start: Path | None = None) -> Path | None:
    """Load ``.env`` from the repo root, then ``code/.env``, without overriding the OS env."""

    try:
        from dotenv import dotenv_values, load_dotenv
    except ImportError:
        return None

    from tools.env import anthropic_api_key

    repo_root = start if start is not None else _REPO_ROOT
    candidates = (
        repo_root / ".env",
        repo_root / "code" / ".env",
        Path.cwd() / ".env",
    )
    loaded: Path | None = None
    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen or not resolved.is_file():
            continue
        seen.add(resolved)
        load_dotenv(resolved, override=False)
        if loaded is None:
            loaded = resolved
    if loaded is not None and anthropic_api_key() is None:
        file_key = (dotenv_values(loaded).get("ANTHROPIC_API_KEY") or "").strip()
        if file_key:
            import os

            os.environ["ANTHROPIC_API_KEY"] = file_key
    return loaded


load_environment()

from tools.data import load_dataset, load_split_manifest  # noqa: E402
from tools.env import anthropic_api_key  # noqa: E402
from tools.orchestrator import BuyOrWaitOrchestrator  # noqa: E402
from tools.usage import METER  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Buy or Wait? financial agent")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Path to the dataset directory (defaults to <repo>/dataset)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to output.csv (defaults to <repo>/output.csv)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N evaluation requests (debug)",
    )
    parser.add_argument(
        "--split",
        choices=("eval", "calibration", "holdout"),
        default="eval",
        help="eval = dataset/requests.csv; calibration = request_01..10; holdout = request_11..25",
    )
    args = parser.parse_args(argv)

    if not anthropic_api_key():
        print(
            "warning: ANTHROPIC_API_KEY is unset or still a placeholder; "
            "vision and explanation will use offline fallbacks",
            file=sys.stderr,
        )

    METER.reset()
    dataset = load_dataset(args.dataset)
    extractor = None
    if anthropic_api_key():
        from tools.vision import ImageAmountExtractor as ClaudeVision

        extractor = ClaudeVision()
    orchestrator = BuyOrWaitOrchestrator(dataset, extractor=extractor)
    manifest = load_split_manifest(dataset.paths.splits_json)
    eval_dir = dataset.paths.repo_root / "code" / "evaluation"
    if args.split == "calibration":
        request_ids = manifest.calibration_ids
        output_path = args.output or (dataset.paths.repo_root / "output.calibration.csv")
        usage_path = eval_dir / "usage_report.calibration.md"
    elif args.split == "holdout":
        request_ids = manifest.holdout_ids
        output_path = args.output or (dataset.paths.repo_root / "output.holdout.csv")
        usage_path = eval_dir / "usage_report.holdout.md"
    else:
        request_ids = dataset.request_order
        if args.limit is not None:
            request_ids = request_ids[: max(0, args.limit)]
            usage_path = eval_dir / "usage_report.limited.md"
        else:
            usage_path = eval_dir / "usage_report.md"
        output_path = args.output
    uses_explicit_ids = args.split in {"calibration", "holdout"} or args.limit is not None
    artifact_name = (
        Path(output_path).name
        if output_path is not None
        else "output.csv"
    )
    METER.begin_run(
        split=args.split,
        output_artifact=artifact_name,
        request_count=len(request_ids),
    )
    path = orchestrator.run(
        output_path=output_path,
        request_ids=request_ids if uses_explicit_ids else None,
        usage_report_path=usage_path,
    )
    print(f"wrote {path} ({len(request_ids) if uses_explicit_ids else len(dataset.request_order)} rows)")
    print(f"wrote usage report {usage_path}")
    print(f"anthropic calls={METER.calls} tokens={METER.total_tokens}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
