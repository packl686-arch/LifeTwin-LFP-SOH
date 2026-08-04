from __future__ import annotations

import argparse
import json
from pathlib import Path

from lifetwin.validation.independent_intake import (
    DEFAULT_CANDIDATE_CONFIG,
    DEFAULT_PROTOCOL_TEMPLATE,
    IndependentLFPIntakeError,
    compile_independent_lfp_intake,
    load_independent_candidate_config,
    load_independent_lfp_intake,
)
from lifetwin.validation.long_term_protocol import (
    IndependentLongTermProtocolValidationError,
)


DEFAULT_OUTPUT_DIRECTORY = Path("artifacts/independent-lfp-intake")


def _write_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compile a metadata-only independent LFP dataset intake into a "
            "fail-closed readiness report and an unfrozen protocol draft"
        )
    )
    parser.add_argument("intake", type=Path)
    parser.add_argument(
        "--candidate",
        type=Path,
        default=DEFAULT_CANDIDATE_CONFIG,
    )
    parser.add_argument(
        "--protocol-template",
        type=Path,
        default=DEFAULT_PROTOCOL_TEMPLATE,
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--require-ready",
        action="store_true",
        help="Return a failing exit status unless the intake is ready for freeze review",
    )
    args = parser.parse_args()
    report_path = args.output_directory / "intake_report.json"
    protocol_path = args.output_directory / "protocol_draft.json"
    existing = [path for path in (report_path, protocol_path) if path.exists()]
    if existing and not args.overwrite:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "error": "Refusing to overwrite existing intake artifacts",
                    "existing": [str(path) for path in existing],
                },
                ensure_ascii=False,
            )
        )
        return 1
    try:
        candidate = load_independent_candidate_config(args.candidate)
        intake = load_independent_lfp_intake(args.intake, candidate=candidate)
        report, protocol = compile_independent_lfp_intake(
            intake,
            candidate,
            protocol_template_path=args.protocol_template,
        )
    except (
        IndependentLFPIntakeError,
        IndependentLongTermProtocolValidationError,
        OSError,
    ) as exc:
        print(
            json.dumps(
                {"status": "failed", "error": str(exc)},
                ensure_ascii=False,
            )
        )
        return 1
    _write_json(report, report_path)
    _write_json(protocol, protocol_path)
    ready = report["readiness_status"] in {
        "ready_for_dataset_specific_freeze_review",
        "ready_for_locked_retrospective_freeze_review",
    }
    result = {
        "status": "passed" if ready else "blocked_as_designed",
        "readiness_status": report["readiness_status"],
        "failure_reasons": report["failure_reasons"],
        "intake_report": str(report_path),
        "protocol_draft": str(protocol_path),
        "report_content_sha256": report["report_content_sha256"],
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 1 if args.require_ready and not ready else 0


if __name__ == "__main__":
    raise SystemExit(main())
