"""Standalone Phase 1 entry point.

Takes a document image path, prints the extracted fields and MRZ validity.
This is the script named in the Phase 1 Definition of Done.

Usage:
    python scripts/run_ocr_mrz.py ../data/samples/specimen_passport_genuine.png
    python scripts/run_ocr_mrz.py --mrz-text "P<UTO..." "L898902C<3UTO..."
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.schemas import MRZCheckResult, ExtractedFields  # noqa: E402

GREEN, RED, DIM, BOLD, RESET = "\033[32m", "\033[31m", "\033[2m", "\033[1m", "\033[0m"


def print_fields(fields: ExtractedFields) -> None:
    print(f"\n{BOLD}Extracted fields{RESET}")
    rows = [
        ("Name", fields.name),
        ("Surname", fields.surname),
        ("Given names", fields.given_names),
        ("Document number", fields.document_number),
        ("Nationality", fields.nationality),
        ("Issuing state", fields.issuing_state),
        ("Date of birth", fields.dob),
        ("Expiry date", fields.expiry_date),
        ("Sex", fields.sex.value if fields.sex else None),
        ("Personal number", fields.personal_number),
    ]
    for label, value in rows:
        shown = value if value not in (None, "") else f"{DIM}--{RESET}"
        print(f"  {label:<18} {shown}")


def print_mrz(check: MRZCheckResult) -> None:
    print(f"\n{BOLD}MRZ validation{RESET}")
    if not check.present:
        print(f"  {RED}No MRZ found{RESET}")
        for err in check.errors:
            print(f"  {DIM}{err}{RESET}")
        return

    print(f"  Format             {check.mrz_format.value if check.mrz_format else '--'}")
    for line in check.raw_lines:
        print(f"  {DIM}{line}{RESET}")

    print()
    for c in check.checks:
        mark = f"{GREEN}PASS{RESET}" if c.passed else f"{RED}FAIL{RESET}"
        print(f"  [{mark}] {c.detail}")

    for err in check.errors:
        print(f"  {RED}{err}{RESET}")

    verdict = f"{GREEN}VALID{RESET}" if check.valid else f"{RED}INVALID{RESET}"
    print(f"\n  Verdict            {verdict}")
    print(f"  {check.summary()}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", nargs="*", help="Image path, or MRZ lines with --mrz-text")
    parser.add_argument("--mrz-text", action="store_true", help="Treat arguments as MRZ lines")
    parser.add_argument("--engine", default=None, help="Force an OCR engine (paddleocr/tesseract)")
    args = parser.parse_args()

    if not args.target:
        parser.error("provide an image path, or MRZ lines with --mrz-text")

    if args.mrz_text:
        from app.modules.ocr_mrz.pipeline import run_mrz_only

        fields, check = run_mrz_only("\n".join(args.target))
        print_fields(fields)
        print_mrz(check)
        return 0 if check.valid else 1

    from app.modules.ocr_mrz.ocr_engine import OCREngineError, get_engine
    from app.modules.ocr_mrz.pipeline import run_ocr_mrz_on_path

    path = Path(args.target[0])
    if not path.exists():
        print(f"{RED}No such file: {path}{RESET}", file=sys.stderr)
        return 2

    try:
        engine = get_engine(args.engine)
    except OCREngineError as exc:
        print(f"{RED}{exc}{RESET}", file=sys.stderr)
        print(f"{DIM}Install one:  pip install paddleocr  |  pip install pytesseract{RESET}")
        return 3

    print(f"{DIM}Reading {path} with {engine.name}...{RESET}")
    result = run_ocr_mrz_on_path(path, engine=engine)

    print_fields(result.extracted_fields)
    print_mrz(result.mrz_check)
    if result.ocr:
        print(
            f"\n{DIM}OCR: {len(result.ocr.fields)} regions, "
            f"mean confidence {result.ocr.mean_confidence:.2f}, "
            f"{result.processing_time_ms} ms total{RESET}"
        )
    return 0 if result.mrz_check.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
