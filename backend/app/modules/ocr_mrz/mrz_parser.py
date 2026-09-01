"""ICAO 9303 machine-readable-zone parser and checksum validator.

The MRZ is a fixed specification, not a learning problem -- so this is entirely
rules-based, fully deterministic, and fully explainable. That matters: a failing
check digit here is the single cheapest, most defensible fraud signal in the
whole pipeline, and we can state exactly why it failed.

Supported layouts (ICAO 9303 Parts 3-5):

    TD1  3 lines x 30 chars  identity cards
    TD2  2 lines x 36 chars  older travel documents
    TD3  2 lines x 44 chars  passports

Check-digit algorithm (9303 Part 3, Section 4.9): each character is weighted
cyclically by 7, 3, 1 and summed modulo 10. Digits score their face value,
letters A-Z score 10-35, and the filler '<' scores 0.
"""

from __future__ import annotations

import re
from datetime import date

from app.core.schemas import (
    CheckDigitResult,
    ExtractedFields,
    MRZCheckResult,
    MRZFormat,
    Sex,
)

FILLER = "<"
_WEIGHTS = (7, 3, 1)
_VALID_CHARS = re.compile(r"^[A-Z0-9<]+$")

# Line lengths that identify each layout.
_LAYOUTS: dict[MRZFormat, tuple[int, int]] = {
    MRZFormat.TD1: (3, 30),
    MRZFormat.TD2: (2, 36),
    MRZFormat.TD3: (2, 44),
}


def char_value(char: str) -> int:
    """Numeric weight of a single MRZ character."""
    if char == FILLER:
        return 0
    if char.isdigit():
        return int(char)
    if "A" <= char <= "Z":
        return ord(char) - ord("A") + 10
    raise ValueError(f"Character {char!r} is not valid in an MRZ")


def compute_check_digit(data: str) -> str:
    """Return the ICAO 9303 check digit for `data` as a single character."""
    total = sum(char_value(c) * _WEIGHTS[i % 3] for i, c in enumerate(data))
    return str(total % 10)


def _check(field: str, raw: str, actual: str, *, filler_ok: bool = False) -> CheckDigitResult:
    """Validate one check digit and record the outcome as evidence.

    `filler_ok` covers the optional-data case: when the underlying field is
    entirely filler, ICAO permits the check digit to be '<' or '0'.
    """
    try:
        expected = compute_check_digit(raw)
    except ValueError:
        expected = "?"

    if filler_ok and set(raw) <= {FILLER} and actual in {FILLER, "0"}:
        passed = True
    else:
        passed = actual == expected

    return CheckDigitResult(
        field=field, raw_value=raw, expected=expected, actual=actual, passed=passed
    )


def _parse_date(value: str, *, is_expiry: bool) -> date | None:
    """Parse a YYMMDD MRZ date into a real date.

    The MRZ carries only two year digits, so the century is inferred: a date of
    birth cannot be in the future, while an expiry date generally is. Documents
    with a >70-year validity window would break this, which does not occur in
    practice for the formats we handle.
    """
    if len(value) != 6 or not value.isdigit():
        return None

    yy, mm, dd = int(value[0:2]), int(value[2:4]), int(value[4:6])
    today = date.today()

    if is_expiry:
        century = 1900 if yy > 70 else 2000
    else:
        century = 2000 if yy <= today.year % 100 else 1900

    try:
        return date(century + yy, mm, dd)
    except ValueError:
        return None


def _clean_field(value: str) -> str | None:
    """Strip MRZ filler characters from a field value."""
    cleaned = value.replace(FILLER, " ").strip()
    return cleaned or None


def _parse_names(value: str) -> tuple[str | None, str | None]:
    """Split an MRZ name field into (surname, given names).

    The separator is a double filler; single fillers inside each part are spaces
    between name components.
    """
    surname_part, _, given_part = value.partition(FILLER * 2)
    surname = _clean_field(surname_part)
    given = _clean_field(given_part)
    return surname, given


def _normalize_lines(text: str) -> list[str]:
    """Extract candidate MRZ lines from raw OCR text."""
    lines: list[str] = []
    for raw in text.replace("\r", "\n").split("\n"):
        line = raw.strip().upper().replace(" ", "")
        if line:
            lines.append(line)
    return lines


def detect_format(lines: list[str]) -> MRZFormat | None:
    """Identify the MRZ layout from line count and length."""
    for fmt, (count, length) in _LAYOUTS.items():
        if len(lines) == count and all(len(line) == length for line in lines):
            return fmt
    return None


def find_mrz_lines(text: str) -> tuple[MRZFormat | None, list[str]]:
    """Locate MRZ lines inside arbitrary OCR output.

    OCR of a full document returns the printed fields as well as the MRZ, so we
    scan for a run of consecutive lines that match a known layout exactly.
    """
    candidates = [line for line in _normalize_lines(text) if _VALID_CHARS.match(line)]

    for fmt, (count, length) in _LAYOUTS.items():
        run: list[str] = []
        for line in candidates:
            if len(line) == length:
                run.append(line)
                if len(run) == count:
                    return fmt, run
            else:
                run = []

    return None, []


def _parse_td3(lines: list[str]) -> tuple[ExtractedFields, list[CheckDigitResult], list[str]]:
    l1, l2 = lines
    errors: list[str] = []

    surname, given = _parse_names(l1[5:44])
    doc_number = l2[0:9]
    nationality = l2[10:13]
    dob_raw = l2[13:19]
    expiry_raw = l2[21:27]
    personal_number = l2[28:42]

    composite_raw = l2[0:10] + l2[13:20] + l2[21:43]

    checks = [
        _check("document_number", doc_number, l2[9]),
        _check("date_of_birth", dob_raw, l2[19]),
        _check("expiry_date", expiry_raw, l2[27]),
        _check("personal_number", personal_number, l2[42], filler_ok=True),
        _check("composite", composite_raw, l2[43]),
    ]

    dob = _parse_date(dob_raw, is_expiry=False)
    expiry = _parse_date(expiry_raw, is_expiry=True)
    if dob is None:
        errors.append(f"Unparseable date of birth {dob_raw!r}")
    if expiry is None:
        errors.append(f"Unparseable expiry date {expiry_raw!r}")

    fields = ExtractedFields(
        surname=surname,
        given_names=given,
        name=" ".join(p for p in (given, surname) if p) or None,
        document_number=_clean_field(doc_number),
        nationality=_clean_field(nationality),
        issuing_state=_clean_field(l1[2:5]),
        dob=dob,
        expiry_date=expiry,
        sex=_to_sex(l2[20]),
        personal_number=_clean_field(personal_number),
    )
    return fields, checks, errors


def _parse_td2(lines: list[str]) -> tuple[ExtractedFields, list[CheckDigitResult], list[str]]:
    l1, l2 = lines
    errors: list[str] = []

    surname, given = _parse_names(l1[5:36])
    doc_number = l2[0:9]
    dob_raw = l2[13:19]
    expiry_raw = l2[21:27]
    optional = l2[28:35]

    composite_raw = l2[0:10] + l2[13:20] + l2[21:35]

    checks = [
        _check("document_number", doc_number, l2[9]),
        _check("date_of_birth", dob_raw, l2[19]),
        _check("expiry_date", expiry_raw, l2[27]),
        _check("composite", composite_raw, l2[35]),
    ]

    dob = _parse_date(dob_raw, is_expiry=False)
    expiry = _parse_date(expiry_raw, is_expiry=True)
    if dob is None:
        errors.append(f"Unparseable date of birth {dob_raw!r}")
    if expiry is None:
        errors.append(f"Unparseable expiry date {expiry_raw!r}")

    fields = ExtractedFields(
        surname=surname,
        given_names=given,
        name=" ".join(p for p in (given, surname) if p) or None,
        document_number=_clean_field(doc_number),
        nationality=_clean_field(l2[10:13]),
        issuing_state=_clean_field(l1[2:5]),
        dob=dob,
        expiry_date=expiry,
        sex=_to_sex(l2[20]),
        personal_number=_clean_field(optional),
    )
    return fields, checks, errors


def _parse_td1(lines: list[str]) -> tuple[ExtractedFields, list[CheckDigitResult], list[str]]:
    l1, l2, l3 = lines
    errors: list[str] = []

    doc_number = l1[5:14]
    optional1 = l1[15:30]
    dob_raw = l2[0:6]
    expiry_raw = l2[8:14]
    optional2 = l2[18:29]

    composite_raw = l1[5:30] + l2[0:7] + l2[8:15] + l2[18:29]

    checks = [
        _check("document_number", doc_number, l1[14]),
        _check("date_of_birth", dob_raw, l2[6]),
        _check("expiry_date", expiry_raw, l2[14]),
        _check("composite", composite_raw, l2[29]),
    ]

    surname, given = _parse_names(l3)

    dob = _parse_date(dob_raw, is_expiry=False)
    expiry = _parse_date(expiry_raw, is_expiry=True)
    if dob is None:
        errors.append(f"Unparseable date of birth {dob_raw!r}")
    if expiry is None:
        errors.append(f"Unparseable expiry date {expiry_raw!r}")

    fields = ExtractedFields(
        surname=surname,
        given_names=given,
        name=" ".join(p for p in (given, surname) if p) or None,
        document_number=_clean_field(doc_number),
        nationality=_clean_field(l2[15:18]),
        issuing_state=_clean_field(l1[2:5]),
        dob=dob,
        expiry_date=expiry,
        sex=_to_sex(l2[7]),
        personal_number=_clean_field(optional1) or _clean_field(optional2),
    )
    return fields, checks, errors


def _to_sex(char: str) -> Sex | None:
    mapping = {"M": Sex.MALE, "F": Sex.FEMALE, "X": Sex.UNSPECIFIED, FILLER: Sex.UNSPECIFIED}
    return mapping.get(char)


_PARSERS = {
    MRZFormat.TD1: _parse_td1,
    MRZFormat.TD2: _parse_td2,
    MRZFormat.TD3: _parse_td3,
}


def parse_mrz(text: str) -> tuple[ExtractedFields, MRZCheckResult]:
    """Parse an MRZ out of `text` and validate every check digit.

    Accepts either the MRZ lines alone or the full OCR output of a document.
    Returns the extracted fields alongside a per-check-digit result set; an
    unparseable or absent MRZ yields `present=False` rather than raising, since
    a missing MRZ is a finding for the risk scorer, not an error.
    """
    fmt, lines = find_mrz_lines(text)

    if fmt is None:
        return ExtractedFields(), MRZCheckResult(
            present=False,
            errors=["No MRZ matching a TD1, TD2, or TD3 layout was found"],
        )

    try:
        fields, checks, errors = _PARSERS[fmt](lines)
    except ValueError as exc:
        return ExtractedFields(), MRZCheckResult(
            present=True,
            mrz_format=fmt,
            raw_lines=lines,
            errors=[f"MRZ contains invalid characters: {exc}"],
        )

    checksum_match = all(c.passed for c in checks)
    result = MRZCheckResult(
        present=True,
        mrz_format=fmt,
        raw_lines=lines,
        checks=checks,
        checksum_match=checksum_match,
        valid=checksum_match and not errors,
        errors=errors,
    )
    return fields, result
