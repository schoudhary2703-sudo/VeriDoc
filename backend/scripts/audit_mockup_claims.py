"""Audit a design mockup or slide export for claims the system cannot back.

Written after four separate classes of error were found in the team's Verify
Screen and Officer Console mockups. Run it on any HTML the team plans to put in
front of a judge.

It checks three things:

1. **Every MRZ is well formed.** 44 characters per line and all five ICAO 9303
   check digits recompute. This is the one a judge can verify with a phone.
2. **Every MRZ agrees with the printed fields on the same page.** An MRZ can
   pass its own check digits and still contradict the printed date of birth or
   expiry -- which is precisely the forgery the cross-check is meant to catch,
   so a screen that contains one is arguing against itself.
3. **No stale claims.** Superseded thresholds, a non-existent "model
   confidence", auto-clear language, checks we do not implement, and timings
   that no build has ever produced.

Usage:
    python scripts/audit_mockup_claims.py <file.html> [<file.html> ...]

Exits non-zero if anything is found, so it can gate a commit.
"""

from __future__ import annotations

import html
import re
import sys
from pathlib import Path

WEIGHTS = (7, 3, 1)
MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")

# Line 2 of a TD3 MRZ: docnum(9) cd nat(3) dob(6) cd sex exp(6) cd ...
TD3_LINE2 = re.compile(r"^[0-9A-Z<]{9}\d[A-Z]{3}\d{6}\d[MF<]\d{6}\d")
MRZ_CANDIDATE = re.compile(r"(?:[A-Z][0-9A-Z]{7})&lt;(?:&lt;|[A-Z0-9])+")

# Claims that outran the system. See docs/UI_REVIEW.md.
STALE_CLAIMS = {
    "0.75": "superseded face threshold (real: 0.40, face_match.py DEFAULT_MATCH_THRESHOLD)",
    "0.85": "auto-clear threshold -- there is no auto-clear",
    "Model confidence": "no such metric; only per-check confidence exists",
    "MODEL CONFIDENCE": "no such metric; only per-check confidence exists",
    "auto-clear threshold. Confidence": "auto-clear language",
    "Stamp and overprint": "this check does not exist in the engine",
}

# Warm pipeline is ~1.9 s, ~2.2 s with face analysis. Anything much past that
# was invented.
MAX_PLAUSIBLE_SECONDS = 3.0


def char_value(c: str) -> int:
    if c == "<":
        return 0
    if c.isdigit():
        return int(c)
    return ord(c) - ord("A") + 10


def check_digit(data: str) -> str:
    return str(sum(char_value(c) * WEIGHTS[i % 3] for i, c in enumerate(data)) % 10)


def validate(line2: str) -> list[str]:
    """Return the names of failing check digits."""
    composite = line2[0:10] + line2[13:20] + line2[21:43]
    fields = [
        ("document_number", line2[0:9], line2[9]),
        ("date_of_birth", line2[13:19], line2[19]),
        ("expiry_date", line2[21:27], line2[27]),
        ("personal_number", line2[28:42], line2[42]),
        ("composite", composite, line2[43]),
    ]
    failed = []
    for name, raw, actual in fields:
        # ICAO permits '<' or '0' when the underlying field is all filler.
        if set(raw) <= {"<"} and actual in "<0":
            continue
        if check_digit(raw) != actual:
            failed.append(name)
    return failed


def decode(line2: str) -> dict:
    yy = int(line2[13:15])
    ey = int(line2[21:23])
    return {
        "document_number": line2[0:9].rstrip("<"),
        "nationality": line2[10:13],
        "date_of_birth": (1900 + yy if yy > 40 else 2000 + yy,
                          int(line2[15:17]), int(line2[17:19])),
        "sex": line2[20],
        "expiry_date": (2000 + ey, int(line2[23:25]), int(line2[25:27])),
    }


def date_forms(t: tuple[int, int, int]) -> list[str]:
    year, month, day = t
    return ["%02d %s %d" % (day, MONTHS[month - 1], year),
            "%02d/%02d/%d" % (day, month, year)]


def audit(path: Path) -> list[str]:
    raw = path.read_text(encoding="utf-8")
    text = html.unescape(re.sub(r"<[^>]*>", " ", raw))
    folded = text.casefold()
    problems: list[str] = []

    print("=" * 70)
    print(path.name)
    print("=" * 70)

    seen: set[str] = set()
    for match in MRZ_CANDIDATE.finditer(raw):
        line2 = html.unescape(match.group(0))
        if line2 in seen or not TD3_LINE2.match(line2):
            continue
        seen.add(line2)

        print("\n  %s" % line2)
        if len(line2) != 44:
            print("      FAIL  length %d, TD3 requires 44" % len(line2))
            problems.append("%s: MRZ length %d: %s" % (path.name, len(line2), line2))
            continue

        failed = validate(line2)
        if failed:
            print("      FAIL  check digits: %s" % ", ".join(failed))
            problems.append("%s: MRZ check digits fail (%s): %s"
                            % (path.name, ",".join(failed), line2))
        else:
            print("      ok    44 chars, all five check digits recompute")

        d = decode(line2)
        print("      decodes: %s  %s  born %s  %s  expires %s"
              % (d["document_number"], d["nationality"],
                 date_forms(d["date_of_birth"])[0], d["sex"],
                 date_forms(d["expiry_date"])[0]))

        # Cross-check against the printed page. Only assert a date format the
        # page actually uses, and compare case-insensitively -- mockups render
        # the same date as "14 Mar 1992" or "14 MAR 1992" depending on the panel.
        if d["document_number"].casefold() not in folded:
            print("      FAIL  document number not printed anywhere on the page")
            problems.append("%s: %s not printed" % (path.name, d["document_number"]))

        for label, value in (("date of birth", d["date_of_birth"]),
                             ("expiry", d["expiry_date"])):
            forms = date_forms(value)
            present = [f for f in forms if f.casefold() in folded]
            if present:
                print("      ok    printed %s agrees: %s" % (label, present[0]))
                continue
            # Does the page show that field in *some* form? If a sibling date in
            # the same style exists, the MRZ and the printed page disagree.
            styles = [r"\d{2} [A-Za-z]{3} \d{4}", r"\d{2}/\d{2}/\d{4}"]
            page_uses = any(re.search(p, text) for p in styles)
            if page_uses:
                print("      FAIL  printed %s does NOT match MRZ (%s)"
                      % (label, " or ".join(forms)))
                problems.append("%s: MRZ %s %s contradicts the printed page"
                                % (path.name, label, forms[0]))
            else:
                print("      --    %s not printed on this page, nothing to compare" % label)

    for needle, why in STALE_CLAIMS.items():
        if needle in raw:
            print("\n  FAIL  stale claim %r -- %s" % (needle, why))
            problems.append("%s: stale claim %r (%s)" % (path.name, needle, why))

    timings = sorted({float(t[:-1]) for t in re.findall(r"\b\d+\.\d+s\b", raw)})
    if timings:
        slow = [t for t in timings if t > MAX_PLAUSIBLE_SECONDS]
        print("\n  timings shown: %s" % timings)
        if slow:
            print("      FAIL  no build produces %s (warm run is ~1.9-2.2 s)" % slow)
            problems.append("%s: implausible timings %s" % (path.name, slow))

    print()
    return problems


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2

    problems: list[str] = []
    for arg in argv[1:]:
        path = Path(arg)
        if not path.is_file():
            print("no such file: %s" % arg)
            return 2
        problems.extend(audit(path))

    print("=" * 70)
    if problems:
        print("%d problem(s) found:" % len(problems))
        for p in problems:
            print("  - %s" % p)
        return 1
    print("clean -- every MRZ validates, agrees with its printed page, "
          "and no stale claims remain")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
