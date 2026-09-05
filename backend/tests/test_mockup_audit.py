"""Tests for the mockup claim auditor.

This script is a gate: it exits non-zero on anything a judge could check and find
wrong. A gate has two ways to fail, and the second is the dangerous one.

**Crying wolf.** It first matched bare "0.75" and "0.85" as substrings of the
whole file, so `opacity: 0.85` in a stylesheet failed a perfectly correct page.
A gate that fires on good input gets switched off.

**Going blind.** Fixing that by extracting "visible text" -- stripping <script>
along with <style> -- was worse. The team's mockups are single-file exports whose
entire UI is built in JavaScript: stripping scripts took one from 706,520
characters to 1,708, and the audit went from reporting 15 real defects to 5 and
exiting 0. It reported success because it could no longer see.

That is this project's recurring defect wearing a different hat, so both
directions are pinned here.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_mockup_claims.py"


def _load():
    spec = importlib.util.spec_from_file_location("audit_mockup_claims", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def audit():
    return _load()


# A correct page: every number quoted is one the system actually uses, and the
# stylesheet is full of values that look like stale claims but are not.
CLEAN_PAGE = """<html><head><style>
.card { opacity: 0.85; transition: transform 3.5s ease; }
.badge { transform: scale(0.75); animation: pulse 4.2s infinite; }
</style></head>
<body>
<h1>VeriDoc</h1>
<p>Face match 0.62 against a 0.40 threshold.</p>
<p>Analysis took 2.1s.</p>
</body></html>"""


# The shape the real mockups take: claims rendered from JavaScript, not present
# as literal HTML body text anywhere in the file.
SCRIPT_RENDERED_PAGE = """<html><head><style>.x { opacity: 0.9; }</style></head>
<body><div id="root"></div>
<script>
  const html = "<span>MODEL CONFIDENCE 0.71 - THRESHOLD 0.85</span>" +
               "<span>Face match 0.62 against a 0.75 threshold</span>" +
               "<span>Stamp and overprint consistency: pass</span>" +
               "<span>Analysis took 6.4s</span>";
  document.getElementById("root").innerHTML = html;
</script>
</body></html>"""


def _write(tmp_path: Path, name: str, body: str) -> Path:
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


class TestDoesNotCryWolf:
    def test_stylesheet_values_are_not_stale_claims(self, audit, tmp_path) -> None:
        """`opacity: 0.85` is not a claim about an auto-clear threshold."""
        problems = audit.audit(_write(tmp_path, "clean.html", CLEAN_PAGE))
        assert problems == [], f"false positives on a correct page: {problems}"

    def test_css_durations_are_not_verification_timings(self, audit, tmp_path) -> None:
        """A 3.5s CSS animation says nothing about how long a document takes."""
        problems = audit.audit(_write(tmp_path, "clean.html", CLEAN_PAGE))
        assert not any("timing" in p for p in problems)


class TestDoesNotGoBlind:
    """The regression that matters: claims live inside <script> in real mockups."""

    def test_claims_rendered_from_javascript_are_still_found(self, audit, tmp_path) -> None:
        path = _write(tmp_path, "mockup.html", SCRIPT_RENDERED_PAGE)
        problems = audit.audit(path)

        joined = " | ".join(problems).lower()
        for expected in ("0.85", "0.75", "model confidence", "stamp and overprint"):
            assert expected in joined, f"{expected!r} missed in: {problems}"
        assert any("timing" in p for p in problems), "6.4s not flagged"

    def test_script_bodies_are_not_stripped(self, audit) -> None:
        """Guards the mechanism directly, not just its outcome."""
        kept = audit.searchable_text(SCRIPT_RENDERED_PAGE)
        assert "MODEL CONFIDENCE" in kept, "script body was stripped"
        assert "opacity" not in kept, "stylesheet body was not stripped"


class TestExitCode:
    def test_a_bad_page_is_a_non_zero_exit(self, audit, tmp_path) -> None:
        path = _write(tmp_path, "mockup.html", SCRIPT_RENDERED_PAGE)
        assert audit.main(["audit_mockup_claims.py", str(path)]) == 1

    def test_a_clean_page_is_a_zero_exit(self, audit, tmp_path) -> None:
        path = _write(tmp_path, "clean.html", CLEAN_PAGE)
        assert audit.main(["audit_mockup_claims.py", str(path)]) == 0
