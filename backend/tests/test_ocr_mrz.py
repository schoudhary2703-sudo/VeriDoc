"""Phase 1 tests: MRZ parsing, checksum validation, and the sample documents.

The checksum tests use the published ICAO 9303 worked example, so they verify
our implementation against the spec rather than against itself.

Tests that need an OCR backend are skipped when none is installed; the MRZ logic
is pure and always runs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.core.schemas import MRZFormat, Sex
from app.modules.ocr_mrz.mrz_parser import (
    char_value,
    compute_check_digit,
    detect_format,
    find_mrz_lines,
    parse_mrz,
)

SAMPLES_DIR = Path(__file__).resolve().parents[2] / "data" / "samples"

# Published ICAO 9303 specimen (Utopian passport, "ANNA MARIA ERIKSSON").
ICAO_LINE_1 = "P<UTOERIKSSON<<ANNA<MARIA".ljust(44, "<")
ICAO_LINE_2 = "L898902C36UTO7408122F1204159ZE184226B<<<<<10"


class TestCheckDigitAlgorithm:
    def test_character_values(self) -> None:
        assert char_value("0") == 0
        assert char_value("9") == 9
        assert char_value("A") == 10
        assert char_value("Z") == 35
        assert char_value("<") == 0

    def test_rejects_invalid_character(self) -> None:
        with pytest.raises(ValueError):
            char_value("!")

    @pytest.mark.parametrize(
        ("data", "expected"),
        [
            ("L898902C3", "6"),   # document number
            ("740812", "2"),      # date of birth
            ("120415", "9"),      # expiry date
            ("ZE184226B<<<<<", "1"),  # personal number
        ],
    )
    def test_icao_worked_examples(self, data: str, expected: str) -> None:
        assert compute_check_digit(data) == expected

    def test_empty_data_is_zero(self) -> None:
        assert compute_check_digit("") == "0"


class TestFormatDetection:
    def test_detects_td3(self) -> None:
        assert detect_format([ICAO_LINE_2, ICAO_LINE_2]) is MRZFormat.TD3

    def test_detects_td2(self) -> None:
        assert detect_format(["A" * 36, "A" * 36]) is MRZFormat.TD2

    def test_detects_td1(self) -> None:
        assert detect_format(["A" * 30] * 3) is MRZFormat.TD1

    def test_rejects_wrong_length(self) -> None:
        assert detect_format(["A" * 41, "A" * 41]) is None

    def test_finds_mrz_inside_noisy_ocr_output(self) -> None:
        """OCR returns the printed fields too; the MRZ must still be located."""
        noisy = "\n".join(
            [
                "SPECIMEN TRAVEL DOCUMENT",
                "SURNAME ERIKSSON",
                "DATE OF BIRTH 12 AUG 1974",
                ICAO_LINE_1,
                ICAO_LINE_2,
            ]
        )
        fmt, lines = find_mrz_lines(noisy)
        assert fmt is MRZFormat.TD3
        assert lines == [ICAO_LINE_1, ICAO_LINE_2]


class TestGenuineMRZ:
    @pytest.fixture
    def parsed(self):
        return parse_mrz(f"{ICAO_LINE_1}\n{ICAO_LINE_2}")

    def test_all_check_digits_pass(self, parsed) -> None:
        _, check = parsed
        assert check.present
        assert check.valid
        assert check.checksum_match
        assert check.failed_checks == []
        assert len(check.checks) == 5

    def test_extracts_identity_fields(self, parsed) -> None:
        fields, _ = parsed
        assert fields.surname == "ERIKSSON"
        assert fields.given_names == "ANNA MARIA"
        assert fields.document_number == "L898902C3"
        assert fields.nationality == "UTO"
        assert fields.issuing_state == "UTO"
        assert fields.sex is Sex.FEMALE

    def test_parses_dates(self, parsed) -> None:
        fields, _ = parsed
        assert fields.dob is not None
        assert (fields.dob.year, fields.dob.month, fields.dob.day) == (1974, 8, 12)
        assert fields.expiry_date is not None
        assert (fields.expiry_date.year, fields.expiry_date.month) == (2012, 4)

    def test_summary_is_human_readable(self, parsed) -> None:
        _, check = parsed
        assert "valid" in check.summary().lower()


class TestTamperedMRZ:
    def test_single_altered_dob_digit_is_caught(self) -> None:
        """Change one DOB digit without fixing the checksum -- the core Phase 1 case."""
        tampered_line_2 = ICAO_LINE_2.replace("7408122", "6408122", 1)
        _, check = parse_mrz(f"{ICAO_LINE_1}\n{tampered_line_2}")

        assert check.present
        assert not check.valid
        assert not check.checksum_match

        failed = {c.field for c in check.failed_checks}
        assert "date_of_birth" in failed

    def test_failure_names_the_field(self) -> None:
        """Evidence must say *what* failed, not merely that something did."""
        tampered_line_2 = ICAO_LINE_2.replace("7408122", "6408122", 1)
        _, check = parse_mrz(f"{ICAO_LINE_1}\n{tampered_line_2}")

        failure = next(c for c in check.failed_checks if c.field == "date_of_birth")
        assert "date_of_birth" in failure.detail
        assert failure.expected != failure.actual
        assert "date_of_birth" in check.summary()

    def test_altered_document_number_is_caught(self) -> None:
        tampered = "L898902C37UTO7408122F1204159ZE184226B<<<<<10"
        _, check = parse_mrz(f"{ICAO_LINE_1}\n{tampered}")
        failed = {c.field for c in check.failed_checks}
        assert "document_number" in failed

    def test_missing_mrz_is_a_finding_not_an_error(self) -> None:
        fields, check = parse_mrz("SURNAME SHARMA\nDATE OF BIRTH 12 JUN 1998")
        assert check.present is False
        assert check.valid is False
        assert check.errors
        assert fields.document_number is None


class TestSpecimenDocuments:
    """Ground-truth checks against the generated sample set."""

    @pytest.fixture
    def manifest(self) -> list[dict]:
        path = SAMPLES_DIR / "manifest.json"
        if not path.exists():
            pytest.skip(
                "Sample documents not generated. Run: "
                "python -m ml.data_prep.generate_specimen_documents"
            )
        return json.loads(path.read_text(encoding="utf-8"))

    def test_manifest_has_genuine_and_tampered(self, manifest: list[dict]) -> None:
        labels = {entry["label"] for entry in manifest}
        assert "genuine" in labels
        assert "mrz_dob_digit_edit" in labels

    def test_each_specimen_matches_its_expected_verdict(self, manifest: list[dict]) -> None:
        for entry in manifest:
            _, check = parse_mrz("\n".join(entry["mrz_lines"]))
            assert check.present, entry["filename"]
            assert check.valid is entry["expect_mrz_valid"], entry["filename"]

            failed = {c.field for c in check.failed_checks}
            assert failed == set(entry["expect_failed_checks"]), entry["filename"]

    def test_specimen_images_exist(self, manifest: list[dict]) -> None:
        for entry in manifest:
            assert (SAMPLES_DIR / entry["filename"]).exists(), entry["filename"]


class TestOCRIntegration:
    """End-to-end OCR runs, skipped when no backend is installed."""

    @pytest.fixture
    def engine(self):
        from app.modules.ocr_mrz.ocr_engine import available_engines, get_engine

        if not available_engines():
            pytest.skip("No OCR backend installed (paddleocr or pytesseract)")
        return get_engine()

    def test_reads_mrz_from_genuine_specimen(self, engine) -> None:
        from app.modules.ocr_mrz.pipeline import run_ocr_mrz_on_path

        path = SAMPLES_DIR / "specimen_passport_genuine.png"
        if not path.exists():
            pytest.skip("Specimen images not generated")

        result = run_ocr_mrz_on_path(path, engine=engine)
        assert result.mrz_check.present, "OCR did not recover a readable MRZ"
        assert result.mrz_check.valid


class TestMRZBandLocalization:
    """The MRZ-band fast path: OCR only the zone that carries the checksum."""

    @pytest.fixture
    def genuine_image(self):
        cv2 = pytest.importorskip("cv2")
        path = SAMPLES_DIR / "specimen_passport_genuine.png"
        if not path.exists():
            pytest.skip("Specimen images not generated")
        return cv2.imread(str(path))

    def test_band_covers_every_mrz_line(self, genuine_image) -> None:
        """Regression: an early version returned a 42 px strip containing only
        the second of the two TD3 lines, silently losing half the zone."""
        from app.modules.preprocessing.normalize import find_mrz_band

        box = find_mrz_band(genuine_image)
        assert box is not None, "MRZ band not located"

        _, y1, _, y2 = box
        height = y2 - y1
        assert height > 60, f"band is {height}px; too short to hold two MRZ lines"
        # The band is a strip, not most of the page.
        assert height < genuine_image.shape[0] * 0.4

    def test_band_sits_in_the_lower_document(self, genuine_image) -> None:
        from app.modules.preprocessing.normalize import find_mrz_band

        _, y1, _, _ = find_mrz_band(genuine_image)
        assert y1 > genuine_image.shape[0] * 0.5


class TestTrailingFillerTolerance:
    """OCR drops trailing '<' fillers from name lines; the parser must cope."""

    def test_name_line_short_by_one_filler_is_accepted(self) -> None:
        truncated = ICAO_LINE_1[:-1]  # 43 chars, still ends in filler
        assert len(truncated) == 43
        _, check = parse_mrz(f"{truncated}\n{ICAO_LINE_2}")
        assert check.present
        assert check.valid

    def test_data_line_is_never_padded(self) -> None:
        """Padding line 2 would invent a check digit and turn a truncated read
        into a confident wrong answer."""
        from app.modules.ocr_mrz.mrz_parser import _fit_to_layout

        # Ends in a check digit, not filler -> must be refused.
        assert _fit_to_layout(ICAO_LINE_2[:-1], 44) is None
        # Ends in filler -> safe to restore.
        assert _fit_to_layout(ICAO_LINE_1[:-1], 44) == ICAO_LINE_1

    def test_excessive_truncation_is_refused(self) -> None:
        from app.modules.ocr_mrz.mrz_parser import _fit_to_layout

        assert _fit_to_layout(ICAO_LINE_1[:-10], 44) is None

# Reads taken verbatim from PP-OCRv5 on genuine SIDTD passports, not constructed.
_LVA_LAMBDA = (
    "P<LVAALKSNIS<<LIVA<<<<<<<<<<<<<<<<<<NΛNK<<<<\n"
    "LV57865342LVA9703083F2903161080397<12462<<22"
)
_AZE_LOGICAL_AND = (
    "P<AZEFARZHALIYEVA<<FATIHA<<<<<<<<<<<<<<<<<∧<\n"
    "C551731541AZE8711072F2911168CL0452U<<<<<<<20"
)
_CLEAN = (
    "P<INDSHARMA<<ANANYA<<<<<<<<<<<<<<<<<<<<<<<<<\n"
    "Z3541287<9IND9203147F3105319<<<<<<<<<<<<<<08"
)
# One digit of the date of birth altered; every other character intact.
_TAMPERED = (
    "P<INDSHARMA<<ANANYA<<<<<<<<<<<<<<<<<<<<<<<<<\n"
    "Z3541287<9IND9203148F3105319<<<<<<<<<<<<<<08"
)
# lva_passport_02: both trailing check digits recognised as filler.
_UNREADABLE = (
    "P<LVAAPSITIS<<AI<IS<<<<<<<<<<<<<<<<<<K<<<<<<\n"
    "LV56986325LVA9805161M2405171160598<16257<<<<"
)


class TestFillerConfusables:
    """Glyphs OCR substitutes for '<' on real passports.

    The original confusable set was all CJK, because our synthetic specimens were
    the only documents this parser had been measured against. Real passports
    produce a different set entirely; adding it took MRZ recovery on SIDTD from
    21% to 62%.
    """

    def test_confusables_repair_in_both_letter_cases(self) -> None:
        """`_normalize_lines` uppercases before translating.

        Adding lowercase 'lambda' alone recovered 2 of 12 Latvian passports;
        including its uppercase form recovered 8. A caseless table half-works
        silently, and the original CJK set could never have exposed it.
        """
        from app.modules.ocr_mrz.mrz_parser import FILLER, _FILLER_TRANSLATION

        for ch in "λΛκΚ∧≤·":
            assert ch.translate(_FILLER_TRANSLATION) == FILLER, f"{ch!r} unrepaired"

    def test_latvian_read_with_greek_lambda(self) -> None:
        _, check = parse_mrz(_LVA_LAMBDA)
        assert check.present
        assert check.mrz_format is MRZFormat.TD3

    def test_azerbaijani_read_with_logical_and(self) -> None:
        fields, check = parse_mrz(_AZE_LOGICAL_AND)
        assert check.present
        assert fields.issuing_state == "AZE"

    def test_repair_restores_a_valid_mrz_rather_than_a_failing_one(self) -> None:
        """The safety argument: substitution must not manufacture a failure.

        A wrong guess would not merely lose a read -- mrz_checksum carries weight
        0.95, so it would accuse a genuine traveller.
        """
        corrupted = _CLEAN.replace("ANANYA<<<", "ANANYA<λΛ")

        _, control = parse_mrz(_CLEAN)
        _, repaired = parse_mrz(corrupted)

        assert control.valid
        assert repaired.present
        assert repaired.valid


class TestUnreadableIsNotTampering:
    """A check digit read as filler means "could not read", not "forged".

    ICAO permits '<' in a check-digit position only when the field it protects is
    entirely filler. Anywhere else it is proof the recogniser missed a character,
    and reporting that as a checksum failure turns a genuine passport into a
    suspected forgery on the heaviest-weighted signal in the system.
    """

    def test_filler_check_digit_reports_not_present(self) -> None:
        _, check = parse_mrz(_UNREADABLE)
        assert check.present is False
        assert "could not be read" in " ".join(check.errors)

    def test_it_does_not_render_as_a_failed_checksum(self) -> None:
        """Guard the rendered status; this project's defects live downstream."""
        from app.core.risk_scoring import _mrz_evidence
        from app.core.schemas import EvidenceStatus

        item, contribution = _mrz_evidence(parse_mrz(_UNREADABLE)[1])

        assert item.status is EvidenceStatus.NOT_APPLICABLE
        assert contribution == 0.0, "an unread MRZ must not move the risk score"

    def test_a_real_check_digit_failure_still_fails(self) -> None:
        """The guard must not become a way for forgeries to slip through."""
        _, check = parse_mrz(_TAMPERED)
        assert check.present is True
        assert check.valid is False

    def test_a_legitimately_empty_optional_field_still_validates(self) -> None:
        """'<' IS allowed when the field it protects is entirely filler."""
        _, check = parse_mrz(_CLEAN)
        assert check.present is True
        assert check.valid is True
