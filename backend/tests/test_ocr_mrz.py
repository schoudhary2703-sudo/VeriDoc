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
