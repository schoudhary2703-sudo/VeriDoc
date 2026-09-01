"""Generate synthetic SPECIMEN passport images for Phase 1 testing.

Every document produced here is fabricated: fictional names, fictional numbers,
a drawn placeholder in place of any photograph, and a SPECIMEN watermark across
the face of the card. No real personal document is ever used in this project --
see the non-negotiables in CLAUDE.md.

The tampered variant edits a date-of-birth digit *without* recomputing the MRZ
check digit, which is precisely the forgery a checksum validator is meant to
catch. That gives Phase 1 a positive control with known ground truth.

Usage:
    python -m ml.data_prep.generate_specimen_documents --out ../data/samples
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# Import path assumes the backend package root is importable.
from app.modules.ocr_mrz.mrz_parser import compute_check_digit

CARD_SIZE = (1000, 660)
BG_COLOR = (243, 240, 232)
INK = (28, 32, 44)
ACCENT = (120, 28, 36)
MRZ_BG = (252, 251, 247)

MONO_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\consola.ttf",
    r"C:\Windows\Fonts\cour.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
]
SANS_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\segoeui.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def _load_font(candidates: list[str], size: int) -> ImageFont.FreeTypeFont:
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default(size)


@dataclass
class Specimen:
    """Ground truth for one generated document."""

    filename: str
    label: str  # "genuine" or the tamper type applied
    surname: str
    given_names: str
    document_number: str
    nationality: str
    issuing_state: str
    dob: str  # YYMMDD as it appears in the MRZ
    sex: str
    expiry: str  # YYMMDD
    printed_dob: str  # human-readable, as printed on the card face
    mrz_lines: list[str]
    expect_mrz_valid: bool
    expect_failed_checks: list[str]


def build_td3_mrz(
    *,
    surname: str,
    given_names: str,
    document_number: str,
    nationality: str,
    issuing_state: str,
    dob: str,
    sex: str,
    expiry: str,
    personal_number: str = "",
    corrupt_dob_digit: str | None = None,
) -> list[str]:
    """Build a valid TD3 (passport) MRZ, optionally with a tampered DOB.

    When `corrupt_dob_digit` is given, the date of birth is rewritten *after* all
    check digits have been computed from the original -- so every digit stays
    self-consistent except the ones that depend on the DOB. That is what a naive
    forger produces, and what the checksum catches.
    """
    name_field = f"{surname}<<{given_names}".replace(" ", "<")
    line1 = f"P<{issuing_state}{name_field}".ljust(44, "<")[:44]

    doc_field = document_number.ljust(9, "<")[:9]
    personal_field = personal_number.ljust(14, "<")[:14]

    doc_check = compute_check_digit(doc_field)
    dob_check = compute_check_digit(dob)
    expiry_check = compute_check_digit(expiry)
    personal_check = compute_check_digit(personal_field)

    mrz_dob = corrupt_dob_digit if corrupt_dob_digit is not None else dob

    # Both check digits are derived from the ORIGINAL date of birth and left
    # untouched when the DOB is rewritten. A forger who edits the visible digits
    # but cannot recompute the checksums leaves exactly this signature, and both
    # the date-of-birth and composite checks then fail.
    composite_check = compute_check_digit(
        doc_field + doc_check + dob + dob_check + expiry + expiry_check
        + personal_field + personal_check
    )

    line2 = (
        f"{doc_field}{doc_check}{nationality}{mrz_dob}{dob_check}{sex}"
        f"{expiry}{expiry_check}{personal_field}{personal_check}{composite_check}"
    )
    assert len(line1) == 44, f"TD3 line 1 is {len(line1)} chars"
    assert len(line2) == 44, f"TD3 line 2 is {len(line2)} chars"
    return [line1, line2]


def _draw_photo_placeholder(image: Image.Image, box: tuple[int, int, int, int]) -> None:
    """Draw an obvious placeholder where a portrait would sit. Never a real face.

    The silhouette is composed on its own tile and pasted, so the shoulders are
    clipped by the photo frame instead of bleeding across the card.
    """
    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1

    tile = Image.new("RGB", (width, height), (214, 210, 200))
    tile_draw = ImageDraw.Draw(tile)

    cx = width // 2
    head_r = width // 6
    head_cy = height // 3
    tile_draw.ellipse(
        (cx - head_r, head_cy - head_r, cx + head_r, head_cy + head_r),
        fill=(178, 174, 166),
    )
    shoulder_w = int(width * 0.38)
    tile_draw.ellipse(
        (cx - shoulder_w, head_cy + head_r, cx + shoulder_w, height + head_r * 2),
        fill=(178, 174, 166),
    )

    image.paste(tile, (x1, y1))
    ImageDraw.Draw(image).rectangle(box, outline=(150, 146, 138), width=2)


def render_specimen(spec: Specimen) -> Image.Image:
    """Render one specimen document image."""
    image = Image.new("RGB", CARD_SIZE, BG_COLOR)
    draw = ImageDraw.Draw(image)

    title_font = _load_font(SANS_FONT_CANDIDATES, 30)
    label_font = _load_font(SANS_FONT_CANDIDATES, 15)
    value_font = _load_font(SANS_FONT_CANDIDATES, 24)
    mrz_font = _load_font(MONO_FONT_CANDIDATES, 30)

    width, height = CARD_SIZE

    draw.rectangle((0, 0, width, 78), fill=ACCENT)
    draw.text((32, 24), "SPECIMEN TRAVEL DOCUMENT", font=title_font, fill=(255, 255, 255))
    draw.text((width - 250, 32), "TYPE P    CODE IND", font=label_font, fill=(255, 230, 230))

    _draw_photo_placeholder(image, (40, 120, 280, 430))
    draw = ImageDraw.Draw(image)

    rows = [
        ("SURNAME", spec.surname),
        ("GIVEN NAMES", spec.given_names),
        ("PASSPORT NO.", spec.document_number),
        ("NATIONALITY", spec.nationality),
        ("DATE OF BIRTH", spec.printed_dob),
        ("SEX", spec.sex),
    ]
    y = 128
    for label, value in rows:
        draw.text((330, y), label, font=label_font, fill=(120, 118, 112))
        draw.text((330, y + 18), value, font=value_font, fill=INK)
        y += 52

    # Watermark: unmistakable, and diagonal so it cannot be cropped away cleanly.
    watermark = Image.new("RGBA", CARD_SIZE, (0, 0, 0, 0))
    wm_draw = ImageDraw.Draw(watermark)
    wm_font = _load_font(SANS_FONT_CANDIDATES, 66)
    wm_draw.text((150, 300), "SPECIMEN - NOT VALID", font=wm_font, fill=(200, 60, 60, 70))
    image = Image.alpha_composite(image.convert("RGBA"), watermark.rotate(12)).convert("RGB")

    draw = ImageDraw.Draw(image)
    mrz_top = height - 150
    draw.rectangle((0, mrz_top, width, height), fill=MRZ_BG)
    draw.line((0, mrz_top, width, mrz_top), fill=(200, 196, 188), width=2)

    y = mrz_top + 26
    for line in spec.mrz_lines:
        draw.text((26, y), line, font=mrz_font, fill=INK)
        y += 52

    return image


def build_specimens() -> list[Specimen]:
    """Define the sample set: one genuine document and one with an edited DOB."""
    common = dict(
        surname="SHARMA",
        given_names="ANANYA DEVI",
        document_number="Z9081234",
        nationality="IND",
        issuing_state="IND",
        sex="F",
        expiry="330415",
    )
    true_dob = "980612"        # 12 June 1998
    forged_dob = "880612"      # aged ten years by a single digit

    genuine_mrz = build_td3_mrz(dob=true_dob, **common)
    tampered_mrz = build_td3_mrz(dob=true_dob, corrupt_dob_digit=forged_dob, **common)

    return [
        Specimen(
            filename="specimen_passport_genuine.png",
            label="genuine",
            dob=true_dob,
            printed_dob="12 JUN 1998",
            mrz_lines=genuine_mrz,
            expect_mrz_valid=True,
            expect_failed_checks=[],
            **common,
        ),
        Specimen(
            filename="specimen_passport_tampered_dob.png",
            label="mrz_dob_digit_edit",
            dob=forged_dob,
            printed_dob="12 JUN 1988",
            mrz_lines=tampered_mrz,
            expect_mrz_valid=False,
            expect_failed_checks=["date_of_birth", "composite"],
            **common,
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[3] / "data" / "samples",
        help="Directory to write specimen images and the ground-truth manifest into",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    specimens = build_specimens()
    for spec in specimens:
        image = render_specimen(spec)
        image.save(args.out / spec.filename)
        print(f"wrote {args.out / spec.filename}  [{spec.label}]")

    manifest = args.out / "manifest.json"
    manifest.write_text(
        json.dumps([asdict(s) for s in specimens], indent=2),
        encoding="utf-8",
    )
    print(f"wrote {manifest}")


if __name__ == "__main__":
    main()
