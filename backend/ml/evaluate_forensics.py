"""Evaluate the forensics engine per tamper type and write a results file.

Per-type reporting is a project non-negotiable: a detector that catches photo
splices but misses date edits is a different and more dangerous tool than one
that is uniformly mediocre, and a single blended accuracy figure hides exactly
that difference.

Usage:
    python -m ml.evaluate_forensics
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import date
from pathlib import Path

import cv2

from app.modules.forensics import copy_move, engine


def evaluate(synthetic_dir: Path) -> list[dict]:
    manifest = json.loads((synthetic_dir / "forgeries.json").read_text(encoding="utf-8"))

    rows: list[dict] = []
    for entry in manifest:
        image = cv2.imread(str(synthetic_dir / entry["filename"]))
        if image is None:
            raise SystemExit(f"could not read {entry['filename']}")

        result = engine.analyze(image)
        expected = entry["tamper_type"] != "genuine"

        rows.append(
            {
                "filename": entry["filename"],
                "tamper_type": entry["tamper_type"],
                "expected_tampered": expected,
                "detected_tampered": result.tampered,
                "correct": result.tampered == expected,
                "score": round(result.score, 3),
                "flagged_checks": [f.check for f in result.flagged_findings],
                "detail": "; ".join(f.detail for f in result.flagged_findings),
                "processing_time_ms": result.processing_time_ms,
            }
        )
    return rows


def render_report(rows: list[dict]) -> str:
    positives = [r for r in rows if r["expected_tampered"]]
    negatives = [r for r in rows if not r["expected_tampered"]]

    true_positives = sum(r["detected_tampered"] for r in positives)
    false_positives = sum(r["detected_tampered"] for r in negatives)

    per_type: dict[str, list[dict]] = defaultdict(list)
    for row in positives:
        per_type[row["tamper_type"]].append(row)

    lines = [
        "# Forensics Results — Classical Detectors (Phase 2)",
        "",
        f"Generated {date.today().isoformat()} by `python -m ml.evaluate_forensics`.",
        "",
        "## Read this before quoting any number below",
        "",
        "These figures come from **one synthetic specimen per tamper type plus one",
        "genuine control** — five images in total. That is a smoke test, not a",
        "benchmark. Detection rates on a sample of one are either 0% or 100% and",
        "carry no confidence interval whatsoever.",
        "",
        "The thresholds were also fitted against this same tiny set, so these",
        "numbers describe the tuning data, not held-out performance. Real accuracy",
        "figures require IDNet for training/validation and SIDTD as a held-out",
        "cross-dataset check (see `docs/DATA_STRATEGY.md`). Nothing here should",
        "appear in a submission as an accuracy claim.",
        "",
        "## Per tamper type",
        "",
        "| Tamper type | Detected | Score | Flagged by |",
        "|---|---|---|---|",
    ]

    for tamper_type in sorted(per_type):
        for row in per_type[tamper_type]:
            checks = ", ".join(row["flagged_checks"]) or "—"
            mark = "yes" if row["detected_tampered"] else "**NO**"
            lines.append(f"| `{tamper_type}` | {mark} | {row['score']:.3f} | {checks} |")

    lines += [
        "",
        "## Negative control",
        "",
        "| Sample | Flagged | Score |",
        "|---|---|---|",
    ]
    for row in negatives:
        mark = "**FALSE POSITIVE**" if row["detected_tampered"] else "no (correct)"
        lines.append(f"| `{row['tamper_type']}` | {mark} | {row['score']:.3f} |")

    detection_rate = (true_positives / len(positives) * 100) if positives else 0.0
    lines += [
        "",
        "## Summary",
        "",
        f"- Tamper types detected: **{true_positives}/{len(positives)}** ({detection_rate:.0f}%)",
        f"- False positives on genuine documents: **{false_positives}/{len(negatives)}**",
        f"- Decision threshold: `{engine.TAMPER_THRESHOLD}`",
        "",
        "## Active checks and their weights",
        "",
        "| Check | Weight | Status |",
        "|---|---|---|",
    ]

    for check, weight in engine.CHECK_WEIGHTS.items():
        if check == "cnn_classifier":
            status = "not yet trained (Phase 2 CNN)"
        elif check == "noise_consistency":
            status = "**advisory — not calibrated on real captures**"
        elif weight > 0:
            status = "active"
        else:
            status = "inactive"
        lines.append(f"| `{check}` | {weight:.2f} | {status} |")

    lines += [
        "",
        "## Known limitations",
        "",
        "**Noise consistency is disabled.** On synthetic renders it does not",
        "discriminate: the genuine control peaked at 47.2 SD above the median block",
        "energy while the four forgeries reached 19.6–46.1, meaning the clean",
        "document looked *more* anomalous than every forgery. Normalizing for local",
        "image activity did not help (51.1 vs 37.3–50.1). The cause is that",
        "synthetic documents have no real sensor-noise field, so JPEG residual",
        "energy tracks local content instead. The method is sound on genuine camera",
        "captures and should be re-validated once IDNet/FantasyID data is available;",
        f"see `NOISE_DETECTOR_VALIDATED` in `{copy_move.__name__}`.",
        "",
        "**Copy-move is dormant on these samples.** It correctly stays silent on the",
        "genuine control — an earlier version matched the MRZ's repeated `<` fillers",
        "against themselves and reported 73 'cloned' keypoints sharing a (+104, 0)",
        "offset. None of the four synthetic forgeries is a clone attack, so the",
        "detector has no positive case here and is currently unverified in the",
        "positive direction.",
        "",
        "**No pixel-level localization is claimed.** Regions are coarse bounding",
        "boxes. Tampered areas on ID documents occupy 0.27–4.17% of the image and",
        "state-of-the-art detectors score near-zero on pixel-level localization",
        "(DocForge-Bench, 2026).",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", type=Path, default=root / "data" / "synthetic")
    parser.add_argument("--out", type=Path, default=root / "docs" / "FORENSICS_RESULTS.md")
    args = parser.parse_args()

    rows = evaluate(args.synthetic)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_report(rows), encoding="utf-8")

    for row in rows:
        status = "OK  " if row["correct"] else "MISS"
        print(f"{status} {row['tamper_type']:18} score={row['score']:.3f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
