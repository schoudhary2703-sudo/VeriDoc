"""Evaluate the forensics engine against SIDTD — the cross-dataset check.

Every accuracy number this project quotes comes from FantasyID. That leaves one
question unanswered, and it is the question a judge asks: did the detectors learn
general tamper cues, or one dataset's signature?

SIDTD answers it. It is built by a different group (CVC, Universitat Autònoma de
Barcelona), from a different source (MIDV-2020 scans), with different forgery
methods (crop-and-replace inpainting on real document photographs) than
FantasyID's generative face swaps and diffusion text edits. Nothing in this
engine has ever seen it.

**There is no train/test split to respect here.** The engine is rules-based, its
thresholds were fixed on FantasyID and synthetic data, and nothing is fitted at
run time — so every SIDTD image is held out by construction. That makes this the
cleanest generalization test available to us, and it cuts both ways: a poor score
cannot be explained away as a tuning artefact.

Read the result against `docs/FORENSICS_FANTASYID.md`. What matters is not
whether SIDTD scores higher or lower, but whether the *shape* holds: near-zero
false positives on genuine documents, and detection well above chance on forged
ones. If the false-positive rate collapses on a new dataset, the zero we quote is
a property of FantasyID, not of the engine.

Usage:
    python -m ml.evaluate_sidtd --limit 150
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import time
from collections import defaultdict
from datetime import date
from pathlib import Path

import cv2

from app.modules.forensics import engine

# SIDTD filenames lead with a three-letter country code: alb_id, esp_id, rus_p.
NATIONALITY = re.compile(r"^([a-z]{3})_")


def find_images(root: Path) -> dict[str, list[Path]]:
    """Locate the reals/ and fakes/ image directories under an extracted SIDTD.

    The archive nests them a few levels down and the exact depth has changed
    between releases, so search rather than hardcode.
    """
    found: dict[str, list[Path]] = {}
    for label in ("reals", "fakes"):
        matches = [d for d in root.rglob(label) if d.is_dir()]
        images: list[Path] = []
        for directory in matches:
            images.extend(
                p for p in directory.iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
            )
        found[label] = sorted(images)
    return found


def nationality_of(path: Path) -> str:
    match = NATIONALITY.match(path.name.lower())
    return match.group(1) if match else "unknown"


def evaluate(paths: list[tuple[Path, bool]]) -> list[dict]:
    results: list[dict] = []
    for path, is_forged in paths:
        image = cv2.imread(str(path))
        if image is None:
            continue

        started = time.perf_counter()
        analysis = engine.analyze(image)
        elapsed = int((time.perf_counter() - started) * 1000)

        results.append(
            {
                "path": str(path),
                "is_forged": is_forged,
                "nationality": nationality_of(path),
                "flagged": analysis.tampered,
                "score": round(analysis.score, 3),
                "checks": [f.check for f in analysis.flagged_findings],
                "ms": elapsed,
            }
        )
    return results


def _rate(rows: list[dict]) -> str:
    if not rows:
        return "n/a"
    return f"{sum(r['flagged'] for r in rows) / len(rows):.0%}"


def render_report(results: list[dict], sampled: bool, seed: int) -> str:
    forged = [r for r in results if r["is_forged"]]
    genuine = [r for r in results if not r["is_forged"]]

    false_positives = sum(r["flagged"] for r in genuine)
    detected = sum(r["flagged"] for r in forged)

    fp_rate = false_positives / max(len(genuine), 1)
    detect_rate = detected / max(len(forged), 1)

    # Which check fired, and how often. On FantasyID the two families catch
    # disjoint sets, so whether that survives a new dataset is worth seeing.
    by_check: dict[str, int] = defaultdict(int)
    for row in forged:
        for check in row["checks"]:
            by_check[check] += 1

    by_nat: dict[str, list[dict]] = defaultdict(list)
    for row in forged:
        by_nat[row["nationality"]].append(row)

    # False positives per nationality. The first version of this report broke out
    # detection per nationality but not this, and so hid its own headline: every
    # false positive in the run came from a single document template. Aggregate
    # rates are where a template-specific failure goes to hide.
    genuine_by_nat: dict[str, list[dict]] = defaultdict(list)
    for row in genuine:
        genuine_by_nat[row["nationality"]].append(row)

    fp_checks: dict[str, int] = defaultdict(int)
    for row in genuine:
        for check in row["checks"]:
            fp_checks[check] += 1

    worst = max(
        genuine_by_nat.items(),
        key=lambda kv: sum(r["flagged"] for r in kv[1]) / max(len(kv[1]), 1),
        default=("", []),
    )
    worst_nat, worst_rows = worst
    worst_hits = sum(r["flagged"] for r in worst_rows)
    clean_rows = [r for r in genuine if r["nationality"] != worst_nat]
    clean_fp = sum(r["flagged"] for r in clean_rows)

    mean_ms = sum(r["ms"] for r in results) / max(len(results), 1)

    lines = [
        "# Forensics Results — SIDTD (cross-dataset validation)",
        "",
        f"Generated {date.today().isoformat()} by `python -m ml.evaluate_sidtd`.",
        "",
        "## What this file answers",
        "",
        "Every other accuracy number in this project comes from FantasyID. This is",
        "the only evidence about whether the detectors generalize.",
        "",
        "SIDTD is built by a different group (CVC, Universitat Autònoma de Barcelona)",
        "from a different source (MIDV-2020 document scans) using different forgery",
        "methods (crop-and-replace inpainting on real photographs) than FantasyID's",
        "generative face swaps and diffusion text edits. No threshold in this engine",
        "was tuned on any of it, and since the engine fits nothing at run time, every",
        "image here is held out by construction.",
        "",
        "## Headline",
        "",
        f"- Images evaluated: **{len(results)}** ({len(genuine)} genuine, {len(forged)} forged)"
        + (f"  — random sample, seed {seed}" if sampled else "  — all images"),
        f"- **False-positive rate on genuine documents: {false_positives}/{len(genuine)} "
        f"({fp_rate:.0%})**",
        f"- **Forgery detection rate: {detected}/{len(forged)} ({detect_rate:.0%})**",
        f"- Mean analysis time: {mean_ms:.0f} ms per image",
        "",
        "> The timing is `engine.analyze` only and reflects whatever the machine had",
        "> spare; it is not a benchmark. See the same note in `FORENSICS_FANTASYID.md`.",
        "",
        "## Which checks fired on forged documents",
        "",
        "| Check | Times flagged |",
        "|---|---|",
    ]

    for check, count in sorted(by_check.items(), key=lambda kv: -kv[1]):
        lines.append(f"| `{check}` | {count} |")
    if not by_check:
        lines.append("| *(none)* | 0 |")

    lines += [
        "",
        "## False positives per nationality — read this before the averages",
        "",
        "| Nationality | False positives | Rate |",
        "|---|---|---|",
    ]
    for nat in sorted(genuine_by_nat):
        rows = genuine_by_nat[nat]
        hits = sum(r["flagged"] for r in rows)
        mark = " **" if hits else " "
        lines.append(
            f"|{mark}`{nat}`{mark.rstrip()} | {hits}/{len(rows)} | {hits / len(rows):.0%} |"
        )

    if worst_rows and worst_hits:
        lines += [
            "",
            f"**{worst_hits} of the {false_positives} false positives are `{worst_nat}`** "
            f"({worst_hits}/{len(worst_rows)} of that template). Excluding it, the rate "
            f"over the remaining {len(clean_rows)} genuine documents is "
            f"**{clean_fp}/{len(clean_rows)} ({clean_fp / max(len(clean_rows), 1):.0%})**.",
            "",
            "That does not rescue the headline claim -- one template in ten producing",
            "false accusations is a deployment blocker, not a footnote -- but it does say",
            "what kind of problem this is. The thresholds are not globally too tight;",
            "one document design defeats them.",
        ]

    lines += [
        "",
        "## Which checks fired on genuine documents",
        "",
        "| Check | Times flagged |",
        "|---|---|",
    ]
    for check, count in sorted(fp_checks.items(), key=lambda kv: -kv[1]):
        lines.append(f"| `{check}` | {count} |")
    if not fp_checks:
        lines.append("| *(none)* | 0 |")

    lines += [
        "",
        "## Detection rate per document nationality",
        "",
        "| Nationality | Detected | Rate |",
        "|---|---|---|",
    ]
    for nat in sorted(by_nat):
        rows = by_nat[nat]
        hits = sum(r["flagged"] for r in rows)
        lines.append(f"| `{nat}` | {hits}/{len(rows)} | {hits / len(rows):.0%} |")

    lines += [
        "",
        "## Compared with FantasyID",
        "",
        "| | FantasyID | SIDTD |",
        "|---|---|---|",
        "| False positives on genuine | 0/150 (0%) | "
        f"{false_positives}/{len(genuine)} ({fp_rate:.0%}) |",
        "| Forgery detection | 82/300 (27%) | "
        f"{detected}/{len(forged)} ({detect_rate:.0%}) |",
        "",
        "FantasyID's 27% is over face swaps and text manipulations in equal number;",
        "SIDTD's forgeries are a different mix, so the two detection figures are not",
        "like for like and should not be averaged. **The false-positive rate is the",
        "comparable one**, and it is the one that matters: it is the property this",
        "project protects, and a rate that holds on an unseen dataset is evidence the",
        "thresholds are conservative rather than fitted.",
        "",
        "## How to read a low detection rate here",
        "",
        "A number well below FantasyID's would not, by itself, mean the engine failed.",
        "SIDTD's forgeries are region-level crop-and-replace edits on already-scanned",
        "documents, and rescanning destroys exactly the compression and sensor-noise",
        "traces ELA and copy-move rely on. What it *would* mean is that the honest",
        "claim is narrower than one dataset suggested — which is the reason to run it.",
        "",
        "A false-positive rate well above zero is the serious outcome. That would say",
        "the zero we quote is a property of FantasyID's captures rather than of the",
        "thresholds, and the headline claim would have to be withdrawn.",
    ]
    return "\n".join(lines)


def main() -> None:
    root_dir = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=root_dir / "data" / "raw" / "SIDTD")
    parser.add_argument("--out", type=Path, default=root_dir / "docs" / "FORENSICS_SIDTD.md")
    parser.add_argument(
        "--limit",
        type=int,
        default=150,
        help="Cap images per class (0 = everything). 150 matches the FantasyID run.",
    )
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    if not args.data.is_dir():
        raise SystemExit(f"SIDTD not found at {args.data}")

    images = find_images(args.data)
    if not images["reals"] or not images["fakes"]:
        raise SystemExit(
            f"found {len(images['reals'])} reals and {len(images['fakes'])} fakes under "
            f"{args.data}; expected both. Is the archive extracted?"
        )

    print(f"found {len(images['reals'])} genuine and {len(images['fakes'])} forged images")

    sampled = args.limit > 0
    if sampled:
        random.seed(args.seed)
        genuine = random.sample(images["reals"], min(args.limit, len(images["reals"])))
        forged = random.sample(images["fakes"], min(args.limit, len(images["fakes"])))
    else:
        genuine, forged = images["reals"], images["fakes"]

    paths = [(p, False) for p in genuine] + [(p, True) for p in forged]
    random.shuffle(paths)

    print(f"evaluating {len(paths)} images ...")
    results = evaluate(paths)

    detected = sum(r["flagged"] for r in results if r["is_forged"])
    false_positives = sum(r["flagged"] for r in results if not r["is_forged"])
    n_forged = sum(r["is_forged"] for r in results)
    n_genuine = len(results) - n_forged
    print(f"detection {detected}/{n_forged}, false positives {false_positives}/{n_genuine}")

    args.out.write_text(render_report(results, sampled, args.seed), encoding="utf-8")
    print(f"wrote {args.out}")

    # Per-image results, so re-rendering the report never needs another full pass.
    results_csv = args.out.with_suffix(".csv")
    with open(results_csv, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["path", "is_forged", "nationality", "flagged", "score", "checks", "ms"]
        )
        writer.writeheader()
        for row in results:
            writer.writerow({**row, "checks": "|".join(row["checks"])})
    print(f"wrote {results_csv}")


if __name__ == "__main__":
    main()
