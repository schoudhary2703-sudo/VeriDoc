"""FantasyID as a torch Dataset, using the dataset's own train/test protocol.

Using the published CSVs rather than re-splitting matters: FantasyID's two splits
deliberately use *different card templates*, so a home-made random split would
leak template identity between train and test and inflate every number we report.

**The task is binary, and that is forced by the data, not chosen for convenience.**
The two splits use different attack taxonomies:

    train : 1266 x "face_text"        face swap AND text edit, combined
    test  :  150 x "face"             face swap alone
             935 x "text"             text inpainting alone

There is no split in which the individual manipulation types are both present as
training labels, so a per-type classifier cannot be fitted here. The model
predicts genuine vs manipulated, and per-type recall is reported at evaluation
time from `attack_type` -- which still satisfies the per-tamper-type reporting
rule, since that rule is about how results are *reported*, not how many output
neurons the model has.

Note this makes the benchmark harder, deliberately: the model trains on combined
manipulations and is tested on each one in isolation.
"""

from __future__ import annotations

import csv
from pathlib import Path

CLASS_LABELS = ["genuine", "manipulated"]


def label_for(row: dict) -> str:
    """Map one CSV row onto a project class label."""
    return "manipulated" if row["is_attack"] == "True" else "genuine"


def read_split(root: Path, split: str) -> list[tuple[Path, int]]:
    """Return [(image_path, class_index)] for the given split."""
    index = {name: i for i, name in enumerate(CLASS_LABELS)}
    with open(root / f"{split}.csv", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    samples: list[tuple[Path, int]] = []
    for row in rows:
        path = root / row["path"]
        if path.exists():
            samples.append((path, index[label_for(row)]))
    return samples


def read_split_with_types(root: Path, split: str) -> list[tuple[Path, int, str]]:
    """As `read_split`, but keeps attack_type for per-type recall reporting."""
    index = {name: i for i, name in enumerate(CLASS_LABELS)}
    with open(root / f"{split}.csv", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    out: list[tuple[Path, int, str]] = []
    for row in rows:
        path = root / row["path"]
        if path.exists():
            attack_type = (row.get("attack_type") or "bonafide").strip() or "bonafide"
            out.append((path, index[label_for(row)], attack_type))
    return out


class FantasyIDDataset:
    """Minimal torch-compatible dataset. torch is imported lazily by the caller."""

    def __init__(self, root: Path, split: str, transform=None) -> None:
        self.root = Path(root)
        self.samples = read_split(self.root, split)
        self.transform = transform
        self.classes = CLASS_LABELS

        if not self.samples:
            raise RuntimeError(f"No images found for split {split!r} under {root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        from PIL import Image

        path, target = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, target

    def class_counts(self) -> dict[str, int]:
        counts = dict.fromkeys(self.classes, 0)
        for _, target in self.samples:
            counts[self.classes[target]] += 1
        return counts
