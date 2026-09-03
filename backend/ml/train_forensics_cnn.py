"""Fine-tune EfficientNet-B0 to classify document tamper types.

Expects a directory of class folders:

    <data>/genuine/*.png
    <data>/photo_splice/*.png
    <data>/field_edit/*.png
    <data>/stamp_overlay/*.png
    <data>/recompression/*.png

Reports **per-class** precision and recall, never one blended accuracy figure --
a model that catches photo splices but misses date edits is a different and more
dangerous tool than one that is uniformly mediocre.

Two deliberate choices worth knowing:

*Input size 384, not the usual 224.* Tampered regions on ID documents are
0.27-4.17% of the image. Downscaling to 224 destroys the very artifacts the model
is meant to see.

*No horizontal flips.* Text on identity documents has a fixed orientation, so a
mirrored passport is not a document the model will ever meet, and training on one
spends capacity on an impossible case.

Usage:
    python -m ml.train_forensics_cnn --data ../data/training --epochs 15
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

CLASS_LABELS = [
    "genuine",
    "photo_splice",
    "field_edit",
    "stamp_overlay",
    "recompression",
]

INPUT_SIZE = 384
MODEL_NAME = "efficientnet_b0"


def build_transforms(train: bool, size: int = INPUT_SIZE):
    from torchvision import transforms

    if train:
        return transforms.Compose(
            [
                transforms.Resize((size, size)),
                # Mild photometric jitter only: models the capture device varying,
                # not the document changing.
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.08),
                transforms.RandomRotation(degrees=2, fill=255),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def _cap_per_class(samples, cap: int):
    """Keep at most `cap` samples of each class, preserving class balance."""
    kept: dict[int, int] = {}
    output = []
    for path, target in samples:
        if kept.get(target, 0) < cap:
            kept[target] = kept.get(target, 0) + 1
            output.append((path, target))
    return output


def per_class_report(true_labels, predicted, labels: list[str]) -> dict:
    """Precision, recall and F1 per class, computed without sklearn."""
    report: dict[str, dict[str, float]] = {}
    for index, name in enumerate(labels):
        true_positive = sum(
            1 for t, p in zip(true_labels, predicted, strict=True) if t == index and p == index
        )
        predicted_positive = sum(1 for p in predicted if p == index)
        actual_positive = sum(1 for t in true_labels if t == index)

        precision = true_positive / predicted_positive if predicted_positive else 0.0
        recall = true_positive / actual_positive if actual_positive else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        report[name] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": actual_positive,
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[2]
    parser.add_argument("--data", type=Path, default=root / "data" / "raw" / "FantasyID")
    parser.add_argument(
        "--dataset",
        choices=["fantasyid", "imagefolder"],
        default="fantasyid",
        help="fantasyid uses the dataset's own train/test CSVs (different templates "
             "per split); imagefolder expects one directory per class",
    )
    parser.add_argument(
        "--input-size", type=int, default=INPUT_SIZE,
        help="224 for a quick pilot; 384 for the real run, since tamper regions are tiny",
    )
    parser.add_argument(
        "--max-per-class", type=int, default=0,
        help="Cap samples per class (0 = all). Use for a fast pipeline smoke test.",
    )
    parser.add_argument(
        "--val-data", type=Path, default=None,
        help="Explicit validation directory (imagefolder mode). Without it the "
             "training directory is randomly split, which leaks template identity "
             "when the two sets come from the same documents.",
    )
    parser.add_argument(
        "--freeze-blocks", type=int, default=0,
        help="Freeze the stem and the first N backbone blocks. With ~2.5k samples "
             "fine-tuning all 5M parameters memorises the training set.",
    )
    parser.add_argument("--out", type=Path, default=Path("./ml/checkpoints/forensics_cnn.pt"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260902)
    args = parser.parse_args()

    if not args.data.is_dir():
        raise SystemExit(
            f"No training data at {args.data}.\n"
            "Phase 2 training data is IDNet plus generated specimens -- see "
            "docs/DATA_STRATEGY.md section 2."
        )

    import timm
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, random_split
    from torchvision.datasets import ImageFolder

    torch.manual_seed(args.seed)

    if args.dataset == "fantasyid":
        from ml.data_prep.fantasyid_dataset import FantasyIDDataset

        train_set = FantasyIDDataset(
            args.data, "train", transform=build_transforms(True, args.input_size)
        )
        val_set = FantasyIDDataset(
            args.data, "test", transform=build_transforms(False, args.input_size)
        )
        labels = train_set.classes
        print(f"classes : {labels}")
        print(f"train   : {len(train_set)}  {train_set.class_counts()}")
        print(f"val     : {len(val_set)}  {val_set.class_counts()}")

        if args.max_per_class:
            train_set.samples = _cap_per_class(train_set.samples, args.max_per_class)
            val_set.samples = _cap_per_class(val_set.samples, args.max_per_class)
            print(f"capped  : train {len(train_set)}, val {len(val_set)}")

        counts = Counter(t for _, t in train_set.samples)
    else:
        full = ImageFolder(str(args.data), transform=build_transforms(True, args.input_size))
        labels = full.classes
        print(f"classes: {labels}")
        print(f"images : {len(full)}  {Counter(t for _, t in full.samples)}")

        if args.val_data is not None:
            train_set = full
            val_set = ImageFolder(
                str(args.val_data), transform=build_transforms(False, args.input_size)
            )
            print(f"val    : {len(val_set)}  {Counter(t for _, t in val_set.samples)}")
        else:
            val_size = int(len(full) * args.val_split)
            train_set, val_set = random_split(
                full, [len(full) - val_size, val_size],
                generator=torch.Generator().manual_seed(args.seed),
            )
            val_set.dataset = ImageFolder(
                str(args.data), transform=build_transforms(False, args.input_size)
            )
        counts = Counter(t for _, t in full.samples)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device : {device}")

    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=len(labels)).to(device)

    if args.freeze_blocks:
        frozen = 0
        for name, param in model.named_parameters():
            if name.startswith(("conv_stem", "bn1")):
                param.requires_grad = False
                frozen += param.numel()
            elif name.startswith("blocks."):
                block_index = int(name.split(".")[1])
                if block_index < args.freeze_blocks:
                    param.requires_grad = False
                    frozen += param.numel()
        total_params = sum(p.numel() for p in model.parameters())
        print(
            f"frozen : stem + blocks[:{args.freeze_blocks}]  "
            f"{frozen:,}/{total_params:,} params ({frozen/total_params:.0%})"
        )

    trainable = [p for p in model.parameters() if p.requires_grad]

    # Class weights counter the imbalance between genuine and each tamper type.
    total = sum(counts.values())
    weights = torch.tensor(
        [total / (len(labels) * max(counts.get(i, 1), 1)) for i in range(len(labels))],
        dtype=torch.float32,
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_macro_f1 = 0.0
    args.out.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), targets)
            loss.backward()
            optimizer.step()
            running += loss.item() * images.size(0)
        scheduler.step()

        model.eval()
        true_labels: list[int] = []
        predicted: list[int] = []
        with torch.no_grad():
            for images, targets in val_loader:
                outputs = model(images.to(device))
                predicted.extend(outputs.argmax(1).cpu().tolist())
                true_labels.extend(targets.tolist())

        report = per_class_report(true_labels, predicted, labels)
        macro_f1 = sum(r["f1"] for r in report.values()) / len(report)
        print(f"epoch {epoch:3d}  loss {running/len(train_set):.4f}  macro-F1 {macro_f1:.4f}")

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "class_labels": labels,
                    "model_name": MODEL_NAME,
                    "input_size": args.input_size,
                    "macro_f1": macro_f1,
                    "per_class": report,
                },
                args.out,
            )

    print(f"\nbest macro-F1 {best_macro_f1:.4f}, checkpoint at {args.out}")
    print("\nPer-class results (never quote a single blended number):")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
