"""EfficientNet-B0 tamper classifier.

The learned half of the forensics engine. It runs *alongside* the classical
checks, never instead of them: `engine.analyze` treats its output as one more
finding, and its weight in `CHECK_WEIGHTS` stays at zero until a checkpoint has
been validated on held-out data. A verdict must never rest on a number nobody
can account for.

Output is deliberately image-level plus a coarse region, not a pixel mask. See
docs/DATA_STRATEGY.md section 2: tampered areas on ID documents occupy
0.27-4.17% of the image and state-of-the-art detectors score near-zero on
pixel-level localization, so a precise heatmap would be a claim we cannot honour.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np

from app.core.schemas import ForensicsFinding, Region, TamperType

# Class order must match the training script's label encoding exactly.
# Binary, matching the training data -- see ml/data_prep/fantasyid_dataset.py.
CLASS_LABELS: list[str] = ["genuine", "manipulated"]

# Class name -> reported tamper type. Anything the model cannot attribute is
# reported as a generic digital manipulation rather than a guessed type.
LABEL_TO_TAMPER_TYPE: dict[str, TamperType] = {
    "manipulated": TamperType.DIGITAL_MANIPULATION,
    TamperType.PHOTO_SPLICE.value: TamperType.PHOTO_SPLICE,
    TamperType.FIELD_EDIT.value: TamperType.FIELD_EDIT,
    TamperType.STAMP_OVERLAY.value: TamperType.STAMP_OVERLAY,
    TamperType.RECOMPRESSION.value: TamperType.RECOMPRESSION,
}

INPUT_SIZE = 384  # larger than the usual 224: tamper cues are small and local
MODEL_NAME = "efficientnet_b0"

# Below this the model is not confident enough to contribute a finding.
MIN_REPORT_CONFIDENCE = 0.60


def model_path() -> Path:
    return Path(os.getenv("FORENSICS_MODEL_PATH", "./ml/checkpoints/forensics_cnn.pt"))


def is_model_available() -> bool:
    """Whether a trained checkpoint exists and torch is importable."""
    from importlib.util import find_spec

    if find_spec("torch") is None or find_spec("timm") is None:
        return False
    return model_path().is_file()


@lru_cache(maxsize=1)
def _load_model():
    """Load the checkpoint once and cache it."""
    import timm
    import torch

    checkpoint = torch.load(model_path(), map_location="cpu", weights_only=False)
    labels = checkpoint.get("class_labels", CLASS_LABELS)

    model = timm.create_model(
        checkpoint.get("model_name", MODEL_NAME),
        pretrained=False,
        num_classes=len(labels),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, labels, checkpoint.get("input_size", INPUT_SIZE)


def _preprocess(image: np.ndarray, size: int):
    import cv2
    import torch

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(resized).float().permute(2, 0, 1) / 255.0

    # ImageNet normalization, matching timm's pretrained weights.
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return ((tensor - mean) / std).unsqueeze(0)


def classify(image: np.ndarray) -> ForensicsFinding:
    """Classify a document image as genuine or one of the tamper types."""
    import torch

    model, labels, size = _load_model()

    with torch.no_grad():
        logits = model(_preprocess(image, size))
        probabilities = torch.softmax(logits, dim=1)[0]

    index = int(probabilities.argmax())
    label = labels[index]
    confidence = float(probabilities[index])

    if label == "genuine":
        return ForensicsFinding(
            check="cnn_classifier",
            tamper_type=None,
            flagged=False,
            confidence=0.0,
            detail=(
                f"Learned classifier rates this document genuine "
                f"({confidence:.0%} confidence)"
            ),
        )

    if confidence < MIN_REPORT_CONFIDENCE:
        return ForensicsFinding(
            check="cnn_classifier",
            tamper_type=None,
            flagged=False,
            confidence=0.0,
            detail=(
                f"Learned classifier leans towards {label.replace('_', ' ')} but at only "
                f"{confidence:.0%} confidence, below the {MIN_REPORT_CONFIDENCE:.0%} "
                f"reporting threshold"
            ),
        )

    height, width = image.shape[:2]
    return ForensicsFinding(
        check="cnn_classifier",
        tamper_type=LABEL_TO_TAMPER_TYPE.get(label, TamperType.DIGITAL_MANIPULATION),
        flagged=True,
        confidence=confidence,
        detail=(
            f"Learned classifier flags digital manipulation at {confidence:.0%} "
            f"confidence. Image-level classification; the region below is the whole "
            f"document, not a localized detection."
        ),
        regions=[Region(x1=0, y1=0, x2=width, y2=height, score=confidence)],
    )
