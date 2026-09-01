"""OCR engine abstraction.

BUILD_PLAN Section 3 calls for PaddleOCR as primary and Tesseract as fallback,
"behind one interface so the engine is swappable". That is what this module is:
a narrow `OCREngine` protocol plus concrete adapters, selected at runtime.

Both backends are imported lazily. Neither is a hard dependency of the package,
so the MRZ parser and the rest of the pipeline stay importable -- and testable --
on a machine where no OCR backend is installed.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from app.core.schemas import OCRField, OCRResult


class OCREngineError(RuntimeError):
    """Raised when no usable OCR backend is available."""


class OCREngine(ABC):
    """One image in, a list of recognized text fields out."""

    name: str = "base"

    @abstractmethod
    def extract_text(self, image: np.ndarray) -> OCRResult:
        """Recognize text in a BGR image array."""

    @staticmethod
    @abstractmethod
    def is_available() -> bool:
        """Whether this backend's dependencies are importable."""

    def extract_from_path(self, path: str | Path) -> OCRResult:
        import cv2

        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        return self.extract_text(image)


class PaddleOCREngine(OCREngine):
    """Primary backend. Better on the dense, low-contrast text of ID documents.

    PaddleOCR 3.x rewrote both the constructor keywords and the result shape:
    `use_angle_cls`/`show_log` became `use_textline_orientation`, and `.ocr()`
    returning nested `[box, (text, score)]` lists became `.predict()` returning
    dict-like results with parallel `rec_texts` / `rec_scores` / `rec_polys`
    arrays. Both generations are supported here, detected at construction, so the
    pinned version can move without the pipeline noticing.
    """

    name = "paddleocr"

    def __init__(self, lang: str = "en") -> None:
        if not self.is_available():
            raise OCREngineError("paddleocr is not installed")
        from paddleocr import PaddleOCR

        # oneDNN is disabled deliberately. paddlepaddle 3.3.1's oneDNN CPU kernels
        # raise NotImplementedError on the PP-OCRv6 detection graph
        # ("ConvertPirAttribute2RuntimeAttribute not support"). The plain CPU
        # kernels produce identical output, so this costs a little speed and
        # buys working inference.
        try:
            self._ocr = PaddleOCR(
                use_textline_orientation=True, lang=lang, enable_mkldnn=False
            )
            self._api_version = 3
        except (TypeError, ValueError):
            self._ocr = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)
            self._api_version = 2

    @staticmethod
    def is_available() -> bool:
        from importlib.util import find_spec

        return find_spec("paddleocr") is not None

    @staticmethod
    def _bbox(polygon: Any) -> tuple[int, int, int, int]:
        xs = [int(point[0]) for point in polygon]
        ys = [int(point[1]) for point in polygon]
        return min(xs), min(ys), max(xs), max(ys)

    def _parse_v3(self, raw: Any) -> list[OCRField]:
        fields: list[OCRField] = []
        for page in raw or []:
            texts = page["rec_texts"]
            scores = page["rec_scores"]
            polys = page.get("rec_polys")
            if polys is None:
                polys = page.get("dt_polys") or [None] * len(texts)

            for text, score, poly in zip(texts, scores, polys, strict=False):
                fields.append(
                    OCRField(
                        text=text,
                        confidence=float(score),
                        bbox=self._bbox(poly) if poly is not None else None,
                    )
                )
        return fields

    def _parse_v2(self, raw: Any) -> list[OCRField]:
        fields: list[OCRField] = []
        # PaddleOCR 2.x returns [[ [box, (text, confidence)], ... ]] per image.
        for page in raw or []:
            for entry in page or []:
                box, (text, confidence) = entry[0], entry[1]
                fields.append(
                    OCRField(
                        text=text,
                        confidence=float(confidence),
                        bbox=self._bbox(box),
                    )
                )
        return fields

    def extract_text(self, image: np.ndarray) -> OCRResult:
        started = time.perf_counter()

        if self._api_version == 3:
            fields = self._parse_v3(self._ocr.predict(image))
        else:
            fields = self._parse_v2(self._ocr.ocr(image, cls=True))

        elapsed = int((time.perf_counter() - started) * 1000)
        return OCRResult(engine=self.name, fields=fields, processing_time_ms=elapsed)


class TesseractEngine(OCREngine):
    """Fallback backend. Weaker on document layouts, but far lighter to install."""

    name = "tesseract"

    # Restricting the alphabet materially improves MRZ character accuracy.
    MRZ_WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"

    def __init__(self, mrz_mode: bool = False) -> None:
        if not self.is_available():
            raise OCREngineError("pytesseract is not installed")
        self._mrz_mode = mrz_mode

    @staticmethod
    def is_available() -> bool:
        from importlib.util import find_spec

        return find_spec("pytesseract") is not None

    def extract_text(self, image: np.ndarray) -> OCRResult:
        import pytesseract
        from pytesseract import Output

        started = time.perf_counter()
        config = "--psm 6"
        if self._mrz_mode:
            config += f" -c tessedit_char_whitelist={self.MRZ_WHITELIST}"

        data = pytesseract.image_to_data(image, config=config, output_type=Output.DICT)

        fields: list[OCRField] = []
        for i, text in enumerate(data["text"]):
            text = text.strip()
            if not text:
                continue
            conf = float(data["conf"][i])
            if conf < 0:  # Tesseract uses -1 for "no confidence available"
                continue
            x, y, w, h = (data["left"][i], data["top"][i], data["width"][i], data["height"][i])
            fields.append(
                OCRField(text=text, confidence=conf / 100.0, bbox=(x, y, x + w, y + h))
            )

        elapsed = int((time.perf_counter() - started) * 1000)
        return OCRResult(engine=self.name, fields=fields, processing_time_ms=elapsed)


def available_engines() -> list[str]:
    """Names of the OCR backends installed on this machine."""
    return [
        engine.name
        for engine in (PaddleOCREngine, TesseractEngine)
        if engine.is_available()
    ]


def get_engine(preferred: str | None = None) -> OCREngine:
    """Return an OCR engine, preferring PaddleOCR then falling back to Tesseract.

    Pass `preferred` to force a specific backend -- useful for benchmarking one
    against the other on the same sample set in Phase 7.
    """
    registry: dict[str, type[OCREngine]] = {
        PaddleOCREngine.name: PaddleOCREngine,
        TesseractEngine.name: TesseractEngine,
    }

    if preferred:
        engine_cls = registry.get(preferred)
        if engine_cls is None:
            raise OCREngineError(f"Unknown OCR engine {preferred!r}")
        if not engine_cls.is_available():
            raise OCREngineError(f"OCR engine {preferred!r} is not installed")
        return engine_cls()

    for engine_cls in (PaddleOCREngine, TesseractEngine):
        if engine_cls.is_available():
            return engine_cls()

    raise OCREngineError(
        "No OCR backend installed. Install paddleocr (preferred) or pytesseract."
    )
