from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher

import re
import threading
import time

import cv2
import easyocr
import numpy as np

from database.medicine_repository import (
    get_all_medicines,
    parse_ocr_keywords,
)


# ============================================================
# OCR THRESHOLDS
# ============================================================

OCR_TEXT_CONFIDENCE_THRESHOLD = 0.22

# Real MemorAi test:
#
# Omez / Omeprazole:
#     strong match around 0.74+
#
# Feviquick against Omez:
#     around 0.19
#
# So 0.72 remains a reasonable strong-text threshold.
OCR_STRONG_IDENTITY_THRESHOLD = 0.72

OCR_SUPPORT_IDENTITY_THRESHOLD = 0.64


# ============================================================
# LIVE OCR IMAGE CONFIG
# ============================================================

# Keep OCR work small on CPU.
OCR_TARGET_WIDTH = 320

OCR_MAX_HEIGHT = 220

OCR_MIN_WIDTH = 220


# ============================================================
# MEDICINE DATABASE CACHE
# ============================================================

MEDICINE_CACHE_SECONDS = 10.0


# ============================================================
# OCR TEXT ITEM
# ============================================================

@dataclass
class OCRTextItem:
    """
    One text result returned by EasyOCR.
    """

    text: str

    confidence: float

    normalized_text: str

    source: str


# ============================================================
# MEDICINE OCR RESULT
# ============================================================

@dataclass
class OCRMedicineMatch:
    """
    Best medicine identity found from printed text.
    """

    medicine_id: int | None

    medicine_name: str

    dosage: str | None

    score: float

    text_score: float

    dosage_score: float

    detected_text: str

    matched_keyword: str | None

    processing_seconds: float

    frame_quality: float

    is_strong: bool


# ============================================================
# MEDICINE TEXT RECOGNIZER
# ============================================================

class MedicineTextRecognizer:
    """
    Fast automatic medicine OCR.

    Production flow:

        recent medicine frames
                ↓
        automatically choose clearest
                ↓
        resize to small bounded image
                ↓
        create overlapping text bands
                ↓
        EasyOCR recognition model ONLY
                ↓
        compare with registered:
            medicine name
            generic/printed aliases
            dosage support

    IMPORTANT:

    We intentionally do NOT use EasyOCR readtext() in the
    live path.

    readtext() first performs neural text detection, which
    was taking far too long on the user's CPU.

    We already know where the medicine is because
    CameraService crops the Medicine Only region.

    Therefore we divide that crop into broad horizontal
    text regions and pass those regions directly to
    EasyOCR's recognizer.

    No keyboard.
    No C/Q.
    No manual capture.
    """

    def __init__(
        self,
        gpu: bool = False,
    ) -> None:

        # ----------------------------------------------------
        # EasyOCR initialization is expensive.
        #
        # Construct exactly once.
        # ----------------------------------------------------

        self.reader = easyocr.Reader(
            [
                "en",
            ],
            gpu=bool(
                gpu
            ),
            verbose=False,
        )

        self._lock = (
            threading.Lock()
        )

        # ----------------------------------------------------
        # MEDICINE DB CACHE
        # ----------------------------------------------------

        self._medicine_cache: list[
            dict
        ] = []

        self._medicine_cache_created_at = (
            0.0
        )


    # ========================================================
    # NORMALIZE TEXT
    # ========================================================

    @staticmethod
    def normalize_text(
        value: str | None,
    ) -> str:

        if not value:

            return ""

        text = (
            str(
                value
            )
            .lower()
            .strip()
        )

        text = re.sub(
            r"[^a-z0-9]+",
            " ",
            text,
        )

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return (
            text.strip()
        )


    # ========================================================
    # COMPACT TEXT
    # ========================================================

    @classmethod
    def compact(
        cls,
        value: str | None,
    ) -> str:

        return (
            cls.normalize_text(
                value
            )
            .replace(
                " ",
                "",
            )
        )


    # ========================================================
    # COMMON OCR CHARACTER ERRORS
    # ========================================================

    @staticmethod
    def normalize_common_ocr_errors(
        value: str,
    ) -> str:

        text = (
            str(
                value
            )
            .upper()
        )

        replacements = {
            "0": "O",
            "|": "I",
        }

        for old, new in replacements.items():

            text = text.replace(
                old,
                new,
            )

        return (
            text
        )


    # ========================================================
    # FUZZY TEXT SCORE
    # ========================================================

    @classmethod
    def fuzzy_score(
        cls,
        detected: str,
        expected: str,
    ) -> float:

        a = (
            cls.compact(
                detected
            )
        )

        b = (
            cls.compact(
                expected
            )
        )

        if (
            not a
            or not b
        ):

            return 0.0

        # ----------------------------------------------------
        # EXACT
        # ----------------------------------------------------

        if a == b:

            return 1.0

        # ----------------------------------------------------
        # EXPECTED WORD INSIDE OCR STRING
        # ----------------------------------------------------

        if b in a:

            return 1.0

        # ----------------------------------------------------
        # OCR RESULT IS LARGE PART OF EXPECTED WORD
        # ----------------------------------------------------

        if (
            a in b
            and len(
                a
            ) >= 4
        ):

            coverage = (
                len(
                    a
                )
                / len(
                    b
                )
            )

            return float(
                min(
                    0.95,
                    coverage,
                )
            )

        # ----------------------------------------------------
        # NORMAL FUZZY SCORE
        # ----------------------------------------------------

        normal_score = float(
            SequenceMatcher(
                None,
                a,
                b,
            )
            .ratio()
        )

        # ----------------------------------------------------
        # COMMON OCR ERROR SCORE
        # ----------------------------------------------------

        corrected_a = (
            cls.compact(
                cls.normalize_common_ocr_errors(
                    detected
                )
            )
        )

        corrected_b = (
            cls.compact(
                cls.normalize_common_ocr_errors(
                    expected
                )
            )
        )

        corrected_score = float(
            SequenceMatcher(
                None,
                corrected_a,
                corrected_b,
            )
            .ratio()
        )

        return float(
            max(
                normal_score,
                corrected_score,
            )
        )


    # ========================================================
    # FRAME QUALITY
    # ========================================================

    @staticmethod
    def frame_quality(
        image: np.ndarray,
    ) -> float:

        if (
            image is None
            or not isinstance(
                image,
                np.ndarray,
            )
            or image.size == 0
        ):

            return 0.0

        if (
            image.ndim != 3
            or image.shape[
                2
            ] != 3
        ):

            return 0.0

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

        # ----------------------------------------------------
        # SHARPNESS
        # ----------------------------------------------------

        sharpness_value = float(
            cv2.Laplacian(
                gray,
                cv2.CV_64F,
            )
            .var()
        )

        sharpness_score = float(
            np.clip(
                sharpness_value
                / 450.0,
                0.0,
                1.0,
            )
        )

        # ----------------------------------------------------
        # CONTRAST
        # ----------------------------------------------------

        contrast_value = float(
            np.std(
                gray
            )
        )

        contrast_score = float(
            np.clip(
                contrast_value
                / 65.0,
                0.0,
                1.0,
            )
        )

        # ----------------------------------------------------
        # BRIGHTNESS
        # ----------------------------------------------------

        brightness_value = float(
            np.mean(
                gray
            )
        )

        brightness_score = (
            1.0
            - min(
                1.0,
                abs(
                    brightness_value
                    - 128.0
                )
                / 128.0,
            )
        )

        # ----------------------------------------------------
        # GLARE
        # ----------------------------------------------------

        glare_ratio = (
            float(
                np.count_nonzero(
                    gray
                    >= 245
                )
            )
            / float(
                gray.size
            )
        )

        glare_score = (
            1.0
            - float(
                np.clip(
                    glare_ratio
                    / 0.30,
                    0.0,
                    1.0,
                )
            )
        )

        # ----------------------------------------------------
        # FINAL QUALITY SCORE
        # ----------------------------------------------------

        score = (
            (
                0.52
                * sharpness_score
            )
            +
            (
                0.23
                * contrast_score
            )
            +
            (
                0.10
                * brightness_score
            )
            +
            (
                0.15
                * glare_score
            )
        )

        return float(
            np.clip(
                score,
                0.0,
                1.0,
            )
        )


    # ========================================================
    # SELECT BEST RECENT FRAME
    # ========================================================

    def select_best_frame(
        self,
        frames: list[
            np.ndarray
        ],
    ) -> tuple[
        np.ndarray,
        float,
    ]:

        candidates: list[
            tuple[
                float,
                np.ndarray,
            ]
        ] = []

        for frame in frames:

            if (
                frame is None
                or not isinstance(
                    frame,
                    np.ndarray,
                )
                or frame.size == 0
            ):

                continue

            quality = (
                self.frame_quality(
                    frame
                )
            )

            candidates.append(
                (
                    quality,
                    frame,
                )
            )

        if not candidates:

            raise ValueError(
                "No usable OCR medicine frames."
            )

        candidates.sort(
            key=lambda item:
                item[
                    0
                ],
            reverse=True,
        )

        (
            quality,
            frame,
        ) = (
            candidates[
                0
            ]
        )

        return (
            frame.copy(),
            float(
                quality
            ),
        )


    # ========================================================
    # PREPARE OCR IMAGE
    # ========================================================

    @staticmethod
    def _prepare_image(
        image: np.ndarray,
    ) -> np.ndarray:

        height, width = (
            image.shape[
                :2
            ]
        )

        if (
            width <= 0
            or height <= 0
        ):

            raise ValueError(
                "OCR image dimensions are invalid."
            )

        # ----------------------------------------------------
        # RESIZE
        # ----------------------------------------------------

        if width < OCR_MIN_WIDTH:

            scale = (
                OCR_MIN_WIDTH
                / float(
                    width
                )
            )

        else:

            scale = (
                OCR_TARGET_WIDTH
                / float(
                    width
                )
            )

        new_width = max(
            1,
            int(
                width
                * scale
            ),
        )

        new_height = max(
            1,
            int(
                height
                * scale
            ),
        )

        if new_height > OCR_MAX_HEIGHT:

            second_scale = (
                OCR_MAX_HEIGHT
                / float(
                    new_height
                )
            )

            new_width = max(
                1,
                int(
                    new_width
                    * second_scale
                ),
            )

            new_height = (
                OCR_MAX_HEIGHT
            )

        interpolation = (
            cv2.INTER_CUBIC
            if new_width > width
            else cv2.INTER_AREA
        )

        resized = cv2.resize(
            image,
            (
                new_width,
                new_height,
            ),
            interpolation=(
                interpolation
            ),
        )

        # ----------------------------------------------------
        # GRAYSCALE
        # ----------------------------------------------------

        gray = cv2.cvtColor(
            resized,
            cv2.COLOR_BGR2GRAY,
        )

        # ----------------------------------------------------
        # LIGHT CLAHE
        #
        # Do not threshold shiny foil.
        # ----------------------------------------------------

        clahe = cv2.createCLAHE(
            clipLimit=1.5,
            tileGridSize=(
                6,
                6,
            ),
        )

        enhanced = (
            clahe.apply(
                gray
            )
        )

        return (
            enhanced
        )


    # ========================================================
    # BUILD TEXT BANDS
    # ========================================================

    @staticmethod
    def _build_text_bands(
        image: np.ndarray,
    ) -> list[
        list[int]
    ]:
        """
        EasyOCR horizontal-list format:

            [x_min, x_max, y_min, y_max]

        We use overlapping bands because text location on
        blister foil can change as the user rotates it.

        One EasyOCR recognize() call processes all bands.
        """

        height, width = (
            image.shape[
                :2
            ]
        )

        if (
            width <= 0
            or height <= 0
        ):

            return []

        # ----------------------------------------------------
        # TOP / MIDDLE / BOTTOM
        #
        # Overlap is intentional.
        # ----------------------------------------------------

        top = [
            0,
            width,
            0,
            max(
                1,
                int(
                    height
                    * 0.48
                ),
            ),
        ]

        middle = [
            0,
            width,
            max(
                0,
                int(
                    height
                    * 0.22
                ),
            ),
            max(
                1,
                int(
                    height
                    * 0.80
                ),
            ),
        ]

        bottom = [
            0,
            width,
            max(
                0,
                int(
                    height
                    * 0.52
                ),
            ),
            height,
        ]

        return [
            top,
            middle,
            bottom,
        ]


    # ========================================================
    # PARSE EASY OCR RESULT
    # ========================================================

    def _parse_results(
        self,
        results,
    ) -> list[
        OCRTextItem
    ]:

        items: list[
            OCRTextItem
        ] = []

        seen: set[
            str
        ] = set()

        for result in results:

            if (
                not isinstance(
                    result,
                    (
                        list,
                        tuple,
                    ),
                )
                or len(
                    result
                )
                < 3
            ):

                continue

            text = str(
                result[
                    1
                ]
                or ""
            ).strip()

            try:

                confidence = float(
                    result[
                        2
                    ]
                )

            except (
                TypeError,
                ValueError,
            ):

                continue

            if (
                not text
                or confidence
                < OCR_TEXT_CONFIDENCE_THRESHOLD
            ):

                continue

            normalized = (
                self.normalize_text(
                    text
                )
            )

            if not normalized:

                continue

            compact = (
                self.compact(
                    text
                )
            )

            if (
                not compact
                or compact in seen
            ):

                continue

            seen.add(
                compact
            )

            items.append(
                OCRTextItem(
                    text=(
                        text
                    ),

                    confidence=(
                        confidence
                    ),

                    normalized_text=(
                        normalized
                    ),

                    source=(
                        "multi_strip"
                    ),
                )
            )

        return (
            items
        )


    # ========================================================
    # DETECTOR-FREE MULTI-STRIP OCR
    # ========================================================

    def _recognize_strips(
        self,
        image: np.ndarray,
    ) -> list[
        OCRTextItem
    ]:

        horizontal_list = (
            self._build_text_bands(
                image
            )
        )

        if not horizontal_list:

            return []

        try:

            with self._lock:

                results = (
                    self.reader.recognize(
                        image,

                        horizontal_list=(
                            horizontal_list
                        ),

                        free_list=[],

                        decoder="greedy",

                        beamWidth=1,

                        batch_size=3,

                        workers=0,

                        allowlist=(
                            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                            "abcdefghijklmnopqrstuvwxyz"
                            "0123456789"
                        ),

                        detail=1,

                        paragraph=False,

                        contrast_ths=0.10,

                        adjust_contrast=0.50,
                    )
                )

        except Exception:

            return []

        return (
            self._parse_results(
                results
            )
        )


    # ========================================================
    # GET ACTIVE MEDICINES
    # ========================================================

    def _get_medicines(
        self,
    ) -> list[
        dict
    ]:

        now = (
            time.monotonic()
        )

        if (
            self._medicine_cache
            and (
                now
                - self._medicine_cache_created_at
            )
            < MEDICINE_CACHE_SECONDS
        ):

            return list(
                self._medicine_cache
            )

        medicines = (
            get_all_medicines(
                active_only=True
            )
        )

        self._medicine_cache = (
            list(
                medicines
            )
        )

        self._medicine_cache_created_at = (
            now
        )

        return list(
            self._medicine_cache
        )


    # ========================================================
    # MEDICINE IDENTITY KEYWORDS
    # ========================================================

    @staticmethod
    def medicine_keywords(
        medicine: dict,
    ) -> list[str]:

        values: list[
            str
        ] = []

        medicine_name = str(
            medicine.get(
                "medicine_name"
            )
            or ""
        ).strip()

        if medicine_name:

            values.append(
                medicine_name
            )

        values.extend(
            parse_ocr_keywords(
                medicine.get(
                    "ocr_keywords"
                )
            )
        )

        seen: set[
            str
        ] = set()

        result: list[
            str
        ] = []

        for value in values:

            cleaned = " ".join(
                str(
                    value
                )
                .strip()
                .split()
            )

            if not cleaned:

                continue

            key = (
                cleaned.lower()
            )

            if key in seen:

                continue

            seen.add(
                key
            )

            result.append(
                cleaned
            )

        return (
            result
        )


    # ========================================================
    # MATCH OCR TEXT TO REGISTERED MEDICINES
    # ========================================================

    def _match_items(
        self,
        items: list[
            OCRTextItem
        ],
        medicines: list[
            dict
        ],
        processing_seconds: float,
        frame_quality: float,
    ) -> OCRMedicineMatch:

        detected_text = " | ".join(
            item.text
            for item
            in items
        )

        best_medicine = None

        best_score = 0.0

        best_text_score = 0.0

        best_dosage_score = 0.0

        best_keyword = None

        for medicine in medicines:

            # =================================================
            # IDENTITY TEXT
            # =================================================

            text_score = 0.0

            matched_keyword = None

            for keyword in (
                self.medicine_keywords(
                    medicine
                )
            ):

                for item in items:

                    raw_score = (
                        self.fuzzy_score(
                            detected=(
                                item.text
                            ),
                            expected=(
                                keyword
                            ),
                        )
                    )

                    weighted_score = (
                        (
                            0.92
                            * raw_score
                        )
                        +
                        (
                            0.08
                            * item.confidence
                        )
                    )

                    if (
                        weighted_score
                        > text_score
                    ):

                        text_score = (
                            weighted_score
                        )

                        matched_keyword = (
                            keyword
                        )

            # =================================================
            # DOSAGE SUPPORT
            # =================================================

            dosage_score = 0.0

            dosage = (
                medicine.get(
                    "dosage"
                )
            )

            if dosage:

                for item in items:

                    dosage_score = max(
                        dosage_score,
                        self.fuzzy_score(
                            detected=(
                                item.text
                            ),
                            expected=str(
                                dosage
                            ),
                        ),
                    )

            # =================================================
            # FINAL OCR MEDICINE SCORE
            #
            # Printed identity dominates.
            # Dosage is supporting evidence only.
            # =================================================

            combined_score = (
                (
                    0.94
                    * text_score
                )
                +
                (
                    0.06
                    * dosage_score
                )
            )

            if (
                combined_score
                > best_score
            ):

                best_score = (
                    combined_score
                )

                best_text_score = (
                    text_score
                )

                best_dosage_score = (
                    dosage_score
                )

                best_keyword = (
                    matched_keyword
                )

                best_medicine = (
                    medicine
                )

        # ====================================================
        # NO MATCH
        # ====================================================

        if best_medicine is None:

            return OCRMedicineMatch(
                medicine_id=None,

                medicine_name=(
                    "unknown"
                ),

                dosage=None,

                score=0.0,

                text_score=0.0,

                dosage_score=0.0,

                detected_text=(
                    detected_text
                ),

                matched_keyword=None,

                processing_seconds=float(
                    processing_seconds
                ),

                frame_quality=float(
                    frame_quality
                ),

                is_strong=False,
            )

        # ====================================================
        # BEST MATCH
        # ====================================================

        return OCRMedicineMatch(
            medicine_id=int(
                best_medicine[
                    "id"
                ]
            ),

            medicine_name=str(
                best_medicine[
                    "medicine_name"
                ]
            ),

            dosage=(
                str(
                    best_medicine[
                        "dosage"
                    ]
                )
                if best_medicine.get(
                    "dosage"
                )
                is not None
                else None
            ),

            score=float(
                np.clip(
                    best_score,
                    0.0,
                    1.0,
                )
            ),

            text_score=float(
                np.clip(
                    best_text_score,
                    0.0,
                    1.0,
                )
            ),

            dosage_score=float(
                np.clip(
                    best_dosage_score,
                    0.0,
                    1.0,
                )
            ),

            detected_text=(
                detected_text
            ),

            matched_keyword=(
                best_keyword
            ),

            processing_seconds=float(
                processing_seconds
            ),

            frame_quality=float(
                frame_quality
            ),

            is_strong=bool(
                best_score
                >= OCR_STRONG_IDENTITY_THRESHOLD
            ),
        )


    # ========================================================
    # PUBLIC LIVE OCR
    # ========================================================

    def match_registered_medicines(
        self,
        frames: list[
            np.ndarray
        ],
    ) -> OCRMedicineMatch:

        started = (
            time.perf_counter()
        )

        # ----------------------------------------------------
        # AUTOMATICALLY SELECT CLEAREST RECENT FRAME
        # ----------------------------------------------------

        (
            best_frame,
            quality,
        ) = (
            self.select_best_frame(
                frames
            )
        )

        # ----------------------------------------------------
        # SMALL OCR IMAGE
        # ----------------------------------------------------

        prepared = (
            self._prepare_image(
                best_frame
            )
        )

        # ----------------------------------------------------
        # REGISTERED MEDICINES
        # ----------------------------------------------------

        medicines = (
            self._get_medicines()
        )

        if not medicines:

            return OCRMedicineMatch(
                medicine_id=None,

                medicine_name="unknown",

                dosage=None,

                score=0.0,

                text_score=0.0,

                dosage_score=0.0,

                detected_text="",

                matched_keyword=None,

                processing_seconds=float(
                    time.perf_counter()
                    - started
                ),

                frame_quality=float(
                    quality
                ),

                is_strong=False,
            )

        # ----------------------------------------------------
        # DETECTOR-FREE OCR
        # ----------------------------------------------------

        items = (
            self._recognize_strips(
                prepared
            )
        )

        elapsed = (
            time.perf_counter()
            - started
        )

        return (
            self._match_items(
                items=(
                    items
                ),

                medicines=(
                    medicines
                ),

                processing_seconds=(
                    elapsed
                ),

                frame_quality=(
                    quality
                ),
            )
        )


    # ========================================================
    # COMPATIBILITY METHOD
    # ========================================================

    def recognize_text(
        self,
        frames: list[
            np.ndarray
        ],
    ) -> tuple[
        list[
            OCRTextItem
        ],
        float,
    ]:
        """
        Used by medicine registration and standalone tests.
        """

        (
            best_frame,
            quality,
        ) = (
            self.select_best_frame(
                frames
            )
        )

        prepared = (
            self._prepare_image(
                best_frame
            )
        )

        items = (
            self._recognize_strips(
                prepared
            )
        )

        return (
            items,
            quality,
        )


    # ========================================================
    # INVALIDATE DB CACHE
    # ========================================================

    def invalidate_medicine_cache(
        self,
    ) -> None:

        self._medicine_cache = []

        self._medicine_cache_created_at = (
            0.0
        )