from __future__ import annotations

from dataclasses import dataclass

from datetime import time

from pathlib import Path

from shutil import copy2

from typing import Iterable

import re

import numpy as np

from config import (
    MEDICINE_REGISTRATION_SAMPLES,
    REGISTERED_MEDICINE_IMAGE_DIR,
)

from database.medicine_repository import (
    add_medicine_reference_image,
    create_medicine,
    deactivate_medicine,
    get_medicine_by_id,
    get_medicine_by_name,
)

from medicine.medicine_recognizer import (
    MedicineRecognitionEngine,
)


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class MedicineRegistrationSample:

    source_path: Path

    stored_path: Path

    embedding: np.ndarray

    reference_image_id: int


@dataclass
class MedicineRegistrationResult:

    medicine_id: int

    medicine_name: str

    dosage: str | None

    instructions: str | None

    schedule_time: time

    schedule_label: str | None

    ocr_keywords: str | None

    primary_image_path: str | None

    sample_count: int

    samples: list[
        MedicineRegistrationSample
    ]


# ============================================================
# REGISTRATION SERVICE
# ============================================================

class MedicineRegistrationService:

    def __init__(
        self,
        recognizer: (
            MedicineRecognitionEngine
            | None
        ) = None,
        required_samples: int | None = None,
    ) -> None:

        self.recognizer = (
            recognizer
            if recognizer is not None
            else MedicineRecognitionEngine()
        )

        self.required_samples = int(
            required_samples
            if required_samples is not None
            else MEDICINE_REGISTRATION_SAMPLES
        )

        if self.required_samples <= 0:

            raise ValueError(
                "required_samples must be "
                "greater than zero."
            )

        REGISTERED_MEDICINE_IMAGE_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )


    # ========================================================
    # SAFE NAME
    # ========================================================

    @staticmethod
    def _safe_folder_name(
        name: str,
    ) -> str:

        cleaned = (
            name.strip()
            .lower()
        )

        safe_chars: list[str] = []

        previous_separator = False

        for char in cleaned:

            if char.isalnum():

                safe_chars.append(
                    char
                )

                previous_separator = False

            elif char in {
                " ",
                "-",
                "_",
            }:

                if not previous_separator:

                    safe_chars.append(
                        "_"
                    )

                    previous_separator = True

        safe_name = "".join(
            safe_chars
        ).strip(
            "_"
        )

        if not safe_name:

            raise ValueError(
                "Could not create medicine folder."
            )

        return safe_name


    # ========================================================
    # VALIDATE
    # ========================================================

    def _validate_sample_paths(
        self,
        sample_paths: Iterable[
            str | Path
        ],
    ) -> list[Path]:

        paths = [
            Path(
                path
            )
            for path
            in sample_paths
        ]

        if (
            len(
                paths
            )
            != self.required_samples
        ):

            raise ValueError(
                "Expected exactly "
                f"{self.required_samples} samples."
            )

        for path in paths:

            if (
                not path.exists()
                or not path.is_file()
            ):

                raise FileNotFoundError(
                    str(
                        path
                    )
                )

        return paths


    # ========================================================
    # DUPLICATE
    # ========================================================

    @staticmethod
    def _ensure_not_registered(
        medicine_name: str,
    ) -> None:

        existing = (
            get_medicine_by_name(
                medicine_name
            )
        )

        if existing is None:

            return

        if bool(
            existing[
                "is_active"
            ]
        ):

            raise ValueError(
                "Medicine is already registered: "
                f"{existing['medicine_name']}"
            )

        raise ValueError(
            "An inactive registration already "
            "exists for this medicine."
        )


    # ========================================================
    # EMBEDDING
    # ========================================================

    def extract_registration_embedding(
        self,
        image_path: str | Path,
    ) -> np.ndarray:

        return (
            self.recognizer
            .generate_embedding(
                image_path
            )
        )


    # ========================================================
    # AUTOMATIC OCR KEYWORDS
    # ========================================================

    def _extract_automatic_ocr_keywords(
        self,
        medicine_name: str,
        dosage: str | None,
        sample_paths: list[
            Path
        ],
    ) -> list[str]:

        keywords = [
            medicine_name
        ]

        if dosage:

            keywords.append(
                dosage
            )

        # OCR only the first few registration images.
        # This happens once during registration, not live.
        frames = []

        import cv2

        for path in sample_paths[
            :3
        ]:

            image = cv2.imread(
                str(
                    path
                )
            )

            if (
                image is not None
                and image.size > 0
            ):

                frames.append(
                    image
                )

        if frames:

            try:

                items, _quality = (
                    self.recognizer
                    .text_recognizer
                    .recognize_text(
                        frames
                    )
                )

                for item in items:

                    words = re.findall(
                        r"[A-Za-z][A-Za-z0-9]{3,}",
                        item.text,
                    )

                    for word in words:

                        keywords.append(
                            word
                        )

            except Exception:

                pass

        seen = set()

        result = []

        for keyword in keywords:

            cleaned = " ".join(
                str(
                    keyword
                )
                .strip()
                .split()
            )

            key = cleaned.lower()

            if (
                not cleaned
                or key in seen
            ):

                continue

            seen.add(
                key
            )

            result.append(
                cleaned
            )

        return result


    # ========================================================
    # SAVE IMAGE
    # ========================================================

    def _save_registration_image(
        self,
        source_path: Path,
        medicine_name: str,
        medicine_id: int,
        sample_number: int,
    ) -> Path:

        medicine_directory = (
            REGISTERED_MEDICINE_IMAGE_DIR
            / (
                f"{self._safe_folder_name(medicine_name)}_"
                f"{medicine_id}"
            )
        )

        medicine_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        suffix = (
            source_path.suffix.lower()
            if source_path.suffix
            else ".jpg"
        )

        destination = (
            medicine_directory
            / (
                f"sample_"
                f"{sample_number:02d}"
                f"{suffix}"
            )
        )

        copy2(
            source_path,
            destination,
        )

        return destination


    # ========================================================
    # REGISTER
    # ========================================================

    def register_medicine(
        self,
        medicine_name: str,
        dosage: str | None,
        schedule_time: time,
        sample_paths: Iterable[
            str | Path
        ],
        schedule_label: str | None = None,
        instructions: str | None = None,
        ocr_keywords: (
            str
            | Iterable[str]
            | None
        ) = None,
    ) -> MedicineRegistrationResult:

        cleaned_name = " ".join(
            medicine_name
            .strip()
            .split()
        )

        if not cleaned_name:

            raise ValueError(
                "Medicine name cannot be empty."
            )

        self._ensure_not_registered(
            cleaned_name
        )

        paths = (
            self._validate_sample_paths(
                sample_paths
            )
        )

        # ----------------------------------------------------
        # OCR KEYWORDS
        # ----------------------------------------------------

        if ocr_keywords is None:

            final_ocr_keywords = (
                self._extract_automatic_ocr_keywords(
                    medicine_name=cleaned_name,
                    dosage=dosage,
                    sample_paths=paths,
                )
            )

        elif isinstance(
            ocr_keywords,
            str,
        ):

            final_ocr_keywords = [
                item.strip()
                for item
                in ocr_keywords.split(
                    ","
                )
                if item.strip()
            ]

        else:

            final_ocr_keywords = list(
                ocr_keywords
            )

        # Always ensure actual registered name is included.
        final_ocr_keywords.insert(
            0,
            cleaned_name,
        )

        # ----------------------------------------------------
        # EMBEDDINGS
        # ----------------------------------------------------

        extracted_samples = []

        for path in paths:

            embedding = (
                self.extract_registration_embedding(
                    path
                )
            )

            extracted_samples.append(
                (
                    path,
                    embedding,
                )
            )

        # ----------------------------------------------------
        # DATABASE
        # ----------------------------------------------------

        medicine_id = (
            create_medicine(
                medicine_name=cleaned_name,
                dosage=dosage,
                schedule_time=schedule_time,
                schedule_label=schedule_label,
                instructions=instructions,
                ocr_keywords=(
                    final_ocr_keywords
                ),
            )
        )

        accepted_samples = []

        primary_image_path = None

        try:

            for (
                index,
                (
                    source_path,
                    embedding,
                ),
            ) in enumerate(
                extracted_samples,
                start=1,
            ):

                stored_path = (
                    self._save_registration_image(
                        source_path=source_path,
                        medicine_name=cleaned_name,
                        medicine_id=medicine_id,
                        sample_number=index,
                    )
                )

                reference_image_id = (
                    add_medicine_reference_image(
                        medicine_id=medicine_id,
                        image_path=str(
                            stored_path
                        ),
                        embedding=(
                            self.recognizer
                            .serialize_embedding(
                                embedding
                            )
                        ),
                        embedding_model=(
                            self.recognizer
                            .model_name
                        ),
                        is_primary=(
                            index == 1
                        ),
                    )
                )

                if index == 1:

                    primary_image_path = (
                        str(
                            stored_path
                        )
                    )

                accepted_samples.append(
                    MedicineRegistrationSample(
                        source_path=source_path,
                        stored_path=stored_path,
                        embedding=embedding,
                        reference_image_id=(
                            reference_image_id
                        ),
                    )
                )

        except Exception:

            deactivate_medicine(
                medicine_id
            )

            raise

        medicine = (
            get_medicine_by_id(
                medicine_id
            )
        )

        if medicine is None:

            raise RuntimeError(
                "Registered medicine could not "
                "be loaded."
            )

        return MedicineRegistrationResult(
            medicine_id=medicine_id,
            medicine_name=str(
                medicine[
                    "medicine_name"
                ]
            ),
            dosage=medicine[
                "dosage"
            ],
            instructions=medicine[
                "instructions"
            ],
            schedule_time=medicine[
                "schedule_time"
            ],
            schedule_label=medicine[
                "schedule_label"
            ],
            ocr_keywords=medicine.get(
                "ocr_keywords"
            ),
            primary_image_path=(
                primary_image_path
            ),
            sample_count=len(
                accepted_samples
            ),
            samples=accepted_samples,
        )