from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from shutil import copy2
from typing import Iterable

import cv2
import numpy as np

from config import (
    FACE_REGISTRATION_SAMPLES,
    REGISTERED_FACE_IMAGE_DIR,
)
from database.person_repository import (
    create_person,
    deactivate_person,
    delete_face_embeddings_for_person,
    get_person_by_id,
    get_person_by_name,
    save_face_embedding,
    update_profile_image_path,
)
from face.face_recognition import (
    DetectedFace,
    FaceRecognitionEngine,
)


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class RegistrationSample:

    source_path: Path
    stored_path: Path
    confidence: float
    embedding: np.ndarray


@dataclass
class RegistrationResult:

    person_id: int
    name: str
    display_name: str
    relation: str | None
    profile_image_path: str | None
    sample_count: int
    samples: list[RegistrationSample]


# ============================================================
# REGISTRATION SERVICE
# ============================================================

class PersonRegistrationService:

    def __init__(
        self,
        face_engine: FaceRecognitionEngine | None = None,
        required_samples: int | None = None,
    ) -> None:

        self.face_engine = (
            face_engine
            if face_engine is not None
            else FaceRecognitionEngine()
        )

        self.required_samples = int(
            required_samples
            if required_samples is not None
            else FACE_REGISTRATION_SAMPLES
        )

        if self.required_samples <= 0:

            raise ValueError(
                "required_samples must be greater than zero."
            )

        REGISTERED_FACE_IMAGE_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ========================================================
    # SAFE FOLDER NAME
    # ========================================================

    def _safe_folder_name(
        self,
        name: str,
    ) -> str:

        cleaned = (
            name.strip().lower()
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
        ).strip("_")

        if not safe_name:

            raise ValueError(
                "Could not create a safe folder "
                "name for this person."
            )

        return safe_name

    # ========================================================
    # LOAD IMAGE
    # ========================================================

    def _load_image(
        self,
        image_path: Path,
    ) -> np.ndarray:

        if not image_path.exists():

            raise FileNotFoundError(
                "Registration image not found: "
                f"{image_path}"
            )

        image = cv2.imread(
            str(
                image_path
            )
        )

        if image is None:

            raise ValueError(
                "OpenCV could not read registration image: "
                f"{image_path}"
            )

        return image

    # ========================================================
    # REQUIRE ONE FACE
    # ========================================================

    def _select_single_face(
        self,
        faces: list[DetectedFace],
        image_path: Path,
    ) -> DetectedFace:

        if not faces:

            raise ValueError(
                "No face detected in registration image: "
                f"{image_path}"
            )

        if len(
            faces
        ) > 1:

            raise ValueError(
                "More than one face was detected in "
                "registration image: "
                f"{image_path}"
            )

        return faces[0]

    # ========================================================
    # EMBEDDING
    # ========================================================

    def extract_registration_embedding(
        self,
        image_path: str | Path,
    ) -> tuple[
        np.ndarray,
        float,
    ]:

        path = Path(
            image_path
        )

        image = self._load_image(
            path
        )

        faces = (
            self.face_engine
            .detect_faces(
                image
            )
        )

        detected_face = (
            self._select_single_face(
                faces,
                path,
            )
        )

        embedding = (
            self.face_engine
            .generate_embedding(
                image,
                detected_face,
            )
        )

        return (
            embedding,
            detected_face.confidence,
        )

    # ========================================================
    # STORE REGISTRATION IMAGE
    # ========================================================

    def _save_registration_image(
        self,
        source_path: Path,
        person_name: str,
        sample_number: int,
    ) -> Path:

        safe_name = (
            self._safe_folder_name(
                person_name
            )
        )

        person_directory = (
            REGISTERED_FACE_IMAGE_DIR
            / safe_name
        )

        person_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        suffix = (
            source_path.suffix.lower()
            if source_path.suffix
            else ".jpg"
        )

        destination = (
            person_directory
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
    # VALIDATE SAMPLE PATHS
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
            for path in sample_paths
        ]

        if len(
            paths
        ) != self.required_samples:

            raise ValueError(
                "Expected exactly "
                f"{self.required_samples} registration "
                f"samples, received {len(paths)}."
            )

        for path in paths:

            if not path.exists():

                raise FileNotFoundError(
                    "Registration image not found: "
                    f"{path}"
                )

        return paths

    # ========================================================
    # DUPLICATE CHECK
    # ========================================================

    def _ensure_person_not_registered(
        self,
        name: str,
    ) -> None:

        existing = get_person_by_name(
            name
        )

        if existing is None:
            return

        if bool(
            existing[
                "is_active"
            ]
        ):

            raise ValueError(
                "Person is already registered: "
                f"{existing['display_name'] or existing['name']}"
            )

        raise ValueError(
            "An inactive registration already exists "
            "for this person."
        )

    # ========================================================
    # REGISTER PERSON
    # ========================================================

    def register_person(
        self,
        name: str,
        sample_paths: Iterable[
            str | Path
        ],
        display_name: str | None = None,
        relation: str | None = None,
        notes: str | None = None,
    ) -> RegistrationResult:

        cleaned_name = " ".join(
            name.strip().split()
        )

        if not cleaned_name:

            raise ValueError(
                "Person name cannot be empty."
            )

        if display_name is None:

            cleaned_display_name = (
                cleaned_name
            )

        else:

            cleaned_display_name = " ".join(
                display_name.strip().split()
            )

            if not cleaned_display_name:

                raise ValueError(
                    "Display name cannot be empty."
                )

        cleaned_relation: (
            str
            | None
        )

        if relation is None:

            cleaned_relation = None

        else:

            cleaned_relation = " ".join(
                relation.strip().split()
            )

            if not cleaned_relation:

                cleaned_relation = None

        self._ensure_person_not_registered(
            cleaned_name
        )

        paths = (
            self._validate_sample_paths(
                sample_paths
            )
        )

        extracted_samples: list[
            tuple[
                Path,
                np.ndarray,
                float,
            ]
        ] = []

        for path in paths:

            (
                embedding,
                confidence,
            ) = (
                self.extract_registration_embedding(
                    path
                )
            )

            extracted_samples.append(
                (
                    path,
                    embedding,
                    confidence,
                )
            )

        person_id = create_person(
            name=(
                cleaned_name
            ),
            display_name=(
                cleaned_display_name
            ),
            relation=(
                cleaned_relation
            ),
            notes=(
                notes
            ),
        )

        accepted_samples: list[
            RegistrationSample
        ] = []

        profile_image_path: (
            str
            | None
        ) = None

        try:

            for (
                index,
                sample,
            ) in enumerate(
                extracted_samples,
                start=1,
            ):

                (
                    source_path,
                    embedding,
                    confidence,
                ) = sample

                stored_path = (
                    self._save_registration_image(
                        source_path=(
                            source_path
                        ),
                        person_name=(
                            cleaned_name
                        ),
                        sample_number=(
                            index
                        ),
                    )
                )

                save_face_embedding(
                    person_id=(
                        person_id
                    ),
                    embedding=(
                        embedding
                    ),
                    sample_image_path=str(
                        stored_path
                    ),
                )

                if index == 1:

                    profile_image_path = str(
                        stored_path
                    )

                    update_profile_image_path(
                        person_id=person_id,
                        profile_image_path=(
                            profile_image_path
                        ),
                    )

                accepted_samples.append(
                    RegistrationSample(
                        source_path=(
                            source_path
                        ),
                        stored_path=(
                            stored_path
                        ),
                        confidence=float(
                            confidence
                        ),
                        embedding=(
                            embedding
                        ),
                    )
                )

        except Exception:

            delete_face_embeddings_for_person(
                person_id
            )

            deactivate_person(
                person_id
            )

            raise

        person = get_person_by_id(
            person_id
        )

        if person is None:

            raise RuntimeError(
                "Registered person could not "
                "be loaded after creation."
            )

        return RegistrationResult(
            person_id=(
                person_id
            ),
            name=str(
                person[
                    "name"
                ]
            ),
            display_name=str(
                person[
                    "display_name"
                ]
                or person[
                    "name"
                ]
            ),
            relation=(
                person[
                    "relation"
                ]
            ),
            profile_image_path=(
                person.get(
                    "profile_image_path"
                )
            ),
            sample_count=len(
                accepted_samples
            ),
            samples=(
                accepted_samples
            ),
        )