from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from config import (
    FACE_DETECTION_MODEL_PATH,
    FACE_DETECTION_NMS_THRESHOLD,
    FACE_DETECTION_SCORE_THRESHOLD,
    FACE_RECOGNITION_COSINE_THRESHOLD,
    FACE_RECOGNITION_MODEL_PATH,
    FACE_RECOGNITION_TOP_K,
)
from database.person_repository import (
    get_all_face_embeddings,
)


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class DetectedFace:
    """
    Represents one face detected by YuNet.
    """

    bbox: tuple[int, int, int, int]

    confidence: float

    raw_face: np.ndarray

    aligned_face: np.ndarray | None = None

    embedding: np.ndarray | None = None


@dataclass
class FaceMatch:
    """
    Final result of identity matching.
    """

    person_id: int | None

    person_name: str

    display_name: str

    similarity: float

    is_known: bool


@dataclass
class PersonSimilarity:
    """
    Internal score calculated for one registered person.
    """

    person_id: int

    person_name: str

    display_name: str

    similarity: float

    sample_similarities: list[float]


# ============================================================
# FACE RECOGNITION ENGINE
# ============================================================

class FaceRecognitionEngine:
    """
    Core MemorAi face recognition engine.

    Responsibilities:

    - Detect faces using YuNet
    - Align faces for SFace
    - Generate embeddings
    - Compare face embeddings
    - Aggregate multiple samples per identity
    - Return Known / Unknown
    """

    def __init__(
        self,
        detection_model_path: str | Path | None = None,
        recognition_model_path: str | Path | None = None,
        detection_score_threshold: float | None = None,
        detection_nms_threshold: float | None = None,
        recognition_threshold: float | None = None,
        recognition_top_k: int | None = None,
    ) -> None:

        # ----------------------------------------------------
        # MODEL PATHS
        # ----------------------------------------------------

        self.detection_model_path = Path(
            detection_model_path
            or FACE_DETECTION_MODEL_PATH
        )

        self.recognition_model_path = Path(
            recognition_model_path
            or FACE_RECOGNITION_MODEL_PATH
        )

        # ----------------------------------------------------
        # DETECTION CONFIGURATION
        # ----------------------------------------------------

        if detection_score_threshold is None:

            detection_score_threshold = (
                FACE_DETECTION_SCORE_THRESHOLD
            )

        if detection_nms_threshold is None:

            detection_nms_threshold = (
                FACE_DETECTION_NMS_THRESHOLD
            )

        # ----------------------------------------------------
        # RECOGNITION CONFIGURATION
        # ----------------------------------------------------

        if recognition_threshold is None:

            recognition_threshold = (
                FACE_RECOGNITION_COSINE_THRESHOLD
            )

        if recognition_top_k is None:

            recognition_top_k = (
                FACE_RECOGNITION_TOP_K
            )

        self.detection_score_threshold = float(
            detection_score_threshold
        )

        self.detection_nms_threshold = float(
            detection_nms_threshold
        )

        self.recognition_threshold = float(
            recognition_threshold
        )

        self.recognition_top_k = int(
            recognition_top_k
        )

        if not (
            0.0
            <= self.detection_score_threshold
            <= 1.0
        ):

            raise ValueError(
                "Face detection threshold must "
                "be between 0 and 1."
            )

        if self.recognition_top_k <= 0:

            raise ValueError(
                "FACE_RECOGNITION_TOP_K must "
                "be greater than zero."
            )

        # ----------------------------------------------------
        # MODEL VALIDATION
        # ----------------------------------------------------

        self._validate_model_files()

        # ----------------------------------------------------
        # YUNET
        # ----------------------------------------------------

        self.detector = (
            cv2.FaceDetectorYN.create(
                str(
                    self.detection_model_path
                ),
                "",
                (320, 320),
                self.detection_score_threshold,
                self.detection_nms_threshold,
                5000,
            )
        )

        # ----------------------------------------------------
        # SFACE
        # ----------------------------------------------------

        self.recognizer = (
            cv2.FaceRecognizerSF.create(
                str(
                    self.recognition_model_path
                ),
                "",
            )
        )

    # ========================================================
    # MODEL VALIDATION
    # ========================================================

    def _validate_model_files(
        self,
    ) -> None:

        if not self.detection_model_path.exists():

            raise FileNotFoundError(
                "YuNet face detection model "
                "was not found:\n"
                f"{self.detection_model_path}"
            )

        if not self.recognition_model_path.exists():

            raise FileNotFoundError(
                "SFace recognition model "
                "was not found:\n"
                f"{self.recognition_model_path}"
            )

    # ========================================================
    # FACE DETECTION
    # ========================================================

    def detect_faces(
        self,
        frame: np.ndarray,
    ) -> list[DetectedFace]:

        if frame is None:

            raise ValueError(
                "Frame cannot be None."
            )

        if not isinstance(
            frame,
            np.ndarray,
        ):

            raise TypeError(
                "Frame must be a NumPy array."
            )

        if frame.size == 0:

            raise ValueError(
                "Frame cannot be empty."
            )

        if frame.ndim != 3:

            raise ValueError(
                "Expected a BGR color image."
            )

        original_height, original_width = (
            frame.shape[:2]
        )

        # ----------------------------------------------------
        # RESIZE LARGE FRAMES
        # ----------------------------------------------------

        max_dimension = 800

        largest_dimension = max(
            original_width,
            original_height,
        )

        if largest_dimension > max_dimension:

            scale = (
                max_dimension
                / largest_dimension
            )

            detection_width = max(
                1,
                int(
                    round(
                        original_width
                        * scale
                    )
                ),
            )

            detection_height = max(
                1,
                int(
                    round(
                        original_height
                        * scale
                    )
                ),
            )

            detection_frame = cv2.resize(
                frame,
                (
                    detection_width,
                    detection_height,
                ),
                interpolation=cv2.INTER_AREA,
            )

        else:

            scale = 1.0

            detection_frame = frame

            detection_width = (
                original_width
            )

            detection_height = (
                original_height
            )

        # ----------------------------------------------------
        # YUNET DETECTION
        # ----------------------------------------------------

        self.detector.setInputSize(
            (
                detection_width,
                detection_height,
            )
        )

        _result, faces = (
            self.detector.detect(
                detection_frame
            )
        )

        if faces is None:

            return []

        inverse_scale = (
            1.0 / scale
        )

        detected_faces: list[
            DetectedFace
        ] = []

        # ----------------------------------------------------
        # CONVERT TO ORIGINAL IMAGE COORDINATES
        # ----------------------------------------------------

        for face in faces:

            mapped_face = (
                face.copy()
            )

            if scale != 1.0:

                mapped_face[
                    :14
                ] *= inverse_scale

            x = int(
                round(
                    float(
                        mapped_face[0]
                    )
                )
            )

            y = int(
                round(
                    float(
                        mapped_face[1]
                    )
                )
            )

            width = int(
                round(
                    float(
                        mapped_face[2]
                    )
                )
            )

            height = int(
                round(
                    float(
                        mapped_face[3]
                    )
                )
            )

            confidence = float(
                mapped_face[-1]
            )

            detected_faces.append(
                DetectedFace(
                    bbox=(
                        x,
                        y,
                        width,
                        height,
                    ),
                    confidence=confidence,
                    raw_face=(
                        mapped_face
                    ),
                )
            )

        return detected_faces

    # ========================================================
    # ALIGNMENT
    # ========================================================

    def align_face(
        self,
        frame: np.ndarray,
        detected_face: DetectedFace,
    ) -> np.ndarray:

        aligned_face = (
            self.recognizer.alignCrop(
                frame,
                detected_face.raw_face,
            )
        )

        if aligned_face is None:

            raise RuntimeError(
                "SFace could not align "
                "the detected face."
            )

        detected_face.aligned_face = (
            aligned_face
        )

        return aligned_face

    # ========================================================
    # EMBEDDING
    # ========================================================

    def generate_embedding(
        self,
        frame: np.ndarray,
        detected_face: DetectedFace,
    ) -> np.ndarray:

        aligned_face = (
            detected_face.aligned_face
        )

        if aligned_face is None:

            aligned_face = (
                self.align_face(
                    frame,
                    detected_face,
                )
            )

        embedding = (
            self.recognizer.feature(
                aligned_face
            )
        )

        if embedding is None:

            raise RuntimeError(
                "SFace failed to generate "
                "a face embedding."
            )

        embedding_array = np.asarray(
            embedding,
            dtype=np.float32,
        ).reshape(-1)

        if embedding_array.size == 0:

            raise RuntimeError(
                "SFace returned an empty "
                "embedding."
            )

        if not np.all(
            np.isfinite(
                embedding_array
            )
        ):

            raise RuntimeError(
                "SFace returned invalid "
                "embedding values."
            )

        detected_face.embedding = (
            embedding_array
        )

        return embedding_array

    # ========================================================
    # PROCESS FRAME
    # ========================================================

    def process_frame(
        self,
        frame: np.ndarray,
    ) -> list[DetectedFace]:

        faces = self.detect_faces(
            frame
        )

        processed_faces: list[
            DetectedFace
        ] = []

        for detected_face in faces:

            try:

                self.generate_embedding(
                    frame,
                    detected_face,
                )

                processed_faces.append(
                    detected_face
                )

            except (
                cv2.error,
                RuntimeError,
            ):

                continue

        return processed_faces

    # ========================================================
    # EMBEDDING COMPARISON
    # ========================================================

    def compare_embeddings(
        self,
        embedding_a: np.ndarray,
        embedding_b: np.ndarray,
    ) -> float:

        face_a = np.asarray(
            embedding_a,
            dtype=np.float32,
        ).reshape(
            1,
            -1,
        )

        face_b = np.asarray(
            embedding_b,
            dtype=np.float32,
        ).reshape(
            1,
            -1,
        )

        if face_a.size == 0:

            raise ValueError(
                "First embedding is empty."
            )

        if face_b.size == 0:

            raise ValueError(
                "Second embedding is empty."
            )

        if face_a.shape != face_b.shape:

            raise ValueError(
                "Face embedding dimensions "
                "do not match."
            )

        similarity = (
            self.recognizer.match(
                face_a,
                face_b,
                cv2.FaceRecognizerSF_FR_COSINE,
            )
        )

        return float(
            similarity
        )

    # ========================================================
    # GROUP STORED EMBEDDINGS
    # ========================================================

    def _group_embeddings_by_person(
        self,
        rows: list[
            dict[str, Any]
        ],
    ) -> dict[
        int,
        list[
            dict[str, Any]
        ],
    ]:

        grouped: dict[
            int,
            list[
                dict[str, Any]
            ],
        ] = {}

        for row in rows:

            person_id = int(
                row[
                    "person_id"
                ]
            )

            grouped.setdefault(
                person_id,
                [],
            ).append(
                row
            )

        return grouped

    # ========================================================
    # PERSON SCORE
    # ========================================================

    def _calculate_person_similarity(
        self,
        query_embedding: np.ndarray,
        samples: list[
            dict[str, Any]
        ],
    ) -> PersonSimilarity:

        if not samples:

            raise ValueError(
                "Person has no stored face samples."
            )

        similarities: list[
            float
        ] = []

        for sample in samples:

            similarity = (
                self.compare_embeddings(
                    query_embedding,
                    sample[
                        "embedding"
                    ],
                )
            )

            similarities.append(
                similarity
            )

        similarities.sort(
            reverse=True
        )

        selected_count = min(
            self.recognition_top_k,
            len(
                similarities
            ),
        )

        strongest = (
            similarities[
                :selected_count
            ]
        )

        aggregated_similarity = (
            sum(
                strongest
            )
            / len(
                strongest
            )
        )

        first_sample = (
            samples[0]
        )

        return PersonSimilarity(
            person_id=int(
                first_sample[
                    "person_id"
                ]
            ),
            person_name=str(
                first_sample[
                    "name"
                ]
            ),
            display_name=str(
                first_sample[
                    "display_name"
                ]
                or first_sample[
                    "name"
                ]
            ),
            similarity=float(
                aggregated_similarity
            ),
            sample_similarities=(
                similarities
            ),
        )

    # ========================================================
    # MATCH IDENTITY
    # ========================================================

    def match_embedding(
        self,
        query_embedding: np.ndarray,
    ) -> FaceMatch:

        rows = (
            get_all_face_embeddings(
                active_people_only=True
            )
        )

        # ----------------------------------------------------
        # NOTHING REGISTERED
        # ----------------------------------------------------

        if not rows:

            return FaceMatch(
                person_id=None,
                person_name="unknown",
                display_name="Unknown",
                similarity=0.0,
                is_known=False,
            )

        # ----------------------------------------------------
        # GROUP SAMPLES BY PERSON
        # ----------------------------------------------------

        grouped = (
            self._group_embeddings_by_person(
                rows
            )
        )

        person_scores: list[
            PersonSimilarity
        ] = []

        # ----------------------------------------------------
        # CALCULATE ONE SCORE PER PERSON
        # ----------------------------------------------------

        for samples in grouped.values():

            person_score = (
                self._calculate_person_similarity(
                    query_embedding,
                    samples,
                )
            )

            person_scores.append(
                person_score
            )

        if not person_scores:

            return FaceMatch(
                person_id=None,
                person_name="unknown",
                display_name="Unknown",
                similarity=0.0,
                is_known=False,
            )

        # ----------------------------------------------------
        # FIND BEST IDENTITY
        # ----------------------------------------------------

        best_person = max(
            person_scores,
            key=lambda item: (
                item.similarity
            ),
        )

        # ----------------------------------------------------
        # UNKNOWN
        # ----------------------------------------------------

        if (
            best_person.similarity
            < self.recognition_threshold
        ):

            return FaceMatch(
                person_id=None,
                person_name="unknown",
                display_name="Unknown",
                similarity=float(
                    best_person.similarity
                ),
                is_known=False,
            )

        # ----------------------------------------------------
        # KNOWN
        # ----------------------------------------------------

        return FaceMatch(
            person_id=(
                best_person.person_id
            ),
            person_name=(
                best_person.person_name
            ),
            display_name=(
                best_person.display_name
            ),
            similarity=float(
                best_person.similarity
            ),
            is_known=True,
        )

    # ========================================================
    # RECOGNIZE DETECTED FACE
    # ========================================================

    def recognize_face(
        self,
        detected_face: DetectedFace,
    ) -> FaceMatch:

        if detected_face.embedding is None:

            raise ValueError(
                "Detected face does not "
                "contain an embedding."
            )

        return self.match_embedding(
            detected_face.embedding
        )


# ============================================================
# IMAGE LOADER
# ============================================================

def load_image(
    image_path: str | Path,
) -> np.ndarray:

    path = Path(
        image_path
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Image not found: {path}"
        )

    image = cv2.imread(
        str(
            path
        )
    )

    if image is None:

        raise ValueError(
            "OpenCV could not load image: "
            f"{path}"
        )

    return image