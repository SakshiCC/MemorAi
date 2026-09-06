from __future__ import annotations

from collections import deque

from concurrent.futures import (
    Future,
    ThreadPoolExecutor,
)

from dataclasses import dataclass

from io import BytesIO

from pathlib import Path

from threading import RLock

from typing import Any

import time

import cv2
import numpy as np
import torch

from PIL import Image

from transformers import (
    AutoImageProcessor,
    AutoModel,
)

from config import (
    MEDICINE_EMBEDDING_DEVICE,
    MEDICINE_EMBEDDING_MODEL_NAME,
    MEDICINE_RECOGNITION_THRESHOLD,
    MEDICINE_RECOGNITION_TOP_K,
)

from database.medicine_repository import (
    get_all_medicine_reference_images,
)

from medicine.medicine_text_recognizer import (
    OCRMedicineMatch,
    OCR_STRONG_IDENTITY_THRESHOLD,
    MedicineTextRecognizer,
)


# ============================================================
# PARALLEL RECOGNITION CONFIG
# ============================================================

DINO_STRONG_MARGIN = 0.075

OCR_STRONG_THRESHOLD = (
    OCR_STRONG_IDENTITY_THRESHOLD
)

DINO_SUPPORT_THRESHOLD = (
    MEDICINE_RECOGNITION_THRESHOLD
)

OCR_SUPPORT_THRESHOLD = 0.64


# Keep results available long enough for asynchronous fusion.
PARALLEL_RESULT_MAX_AGE_SECONDS = 6.0


# OCR automatically chooses the clearest recent frame.
OCR_FRAME_BUFFER_SIZE = 4


# ============================================================
# ONE NEW RECOGNITION EVIDENCE EVENT
# ============================================================

@dataclass(frozen=True)
class RecognitionEvidenceEvent:
    """
    One genuinely new recognition result.

    event_key is globally unique for the current recognizer
    session.

    Examples:

        dino:4:17
        ocr:4:8
        fused:4:17:8

    MedicineMonitor uses this key to ensure the same cached
    result cannot be counted multiple times.
    """

    event_key: str

    medicine_id: int | None

    accepted: bool

    score: float

    source: str


# ============================================================
# FINAL MEDICINE MATCH
# ============================================================

@dataclass
class MedicineMatch:

    medicine_id: int | None

    medicine_name: str

    dosage: str | None

    similarity: float

    is_known: bool

    threshold: float = (
        MEDICINE_RECOGNITION_THRESHOLD
    )

    # --------------------------------------------------------
    # LEGACY / COMPATIBILITY ORB FIELDS
    # --------------------------------------------------------

    feature_verified: bool = False

    feature_score: float = 0.0

    feature_good_matches: int = 0

    feature_inliers: int = 0

    feature_inlier_ratio: float = 0.0

    # --------------------------------------------------------
    # DINO
    # --------------------------------------------------------

    dino_available: bool = False

    dino_score: float = 0.0

    dino_strong: bool = False

    dino_sequence: int = 0

    # --------------------------------------------------------
    # OCR
    # --------------------------------------------------------

    ocr_available: bool = False

    ocr_score: float = 0.0

    ocr_text_score: float = 0.0

    ocr_dosage_score: float = 0.0

    ocr_text: str = ""

    ocr_keyword: str | None = None

    ocr_seconds: float = 0.0

    ocr_frame_quality: float = 0.0

    ocr_strong: bool = False

    ocr_sequence: int = 0

    # --------------------------------------------------------
    # FINAL DECISION
    # --------------------------------------------------------

    recognition_source: str = "none"

    combined_score: float = 0.0

    acceptance_reason: str = ""

    # --------------------------------------------------------
    # NEW UNIQUE TEMPORAL EVIDENCE
    # --------------------------------------------------------

    evidence_events: tuple[
        RecognitionEvidenceEvent,
        ...
    ] = ()


# ============================================================
# DINO BACKGROUND RESULT
# ============================================================

@dataclass
class DINOResult:

    medicine_id: int | None

    medicine_name: str

    dosage: str | None

    score: float

    samples: list[
        dict[str, Any]
    ]

    completed_at: float

    sequence: int

    generation: int


# ============================================================
# OCR BACKGROUND RESULT
# ============================================================

@dataclass
class OCRWorkerResult:

    match: OCRMedicineMatch

    completed_at: float

    sequence: int

    generation: int


# ============================================================
# RECOGNITION ENGINE
# ============================================================

class MedicineRecognitionEngine:
    """
    Parallel medicine recognition.

    DINO and EasyOCR are independent background jobs.

    Neither waits for the other.

    Very important:

    A completed worker result receives a unique sequence ID.

    Therefore repeated UI/camera polling may display the
    same result repeatedly, but MedicineMonitor can count it
    only once.
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        recognition_threshold: float | None = None,
        recognition_top_k: int | None = None,
    ) -> None:

        # ====================================================
        # CONFIG
        # ====================================================

        self.model_name = str(
            model_name
            or MEDICINE_EMBEDDING_MODEL_NAME
        ).strip()

        self.device_name = str(
            device
            or MEDICINE_EMBEDDING_DEVICE
        ).strip().lower()

        self.recognition_threshold = float(
            recognition_threshold
            if recognition_threshold is not None
            else MEDICINE_RECOGNITION_THRESHOLD
        )

        self.recognition_top_k = int(
            recognition_top_k
            if recognition_top_k is not None
            else MEDICINE_RECOGNITION_TOP_K
        )

        if not (
            0.0
            <= self.recognition_threshold
            <= 1.0
        ):

            raise ValueError(
                "Medicine recognition threshold must "
                "be between 0 and 1."
            )

        if self.recognition_top_k <= 0:

            raise ValueError(
                "Medicine recognition top-k must "
                "be greater than zero."
            )

        # ====================================================
        # DEVICE
        # ====================================================

        if (
            self.device_name.startswith(
                "cuda"
            )
            and not torch.cuda.is_available()
        ):

            raise RuntimeError(
                "CUDA was requested for medicine "
                "recognition but CUDA is unavailable."
            )

        self.device = torch.device(
            self.device_name
        )

        # ====================================================
        # DINO
        # ====================================================

        self.processor = (
            AutoImageProcessor.from_pretrained(
                self.model_name
            )
        )

        self.model = (
            AutoModel.from_pretrained(
                self.model_name
            )
        )

        self.model.to(
            self.device
        )

        self.model.eval()

        # ====================================================
        # OCR
        # ====================================================

        self.text_recognizer = (
            MedicineTextRecognizer(
                gpu=False
            )
        )

        # ====================================================
        # TWO WORKER THREADS
        # ====================================================

        self._executor = (
            ThreadPoolExecutor(
                max_workers=2,
                thread_name_prefix=(
                    "MemorAiMedicine"
                ),
            )
        )

        self._lock = (
            RLock()
        )

        self._dino_future: (
            Future
            | None
        ) = None

        self._ocr_future: (
            Future
            | None
        ) = None

        # ====================================================
        # MOST RECENT COMPLETED RESULTS
        # ====================================================

        self._latest_dino: (
            DINOResult
            | None
        ) = None

        self._latest_ocr: (
            OCRWorkerResult
            | None
        ) = None

        # ====================================================
        # OCR RECENT FRAME BUFFER
        # ====================================================

        self._ocr_frames: deque[
            np.ndarray
        ] = deque(
            maxlen=(
                OCR_FRAME_BUFFER_SIZE
            )
        )

        # ====================================================
        # UNIQUE SEQUENCES
        # ====================================================

        self._dino_sequence = 0

        self._ocr_sequence = 0

        # ----------------------------------------------------
        # Generation protects reset().
        #
        # An old worker may still finish after reset.
        # Its generation will no longer match and the result
        # will be ignored.
        # ----------------------------------------------------

        self._generation = 1


    # ========================================================
    # LOAD IMAGE
    # ========================================================

    def load_image(
        self,
        image: (
            str
            | Path
            | np.ndarray
            | Image.Image
        ),
    ) -> Image.Image:

        if isinstance(
            image,
            Image.Image,
        ):

            return image.convert(
                "RGB"
            )

        if isinstance(
            image,
            np.ndarray,
        ):

            if image.size == 0:

                raise ValueError(
                    "Medicine image cannot be empty."
                )

            if (
                image.ndim != 3
                or image.shape[
                    2
                ] != 3
            ):

                raise ValueError(
                    "Medicine NumPy image must "
                    "be BGR colour."
                )

            rgb = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB,
            )

            return Image.fromarray(
                rgb
            )

        path = Path(
            image
        )

        if not path.exists():

            raise FileNotFoundError(
                f"Medicine image not found: {path}"
            )

        with Image.open(
            path
        ) as loaded:

            return loaded.convert(
                "RGB"
            )


    # ========================================================
    # LOAD BGR
    # ========================================================

    def _load_bgr_image(
        self,
        image: (
            str
            | Path
            | np.ndarray
            | Image.Image
        ),
    ) -> np.ndarray:

        if isinstance(
            image,
            np.ndarray,
        ):

            if image.size == 0:

                raise ValueError(
                    "Medicine image cannot be empty."
                )

            return (
                image.copy()
            )

        rgb = np.asarray(
            self.load_image(
                image
            ),
            dtype=np.uint8,
        )

        return cv2.cvtColor(
            rgb,
            cv2.COLOR_RGB2BGR,
        )


    # ========================================================
    # GENERATE DINO EMBEDDING
    # ========================================================

    def generate_embedding(
        self,
        image: (
            str
            | Path
            | np.ndarray
            | Image.Image
        ),
    ) -> np.ndarray:

        pil_image = (
            self.load_image(
                image
            )
        )

        inputs = (
            self.processor(
                images=(
                    pil_image
                ),
                return_tensors=(
                    "pt"
                ),
            )
        )

        inputs = {
            key:
                value.to(
                    self.device
                )

            for (
                key,
                value,
            )
            in inputs.items()
        }

        with torch.inference_mode():

            outputs = (
                self.model(
                    **inputs
                )
            )

        hidden = (
            outputs.last_hidden_state
        )

        if hidden is None:

            raise RuntimeError(
                "DINO returned no hidden state."
            )

        embedding = (
            hidden[
                0,
                0,
                :
            ]
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.float32
            )
            .reshape(
                -1
            )
        )

        norm = float(
            np.linalg.norm(
                embedding
            )
        )

        if norm <= 0.0:

            raise RuntimeError(
                "DINO embedding has zero norm."
            )

        return (
            embedding
            / norm
        ).astype(
            np.float32
        )


    # ========================================================
    # SERIALIZE EMBEDDING
    # ========================================================

    @staticmethod
    def serialize_embedding(
        embedding: np.ndarray,
    ) -> bytes:

        array = np.asarray(
            embedding,
            dtype=np.float32,
        ).reshape(
            -1
        )

        buffer = BytesIO()

        np.save(
            buffer,
            array,
            allow_pickle=False,
        )

        return (
            buffer.getvalue()
        )


    # ========================================================
    # DESERIALIZE EMBEDDING
    # ========================================================

    @staticmethod
    def deserialize_embedding(
        embedding_bytes: bytes,
    ) -> np.ndarray:

        if not embedding_bytes:

            raise ValueError(
                "Stored medicine embedding is empty."
            )

        array = np.load(
            BytesIO(
                embedding_bytes
            ),
            allow_pickle=False,
        ).astype(
            np.float32
        ).reshape(
            -1
        )

        norm = float(
            np.linalg.norm(
                array
            )
        )

        if norm <= 0.0:

            raise ValueError(
                "Stored medicine embedding has "
                "zero norm."
            )

        return (
            array
            / norm
        ).astype(
            np.float32
        )


    # ========================================================
    # COSINE SIMILARITY
    # ========================================================

    @staticmethod
    def compare_embeddings(
        embedding_a: np.ndarray,
        embedding_b: np.ndarray,
    ) -> float:

        a = np.asarray(
            embedding_a,
            dtype=np.float32,
        ).reshape(
            -1
        )

        b = np.asarray(
            embedding_b,
            dtype=np.float32,
        ).reshape(
            -1
        )

        if (
            a.size == 0
            or b.size == 0
        ):

            raise ValueError(
                "Medicine embeddings cannot be empty."
            )

        if a.shape != b.shape:

            raise ValueError(
                "Medicine embedding dimensions "
                "do not match."
            )

        a_norm = float(
            np.linalg.norm(
                a
            )
        )

        b_norm = float(
            np.linalg.norm(
                b
            )
        )

        if (
            a_norm <= 0.0
            or b_norm <= 0.0
        ):

            raise ValueError(
                "Medicine embeddings cannot have "
                "zero norm."
            )

        similarity = float(
            np.dot(
                a,
                b,
            )
            / (
                a_norm
                * b_norm
            )
        )

        return float(
            np.clip(
                similarity,
                -1.0,
                1.0,
            )
        )


    # ========================================================
    # GROUP REGISTERED DINO REFERENCES
    # ========================================================

    def _group_references(
        self,
    ) -> dict[
        int,
        list[
            dict[str, Any]
        ],
    ]:

        rows = (
            get_all_medicine_reference_images(
                active_medicines_only=True
            )
        )

        grouped: dict[
            int,
            list[
                dict[str, Any]
            ],
        ] = {}

        for row in rows:

            embedding_blob = (
                row.get(
                    "embedding"
                )
            )

            if not embedding_blob:

                continue

            embedding_model = str(
                row.get(
                    "embedding_model"
                )
                or ""
            ).strip()

            if (
                embedding_model
                != self.model_name
            ):

                continue

            medicine_id = int(
                row[
                    "medicine_id"
                ]
            )

            grouped.setdefault(
                medicine_id,
                [],
            ).append(
                row
            )

        return (
            grouped
        )


    # ========================================================
    # NEXT DINO SEQUENCE
    # ========================================================

    def _next_dino_sequence(
        self,
    ) -> int:

        with self._lock:

            self._dino_sequence += 1

            return (
                self._dino_sequence
            )


    # ========================================================
    # NEXT OCR SEQUENCE
    # ========================================================

    def _next_ocr_sequence(
        self,
    ) -> int:

        with self._lock:

            self._ocr_sequence += 1

            return (
                self._ocr_sequence
            )


    # ========================================================
    # DINO WORKER
    # ========================================================

    def _run_dino(
        self,
        frame: np.ndarray,
        generation: int,
    ) -> DINOResult:

        embedding = (
            self.generate_embedding(
                frame
            )
        )

        grouped = (
            self._group_references()
        )

        best_id = None

        best_name = "unknown"

        best_dosage = None

        best_score = 0.0

        best_samples: list[
            dict[str, Any]
        ] = []

        for (
            medicine_id,
            samples,
        ) in grouped.items():

            scores: list[
                float
            ] = []

            for sample in samples:

                try:

                    stored = (
                        self.deserialize_embedding(
                            sample[
                                "embedding"
                            ]
                        )
                    )

                    score = (
                        self.compare_embeddings(
                            embedding,
                            stored,
                        )
                    )

                    scores.append(
                        score
                    )

                except Exception:

                    continue

            if not scores:

                continue

            scores.sort(
                reverse=True
            )

            selected_count = min(
                self.recognition_top_k,
                len(
                    scores
                ),
            )

            selected = (
                scores[
                    :selected_count
                ]
            )

            score = float(
                sum(
                    selected
                )
                / len(
                    selected
                )
            )

            if score > best_score:

                first = (
                    samples[
                        0
                    ]
                )

                best_id = (
                    medicine_id
                )

                best_name = str(
                    first[
                        "medicine_name"
                    ]
                )

                best_dosage = (
                    str(
                        first[
                            "dosage"
                        ]
                    )
                    if first.get(
                        "dosage"
                    )
                    is not None
                    else None
                )

                best_score = (
                    score
                )

                best_samples = (
                    samples
                )

        sequence = (
            self._next_dino_sequence()
        )

        return DINOResult(
            medicine_id=(
                best_id
            ),

            medicine_name=(
                best_name
            ),

            dosage=(
                best_dosage
            ),

            score=float(
                best_score
            ),

            samples=(
                best_samples
            ),

            completed_at=(
                time.monotonic()
            ),

            sequence=(
                sequence
            ),

            generation=(
                generation
            ),
        )


    # ========================================================
    # OCR WORKER
    # ========================================================

    def _run_ocr(
        self,
        frames: list[
            np.ndarray
        ],
        generation: int,
    ) -> OCRWorkerResult:

        result = (
            self.text_recognizer
            .match_registered_medicines(
                frames
            )
        )

        sequence = (
            self._next_ocr_sequence()
        )

        return OCRWorkerResult(
            match=(
                result
            ),

            completed_at=(
                time.monotonic()
            ),

            sequence=(
                sequence
            ),

            generation=(
                generation
            ),
        )


    # ========================================================
    # HARVEST COMPLETED WORKERS
    # ========================================================

    def _harvest(
        self,
    ) -> None:

        now = (
            time.monotonic()
        )

        with self._lock:

            current_generation = (
                self._generation
            )

            # =================================================
            # DINO
            # =================================================

            if (
                self._dino_future
                is not None
                and self._dino_future.done()
            ):

                try:

                    result = (
                        self._dino_future.result()
                    )

                    if (
                        result.generation
                        == current_generation
                    ):

                        self._latest_dino = (
                            result
                        )

                except Exception:

                    pass

                self._dino_future = (
                    None
                )

            # =================================================
            # OCR
            # =================================================

            if (
                self._ocr_future
                is not None
                and self._ocr_future.done()
            ):

                try:

                    result = (
                        self._ocr_future.result()
                    )

                    if (
                        result.generation
                        == current_generation
                    ):

                        self._latest_ocr = (
                            result
                        )

                except Exception:

                    pass

                self._ocr_future = (
                    None
                )

            # =================================================
            # EXPIRE DINO
            # =================================================

            if (
                self._latest_dino
                is not None
                and (
                    now
                    - self._latest_dino.completed_at
                )
                > PARALLEL_RESULT_MAX_AGE_SECONDS
            ):

                self._latest_dino = (
                    None
                )

            # =================================================
            # EXPIRE OCR
            # =================================================

            if (
                self._latest_ocr
                is not None
                and (
                    now
                    - self._latest_ocr.completed_at
                )
                > PARALLEL_RESULT_MAX_AGE_SECONDS
            ):

                self._latest_ocr = (
                    None
                )


    # ========================================================
    # SUBMIT NEW INDEPENDENT WORK
    # ========================================================

    def _submit_workers(
        self,
        frame: np.ndarray,
    ) -> None:

        with self._lock:

            generation = (
                self._generation
            )

            # ------------------------------------------------
            # OCR best-frame buffer
            # ------------------------------------------------

            self._ocr_frames.append(
                frame.copy()
            )

            # ------------------------------------------------
            # DINO
            # ------------------------------------------------

            if self._dino_future is None:

                self._dino_future = (
                    self._executor.submit(
                        self._run_dino,
                        frame.copy(),
                        generation,
                    )
                )

            # ------------------------------------------------
            # OCR
            # ------------------------------------------------

            if self._ocr_future is None:

                buffered_frames = [
                    buffered.copy()

                    for buffered
                    in self._ocr_frames
                ]

                self._ocr_future = (
                    self._executor.submit(
                        self._run_ocr,
                        buffered_frames,
                        generation,
                    )
                )


    # ========================================================
    # BUILD UNIQUE EVIDENCE EVENT
    # ========================================================

    @staticmethod
    def _event(
        key: str,
        medicine_id: int | None,
        accepted: bool,
        score: float,
        source: str,
    ) -> RecognitionEvidenceEvent:

        return RecognitionEvidenceEvent(
            event_key=(
                key
            ),

            medicine_id=(
                medicine_id
            ),

            accepted=bool(
                accepted
            ),

            score=float(
                np.clip(
                    score,
                    0.0,
                    1.0,
                )
            ),

            source=(
                source
            ),
        )


    # ========================================================
    # COMBINE CURRENT DINO + OCR RESULTS
    # ========================================================

    def _combine(
        self,
    ) -> MedicineMatch:

        with self._lock:

            dino = (
                self._latest_dino
            )

            ocr_worker = (
                self._latest_ocr
            )

            generation = (
                self._generation
            )

        ocr = (
            ocr_worker.match
            if ocr_worker is not None
            else None
        )

        # ====================================================
        # AVAILABILITY
        # ====================================================

        dino_available = bool(
            dino is not None
            and dino.medicine_id
            is not None
        )

        ocr_available = bool(
            ocr is not None
            and ocr.medicine_id
            is not None
        )

        dino_score = (
            float(
                dino.score
            )
            if dino is not None
            else 0.0
        )

        ocr_score = (
            float(
                ocr.score
            )
            if ocr is not None
            else 0.0
        )

        dino_sequence = (
            int(
                dino.sequence
            )
            if dino is not None
            else 0
        )

        ocr_sequence = (
            int(
                ocr_worker.sequence
            )
            if ocr_worker is not None
            else 0
        )

        # ====================================================
        # STRONG SIGNALS
        # ====================================================

        dino_strong = bool(
            dino_available
            and dino_score
            >= (
                self.recognition_threshold
                + DINO_STRONG_MARGIN
            )
        )

        ocr_strong = bool(
            ocr_available
            and ocr_score
            >= OCR_STRONG_THRESHOLD
        )

        same_medicine = bool(
            dino_available
            and ocr_available
            and dino.medicine_id
            == ocr.medicine_id
        )

        # ====================================================
        # DEFAULT CANDIDATE
        # ====================================================

        if ocr_strong:

            medicine_id = (
                ocr.medicine_id
            )

            medicine_name = (
                ocr.medicine_name
            )

            dosage = (
                ocr.dosage
            )

        elif dino_strong:

            medicine_id = (
                dino.medicine_id
            )

            medicine_name = (
                dino.medicine_name
            )

            dosage = (
                dino.dosage
            )

        elif dino_available:

            medicine_id = (
                dino.medicine_id
            )

            medicine_name = (
                dino.medicine_name
            )

            dosage = (
                dino.dosage
            )

        elif ocr_available:

            medicine_id = (
                ocr.medicine_id
            )

            medicine_name = (
                ocr.medicine_name
            )

            dosage = (
                ocr.dosage
            )

        else:

            medicine_id = None

            medicine_name = (
                "unknown"
            )

            dosage = None

        # ====================================================
        # DECISION
        # ====================================================

        accepted = False

        source = (
            "none"
        )

        reason = (
            "Waiting for parallel recognition results."
        )

        evidence_events: list[
            RecognitionEvidenceEvent
        ] = []

        # ====================================================
        # STRONG CONFLICT
        # ====================================================

        if (
            dino_strong
            and ocr_strong
            and not same_medicine
        ):

            accepted = False

            source = (
                "conflict"
            )

            reason = (
                "DINO and OCR strongly disagree. "
                "Waiting for a clearer medicine view."
            )

            # Neither result should contribute a positive
            # temporal vote while they disagree.

            evidence_events.append(
                self._event(
                    key=(
                        f"dino:{generation}:"
                        f"{dino_sequence}"
                    ),

                    medicine_id=(
                        dino.medicine_id
                    ),

                    accepted=False,

                    score=(
                        dino_score
                    ),

                    source=(
                        "visual"
                    ),
                )
            )

            evidence_events.append(
                self._event(
                    key=(
                        f"ocr:{generation}:"
                        f"{ocr_sequence}"
                    ),

                    medicine_id=(
                        ocr.medicine_id
                    ),

                    accepted=False,

                    score=(
                        ocr_score
                    ),

                    source=(
                        "text"
                    ),
                )
            )

        # ====================================================
        # BOTH STRONG AND AGREE
        # ====================================================

        elif (
            dino_strong
            and ocr_strong
            and same_medicine
        ):

            accepted = True

            source = (
                "both"
            )

            reason = (
                "DINO and OCR independently "
                "recognized the same medicine."
            )

            # Two independent newly completed models are two
            # independent temporal evidence events.

            evidence_events.append(
                self._event(
                    key=(
                        f"dino:{generation}:"
                        f"{dino_sequence}"
                    ),

                    medicine_id=(
                        dino.medicine_id
                    ),

                    accepted=True,

                    score=(
                        dino_score
                    ),

                    source=(
                        "visual"
                    ),
                )
            )

            evidence_events.append(
                self._event(
                    key=(
                        f"ocr:{generation}:"
                        f"{ocr_sequence}"
                    ),

                    medicine_id=(
                        ocr.medicine_id
                    ),

                    accepted=True,

                    score=(
                        ocr_score
                    ),

                    source=(
                        "text"
                    ),
                )
            )

        # ====================================================
        # STRONG OCR
        # ====================================================

        elif ocr_strong:

            accepted = True

            source = (
                "text"
            )

            reason = (
                "Strong OCR medicine text match."
            )

            evidence_events.append(
                self._event(
                    key=(
                        f"ocr:{generation}:"
                        f"{ocr_sequence}"
                    ),

                    medicine_id=(
                        ocr.medicine_id
                    ),

                    accepted=True,

                    score=(
                        ocr_score
                    ),

                    source=(
                        "text"
                    ),
                )
            )

        # ====================================================
        # STRONG DINO
        # ====================================================

        elif dino_strong:

            accepted = True

            source = (
                "visual"
            )

            reason = (
                "Strong DINO visual medicine match."
            )

            evidence_events.append(
                self._event(
                    key=(
                        f"dino:{generation}:"
                        f"{dino_sequence}"
                    ),

                    medicine_id=(
                        dino.medicine_id
                    ),

                    accepted=True,

                    score=(
                        dino_score
                    ),

                    source=(
                        "visual"
                    ),
                )
            )

        # ====================================================
        # MODERATE DINO + OCR AGREE
        # ====================================================

        elif (
            same_medicine
            and dino_score
            >= DINO_SUPPORT_THRESHOLD
            and ocr_score
            >= OCR_SUPPORT_THRESHOLD
        ):

            accepted = True

            source = (
                "both"
            )

            reason = (
                "DINO and OCR supporting evidence "
                "agree on the same medicine."
            )

            # Moderate evidence is treated as one fused vote,
            # not two independent strong votes.

            evidence_events.append(
                self._event(
                    key=(
                        f"fused:{generation}:"
                        f"{dino_sequence}:"
                        f"{ocr_sequence}"
                    ),

                    medicine_id=(
                        dino.medicine_id
                    ),

                    accepted=True,

                    score=(
                        (
                            0.55
                            * dino_score
                        )
                        +
                        (
                            0.45
                            * ocr_score
                        )
                    ),

                    source=(
                        "both"
                    ),
                )
            )

        # ====================================================
        # AVAILABLE BUT NOT STRONG
        # ====================================================

        else:

            if (
                dino_available
                or ocr_available
            ):

                reason = (
                    "Recognition evidence is available, "
                    "but it is not strong enough yet."
                )

            # Add negative evidence only for the currently
            # selected candidate.

            if (
                medicine_id is not None
                and dino_available
                and dino.medicine_id
                == medicine_id
            ):

                evidence_events.append(
                    self._event(
                        key=(
                            f"dino:{generation}:"
                            f"{dino_sequence}"
                        ),

                        medicine_id=(
                            dino.medicine_id
                        ),

                        accepted=False,

                        score=(
                            dino_score
                        ),

                        source=(
                            "visual"
                        ),
                    )
                )

            if (
                medicine_id is not None
                and ocr_available
                and ocr.medicine_id
                == medicine_id
            ):

                evidence_events.append(
                    self._event(
                        key=(
                            f"ocr:{generation}:"
                            f"{ocr_sequence}"
                        ),

                        medicine_id=(
                            ocr.medicine_id
                        ),

                        accepted=False,

                        score=(
                            ocr_score
                        ),

                        source=(
                            "text"
                        ),
                    )
                )

        # ====================================================
        # DISPLAY / COMBINED SCORE
        # ====================================================

        combined_score = max(
            dino_score,
            ocr_score,
        )

        if same_medicine:

            combined_score = (
                (
                    0.55
                    * dino_score
                )
                +
                (
                    0.45
                    * ocr_score
                )
            )

        # ====================================================
        # MATCH
        # ====================================================

        return MedicineMatch(
            medicine_id=(
                medicine_id
            ),

            medicine_name=(
                medicine_name
            ),

            dosage=(
                dosage
            ),

            similarity=float(
                np.clip(
                    combined_score,
                    0.0,
                    1.0,
                )
            ),

            is_known=bool(
                accepted
            ),

            threshold=(
                self.recognition_threshold
            ),

            # Legacy fields.
            feature_verified=False,

            feature_score=0.0,

            feature_good_matches=0,

            feature_inliers=0,

            feature_inlier_ratio=0.0,

            # DINO
            dino_available=(
                dino_available
            ),

            dino_score=(
                dino_score
            ),

            dino_strong=(
                dino_strong
            ),

            dino_sequence=(
                dino_sequence
            ),

            # OCR
            ocr_available=(
                ocr_available
            ),

            ocr_score=(
                ocr_score
            ),

            ocr_text_score=(
                float(
                    ocr.text_score
                )
                if ocr is not None
                else 0.0
            ),

            ocr_dosage_score=(
                float(
                    ocr.dosage_score
                )
                if ocr is not None
                else 0.0
            ),

            ocr_text=(
                str(
                    ocr.detected_text
                )
                if ocr is not None
                else ""
            ),

            ocr_keyword=(
                ocr.matched_keyword
                if ocr is not None
                else None
            ),

            ocr_seconds=(
                float(
                    ocr.processing_seconds
                )
                if ocr is not None
                else 0.0
            ),

            ocr_frame_quality=(
                float(
                    ocr.frame_quality
                )
                if ocr is not None
                else 0.0
            ),

            ocr_strong=(
                ocr_strong
            ),

            ocr_sequence=(
                ocr_sequence
            ),

            # Final decision
            recognition_source=(
                source
            ),

            combined_score=float(
                np.clip(
                    combined_score,
                    0.0,
                    1.0,
                )
            ),

            acceptance_reason=(
                reason
            ),

            evidence_events=tuple(
                evidence_events
            ),
        )


    # ========================================================
    # PUBLIC LIVE RECOGNITION
    # ========================================================

    def recognize_image(
        self,
        image: (
            str
            | Path
            | np.ndarray
            | Image.Image
        ),
    ) -> MedicineMatch:
        """
        Non-blocking live recognition.

        DINO and OCR work simultaneously.

        This method only:
            harvests
            submits
            combines

        It never waits for either model to finish.
        """

        frame = (
            self._load_bgr_image(
                image
            )
        )

        # ----------------------------------------------------
        # Collect completed jobs.
        # ----------------------------------------------------

        self._harvest()

        # ----------------------------------------------------
        # Start another job for every idle worker.
        # ----------------------------------------------------

        self._submit_workers(
            frame
        )

        # ----------------------------------------------------
        # Harvest again in case something completed quickly.
        # ----------------------------------------------------

        self._harvest()

        return (
            self._combine()
        )


    # ========================================================
    # EMBEDDING-ONLY COMPATIBILITY METHOD
    # ========================================================

    def match_embedding(
        self,
        query_embedding: np.ndarray,
    ) -> MedicineMatch:

        grouped = (
            self._group_references()
        )

        best_id = None

        best_name = (
            "unknown"
        )

        best_dosage = None

        best_score = 0.0

        for samples in grouped.values():

            scores: list[
                float
            ] = []

            for sample in samples:

                try:

                    stored = (
                        self.deserialize_embedding(
                            sample[
                                "embedding"
                            ]
                        )
                    )

                    scores.append(
                        self.compare_embeddings(
                            query_embedding,
                            stored,
                        )
                    )

                except Exception:

                    continue

            if not scores:

                continue

            scores.sort(
                reverse=True
            )

            selected = scores[
                :min(
                    self.recognition_top_k,
                    len(
                        scores
                    ),
                )
            ]

            score = float(
                sum(
                    selected
                )
                / len(
                    selected
                )
            )

            if score > best_score:

                first = (
                    samples[
                        0
                    ]
                )

                best_id = int(
                    first[
                        "medicine_id"
                    ]
                )

                best_name = str(
                    first[
                        "medicine_name"
                    ]
                )

                best_dosage = (
                    str(
                        first[
                            "dosage"
                        ]
                    )
                    if first.get(
                        "dosage"
                    )
                    is not None
                    else None
                )

                best_score = (
                    score
                )

        accepted = bool(
            best_id is not None
            and best_score
            >= self.recognition_threshold
        )

        return MedicineMatch(
            medicine_id=(
                best_id
            ),

            medicine_name=(
                best_name
            ),

            dosage=(
                best_dosage
            ),

            similarity=(
                best_score
            ),

            is_known=(
                accepted
            ),

            threshold=(
                self.recognition_threshold
            ),

            dino_available=(
                best_id is not None
            ),

            dino_score=(
                best_score
            ),

            dino_strong=(
                accepted
            ),

            recognition_source=(
                "visual"
                if accepted
                else "none"
            ),

            combined_score=(
                best_score
            ),

            acceptance_reason=(
                "Embedding-only compatibility result."
            ),
        )


    # ========================================================
    # RESET LIVE STATE
    # ========================================================

    def reset_live_state(
        self,
    ) -> None:
        """
        Clear completed live recognition state.

        Existing worker threads may still finish, but their
        old generation number causes their results to be
        ignored.
        """

        with self._lock:

            self._generation += 1

            self._latest_dino = (
                None
            )

            self._latest_ocr = (
                None
            )

            self._ocr_frames.clear()


    # ========================================================
    # CLOSE
    # ========================================================

    def close(
        self,
    ) -> None:

        self._executor.shutdown(
            wait=False,
            cancel_futures=True,
        )