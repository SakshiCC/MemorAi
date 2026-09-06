from __future__ import annotations

from collections import deque

from dataclasses import dataclass

from datetime import datetime

import time

import numpy as np

from config import (
    MEDICINE_CONFIRMATION_FRAMES,
    MEDICINE_RECOGNITION_COOLDOWN_SECONDS,
    MEDICINE_RECOGNITION_THRESHOLD,
)

from medicine.medicine_recognizer import (
    MedicineMatch,
    MedicineRecognitionEngine,
    RecognitionEvidenceEvent,
)

from utils.logger import (
    get_logger,
)


# ============================================================
# LOGGER
# ============================================================

logger = get_logger(
    "medicine.monitor",
    "vision.log",
)


# ============================================================
# TEMPORAL EVIDENCE
# ============================================================

DEFAULT_MEDICINE_EVIDENCE_WINDOW_SIZE = 5

DEFAULT_MEDICINE_EVIDENCE_TIMEOUT_SECONDS = 6.0


# ============================================================
# ONE TEMPORAL EVIDENCE ITEM
# ============================================================

@dataclass
class MedicineEvidenceItem:

    event_key: str

    accepted: bool

    similarity: float

    source: str

    observed_monotonic: float


# ============================================================
# ONE MEDICINE OBSERVATION
# ============================================================

@dataclass
class MedicineObservation:

    medicine_id: int | None

    medicine_name: str

    dosage: str | None

    similarity: float

    is_known: bool

    observed_at: datetime

    threshold: float = (
        MEDICINE_RECOGNITION_THRESHOLD
    )

    confirmation_count: int = 0

    confirmation_required: int = (
        MEDICINE_CONFIRMATION_FRAMES
    )

    # --------------------------------------------------------
    # LEGACY ORB FIELDS
    # --------------------------------------------------------

    feature_verified: bool = False

    feature_score: float = 0.0

    feature_good_matches: int = 0

    feature_inliers: int = 0

    feature_inlier_ratio: float = 0.0

    # --------------------------------------------------------
    # TEMPORAL WINDOW
    # --------------------------------------------------------

    evidence_window_count: int = 0

    evidence_window_size: int = (
        DEFAULT_MEDICINE_EVIDENCE_WINDOW_SIZE
    )

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
    # FINAL RECOGNIZER DECISION
    # --------------------------------------------------------

    recognition_source: str = (
        "none"
    )

    combined_score: float = 0.0

    acceptance_reason: str = ""

    # --------------------------------------------------------
    # DIAGNOSTICS
    # --------------------------------------------------------

    new_evidence_count: int = 0


# ============================================================
# CONFIRMED PACKAGE
# ============================================================

@dataclass
class ConfirmedMedicineRecognition:
    """
    Medicine PACKAGE identity confirmation.

    This does not prove medicine swallowing.
    """

    medicine_id: int

    medicine_name: str

    dosage: str | None

    similarity: float

    observed_at: datetime

    confirmation_frames: int


# ============================================================
# MEDICINE MONITOR
# ============================================================

class MedicineMonitor:
    """
    Temporal confirmation over genuinely NEW DINO/OCR
    recognition events.

    The camera service may poll the recognizer several times
    while a completed DINO result is cached.

    OLD BUG:

        DINO result #12
        poll → PASS
        poll → PASS
        poll → PASS

        incorrectly became 3/3.

    NEW:

        DINO #12
            → counted once

        OCR #5
            → counted once

        DINO #13
            → counted once

        now 3/3.

    Strong DINO and OCR still work independently.

    Moderate DINO+OCR agreement is emitted by the recognizer
    as a single fused evidence event.
    """

    def __init__(
        self,
        recognizer: (
            MedicineRecognitionEngine
            | None
        ) = None,
        confirmation_frames: int | None = None,
        cooldown_seconds: float | None = None,
        evidence_window_size: int = (
            DEFAULT_MEDICINE_EVIDENCE_WINDOW_SIZE
        ),
        evidence_timeout_seconds: float = (
            DEFAULT_MEDICINE_EVIDENCE_TIMEOUT_SECONDS
        ),
    ) -> None:

        # ====================================================
        # RECOGNIZER
        # ====================================================

        self.recognizer = (
            recognizer
            if recognizer is not None
            else MedicineRecognitionEngine()
        )

        # ====================================================
        # REQUIRED POSITIVE EVENTS
        # ====================================================

        if confirmation_frames is None:

            confirmation_frames = (
                MEDICINE_CONFIRMATION_FRAMES
            )

        self.confirmation_frames = max(
            1,
            int(
                confirmation_frames
            ),
        )

        # ====================================================
        # WINDOW SIZE
        # ====================================================

        self.evidence_window_size = max(
            self.confirmation_frames,
            int(
                evidence_window_size
            ),
        )

        # ====================================================
        # TIMEOUT
        # ====================================================

        self.evidence_timeout_seconds = max(
            0.1,
            float(
                evidence_timeout_seconds
            ),
        )

        # ====================================================
        # COOLDOWN
        # ====================================================

        if cooldown_seconds is None:

            cooldown_seconds = (
                MEDICINE_RECOGNITION_COOLDOWN_SECONDS
            )

        self.cooldown_seconds = max(
            0.0,
            float(
                cooldown_seconds
            ),
        )

        # ====================================================
        # TEMPORAL WINDOW
        # ====================================================

        self._evidence_window: deque[
            MedicineEvidenceItem
        ] = deque(
            maxlen=(
                self.evidence_window_size
            )
        )

        # ====================================================
        # CONSUMED UNIQUE WORKER RESULTS
        # ====================================================

        self._consumed_event_keys: set[
            str
        ] = set()

        # ====================================================
        # ACTIVE MEDICINE
        # ====================================================

        self._active_medicine_id: (
            int
            | None
        ) = None

        self._last_candidate_seen_at: (
            float
            | None
        ) = None

        # ====================================================
        # CONFIRMATION COOLDOWN
        # ====================================================

        self._last_confirmed_at: dict[
            int,
            float,
        ] = {}


    # ========================================================
    # CLEAR TEMPORAL EVIDENCE
    # ========================================================

    def _clear_evidence(
        self,
    ) -> None:

        self._evidence_window.clear()

        self._consumed_event_keys.clear()

        self._active_medicine_id = (
            None
        )

        self._last_candidate_seen_at = (
            None
        )


    # ========================================================
    # FULL RESET
    # ========================================================

    def reset(
        self,
    ) -> None:

        self._clear_evidence()

        self._last_confirmed_at.clear()

        try:

            self.recognizer.reset_live_state()

        except Exception:

            logger.exception(
                "Could not reset live medicine "
                "recognizer state."
            )


    # ========================================================
    # COOLDOWN
    # ========================================================

    def _is_in_cooldown(
        self,
        medicine_id: int,
    ) -> bool:

        last = (
            self._last_confirmed_at.get(
                medicine_id
            )
        )

        if last is None:

            return False

        return (
            (
                time.monotonic()
                - last
            )
            < self.cooldown_seconds
        )


    # ========================================================
    # RECOGNIZE IMAGE
    # ========================================================

    def recognize_image(
        self,
        image: np.ndarray,
    ) -> MedicineObservation:

        if image is None:

            raise ValueError(
                "Medicine image cannot be None."
            )

        if not isinstance(
            image,
            np.ndarray,
        ):

            raise TypeError(
                "Medicine image must be "
                "a NumPy array."
            )

        if image.size == 0:

            raise ValueError(
                "Medicine image cannot be empty."
            )

        match = (
            self.recognizer
            .recognize_image(
                image
            )
        )

        return (
            self._match_to_observation(
                match
            )
        )


    # ========================================================
    # MATCH -> OBSERVATION
    # ========================================================

    def _match_to_observation(
        self,
        match: MedicineMatch,
    ) -> MedicineObservation:

        observation = (
            MedicineObservation(
                medicine_id=(
                    match.medicine_id
                ),

                medicine_name=(
                    match.medicine_name
                ),

                dosage=(
                    match.dosage
                ),

                similarity=float(
                    match.similarity
                ),

                is_known=bool(
                    match.is_known
                ),

                observed_at=(
                    datetime.now()
                ),

                threshold=float(
                    match.threshold
                ),

                confirmation_count=0,

                confirmation_required=(
                    self.confirmation_frames
                ),

                feature_verified=bool(
                    match.feature_verified
                ),

                feature_score=float(
                    match.feature_score
                ),

                feature_good_matches=int(
                    match.feature_good_matches
                ),

                feature_inliers=int(
                    match.feature_inliers
                ),

                feature_inlier_ratio=float(
                    match.feature_inlier_ratio
                ),

                evidence_window_count=0,

                evidence_window_size=(
                    self.evidence_window_size
                ),

                dino_available=bool(
                    match.dino_available
                ),

                dino_score=float(
                    match.dino_score
                ),

                dino_strong=bool(
                    match.dino_strong
                ),

                dino_sequence=int(
                    match.dino_sequence
                ),

                ocr_available=bool(
                    match.ocr_available
                ),

                ocr_score=float(
                    match.ocr_score
                ),

                ocr_text_score=float(
                    match.ocr_text_score
                ),

                ocr_dosage_score=float(
                    match.ocr_dosage_score
                ),

                ocr_text=str(
                    match.ocr_text
                ),

                ocr_keyword=(
                    match.ocr_keyword
                ),

                ocr_seconds=float(
                    match.ocr_seconds
                ),

                ocr_frame_quality=float(
                    match.ocr_frame_quality
                ),

                ocr_strong=bool(
                    match.ocr_strong
                ),

                ocr_sequence=int(
                    match.ocr_sequence
                ),

                recognition_source=str(
                    match.recognition_source
                ),

                combined_score=float(
                    match.combined_score
                ),

                acceptance_reason=str(
                    match.acceptance_reason
                ),

                new_evidence_count=0,
            )
        )

        # Store the recognizer's unique evidence events only
        # for the current processing call.
        observation._recognition_events = (  # type: ignore[attr-defined]
            match.evidence_events
        )

        return (
            observation
        )


    # ========================================================
    # PRUNE OLD TEMPORAL EVIDENCE
    # ========================================================

    def _prune_expired_evidence(
        self,
        now: float,
    ) -> None:

        while self._evidence_window:

            oldest = (
                self._evidence_window[
                    0
                ]
            )

            age = (
                now
                - oldest.observed_monotonic
            )

            if (
                age
                <= self.evidence_timeout_seconds
            ):

                break

            self._evidence_window.popleft()

        if (
            not self._evidence_window
            and self._last_candidate_seen_at
            is not None
        ):

            age = (
                now
                - self._last_candidate_seen_at
            )

            if (
                age
                > self.evidence_timeout_seconds
            ):

                self._active_medicine_id = (
                    None
                )

                self._last_candidate_seen_at = (
                    None
                )


    # ========================================================
    # POSITIVE COUNT
    # ========================================================

    def _verified_count(
        self,
    ) -> int:

        return sum(
            1

            for item
            in self._evidence_window

            if item.accepted
        )


    # ========================================================
    # ACCEPTED SCORES
    # ========================================================

    def _accepted_similarities(
        self,
    ) -> list[
        float
    ]:

        return [
            float(
                item.similarity
            )

            for item
            in self._evidence_window

            if item.accepted
        ]


    # ========================================================
    # ADD ONE UNIQUE EVIDENCE EVENT
    # ========================================================

    def _consume_event(
        self,
        event: RecognitionEvidenceEvent,
        medicine_id: int,
        now: float,
    ) -> bool:
        """
        Returns True only when this event was genuinely new.
        """

        if (
            not event.event_key
            or event.event_key
            in self._consumed_event_keys
        ):

            return False

        # Mark consumed even if it belongs to another
        # medicine, so the same cached event cannot appear
        # repeatedly later.
        self._consumed_event_keys.add(
            event.event_key
        )

        if (
            event.medicine_id
            is None
            or int(
                event.medicine_id
            )
            != medicine_id
        ):

            return False

        self._evidence_window.append(
            MedicineEvidenceItem(
                event_key=(
                    event.event_key
                ),

                accepted=bool(
                    event.accepted
                ),

                similarity=float(
                    event.score
                ),

                source=str(
                    event.source
                ),

                observed_monotonic=(
                    now
                ),
            )
        )

        return True


    # ========================================================
    # UPDATE CONFIRMATION
    # ========================================================

    def _update_confirmation(
        self,
        observation: MedicineObservation,
    ) -> bool:

        now = (
            time.monotonic()
        )

        self._prune_expired_evidence(
            now
        )

        # ====================================================
        # NO CANDIDATE
        # ====================================================

        if (
            observation.medicine_id
            is None
        ):

            observation.confirmation_count = (
                self._verified_count()
            )

            observation.confirmation_required = (
                self.confirmation_frames
            )

            observation.evidence_window_count = (
                len(
                    self._evidence_window
                )
            )

            observation.evidence_window_size = (
                self.evidence_window_size
            )

            return False

        medicine_id = int(
            observation.medicine_id
        )

        # ====================================================
        # CANDIDATE CHANGED
        # ====================================================

        if (
            self._active_medicine_id
            is not None
            and self._active_medicine_id
            != medicine_id
        ):

            logger.info(
                (
                    "Medicine candidate changed. "
                    "old_id=%s new_id=%s"
                ),
                self._active_medicine_id,
                medicine_id,
            )

            self._clear_evidence()

        self._active_medicine_id = (
            medicine_id
        )

        self._last_candidate_seen_at = (
            now
        )

        # ====================================================
        # CONSUME ONLY NEW DINO/OCR RESULTS
        # ====================================================

        events = getattr(
            observation,
            "_recognition_events",
            (),
        )

        new_count = 0

        for event in events:

            if self._consume_event(
                event=(
                    event
                ),
                medicine_id=(
                    medicine_id
                ),
                now=(
                    now
                ),
            ):

                new_count += 1

        observation.new_evidence_count = (
            new_count
        )

        # ====================================================
        # UPDATE UI COUNTS
        # ====================================================

        observation.confirmation_count = (
            self._verified_count()
        )

        observation.confirmation_required = (
            self.confirmation_frames
        )

        observation.evidence_window_count = (
            len(
                self._evidence_window
            )
        )

        observation.evidence_window_size = (
            self.evidence_window_size
        )

        return bool(
            observation.confirmation_count
            >= self.confirmation_frames
        )


    # ========================================================
    # PROCESS IMAGE
    # ========================================================

    def process_image(
        self,
        image: np.ndarray,
    ) -> tuple[
        MedicineObservation,
        ConfirmedMedicineRecognition | None,
    ]:

        observation = (
            self.recognize_image(
                image
            )
        )

        confirmed = (
            self._update_confirmation(
                observation
            )
        )

        # ====================================================
        # DIAGNOSTICS
        # ====================================================

        logger.info(
            (
                "Medicine candidate: "
                "name=%s "
                "source=%s "
                "dino_seq=%s "
                "dino_available=%s "
                "dino=%.4f "
                "dino_strong=%s "
                "ocr_seq=%s "
                "ocr_available=%s "
                "ocr=%.4f "
                "ocr_strong=%s "
                "ocr_text=%r "
                "ocr_keyword=%r "
                "ocr_time=%.3fs "
                "frame_quality=%.3f "
                "combined=%.4f "
                "accepted=%s "
                "new_evidence=%s "
                "reason=%s "
                "evidence=%s/%s "
                "window=%s/%s"
            ),
            observation.medicine_name,
            observation.recognition_source,
            observation.dino_sequence,
            observation.dino_available,
            observation.dino_score,
            observation.dino_strong,
            observation.ocr_sequence,
            observation.ocr_available,
            observation.ocr_score,
            observation.ocr_strong,
            observation.ocr_text,
            observation.ocr_keyword,
            observation.ocr_seconds,
            observation.ocr_frame_quality,
            observation.combined_score,
            observation.is_known,
            observation.new_evidence_count,
            observation.acceptance_reason,
            observation.confirmation_count,
            observation.confirmation_required,
            observation.evidence_window_count,
            observation.evidence_window_size,
        )

        # ====================================================
        # NOT YET CONFIRMED
        # ====================================================

        if (
            not confirmed
            or observation.medicine_id
            is None
        ):

            return (
                observation,
                None,
            )

        medicine_id = int(
            observation.medicine_id
        )

        # ====================================================
        # COOLDOWN
        # ====================================================

        if self._is_in_cooldown(
            medicine_id
        ):

            return (
                observation,
                None,
            )

        # ====================================================
        # POSITIVE SCORES
        # ====================================================

        similarities = (
            self._accepted_similarities()
        )

        if not similarities:

            return (
                observation,
                None,
            )

        selected = similarities[
            -self.confirmation_frames:
        ]

        average_similarity = float(
            np.mean(
                selected
            )
        )

        # ====================================================
        # CONFIRMED PACKAGE
        # ====================================================

        result = (
            ConfirmedMedicineRecognition(
                medicine_id=(
                    medicine_id
                ),

                medicine_name=(
                    observation.medicine_name
                ),

                dosage=(
                    observation.dosage
                ),

                similarity=float(
                    average_similarity
                ),

                observed_at=(
                    observation.observed_at
                ),

                confirmation_frames=(
                    self.confirmation_frames
                ),
            )
        )

        self._last_confirmed_at[
            medicine_id
        ] = (
            time.monotonic()
        )

        logger.info(
            (
                "Medicine package confirmed. "
                "medicine_id=%s "
                "name=%s "
                "similarity=%.4f"
            ),
            result.medicine_id,
            result.medicine_name,
            result.similarity,
        )

        # ----------------------------------------------------
        # Clear consumed result IDs and increment recognizer
        # generation so old DINO/OCR results cannot leak into
        # the next medication interaction.
        # ----------------------------------------------------

        self._clear_evidence()

        try:

            self.recognizer.reset_live_state()

        except Exception:

            logger.exception(
                "Could not reset recognizer after "
                "medicine package confirmation."
            )

        return (
            observation,
            result,
        )