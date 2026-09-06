from __future__ import annotations

from collections import (
    defaultdict,
    deque,
)

from dataclasses import dataclass

from datetime import datetime

import time

import cv2
import numpy as np

from config import (
    FACE_CONFIRMATION_FRAMES,
    FACE_EVENT_COOLDOWN_SECONDS,
    PEOPLE_EVIDENCE_DIR,
)

from database.person_repository import (
    get_person_event_by_id,
    save_person_event,
)

from database.person_vector_memory import (
    PersonVectorMemory,
)

from face.face_recognition import (
    FaceRecognitionEngine,
)

from utils.logger import (
    get_logger,
)


# ============================================================
# LOGGER
# ============================================================

logger = get_logger(
    "face.person_monitor",
    "vision.log",
)


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class PersonObservation:
    """
    One recognition result from a single frame.
    """

    person_id: int | None

    person_name: str

    display_name: str

    similarity: float

    is_known: bool

    bbox: tuple[
        int,
        int,
        int,
        int,
    ]

    detected_at: datetime


@dataclass
class ConfirmedPersonEncounter:
    """
    A person encounter confirmed across multiple frames.
    """

    person_id: int

    person_name: str

    display_name: str

    similarity: float

    bbox: tuple[
        int,
        int,
        int,
        int,
    ]

    detected_at: datetime

    image_path: str

    event_id: int


# ============================================================
# PERSON MONITOR
# ============================================================

class PersonMonitor:
    """
    Live person-memory monitor.

    Responsibilities:

    - Process camera frames
    - Recognize known people
    - Ignore Unknown for database storage
    - Confirm identities over multiple frames
    - Apply per-person cooldown
    - Save evidence images
    - Save confirmed encounters to MySQL
    - Index confirmed encounters into person ChromaDB
    """

    def __init__(
        self,
        face_engine: (
            FaceRecognitionEngine
            | None
        ) = None,
        confirmation_frames: int = (
            FACE_CONFIRMATION_FRAMES
        ),
        cooldown_seconds: float = (
            FACE_EVENT_COOLDOWN_SECONDS
        ),
    ) -> None:

        # ----------------------------------------------------
        # FACE ENGINE
        # ----------------------------------------------------

        self.face_engine = (
            face_engine
            if face_engine is not None
            else FaceRecognitionEngine()
        )

        # ----------------------------------------------------
        # PERSON VECTOR MEMORY
        # ----------------------------------------------------

        self.person_vector_memory = (
            PersonVectorMemory()
        )

        # ----------------------------------------------------
        # CONFIRMATION SETTINGS
        # ----------------------------------------------------

        self.confirmation_frames = max(
            1,
            int(
                confirmation_frames
            ),
        )

        self.cooldown_seconds = max(
            0.0,
            float(
                cooldown_seconds
            ),
        )

        # ----------------------------------------------------
        # PERSON HISTORY
        # ----------------------------------------------------

        self._history: dict[
            int,
            deque[float],
        ] = defaultdict(
            lambda: deque(
                maxlen=(
                    self.confirmation_frames
                )
            )
        )

        # ----------------------------------------------------
        # LAST SAVED EVENT TIMES
        # ----------------------------------------------------

        self._last_saved_at: dict[
            int,
            float,
        ] = {}

        # ----------------------------------------------------
        # EVIDENCE DIRECTORY
        # ----------------------------------------------------

        PEOPLE_EVIDENCE_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ========================================================
    # RESET
    # ========================================================

    def reset(
        self,
    ) -> None:
        """
        Reset confirmation history and cooldown state.
        """

        self._history.clear()

        self._last_saved_at.clear()

    # ========================================================
    # COOLDOWN
    # ========================================================

    def _is_in_cooldown(
        self,
        person_id: int,
    ) -> bool:

        last_saved = (
            self._last_saved_at.get(
                person_id
            )
        )

        if last_saved is None:

            return False

        elapsed = (
            time.monotonic()
            - last_saved
        )

        return (
            elapsed
            < self.cooldown_seconds
        )

    # ========================================================
    # SAFE FILE NAME
    # ========================================================

    def _safe_name(
        self,
        name: str,
    ) -> str:

        cleaned = (
            name.strip()
            .lower()
        )

        chars: list[str] = []

        previous_separator = False

        for char in cleaned:

            if char.isalnum():

                chars.append(
                    char
                )

                previous_separator = False

            elif char in {
                " ",
                "-",
                "_",
            }:

                if not previous_separator:

                    chars.append(
                        "_"
                    )

                    previous_separator = True

        safe = "".join(
            chars
        ).strip(
            "_"
        )

        return (
            safe
            or "person"
        )

    # ========================================================
    # SAVE EVIDENCE
    # ========================================================

    def _save_evidence_image(
        self,
        frame: np.ndarray,
        display_name: str,
        detected_at: datetime,
    ) -> str:
        """
        Save full-frame encounter evidence.
        """

        safe_name = (
            self._safe_name(
                display_name
            )
        )

        timestamp = (
            detected_at.strftime(
                "%Y%m%d_%H%M%S_%f"
            )
        )

        filename = (
            f"{safe_name}_"
            f"{timestamp}.jpg"
        )

        path = (
            PEOPLE_EVIDENCE_DIR
            / filename
        )

        success = (
            cv2.imwrite(
                str(
                    path
                ),
                frame,
            )
        )

        if not success:

            raise RuntimeError(
                "Failed to save person "
                "evidence image."
            )

        return str(
            path
        )

    # ========================================================
    # DRAW FACE LABEL
    # ========================================================

    def _draw_person_annotation(
        self,
        frame: np.ndarray,
        observation: PersonObservation,
    ) -> np.ndarray:
        """
        Draw one face box and label onto a copy.
        """

        annotated = (
            frame.copy()
        )

        (
            x,
            y,
            w,
            h,
        ) = observation.bbox

        x2 = (
            x + w
        )

        y2 = (
            y + h
        )

        cv2.rectangle(
            annotated,
            (
                x,
                y,
            ),
            (
                x2,
                y2,
            ),
            (
                255,
                255,
                255,
            ),
            2,
        )

        if observation.is_known:

            label = (
                f"{observation.display_name} "
                f"{observation.similarity:.2f}"
            )

        else:

            label = (
                "Unknown"
            )

        cv2.putText(
            annotated,
            label,
            (
                x,
                max(
                    20,
                    y - 8,
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (
                255,
                255,
                255,
            ),
            2,
        )

        return annotated

    # ========================================================
    # RECOGNIZE FRAME
    # ========================================================

    def recognize_frame(
        self,
        frame: np.ndarray,
    ) -> list[
        PersonObservation
    ]:
        """
        Detect and recognize all faces in one frame.

        This method does not save anything to MySQL
        or ChromaDB.
        """

        processed_faces = (
            self.face_engine
            .process_frame(
                frame
            )
        )

        detected_at = (
            datetime.now()
        )

        observations: list[
            PersonObservation
        ] = []

        for face in processed_faces:

            match = (
                self.face_engine
                .recognize_face(
                    face
                )
            )

            observations.append(
                PersonObservation(
                    person_id=(
                        match.person_id
                    ),
                    person_name=(
                        match.person_name
                    ),
                    display_name=(
                        match.display_name
                    ),
                    similarity=float(
                        match.similarity
                    ),
                    is_known=(
                        match.is_known
                    ),
                    bbox=(
                        face.bbox
                    ),
                    detected_at=(
                        detected_at
                    ),
                )
            )

        return observations

    # ========================================================
    # CONFIRM ONE OBSERVATION
    # ========================================================

    def _update_confirmation(
        self,
        observation: PersonObservation,
    ) -> bool:
        """
        Return True once the same known person has been
        recognized enough consecutive times.
        """

        if not observation.is_known:

            return False

        if observation.person_id is None:

            return False

        person_id = (
            observation.person_id
        )

        history = (
            self._history[
                person_id
            ]
        )

        history.append(
            observation.similarity
        )

        return (
            len(
                history
            )
            >= self.confirmation_frames
        )

    # ========================================================
    # CLEAR OTHER IDENTITIES
    # ========================================================

    def _clear_unseen_people(
        self,
        seen_person_ids: set[int],
    ) -> None:
        """
        Clear histories of identities not seen in this frame.

        This makes confirmation effectively consecutive.
        """

        stored_ids = list(
            self._history.keys()
        )

        for person_id in stored_ids:

            if (
                person_id
                not in seen_person_ids
            ):

                self._history.pop(
                    person_id,
                    None,
                )

    # ========================================================
    # INDEX PERSON EVENT
    # ========================================================

    def _index_person_event(
        self,
        event_id: int,
    ) -> bool:
        """
        Load the newly stored MySQL encounter and index the
        exact stored record into the person Chroma collection.

        MySQL is authoritative. We fetch the row back after
        insertion rather than constructing a second copy of
        the event independently.

        Vector indexing is intentionally non-fatal. If
        Chroma/BGE indexing fails, the confirmed MySQL
        encounter remains valid and camera monitoring
        continues.
        """

        try:

            event = (
                get_person_event_by_id(
                    event_id
                )
            )

            if not event:

                logger.warning(
                    "Person event %s was saved "
                    "but could not be loaded "
                    "for vector indexing.",
                    event_id,
                )

                return False

            self.person_vector_memory.add_person_event(
                event
            )

            logger.info(
                "Person event %s indexed "
                "into person vector memory.",
                event_id,
            )

            return True

        except Exception:

            logger.exception(
                "Person event %s was saved "
                "to MySQL but could not be "
                "indexed into person vector "
                "memory.",
                event_id,
            )

            return False

    # ========================================================
    # PROCESS FRAME
    # ========================================================

    def process_frame(
        self,
        frame: np.ndarray,
    ) -> tuple[
        list[
            PersonObservation
        ],
        list[
            ConfirmedPersonEncounter
        ],
    ]:
        """
        Full live-monitoring operation.

        Returns:

        observations
            Every recognized face in the frame.

        confirmed_encounters
            Newly stored encounter events.

        Storage order:

        1. Save evidence image.
        2. Save authoritative encounter to MySQL.
        3. Load the stored MySQL record.
        4. Index that record into person ChromaDB.
        5. Start encounter cooldown.
        """

        observations = (
            self.recognize_frame(
                frame
            )
        )

        seen_known_ids: set[
            int
        ] = set()

        confirmed_encounters: list[
            ConfirmedPersonEncounter
        ] = []

        # ----------------------------------------------------
        # FIND KNOWN IDENTITIES PRESENT IN FRAME
        # ----------------------------------------------------

        for observation in observations:

            if (
                observation.is_known
                and observation.person_id
                is not None
            ):

                seen_known_ids.add(
                    observation.person_id
                )

        # ----------------------------------------------------
        # CLEAR NON-CONSECUTIVE HISTORY
        # ----------------------------------------------------

        self._clear_unseen_people(
            seen_known_ids
        )

        # ----------------------------------------------------
        # PROCESS OBSERVATIONS
        # ----------------------------------------------------

        for observation in observations:

            # ------------------------------------------------
            # UNKNOWN
            # ------------------------------------------------

            if not observation.is_known:

                continue

            if observation.person_id is None:

                continue

            # ------------------------------------------------
            # CONFIRMATION
            # ------------------------------------------------

            confirmed = (
                self._update_confirmation(
                    observation
                )
            )

            if not confirmed:

                continue

            person_id = (
                observation.person_id
            )

            # ------------------------------------------------
            # COOLDOWN
            # ------------------------------------------------

            if self._is_in_cooldown(
                person_id
            ):

                continue

            # ------------------------------------------------
            # AGGREGATED SIMILARITY
            # ------------------------------------------------

            history = (
                self._history[
                    person_id
                ]
            )

            average_similarity = (
                sum(
                    history
                )
                / len(
                    history
                )
            )

            # ------------------------------------------------
            # EVIDENCE
            # ------------------------------------------------

            evidence_frame = (
                self._draw_person_annotation(
                    frame,
                    observation,
                )
            )

            image_path = (
                self._save_evidence_image(
                    evidence_frame,
                    observation.display_name,
                    observation.detected_at,
                )
            )

            # ------------------------------------------------
            # MYSQL DATABASE EVENT
            # ------------------------------------------------

            event_id = (
                save_person_event(
                    person_id=(
                        person_id
                    ),
                    similarity=(
                        average_similarity
                    ),
                    detected_at=(
                        observation.detected_at
                    ),
                    image_path=(
                        image_path
                    ),
                )
            )

            logger.info(
                "Confirmed person encounter "
                "saved to MySQL. "
                "person_id=%s event_id=%s",
                person_id,
                event_id,
            )

            # ------------------------------------------------
            # PERSON VECTOR MEMORY
            # ------------------------------------------------

            self._index_person_event(
                event_id
            )

            # ------------------------------------------------
            # START COOLDOWN
            # ------------------------------------------------

            self._last_saved_at[
                person_id
            ] = time.monotonic()

            # ------------------------------------------------
            # CLEAR HISTORY AFTER SAVE
            # ------------------------------------------------

            self._history.pop(
                person_id,
                None,
            )

            # ------------------------------------------------
            # RETURN CONFIRMED ENCOUNTER
            # ------------------------------------------------

            confirmed_encounters.append(
                ConfirmedPersonEncounter(
                    person_id=(
                        person_id
                    ),
                    person_name=(
                        observation.person_name
                    ),
                    display_name=(
                        observation.display_name
                    ),
                    similarity=float(
                        average_similarity
                    ),
                    bbox=(
                        observation.bbox
                    ),
                    detected_at=(
                        observation.detected_at
                    ),
                    image_path=(
                        image_path
                    ),
                    event_id=(
                        event_id
                    ),
                )
            )

        return (
            observations,
            confirmed_encounters,
        )