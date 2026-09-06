from __future__ import annotations

from dataclasses import dataclass

from datetime import datetime

from enum import Enum

from math import sqrt

import time

import numpy as np

from config import (
    MEDICINE_ACTION_MOUTH_EXIT_RATIO,
    MEDICINE_ACTION_MOUTH_NEAR_RATIO,
    MEDICINE_ACTION_NEAR_FRAMES,
    MEDICINE_ACTION_RETREAT_FRAMES,
    MEDICINE_ACTION_WINDOW_SECONDS,
)

from medicine.hand_landmarker import (
    HandLandmarkDetector,
    HandObservation,
)

from utils.logger import (
    get_logger,
)


# ============================================================
# LOGGER
# ============================================================

logger = get_logger(
    "medicine.action_tracker",
    "vision.log",
)


# ============================================================
# ACTION CONFIGURATION
# ============================================================

# ------------------------------------------------------------
# After the hand-to-mouth stage has already been established,
# MemorAi may accept sustained disappearance of the hand as
# retreat evidence.
#
# Why?
#
# A real user may naturally lower their hand below the webcam
# after bringing medicine toward the mouth.
#
# IMPORTANT:
#
# A SINGLE missing frame is NOT enough.
#
# We require several consecutive missing-hand observations.
# At the current action-processing speed this represents
# roughly less than one second of sustained disappearance.
# ------------------------------------------------------------

MEDICINE_ACTION_MISSING_HAND_RETREAT_FRAMES = 5


# ============================================================
# ACTION STATE
# ============================================================

class MedicationActionState(
    str,
    Enum,
):

    IDLE = (
        "idle"
    )

    WAITING_FOR_APPROACH = (
        "waiting_for_approach"
    )

    HAND_NEAR_MOUTH = (
        "hand_near_mouth"
    )

    WAITING_FOR_RETREAT = (
        "waiting_for_retreat"
    )

    CONFIRMED = (
        "confirmed"
    )

    EXPIRED = (
        "expired"
    )


# ============================================================
# ACTIVE SESSION
# ============================================================

@dataclass
class MedicationActionSession:

    medicine_id: int

    medicine_name: str

    dosage: str | None

    medicine_similarity: float

    person_id: int | None

    person_name: str | None

    person_similarity: float | None

    armed_at: datetime

    armed_monotonic: float

    state: MedicationActionState

    # --------------------------------------------------------
    # APPROACH EVIDENCE
    # --------------------------------------------------------

    near_frames: int = 0

    # --------------------------------------------------------
    # VISIBLE RETREAT EVIDENCE
    # --------------------------------------------------------

    retreat_frames: int = 0

    # --------------------------------------------------------
    # SUSTAINED MISSING-HAND RETREAT EVIDENCE
    # --------------------------------------------------------

    missing_hand_frames: int = 0

    # --------------------------------------------------------
    # CLOSEST HAND-TO-MOUTH DISTANCE
    # --------------------------------------------------------

    closest_normalized_distance: float = float(
        "inf"
    )

    # --------------------------------------------------------
    # IMPORTANT SAFETY STATE
    #
    # Missing-hand evidence is allowed ONLY after the system
    # has already positively established a genuine near-mouth
    # stage.
    # --------------------------------------------------------

    near_mouth_established: bool = False


# ============================================================
# ACTION OBSERVATION
# ============================================================

@dataclass
class MedicationActionObservation:

    state: MedicationActionState

    hand_detected: bool

    normalized_hand_mouth_distance: (
        float
        | None
    )

    near_mouth: bool

    retreat_detected: bool

    observed_at: datetime


# ============================================================
# CONFIRMED ACTION
# ============================================================

@dataclass
class ConfirmedMedicationAction:
    """
    A visual gesture consistent with a medicine-taking action.

    This is NOT proof that medicine was swallowed.

    MemorAi records only that the observed visual action was
    consistent with a medication-taking gesture.
    """

    medicine_id: int

    medicine_name: str

    dosage: str | None

    medicine_similarity: float

    person_id: int | None

    person_name: str | None

    person_similarity: float | None

    action_confidence: float

    closest_normalized_distance: float

    confirmed_at: datetime


# ============================================================
# TRACKER
# ============================================================

class MedicationActionTracker:
    """
    Temporal medication-action tracker.

    The medicine package has ALREADY been recognized before
    this tracker starts.

    Normal sequence:

        medicine confirmed
            ↓
        tracker armed
            ↓
        hand approaches mouth
            ↓
        hand remains near mouth for N frames
            ↓
        near-mouth stage established
            ↓
        wait for retreat
            ↓
        ┌──────────────────────────────────────┐
        │                                      │
        ▼                                      ▼

    PATH A                              PATH B

    Hand remains visible               Hand disappears
    and moves far from mouth           after near-mouth stage
        ↓                                      ↓
    visible retreat frames             missing for 5 frames
        ↓                                      ↓
        └──────────────────┬───────────────────┘
                           ↓
                     ACTION CONFIRMED


    IMPORTANT:

    Missing-hand retreat is NEVER accepted before the
    near-mouth stage has been established.

    Therefore:

        hand not visible from beginning
                ≠
        medication action

    and:

        one accidental MediaPipe miss
                ≠
        medication action
    """

    def __init__(
        self,
        hand_detector: (
            HandLandmarkDetector
            | None
        ) = None,
        action_window_seconds: float = (
            MEDICINE_ACTION_WINDOW_SECONDS
        ),
        mouth_near_ratio: float = (
            MEDICINE_ACTION_MOUTH_NEAR_RATIO
        ),
        mouth_exit_ratio: float = (
            MEDICINE_ACTION_MOUTH_EXIT_RATIO
        ),
        required_near_frames: int = (
            MEDICINE_ACTION_NEAR_FRAMES
        ),
        required_retreat_frames: int = (
            MEDICINE_ACTION_RETREAT_FRAMES
        ),
        required_missing_hand_frames: int = (
            MEDICINE_ACTION_MISSING_HAND_RETREAT_FRAMES
        ),
    ) -> None:

        # ====================================================
        # HAND DETECTOR
        # ====================================================

        self.hand_detector = (
            hand_detector
            if hand_detector is not None
            else HandLandmarkDetector()
        )

        # ====================================================
        # ACTION WINDOW
        # ====================================================

        self.action_window_seconds = max(
            1.0,
            float(
                action_window_seconds
            ),
        )

        # ====================================================
        # MOUTH DISTANCE THRESHOLDS
        # ====================================================

        self.mouth_near_ratio = float(
            mouth_near_ratio
        )

        self.mouth_exit_ratio = float(
            mouth_exit_ratio
        )

        if (
            self.mouth_near_ratio
            <= 0.0
        ):

            raise ValueError(
                "mouth_near_ratio must be "
                "greater than zero."
            )

        if (
            self.mouth_exit_ratio
            <= self.mouth_near_ratio
        ):

            raise ValueError(
                "mouth_exit_ratio must be "
                "greater than mouth_near_ratio."
            )

        # ====================================================
        # REQUIRED TEMPORAL EVIDENCE
        # ====================================================

        self.required_near_frames = max(
            1,
            int(
                required_near_frames
            ),
        )

        self.required_retreat_frames = max(
            1,
            int(
                required_retreat_frames
            ),
        )

        self.required_missing_hand_frames = max(
            2,
            int(
                required_missing_hand_frames
            ),
        )

        # ====================================================
        # SESSION
        # ====================================================

        self._session: (
            MedicationActionSession
            | None
        ) = None


    # ========================================================
    # STATE
    # ========================================================

    @property
    def state(
        self,
    ) -> MedicationActionState:

        if self._session is None:

            return (
                MedicationActionState.IDLE
            )

        return (
            self._session.state
        )


    # ========================================================
    # ACTIVE
    # ========================================================

    @property
    def is_active(
        self,
    ) -> bool:

        return (
            self._session
            is not None
            and self._session.state
            not in {
                MedicationActionState.CONFIRMED,
                MedicationActionState.EXPIRED,
            }
        )


    # ========================================================
    # ARM
    # ========================================================

    def arm(
        self,
        medicine_id: int,
        medicine_name: str,
        medicine_similarity: float,
        dosage: str | None = None,
        person_id: int | None = None,
        person_name: str | None = None,
        person_similarity: float | None = None,
        armed_at: datetime | None = None,
    ) -> None:
        """
        Begin observing the medication-taking gesture.

        This method must only be called AFTER the medicine
        package itself has already been confirmed.
        """

        # ====================================================
        # MEDICINE ID
        # ====================================================

        if medicine_id <= 0:

            raise ValueError(
                "medicine_id must be "
                "greater than zero."
            )

        # ====================================================
        # MEDICINE NAME
        # ====================================================

        cleaned_name = " ".join(
            medicine_name
            .strip()
            .split()
        )

        if not cleaned_name:

            raise ValueError(
                "medicine_name cannot be empty."
            )

        # ====================================================
        # MEDICINE SIMILARITY
        # ====================================================

        medicine_similarity = (
            self._validate_probability(
                medicine_similarity,
                "medicine_similarity",
            )
        )

        # ====================================================
        # PERSON SIMILARITY
        # ====================================================

        if person_similarity is not None:

            person_similarity = (
                self._validate_probability(
                    person_similarity,
                    "person_similarity",
                )
            )

        # ====================================================
        # ARM TIME
        # ====================================================

        if armed_at is None:

            armed_at = (
                datetime.now()
            )

        # ====================================================
        # CREATE SESSION
        # ====================================================

        self._session = (
            MedicationActionSession(
                medicine_id=(
                    int(
                        medicine_id
                    )
                ),

                medicine_name=(
                    cleaned_name
                ),

                dosage=(
                    dosage
                ),

                medicine_similarity=(
                    medicine_similarity
                ),

                person_id=(
                    person_id
                ),

                person_name=(
                    person_name
                ),

                person_similarity=(
                    person_similarity
                ),

                armed_at=(
                    armed_at
                ),

                armed_monotonic=(
                    time.monotonic()
                ),

                state=(
                    MedicationActionState
                    .WAITING_FOR_APPROACH
                ),

                near_frames=0,

                retreat_frames=0,

                missing_hand_frames=0,

                closest_normalized_distance=(
                    float(
                        "inf"
                    )
                ),

                near_mouth_established=False,
            )
        )

        logger.info(
            (
                "Medication action tracker armed. "
                "medicine_id=%s medicine=%s "
                "person_id=%s"
            ),
            medicine_id,
            cleaned_name,
            person_id,
        )


    # ========================================================
    # CANCEL / RESET
    # ========================================================

    def reset(
        self,
    ) -> None:

        self._session = (
            None
        )


    # ========================================================
    # PROBABILITY
    # ========================================================

    @staticmethod
    def _validate_probability(
        value: float,
        field_name: str,
    ) -> float:

        number = float(
            value
        )

        if not (
            0.0
            <= number
            <= 1.0
        ):

            raise ValueError(
                f"{field_name} must be "
                "between 0.0 and 1.0."
            )

        return (
            number
        )


    # ========================================================
    # EXPIRED
    # ========================================================

    def _is_expired(
        self,
    ) -> bool:

        if self._session is None:

            return (
                False
            )

        elapsed = (
            time.monotonic()
            - self._session.armed_monotonic
        )

        return (
            elapsed
            > self.action_window_seconds
        )


    # ========================================================
    # MOUTH CENTER
    # ========================================================

    @staticmethod
    def _mouth_center_from_face_bbox(
        face_bbox: tuple[
            int,
            int,
            int,
            int,
        ],
        frame_width: int,
        frame_height: int,
    ) -> tuple[
        float,
        float,
        float,
    ]:
        """
        Approximate the mouth region using the already
        available face bounding box.

        face_bbox:

            x, y, width, height

        Returns normalized:

            mouth_x
            mouth_y
            face_scale
        """

        (
            x,
            y,
            width,
            height,
        ) = (
            face_bbox
        )

        if (
            width <= 0
            or height <= 0
        ):

            raise ValueError(
                "Face bounding box has "
                "invalid dimensions."
            )

        if (
            frame_width <= 0
            or frame_height <= 0
        ):

            raise ValueError(
                "Frame dimensions must "
                "be greater than zero."
            )

        # ----------------------------------------------------
        # APPROXIMATE MOUTH LOCATION
        # ----------------------------------------------------

        mouth_pixel_x = (
            float(
                x
            )
            +
            (
                float(
                    width
                )
                * 0.50
            )
        )

        mouth_pixel_y = (
            float(
                y
            )
            +
            (
                float(
                    height
                )
                * 0.72
            )
        )

        # ----------------------------------------------------
        # NORMALIZE
        # ----------------------------------------------------

        mouth_x = (
            mouth_pixel_x
            / float(
                frame_width
            )
        )

        mouth_y = (
            mouth_pixel_y
            / float(
                frame_height
            )
        )

        normalized_face_width = (
            float(
                width
            )
            / float(
                frame_width
            )
        )

        normalized_face_height = (
            float(
                height
            )
            / float(
                frame_height
            )
        )

        face_scale = max(
            normalized_face_width,
            normalized_face_height,
        )

        return (
            mouth_x,
            mouth_y,
            face_scale,
        )


    # ========================================================
    # DISTANCE
    # ========================================================

    @staticmethod
    def _distance(
        x1: float,
        y1: float,
        x2: float,
        y2: float,
    ) -> float:

        return sqrt(
            (
                x1
                - x2
            )
            ** 2
            +
            (
                y1
                - y2
            )
            ** 2
        )


    # ========================================================
    # NEAREST HAND
    # ========================================================

    def _nearest_hand_distance(
        self,
        hands: list[
            HandObservation
        ],
        mouth_x: float,
        mouth_y: float,
        face_scale: float,
    ) -> float | None:

        if not hands:

            return (
                None
            )

        if face_scale <= 0.0:

            return (
                None
            )

        distances: list[
            float
        ] = []

        for hand in hands:

            raw_distance = (
                self._distance(
                    hand.focus_x,
                    hand.focus_y,
                    mouth_x,
                    mouth_y,
                )
            )

            normalized_distance = (
                raw_distance
                / face_scale
            )

            distances.append(
                normalized_distance
            )

        if not distances:

            return (
                None
            )

        return float(
            min(
                distances
            )
        )


    # ========================================================
    # ACTION CONFIDENCE
    # ========================================================

    def _calculate_action_confidence(
        self,
        closest_distance: float,
    ) -> float:
        """
        Confidence is based on the strongest observed
        hand-to-mouth proximity plus completion of the
        temporal gesture.

        Medicine identity and person identity remain separate
        confidence signals and are combined later by the
        MedicineService.
        """

        if (
            closest_distance
            == float(
                "inf"
            )
        ):

            return (
                0.0
            )

        proximity_score = (
            1.0
            -
            (
                closest_distance
                / self.mouth_near_ratio
            )
        )

        proximity_score = float(
            np.clip(
                proximity_score,
                0.0,
                1.0,
            )
        )

        # ----------------------------------------------------
        # APPROACH + NEAR-MOUTH + RETREAT/DISAPPEARANCE
        # completed.
        # ----------------------------------------------------

        sequence_score = (
            1.0
        )

        confidence = (
            (
                0.60
                * proximity_score
            )
            +
            (
                0.40
                * sequence_score
            )
        )

        return float(
            np.clip(
                confidence,
                0.0,
                1.0,
            )
        )


    # ========================================================
    # CREATE CONFIRMED ACTION
    # ========================================================

    def _create_confirmed_action(
        self,
        observed_at: datetime,
    ) -> ConfirmedMedicationAction:

        if self._session is None:

            raise RuntimeError(
                "Medication action tracker "
                "is not armed."
            )

        action_confidence = (
            self._calculate_action_confidence(
                self._session
                .closest_normalized_distance
            )
        )

        return (
            ConfirmedMedicationAction(
                medicine_id=(
                    self._session
                    .medicine_id
                ),

                medicine_name=(
                    self._session
                    .medicine_name
                ),

                dosage=(
                    self._session
                    .dosage
                ),

                medicine_similarity=(
                    self._session
                    .medicine_similarity
                ),

                person_id=(
                    self._session
                    .person_id
                ),

                person_name=(
                    self._session
                    .person_name
                ),

                person_similarity=(
                    self._session
                    .person_similarity
                ),

                action_confidence=(
                    action_confidence
                ),

                closest_normalized_distance=(
                    self._session
                    .closest_normalized_distance
                ),

                confirmed_at=(
                    observed_at
                ),
            )
        )


    # ========================================================
    # RETURN CONFIRMED
    # ========================================================

    def _confirm_action(
        self,
        observed_at: datetime,
        hand_detected: bool,
        normalized_distance: float | None,
        near_mouth: bool,
        retreat_detected: bool,
        confirmation_reason: str,
    ) -> tuple[
        MedicationActionObservation,
        ConfirmedMedicationAction,
    ]:

        if self._session is None:

            raise RuntimeError(
                "Medication action tracker "
                "is not armed."
            )

        self._session.state = (
            MedicationActionState.CONFIRMED
        )

        confirmed = (
            self._create_confirmed_action(
                observed_at
            )
        )

        logger.info(
            (
                "Medication action candidate confirmed. "
                "medicine=%s "
                "medicine_id=%s "
                "person_id=%s "
                "action_confidence=%.4f "
                "closest_distance=%.4f "
                "reason=%s"
            ),
            confirmed.medicine_name,
            confirmed.medicine_id,
            confirmed.person_id,
            confirmed.action_confidence,
            confirmed.closest_normalized_distance,
            confirmation_reason,
        )

        observation = (
            MedicationActionObservation(
                state=(
                    MedicationActionState
                    .CONFIRMED
                ),

                hand_detected=(
                    hand_detected
                ),

                normalized_hand_mouth_distance=(
                    normalized_distance
                ),

                near_mouth=(
                    near_mouth
                ),

                retreat_detected=(
                    retreat_detected
                ),

                observed_at=(
                    observed_at
                ),
            )
        )

        return (
            observation,
            confirmed,
        )


    # ========================================================
    # PROCESS FRAME
    # ========================================================

    def process_frame(
        self,
        frame: np.ndarray,
        face_bbox: tuple[
            int,
            int,
            int,
            int,
        ],
        observed_at: datetime | None = None,
    ) -> tuple[
        MedicationActionObservation,
        ConfirmedMedicationAction | None,
    ]:

        # ====================================================
        # SESSION
        # ====================================================

        if self._session is None:

            raise RuntimeError(
                "Medication action tracker "
                "is not armed."
            )

        # ====================================================
        # FRAME VALIDATION
        # ====================================================

        if frame is None:

            raise ValueError(
                "Action frame cannot be None."
            )

        if not isinstance(
            frame,
            np.ndarray,
        ):

            raise TypeError(
                "Action frame must be "
                "a NumPy array."
            )

        if frame.size == 0:

            raise ValueError(
                "Action frame cannot be empty."
            )

        # ====================================================
        # TIME
        # ====================================================

        if observed_at is None:

            observed_at = (
                datetime.now()
            )

        # ====================================================
        # TIMEOUT
        # ====================================================

        if self._is_expired():

            self._session.state = (
                MedicationActionState.EXPIRED
            )

            logger.info(
                (
                    "Medication action tracker expired. "
                    "medicine=%s person=%s"
                ),
                self._session.medicine_name,
                self._session.person_name,
            )

            observation = (
                MedicationActionObservation(
                    state=(
                        MedicationActionState
                        .EXPIRED
                    ),

                    hand_detected=False,

                    normalized_hand_mouth_distance=None,

                    near_mouth=False,

                    retreat_detected=False,

                    observed_at=(
                        observed_at
                    ),
                )
            )

            return (
                observation,
                None,
            )

        # ====================================================
        # FRAME SIZE
        # ====================================================

        frame_height, frame_width = (
            frame.shape[
                :2
            ]
        )

        # ====================================================
        # MOUTH REGION
        # ====================================================

        (
            mouth_x,
            mouth_y,
            face_scale,
        ) = (
            self._mouth_center_from_face_bbox(
                face_bbox=(
                    face_bbox
                ),

                frame_width=(
                    frame_width
                ),

                frame_height=(
                    frame_height
                ),
            )
        )

        # ====================================================
        # HAND LANDMARKS
        # ====================================================

        hands = (
            self.hand_detector
            .detect(
                frame
            )
        )

        normalized_distance = (
            self._nearest_hand_distance(
                hands=(
                    hands
                ),

                mouth_x=(
                    mouth_x
                ),

                mouth_y=(
                    mouth_y
                ),

                face_scale=(
                    face_scale
                ),
            )
        )

        hand_detected = (
            normalized_distance
            is not None
        )

        near_mouth = (
            False
        )

        retreat_detected = (
            False
        )

        # ====================================================
        # DISTANCE CLASSIFICATION
        # ====================================================

        if normalized_distance is not None:

            near_mouth = (
                normalized_distance
                <= self.mouth_near_ratio
            )

            retreat_detected = (
                normalized_distance
                >= self.mouth_exit_ratio
            )

            self._session.closest_normalized_distance = min(
                self._session
                .closest_normalized_distance,
                normalized_distance,
            )

        # ====================================================
        # WAITING FOR APPROACH
        # ====================================================

        if (
            self._session.state
            == MedicationActionState
            .WAITING_FOR_APPROACH
        ):

            # ------------------------------------------------
            # Missing-hand evidence has no meaning yet.
            # ------------------------------------------------

            self._session.missing_hand_frames = (
                0
            )

            self._session.retreat_frames = (
                0
            )

            if near_mouth:

                self._session.near_frames += 1

                logger.debug(
                    (
                        "Medication near-mouth evidence "
                        "%s/%s distance=%.4f"
                    ),
                    self._session.near_frames,
                    self.required_near_frames,
                    normalized_distance,
                )

            else:

                self._session.near_frames = (
                    0
                )

            # ------------------------------------------------
            # APPROACH CONFIRMED
            # ------------------------------------------------

            if (
                self._session.near_frames
                >= self.required_near_frames
            ):

                self._session.near_mouth_established = (
                    True
                )

                self._session.state = (
                    MedicationActionState
                    .HAND_NEAR_MOUTH
                )

                self._session.retreat_frames = (
                    0
                )

                self._session.missing_hand_frames = (
                    0
                )

                logger.info(
                    (
                        "Medication action near-mouth "
                        "stage established. "
                        "medicine=%s person=%s "
                        "near_frames=%s"
                    ),
                    self._session.medicine_name,
                    self._session.person_name,
                    self._session.near_frames,
                )

        # ====================================================
        # HAND NEAR MOUTH
        # ====================================================

        if (
            self._session.state
            == MedicationActionState
            .HAND_NEAR_MOUTH
        ):

            self._session.state = (
                MedicationActionState
                .WAITING_FOR_RETREAT
            )

            logger.info(
                (
                    "Medication action tracker now "
                    "waiting for retreat. "
                    "medicine=%s person=%s"
                ),
                self._session.medicine_name,
                self._session.person_name,
            )

        # ====================================================
        # WAITING FOR RETREAT
        # ====================================================

        if (
            self._session.state
            == MedicationActionState
            .WAITING_FOR_RETREAT
        ):

            # =================================================
            # PATH A:
            # VISIBLE RETREAT
            # =================================================

            if retreat_detected:

                self._session.retreat_frames += 1

                # Hand is visible again, therefore this is not
                # sustained disappearance.
                self._session.missing_hand_frames = (
                    0
                )

                logger.info(
                    (
                        "Medication visible retreat "
                        "evidence: %s/%s "
                        "distance=%.4f"
                    ),
                    self._session.retreat_frames,
                    self.required_retreat_frames,
                    normalized_distance,
                )

            # =================================================
            # HAND VISIBLE, BUT NOT RETREATED
            # =================================================

            elif hand_detected:

                # A visible hand means the missing-hand
                # sequence has been broken.
                self._session.missing_hand_frames = (
                    0
                )

                # ------------------------------------------------
                # If the hand is still near the mouth, keep
                # waiting.
                #
                # If it is between near and exit thresholds,
                # also keep waiting.
                # ------------------------------------------------

                self._session.retreat_frames = (
                    0
                )

            # =================================================
            # PATH B:
            # SUSTAINED HAND DISAPPEARANCE
            # =================================================

            else:

                # ------------------------------------------------
                # This path is permitted ONLY because genuine
                # near-mouth evidence has already occurred.
                # ------------------------------------------------

                if (
                    self._session
                    .near_mouth_established
                ):

                    self._session.missing_hand_frames += 1

                    logger.info(
                        (
                            "Medication hand missing after "
                            "near-mouth stage: %s/%s"
                        ),
                        self._session.missing_hand_frames,
                        self.required_missing_hand_frames,
                    )

                else:

                    self._session.missing_hand_frames = (
                        0
                    )

            # =================================================
            # CONFIRM PATH A
            # =================================================

            if (
                self._session.retreat_frames
                >= self.required_retreat_frames
            ):

                return (
                    self._confirm_action(
                        observed_at=(
                            observed_at
                        ),

                        hand_detected=(
                            hand_detected
                        ),

                        normalized_distance=(
                            normalized_distance
                        ),

                        near_mouth=(
                            near_mouth
                        ),

                        retreat_detected=True,

                        confirmation_reason=(
                            "visible_retreat"
                        ),
                    )
                )

            # =================================================
            # CONFIRM PATH B
            # =================================================

            if (
                self._session
                .near_mouth_established

                and

                self._session
                .missing_hand_frames
                >= self.required_missing_hand_frames
            ):

                logger.info(
                    (
                        "Sustained hand disappearance "
                        "accepted as retreat evidence. "
                        "missing_frames=%s"
                    ),
                    self._session.missing_hand_frames,
                )

                return (
                    self._confirm_action(
                        observed_at=(
                            observed_at
                        ),

                        hand_detected=False,

                        normalized_distance=None,

                        near_mouth=False,

                        # Semantically the tracker has now
                        # concluded retreat occurred.
                        retreat_detected=True,

                        confirmation_reason=(
                            "sustained_hand_disappearance"
                        ),
                    )
                )

        # ====================================================
        # NORMAL OBSERVATION
        # ====================================================

        observation = (
            MedicationActionObservation(
                state=(
                    self._session.state
                ),

                hand_detected=(
                    hand_detected
                ),

                normalized_hand_mouth_distance=(
                    normalized_distance
                ),

                near_mouth=(
                    near_mouth
                ),

                retreat_detected=(
                    retreat_detected
                ),

                observed_at=(
                    observed_at
                ),
            )
        )

        return (
            observation,
            None,
        )