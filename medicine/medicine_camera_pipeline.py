from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import cv2
import numpy as np

from config import (
    MEDICINE_EVIDENCE_DIR,
)

from medicine.medicine_action_tracker import (
    ConfirmedMedicationAction,
    MedicationActionState,
    MedicationActionTracker,
)

from medicine.medicine_monitor import (
    ConfirmedMedicineRecognition,
    MedicineMonitor,
    MedicineObservation,
)

from medicine.medicine_service import (
    MedicineService,
)

from database.medicine_repository import (
    get_medicine_event_by_id,
)

from utils.logger import (
    get_logger,
)


# ============================================================
# LOGGER
# ============================================================

logger = get_logger(
    "medicine.camera_pipeline",
    "vision.log",
)


# ============================================================
# PERSON CONTEXT
# ============================================================

@dataclass
class ActivePersonContext:

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

    observed_at: datetime


# ============================================================
# LIVE STATUS
# ============================================================

@dataclass
class MedicinePipelineStatus:

    phase: str

    medicine_name: str | None

    medicine_similarity: float | None

    medicine_threshold: float | None

    medicine_is_known: bool

    confirmation_count: int

    confirmation_required: int

    person_name: str | None

    person_similarity: float | None

    action_confidence: float | None

    message: str


# ============================================================
# FINAL MEDICINE EVENT
# ============================================================

@dataclass
class ConfirmedMedicineEvent:

    event_id: int

    medicine_id: int

    medicine_name: str

    dosage: str | None

    person_id: int

    person_name: str

    medicine_similarity: float

    person_similarity: float

    action_confidence: float

    overall_confidence: float

    observed_at: datetime

    evidence_image_path: str

    status: str

    memory_text: str | None


# ============================================================
# CAMERA PIPELINE
# ============================================================

class MedicineCameraPipeline:

    def __init__(
        self,
        medicine_monitor: (
            MedicineMonitor
            | None
        ) = None,
        action_tracker: (
            MedicationActionTracker
            | None
        ) = None,
        medicine_service: (
            MedicineService
            | None
        ) = None,
    ) -> None:

        self.medicine_monitor = (
            medicine_monitor
            if medicine_monitor is not None
            else MedicineMonitor()
        )

        self.action_tracker = (
            action_tracker
            if action_tracker is not None
            else MedicationActionTracker()
        )

        self.medicine_service = (
            medicine_service
            if medicine_service is not None
            else MedicineService()
        )

        MEDICINE_EVIDENCE_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._confirmed_medicine: (
            ConfirmedMedicineRecognition
            | None
        ) = None

        self._active_person: (
            ActivePersonContext
            | None
        ) = None

        self._status = (
            self._idle_status()
        )


    # ========================================================
    # IDLE
    # ========================================================

    @staticmethod
    def _idle_status(
    ) -> MedicinePipelineStatus:

        return MedicinePipelineStatus(
            phase="idle",

            medicine_name=None,

            medicine_similarity=None,

            medicine_threshold=None,

            medicine_is_known=False,

            confirmation_count=0,

            confirmation_required=0,

            person_name=None,

            person_similarity=None,

            action_confidence=None,

            message=(
                "Waiting for a registered medicine."
            ),
        )


    # ========================================================
    # STATUS COPY
    # ========================================================

    @property
    def status(
        self,
    ) -> MedicinePipelineStatus:

        return MedicinePipelineStatus(
            phase=(
                self._status.phase
            ),

            medicine_name=(
                self._status.medicine_name
            ),

            medicine_similarity=(
                self._status.medicine_similarity
            ),

            medicine_threshold=(
                self._status.medicine_threshold
            ),

            medicine_is_known=(
                self._status.medicine_is_known
            ),

            confirmation_count=(
                self._status.confirmation_count
            ),

            confirmation_required=(
                self._status.confirmation_required
            ),

            person_name=(
                self._status.person_name
            ),

            person_similarity=(
                self._status.person_similarity
            ),

            action_confidence=(
                self._status.action_confidence
            ),

            message=(
                self._status.message
            ),
        )


    # ========================================================
    # ACTION ACTIVE
    # ========================================================

    @property
    def action_active(
        self,
    ) -> bool:

        return (
            self._confirmed_medicine
            is not None

            and self._active_person
            is not None

            and self.action_tracker.is_active
        )


    # ========================================================
    # WAITING FOR PERSON
    # ========================================================

    @property
    def waiting_for_person(
        self,
    ) -> bool:

        return (
            self._confirmed_medicine
            is not None

            and self._active_person
            is None

            and self._status.phase
            == "waiting_for_person"
        )


    # ========================================================
    # RESET
    # ========================================================

    def reset(
        self,
    ) -> None:

        self.medicine_monitor.reset()

        self.action_tracker.reset()

        self._confirmed_medicine = (
            None
        )

        self._active_person = (
            None
        )

        self._status = (
            self._idle_status()
        )


    # ========================================================
    # VALIDATE PERSON
    # ========================================================

    @staticmethod
    def _validate_person(
        person: ActivePersonContext,
    ) -> None:

        if person.person_id <= 0:

            raise ValueError(
                "person_id must be greater than zero."
            )

        if not (
            0.0
            <= float(
                person.similarity
            )
            <= 1.0
        ):

            raise ValueError(
                "Person similarity must be between "
                "0.0 and 1.0."
            )

        (
            _x,
            _y,
            width,
            height,
        ) = (
            person.bbox
        )

        if (
            width <= 0
            or height <= 0
        ):

            raise ValueError(
                "Person face bounding box is invalid."
            )


    # ========================================================
    # ARM ACTION TRACKER
    # ========================================================

    def _arm_with_person(
        self,
        person: ActivePersonContext,
    ) -> None:

        if (
            self._confirmed_medicine
            is None
        ):

            raise RuntimeError(
                "No confirmed medicine is waiting "
                "for a person."
            )

        self._validate_person(
            person
        )

        confirmed = (
            self._confirmed_medicine
        )

        # ----------------------------------------------------
        # IMPORTANT
        #
        # Freeze a valid person association when action
        # monitoring begins.
        #
        # Later face detections may refresh this context,
        # but temporary loss of the face will NOT stop
        # hand tracking.
        # ----------------------------------------------------

        self._active_person = (
            person
        )

        self.action_tracker.arm(
            medicine_id=(
                confirmed.medicine_id
            ),

            medicine_name=(
                confirmed.medicine_name
            ),

            dosage=(
                confirmed.dosage
            ),

            medicine_similarity=(
                confirmed.similarity
            ),

            person_id=(
                person.person_id
            ),

            person_name=(
                person.display_name
            ),

            person_similarity=(
                person.similarity
            ),

            armed_at=(
                datetime.now()
            ),
        )

        self._status = (
            MedicinePipelineStatus(
                phase="watching_action",

                medicine_name=(
                    confirmed.medicine_name
                ),

                medicine_similarity=(
                    confirmed.similarity
                ),

                medicine_threshold=(
                    self._status.medicine_threshold
                ),

                medicine_is_known=True,

                confirmation_count=(
                    self._status
                    .confirmation_required
                ),

                confirmation_required=(
                    self._status
                    .confirmation_required
                ),

                person_name=(
                    person.display_name
                ),

                person_similarity=(
                    person.similarity
                ),

                action_confidence=None,

                message=(
                    "Medicine recognized. "
                    "You may move the package away "
                    "from the medicine box. "
                    "MemorAi is now watching the "
                    "hand-to-mouth medication action."
                ),
            )
        )

        logger.info(
            (
                "Medicine action monitoring armed. "
                "medicine_id=%s "
                "medicine=%s "
                "person_id=%s "
                "person=%s"
            ),
            confirmed.medicine_id,
            confirmed.medicine_name,
            person.person_id,
            person.display_name,
        )


    # ========================================================
    # ASSOCIATE PERSON LATER
    # ========================================================

    def associate_person(
        self,
        person: ActivePersonContext,
    ) -> bool:

        if not self.waiting_for_person:

            return False

        self._arm_with_person(
            person
        )

        return True


    # ========================================================
    # CONFIDENCE
    # ========================================================

    @staticmethod
    def calculate_overall_confidence(
        medicine_similarity: float,
        person_similarity: float,
        action_confidence: float,
    ) -> float:

        for value in (
            medicine_similarity,
            person_similarity,
            action_confidence,
        ):

            if not (
                0.0
                <= float(
                    value
                )
                <= 1.0
            ):

                raise ValueError(
                    "Confidence values must be between "
                    "0.0 and 1.0."
                )

        overall = (
            (
                0.40
                * float(
                    medicine_similarity
                )
            )
            +
            (
                0.40
                * float(
                    action_confidence
                )
            )
            +
            (
                0.20
                * float(
                    person_similarity
                )
            )
        )

        return float(
            np.clip(
                overall,
                0.0,
                1.0,
            )
        )


    # ========================================================
    # SAFE FILE NAME
    # ========================================================

    @staticmethod
    def _safe_filename(
        value: str,
    ) -> str:

        cleaned = (
            value
            .strip()
            .lower()
        )

        result: list[str] = []

        previous_separator = False

        for character in cleaned:

            if character.isalnum():

                result.append(
                    character
                )

                previous_separator = False

            elif character in {
                " ",
                "-",
                "_",
            }:

                if not previous_separator:

                    result.append(
                        "_"
                    )

                    previous_separator = True

        return (
            "".join(
                result
            )
            .strip(
                "_"
            )
            or "medicine"
        )


    # ========================================================
    # SAVE EVIDENCE
    # ========================================================

    def _save_evidence_image(
        self,
        frame: np.ndarray,
        medicine_name: str,
        observed_at: datetime,
    ) -> str:

        if frame is None:

            raise ValueError(
                "Evidence frame cannot be None."
            )

        if not isinstance(
            frame,
            np.ndarray,
        ):

            raise TypeError(
                "Evidence frame must be a NumPy array."
            )

        if frame.size == 0:

            raise ValueError(
                "Evidence frame cannot be empty."
            )

        safe_name = (
            self._safe_filename(
                medicine_name
            )
        )

        timestamp = (
            observed_at.strftime(
                "%Y%m%d_%H%M%S_%f"
            )
        )

        path = (
            MEDICINE_EVIDENCE_DIR
            / (
                f"{safe_name}_"
                f"{timestamp}.jpg"
            )
        )

        success = cv2.imwrite(
            str(
                path
            ),
            frame,
        )

        if not success:

            raise RuntimeError(
                "Failed to save medicine evidence image."
            )

        return str(
            path
        )


    # ========================================================
    # DRAW EVIDENCE
    # ========================================================

    @staticmethod
    def _draw_evidence(
        frame: np.ndarray,
        medicine: ConfirmedMedicineRecognition,
        person: ActivePersonContext,
        action: ConfirmedMedicationAction,
    ) -> np.ndarray:

        evidence = (
            frame.copy()
        )

        (
            x,
            y,
            width,
            height,
        ) = (
            person.bbox
        )

        x2 = (
            x
            + width
        )

        y2 = (
            y
            + height
        )

        cv2.rectangle(
            evidence,
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

        cv2.putText(
            evidence,
            (
                f"{person.display_name} "
                f"{person.similarity:.2f}"
            ),
            (
                x,
                max(
                    20,
                    y - 8,
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (
                255,
                255,
                255,
            ),
            2,
        )

        label = (
            medicine.medicine_name
        )

        if medicine.dosage:

            label += (
                " "
                + medicine.dosage
            )

        cv2.putText(
            evidence,
            (
                "Medicine: "
                f"{label} "
                f"{medicine.similarity:.2f}"
            ),
            (
                20,
                30,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (
                255,
                255,
                255,
            ),
            2,
        )

        cv2.putText(
            evidence,
            (
                "Action evidence: "
                f"{action.action_confidence:.2f}"
            ),
            (
                20,
                60,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (
                255,
                255,
                255,
            ),
            2,
        )

        return (
            evidence
        )


    # ========================================================
    # PROCESS MEDICINE ROI
    #
    # IMPORTANT:
    # THIS MEDICINE RECOGNITION FLOW IS UNCHANGED.
    # ========================================================

    def process_medicine_image(
        self,
        medicine_image: np.ndarray,
        person: ActivePersonContext | None,
    ) -> tuple[
        MedicineObservation,
        ConfirmedMedicineRecognition | None,
    ]:

        (
            observation,
            confirmed,
        ) = (
            self.medicine_monitor
            .process_image(
                medicine_image
            )
        )

        # ====================================================
        # IDENTITY FOUND, TEMPORAL CONFIRMATION STILL RUNNING
        # ====================================================

        if confirmed is None:

            if (
                observation.medicine_id
                is None
            ):

                message = (
                    "MemorAi is checking the medicine "
                    "package. Keep it inside the box."
                )

            elif observation.is_known:

                if (
                    observation.recognition_source
                    == "text"
                ):

                    message = (
                        "Medicine recognized from printed "
                        "package text. MemorAi is completing "
                        "its short confirmation check."
                    )

                elif (
                    observation.recognition_source
                    == "visual"
                ):

                    message = (
                        "Medicine recognized from the visual "
                        "package appearance. MemorAi is "
                        "completing its short confirmation "
                        "check."
                    )

                elif (
                    observation.recognition_source
                    == "both"
                ):

                    message = (
                        "Visual and text recognition agree "
                        "on this medicine. MemorAi is "
                        "completing its short confirmation "
                        "check."
                    )

                elif (
                    observation.recognition_source
                    == "conflict"
                ):

                    message = (
                        "Visual and text recognition disagree. "
                        "Keep the package inside the box "
                        "for a clearer reading."
                    )

                else:

                    message = (
                        "Medicine identity recognized. "
                        "MemorAi is completing its short "
                        "confirmation check."
                    )

            elif (
                observation.dino_available
                or observation.ocr_available
            ):

                message = (
                    "MemorAi sees a possible medicine and "
                    "is comparing its visual appearance "
                    "and printed text."
                )

            else:

                message = (
                    "MemorAi is checking the medicine "
                    "package. Keep it inside the box."
                )

            self._status = (
                MedicinePipelineStatus(
                    phase="scanning_medicine",

                    medicine_name=(
                        observation.medicine_name
                        if observation.medicine_id
                        is not None
                        else None
                    ),

                    medicine_similarity=(
                        observation.similarity
                        if observation.medicine_id
                        is not None
                        else None
                    ),

                    medicine_threshold=(
                        observation.threshold
                    ),

                    medicine_is_known=(
                        observation.is_known
                    ),

                    confirmation_count=(
                        observation.confirmation_count
                    ),

                    confirmation_required=(
                        observation.confirmation_required
                    ),

                    person_name=(
                        person.display_name
                        if person is not None
                        else None
                    ),

                    person_similarity=(
                        person.similarity
                        if person is not None
                        else None
                    ),

                    action_confidence=None,

                    message=(
                        message
                    ),
                )
            )

            return (
                observation,
                None,
            )

        # ====================================================
        # PACKAGE TEMPORALLY CONFIRMED
        # ====================================================

        self._confirmed_medicine = (
            confirmed
        )

        # ====================================================
        # NO CURRENT PERSON
        # ====================================================

        if person is None:

            self._active_person = (
                None
            )

            self._status = (
                MedicinePipelineStatus(
                    phase="waiting_for_person",

                    medicine_name=(
                        confirmed.medicine_name
                    ),

                    medicine_similarity=(
                        confirmed.similarity
                    ),

                    medicine_threshold=(
                        observation.threshold
                    ),

                    medicine_is_known=True,

                    confirmation_count=(
                        observation.confirmation_required
                    ),

                    confirmation_required=(
                        observation.confirmation_required
                    ),

                    person_name=None,

                    person_similarity=None,

                    action_confidence=None,

                    message=(
                        "Medicine recognized. "
                        "You may move the package away "
                        "from the camera. MemorAi is "
                        "waiting to associate a registered "
                        "person before watching the "
                        "medication action."
                    ),
                )
            )

            return (
                observation,
                confirmed,
            )

        # ====================================================
        # PERSON AVAILABLE
        # ====================================================

        self._arm_with_person(
            person
        )

        return (
            observation,
            confirmed,
        )


    # ========================================================
    # PROCESS ACTION
    #
    # THIS IS THE IMPORTANT FIX.
    # ========================================================

    def process_action_frame(
        self,
        frame: np.ndarray,
        person: ActivePersonContext | None = None,
    ) -> ConfirmedMedicineEvent | None:
        """
        Process one frame after medicine recognition.

        A fresh face recognition result is optional.

        If the same person is detected again:
            refresh face/mouth location.

        If the face temporarily disappears:
            continue using the stored person context.

        This allows the hand detector to keep running while
        the hand or medicine partly covers the face.
        """

        # ====================================================
        # MUST ALREADY BE ARMED
        # ====================================================

        if (
            self._confirmed_medicine
            is None
            or self._active_person
            is None
        ):

            raise RuntimeError(
                "Medicine camera pipeline is not armed."
            )

        # ====================================================
        # OPTIONAL FRESH PERSON UPDATE
        # ====================================================

        if person is not None:

            self._validate_person(
                person
            )

            # ------------------------------------------------
            # DIFFERENT REGISTERED PERSON
            # ------------------------------------------------

            if (
                person.person_id
                != self._active_person.person_id
            ):

                logger.warning(
                    (
                        "Medication action cancelled because "
                        "person identity changed. "
                        "expected=%s actual=%s"
                    ),
                    self._active_person.person_id,
                    person.person_id,
                )

                self.reset()

                return None

            # ------------------------------------------------
            # SAME PERSON:
            # Refresh latest face bbox.
            # ------------------------------------------------

            self._active_person = (
                person
            )

        # ====================================================
        # STORED PERSON CONTEXT
        # ====================================================

        active_person = (
            self._active_person
        )

        medicine = (
            self._confirmed_medicine
        )

        # ====================================================
        # PROCESS HAND / MOUTH ACTION
        # ====================================================

        (
            action_observation,
            confirmed_action,
        ) = (
            self.action_tracker
            .process_frame(
                frame=(
                    frame
                ),

                face_bbox=(
                    active_person.bbox
                ),

                observed_at=(
                    datetime.now()
                ),
            )
        )

        # ====================================================
        # ACTION DIAGNOSTICS
        # ====================================================

        distance_text = (
            f"{action_observation.normalized_hand_mouth_distance:.3f}"

            if (
                action_observation
                .normalized_hand_mouth_distance
                is not None
            )

            else "none"
        )

        logger.info(
            (
                "Medication action frame: "
                "state=%s "
                "hand_detected=%s "
                "distance=%s "
                "near_mouth=%s "
                "retreat=%s"
            ),
            action_observation.state.value,
            action_observation.hand_detected,
            distance_text,
            action_observation.near_mouth,
            action_observation.retreat_detected,
        )

        # ====================================================
        # ACTION EXPIRED
        # ====================================================

        if (
            action_observation.state
            == MedicationActionState.EXPIRED
        ):

            old_status = (
                self._status
            )

            self._status = (
                MedicinePipelineStatus(
                    phase="expired",

                    medicine_name=(
                        medicine.medicine_name
                    ),

                    medicine_similarity=(
                        medicine.similarity
                    ),

                    medicine_threshold=(
                        old_status
                        .medicine_threshold
                    ),

                    medicine_is_known=True,

                    confirmation_count=(
                        old_status
                        .confirmation_required
                    ),

                    confirmation_required=(
                        old_status
                        .confirmation_required
                    ),

                    person_name=(
                        active_person.display_name
                    ),

                    person_similarity=(
                        active_person.similarity
                    ),

                    action_confidence=None,

                    message=(
                        "Medicine was recognized, but "
                        "MemorAi did not observe enough "
                        "hand-to-mouth action evidence."
                    ),
                )
            )

            logger.info(
                (
                    "Medication action monitoring expired. "
                    "medicine=%s person=%s"
                ),
                medicine.medicine_name,
                active_person.display_name,
            )

            return None

        # ====================================================
        # ACTION NOT YET CONFIRMED
        # ====================================================

        if confirmed_action is None:

            old_status = (
                self._status
            )

            # ------------------------------------------------
            # FRIENDLY ACTION MESSAGE
            # ------------------------------------------------

            if not action_observation.hand_detected:

                action_message = (
                    "Watching for your hand. "
                    "Move the medicine toward your mouth."
                )

            elif action_observation.near_mouth:

                action_message = (
                    "Hand detected near your mouth. "
                    "Now move your hand away normally."
                )

            elif action_observation.retreat_detected:

                action_message = (
                    "Hand moved away from your mouth. "
                    "MemorAi is checking the medication "
                    "gesture."
                )

            else:

                distance = (
                    action_observation
                    .normalized_hand_mouth_distance
                )

                if distance is not None:

                    action_message = (
                        "Hand detected. "
                        "Move it toward your mouth. "
                        f"Distance: {distance:.2f}"
                    )

                else:

                    action_message = (
                        "Hand detected. MemorAi is watching "
                        "the hand-to-mouth medication gesture."
                    )

            self._status = (
                MedicinePipelineStatus(
                    phase="watching_action",

                    medicine_name=(
                        medicine.medicine_name
                    ),

                    medicine_similarity=(
                        medicine.similarity
                    ),

                    medicine_threshold=(
                        old_status
                        .medicine_threshold
                    ),

                    medicine_is_known=True,

                    confirmation_count=(
                        old_status
                        .confirmation_required
                    ),

                    confirmation_required=(
                        old_status
                        .confirmation_required
                    ),

                    person_name=(
                        active_person.display_name
                    ),

                    person_similarity=(
                        active_person.similarity
                    ),

                    action_confidence=None,

                    message=(
                        action_message
                    ),
                )
            )

            return None

        # ====================================================
        # ACTION CONFIRMED
        # ====================================================

        overall_confidence = (
            self.calculate_overall_confidence(
                medicine_similarity=(
                    medicine.similarity
                ),

                person_similarity=(
                    active_person.similarity
                ),

                action_confidence=(
                    confirmed_action
                    .action_confidence
                ),
            )
        )

        # ====================================================
        # EVIDENCE FRAME
        # ====================================================

        evidence_frame = (
            self._draw_evidence(
                frame=(
                    frame
                ),

                medicine=(
                    medicine
                ),

                person=(
                    active_person
                ),

                action=(
                    confirmed_action
                ),
            )
        )

        evidence_path = (
            self._save_evidence_image(
                frame=(
                    evidence_frame
                ),

                medicine_name=(
                    medicine.medicine_name
                ),

                observed_at=(
                    confirmed_action
                    .confirmed_at
                ),
            )
        )

        # ====================================================
        # STORE MEDICATION EVENT
        # ====================================================

        event_id = (
            self.medicine_service
            .record_camera_observation(
                medicine_id=(
                    medicine.medicine_id
                ),

                observed_at=(
                    confirmed_action
                    .confirmed_at
                ),

                confidence=(
                    overall_confidence
                ),

                medicine_similarity=(
                    medicine.similarity
                ),

                action_confidence=(
                    confirmed_action
                    .action_confidence
                ),

                person_id=(
                    active_person.person_id
                ),

                person_similarity=(
                    active_person.similarity
                ),

                evidence_image_path=(
                    evidence_path
                ),

                scheduled_for=None,
            )
        )

        stored_event = (
            get_medicine_event_by_id(
                event_id
            )
        )

        if stored_event is None:

            raise RuntimeError(
                "Medicine event was inserted "
                "but could not be loaded."
            )

        result = (
            ConfirmedMedicineEvent(
                event_id=(
                    event_id
                ),

                medicine_id=(
                    medicine.medicine_id
                ),

                medicine_name=(
                    medicine.medicine_name
                ),

                dosage=(
                    medicine.dosage
                ),

                person_id=(
                    active_person.person_id
                ),

                person_name=(
                    active_person.display_name
                ),

                medicine_similarity=float(
                    medicine.similarity
                ),

                person_similarity=float(
                    active_person.similarity
                ),

                action_confidence=float(
                    confirmed_action
                    .action_confidence
                ),

                overall_confidence=float(
                    overall_confidence
                ),

                observed_at=(
                    confirmed_action
                    .confirmed_at
                ),

                evidence_image_path=(
                    evidence_path
                ),

                status=str(
                    stored_event[
                        "status"
                    ]
                ),

                memory_text=(
                    stored_event.get(
                        "memory_text"
                    )
                ),
            )
        )

        # ====================================================
        # UPDATE LIVE STATUS
        # ====================================================

        old_status = (
            self._status
        )

        self._status = (
            MedicinePipelineStatus(
                phase="event_saved",

                medicine_name=(
                    result.medicine_name
                ),

                medicine_similarity=(
                    result.medicine_similarity
                ),

                medicine_threshold=(
                    old_status
                    .medicine_threshold
                ),

                medicine_is_known=True,

                confirmation_count=(
                    old_status
                    .confirmation_required
                ),

                confirmation_required=(
                    old_status
                    .confirmation_required
                ),

                person_name=(
                    result.person_name
                ),

                person_similarity=(
                    result.person_similarity
                ),

                action_confidence=(
                    result.action_confidence
                ),

                message=(
                    "Medication event confirmed. "
                    "MemorAi stored the camera-confirmed "
                    "medication event."
                ),
            )
        )

        logger.info(
            (
                "Camera medication event stored. "
                "event_id=%s "
                "medicine_id=%s "
                "medicine=%s "
                "person_id=%s "
                "person=%s "
                "action=%.4f "
                "overall=%.4f "
                "status=%s"
            ),
            result.event_id,
            result.medicine_id,
            result.medicine_name,
            result.person_id,
            result.person_name,
            result.action_confidence,
            result.overall_confidence,
            result.status,
        )

        # ====================================================
        # RESET AFTER SUCCESS
        # ====================================================

        self.medicine_monitor.reset()

        self.action_tracker.reset()

        self._confirmed_medicine = (
            None
        )

        self._active_person = (
            None
        )

        return result