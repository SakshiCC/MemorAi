from __future__ import annotations

from datetime import (
    date,
    datetime,
    time,
)

from typing import Any

from database.medicine_repository import (
    STATUS_CAMERA_CONFIRMED,
    STATUS_LOW_CONFIDENCE,
    STATUS_MISSED,
    STATUS_UNCONFIRMED,
    add_medicine_schedule,
    create_medicine,
    deactivate_medicine,
    get_all_active_schedules,
    get_all_medicines,
    get_latest_medicine_event,
    get_latest_medicine_event_by_name,
    get_medicine_by_id,
    get_medicine_by_name,
    get_medicine_event_by_id,
    get_medicine_events_for_date,
    get_medicine_schedules,
    get_today_medicine_events,
    reactivate_medicine,
    save_medicine_event,
)

from database.medicine_vector_memory import (
    MedicineVectorMemory,
)

from utils.logger import (
    get_logger,
)


# ============================================================
# LOGGER
# ============================================================

logger = get_logger(
    "medicine.service",
    "app.log",
)


# ============================================================
# CONSTANTS
# ============================================================

# ------------------------------------------------------------
# OVERALL CONFIDENCE
# ------------------------------------------------------------

DEFAULT_CONFIRMATION_THRESHOLD = 0.75


# ------------------------------------------------------------
# FINAL MEDICINE SIMILARITY THRESHOLD
# ------------------------------------------------------------

DEFAULT_MEDICINE_SIMILARITY_THRESHOLD = 0.65


# ------------------------------------------------------------
# PERSON SIMILARITY THRESHOLD
# ------------------------------------------------------------

DEFAULT_PERSON_SIMILARITY_THRESHOLD = 0.50


# ------------------------------------------------------------
# MEDICATION ACTION THRESHOLD
# ------------------------------------------------------------

DEFAULT_ACTION_CONFIDENCE_THRESHOLD = 0.70


# ============================================================
# SERVICE
# ============================================================

class MedicineService:
    """
    High-level Medicine Memory service.

    Important safety rule:

    MemorAi records visual medication confirmation.

    It must NOT claim that a medicine was definitely
    swallowed or medically taken.

    Storage flow:

        Medicine event
            ↓
        MySQL medicine_events
            ↓
        MedicineVectorMemory
            ↓
        ChromaDB medicine_memories

    MySQL remains the authoritative data source.

    ChromaDB is used only for semantic memory retrieval.
    """

    # ========================================================
    # INITIALIZE
    # ========================================================

    def __init__(
        self,
        confirmation_threshold: float = (
            DEFAULT_CONFIRMATION_THRESHOLD
        ),
        medicine_similarity_threshold: float = (
            DEFAULT_MEDICINE_SIMILARITY_THRESHOLD
        ),
        person_similarity_threshold: float = (
            DEFAULT_PERSON_SIMILARITY_THRESHOLD
        ),
        action_confidence_threshold: float = (
            DEFAULT_ACTION_CONFIDENCE_THRESHOLD
        ),
    ) -> None:

        # ----------------------------------------------------
        # INFORMATIONAL OVERALL THRESHOLD
        # ----------------------------------------------------

        self.confirmation_threshold = (
            self._validate_threshold(
                confirmation_threshold,
                "confirmation_threshold",
            )
        )

        # ----------------------------------------------------
        # MEDICINE THRESHOLD
        # ----------------------------------------------------

        self.medicine_similarity_threshold = (
            self._validate_threshold(
                medicine_similarity_threshold,
                "medicine_similarity_threshold",
            )
        )

        # ----------------------------------------------------
        # PERSON THRESHOLD
        # ----------------------------------------------------

        self.person_similarity_threshold = (
            self._validate_threshold(
                person_similarity_threshold,
                "person_similarity_threshold",
            )
        )

        # ----------------------------------------------------
        # ACTION THRESHOLD
        # ----------------------------------------------------

        self.action_confidence_threshold = (
            self._validate_threshold(
                action_confidence_threshold,
                "action_confidence_threshold",
            )
        )

        # ----------------------------------------------------
        # MEDICINE VECTOR MEMORY
        #
        # This uses the same shared BGE embedding layer as
        # the other MemorAi semantic memories.
        # ----------------------------------------------------

        self._medicine_vector_memory: (
            MedicineVectorMemory
            | None
        ) = None

    # ========================================================
    # LAZY VECTOR MEMORY
    # ========================================================

    def _get_vector_memory(
        self,
    ) -> MedicineVectorMemory:
        """
        Initialize MedicineVectorMemory only when an event
        actually needs to be indexed.

        This avoids loading the embedding model merely
        because MedicineService was constructed.
        """

        if (
            self._medicine_vector_memory
            is None
        ):

            self._medicine_vector_memory = (
                MedicineVectorMemory()
            )

        return (
            self._medicine_vector_memory
        )

    # ========================================================
    # INDEX STORED EVENT
    # ========================================================

    def _index_medicine_event(
        self,
        event_id: int,
    ) -> None:
        """
        Index the authoritative MySQL medicine event into
        the dedicated medicine ChromaDB collection.

        IMPORTANT:

        Failure to index into ChromaDB must NOT undo or
        invalidate a successfully stored MySQL event.

        MySQL is authoritative.
        """

        try:

            event = (
                get_medicine_event_by_id(
                    event_id
                )
            )

            if event is None:

                logger.warning(
                    (
                        "Medicine event could not be "
                        "indexed because the MySQL row "
                        "was not found. event_id=%s"
                    ),
                    event_id,
                )

                return

            vector_memory = (
                self._get_vector_memory()
            )

            vector_memory.add_event(
                event
            )

            logger.info(
                (
                    "Medicine event indexed into "
                    "medicine vector memory. "
                    "event_id=%s medicine_id=%s "
                    "status=%s"
                ),
                event_id,
                event.get(
                    "medicine_id"
                ),
                event.get(
                    "status"
                ),
            )

        except Exception as exc:

            # ------------------------------------------------
            # IMPORTANT:
            #
            # Medicine event is already safely stored in
            # MySQL. Do not break camera confirmation merely
            # because semantic indexing failed.
            # ------------------------------------------------

            logger.exception(
                (
                    "Medicine event was stored in MySQL "
                    "but vector indexing failed. "
                    "event_id=%s error=%s"
                ),
                event_id,
                exc,
            )

    # ========================================================
    # VALIDATION
    # ========================================================

    @staticmethod
    def _validate_threshold(
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
                f"{field_name} must be between "
                "0.0 and 1.0."
            )

        return number

    # ========================================================
    # REGISTER MEDICINE
    # ========================================================

    def register_medicine(
        self,
        medicine_name: str,
        dosage: str | None,
        schedule_time: time,
        schedule_label: str | None = None,
        instructions: str | None = None,
    ) -> int:

        existing = (
            get_medicine_by_name(
                medicine_name
            )
        )

        if existing is not None:

            raise ValueError(
                "Medicine is already registered: "
                f"{medicine_name}"
            )

        medicine_id = (
            create_medicine(
                medicine_name=(
                    medicine_name
                ),
                dosage=(
                    dosage
                ),
                schedule_time=(
                    schedule_time
                ),
                schedule_label=(
                    schedule_label
                ),
                instructions=(
                    instructions
                ),
            )
        )

        logger.info(
            (
                "Medicine registered. "
                "medicine_id=%s name=%s"
            ),
            medicine_id,
            medicine_name,
        )

        return medicine_id

    # ========================================================
    # ADD ANOTHER SCHEDULE
    # ========================================================

    def add_schedule(
        self,
        medicine_id: int,
        schedule_time: time,
        schedule_label: str | None = None,
    ) -> int:

        medicine = (
            get_medicine_by_id(
                medicine_id
            )
        )

        if medicine is None:

            raise ValueError(
                "Medicine does not exist."
            )

        existing_schedules = (
            get_medicine_schedules(
                medicine_id=(
                    medicine_id
                ),
                active_only=False,
            )
        )

        for schedule in existing_schedules:

            if (
                schedule[
                    "schedule_time"
                ]
                == schedule_time
                and bool(
                    schedule[
                        "is_active"
                    ]
                )
            ):

                raise ValueError(
                    "An active schedule already "
                    "exists for this time."
                )

        schedule_id = (
            add_medicine_schedule(
                medicine_id=(
                    medicine_id
                ),
                schedule_time=(
                    schedule_time
                ),
                schedule_label=(
                    schedule_label
                ),
            )
        )

        logger.info(
            (
                "Medicine schedule added. "
                "medicine_id=%s schedule_id=%s"
            ),
            medicine_id,
            schedule_id,
        )

        return schedule_id

    # ========================================================
    # ACTIVE MEDICINES
    # ========================================================

    def get_registered_medicines(
        self,
        active_only: bool = True,
    ) -> list[dict[str, Any]]:

        return (
            get_all_medicines(
                active_only=(
                    active_only
                ),
            )
        )

    # ========================================================
    # ACTIVATE / DEACTIVATE
    # ========================================================

    def deactivate(
        self,
        medicine_id: int,
    ) -> None:

        deactivate_medicine(
            medicine_id
        )

        logger.info(
            (
                "Medicine deactivated. "
                "medicine_id=%s"
            ),
            medicine_id,
        )

    def reactivate(
        self,
        medicine_id: int,
    ) -> None:

        reactivate_medicine(
            medicine_id
        )

        logger.info(
            (
                "Medicine reactivated. "
                "medicine_id=%s"
            ),
            medicine_id,
        )

    # ========================================================
    # CAMERA CONFIDENCE LOGIC
    # ========================================================

    def determine_camera_status(
        self,
        confidence: float,
        medicine_similarity: float,
        action_confidence: float,
        person_similarity: float | None = None,
    ) -> str:
        """
        Decide whether a camera medication observation has
        enough independent evidence to be stored as
        camera_confirmed.

        Final requirements:

            medicine similarity >= 0.65

            action confidence >= 0.70

            person similarity >= 0.50
                when person evidence is available

        Important:

        camera_confirmed means MemorAi observed sufficient
        visual evidence of a medication-taking event.

        It does NOT prove swallowing.
        """

        # ----------------------------------------------------
        # VALIDATE ALL SCORES
        # ----------------------------------------------------

        confidence = (
            self._validate_threshold(
                confidence,
                "confidence",
            )
        )

        medicine_similarity = (
            self._validate_threshold(
                medicine_similarity,
                "medicine_similarity",
            )
        )

        action_confidence = (
            self._validate_threshold(
                action_confidence,
                "action_confidence",
            )
        )

        if (
            person_similarity
            is not None
        ):

            person_similarity = (
                self._validate_threshold(
                    person_similarity,
                    "person_similarity",
                )
            )

        # ----------------------------------------------------
        # MEDICINE IDENTITY
        # ----------------------------------------------------

        if (
            medicine_similarity
            < self.medicine_similarity_threshold
        ):

            logger.info(
                (
                    "Medicine event rejected: "
                    "medicine similarity %.4f "
                    "< required %.4f."
                ),
                medicine_similarity,
                self.medicine_similarity_threshold,
            )

            return (
                STATUS_LOW_CONFIDENCE
            )

        # ----------------------------------------------------
        # MEDICATION ACTION
        # ----------------------------------------------------

        if (
            action_confidence
            < self.action_confidence_threshold
        ):

            logger.info(
                (
                    "Medicine event rejected: "
                    "action confidence %.4f "
                    "< required %.4f."
                ),
                action_confidence,
                self.action_confidence_threshold,
            )

            return (
                STATUS_LOW_CONFIDENCE
            )

        # ----------------------------------------------------
        # PERSON IDENTITY
        # ----------------------------------------------------

        if (
            person_similarity
            is not None
            and person_similarity
            < self.person_similarity_threshold
        ):

            logger.info(
                (
                    "Medicine event rejected: "
                    "person similarity %.4f "
                    "< required %.4f."
                ),
                person_similarity,
                self.person_similarity_threshold,
            )

            return (
                STATUS_LOW_CONFIDENCE
            )

        # ----------------------------------------------------
        # ALL INDEPENDENT SIGNALS PASSED
        # ----------------------------------------------------

        logger.info(
            (
                "Medicine event accepted. "
                "overall=%.4f "
                "medicine=%.4f "
                "action=%.4f "
                "person=%s"
            ),
            confidence,
            medicine_similarity,
            action_confidence,
            (
                f"{person_similarity:.4f}"
                if person_similarity
                is not None
                else "none"
            ),
        )

        return (
            STATUS_CAMERA_CONFIRMED
        )

    # ========================================================
    # RECORD CAMERA OBSERVATION
    # ========================================================

    def record_camera_observation(
        self,
        medicine_id: int,
        observed_at: datetime,
        confidence: float,
        medicine_similarity: float,
        action_confidence: float,
        person_id: int | None = None,
        person_similarity: float | None = None,
        evidence_image_path: str | None = None,
        scheduled_for: datetime | None = None,
    ) -> int:

        medicine = (
            get_medicine_by_id(
                medicine_id
            )
        )

        if medicine is None:

            raise ValueError(
                "Medicine does not exist."
            )

        if not bool(
            medicine[
                "is_active"
            ]
        ):

            raise ValueError(
                "Cannot record confirmation "
                "for an inactive medicine."
            )

        # ----------------------------------------------------
        # DETERMINE STATUS
        # ----------------------------------------------------

        status = (
            self.determine_camera_status(
                confidence=(
                    confidence
                ),
                medicine_similarity=(
                    medicine_similarity
                ),
                person_similarity=(
                    person_similarity
                ),
                action_confidence=(
                    action_confidence
                ),
            )
        )

        # ----------------------------------------------------
        # SAVE TO MYSQL FIRST
        # ----------------------------------------------------

        event_id = (
            save_medicine_event(
                medicine_id=(
                    medicine_id
                ),
                observed_at=(
                    observed_at
                ),
                status=(
                    status
                ),
                person_id=(
                    person_id
                ),
                confidence=(
                    confidence
                ),
                medicine_similarity=(
                    medicine_similarity
                ),
                person_similarity=(
                    person_similarity
                ),
                action_confidence=(
                    action_confidence
                ),
                evidence_image_path=(
                    evidence_image_path
                ),
                scheduled_for=(
                    scheduled_for
                ),
            )
        )

        logger.info(
            (
                "Medicine camera observation stored. "
                "medicine_id=%s event_id=%s "
                "status=%s overall=%.4f "
                "medicine=%.4f action=%.4f "
                "person=%s"
            ),
            medicine_id,
            event_id,
            status,
            confidence,
            medicine_similarity,
            action_confidence,
            (
                f"{person_similarity:.4f}"
                if person_similarity
                is not None
                else "none"
            ),
        )

        # ----------------------------------------------------
        # INDEX INTO MEDICINE VECTOR MEMORY
        # ----------------------------------------------------

        self._index_medicine_event(
            event_id
        )

        return event_id

    # ========================================================
    # CONFIRMATION FOR DATE
    # ========================================================

    def was_medicine_confirmed_on_date(
        self,
        medicine_id: int,
        target_date: date,
    ) -> bool:

        events = (
            get_medicine_events_for_date(
                target_date=(
                    target_date
                ),
                medicine_id=(
                    medicine_id
                ),
                confirmed_only=True,
            )
        )

        return bool(
            events
        )

    # ========================================================
    # CONFIRMATION TODAY
    # ========================================================

    def was_medicine_confirmed_today(
        self,
        medicine_id: int,
    ) -> bool:

        return (
            self.was_medicine_confirmed_on_date(
                medicine_id=(
                    medicine_id
                ),
                target_date=(
                    date.today()
                ),
            )
        )

    # ========================================================
    # GET TODAY HISTORY
    # ========================================================

    def get_today_history(
        self,
        medicine_id: int | None = None,
        confirmed_only: bool = False,
    ) -> list[dict[str, Any]]:

        return (
            get_today_medicine_events(
                medicine_id=(
                    medicine_id
                ),
                confirmed_only=(
                    confirmed_only
                ),
            )
        )

    # ========================================================
    # LAST CONFIRMATION
    # ========================================================

    def get_last_confirmation(
        self,
        medicine_id: int,
    ) -> dict[str, Any] | None:

        return (
            get_latest_medicine_event(
                medicine_id=(
                    medicine_id
                ),
                confirmed_only=True,
            )
        )

    def get_last_confirmation_by_name(
        self,
        medicine_name: str,
    ) -> dict[str, Any] | None:

        return (
            get_latest_medicine_event_by_name(
                medicine_name=(
                    medicine_name
                ),
                confirmed_only=True,
            )
        )

    # ========================================================
    # TODAY'S SCHEDULE
    # ========================================================

    def get_todays_schedule(
        self,
    ) -> list[dict[str, Any]]:

        return (
            get_all_active_schedules()
        )

    # ========================================================
    # TODAY STATUS
    # ========================================================

    def get_todays_medicine_status(
        self,
    ) -> list[dict[str, Any]]:

        schedules = (
            self.get_todays_schedule()
        )

        today = (
            date.today()
        )

        results: list[
            dict[str, Any]
        ] = []

        for schedule in schedules:

            medicine_id = int(
                schedule[
                    "medicine_id"
                ]
            )

            confirmed = (
                self.was_medicine_confirmed_on_date(
                    medicine_id=(
                        medicine_id
                    ),
                    target_date=(
                        today
                    ),
                )
            )

            item = dict(
                schedule
            )

            item[
                "confirmation_status"
            ] = (
                STATUS_CAMERA_CONFIRMED
                if confirmed
                else STATUS_UNCONFIRMED
            )

            results.append(
                item
            )

        return results

    # ========================================================
    # SAFE ANSWER: TODAY CONFIRMATION
    # ========================================================

    def build_today_confirmation_answer(
        self,
        medicine_name: str,
    ) -> str:

        medicine = (
            get_medicine_by_name(
                medicine_name
            )
        )

        if medicine is None:

            return (
                "I do not have that medicine "
                "registered in MemorAi."
            )

        events = (
            get_medicine_events_for_date(
                target_date=(
                    date.today()
                ),
                medicine_id=int(
                    medicine[
                        "id"
                    ]
                ),
                confirmed_only=True,
            )
        )

        display_name = (
            medicine[
                "medicine_name"
            ]
        )

        dosage = (
            medicine[
                "dosage"
            ]
        )

        description = (
            display_name
        )

        if dosage:

            description += (
                f" {dosage}"
            )

        if not events:

            return (
                "MemorAi has no camera-confirmed "
                f"record for {description} today."
            )

        latest = (
            events[
                0
            ]
        )

        observed_at = (
            latest[
                "observed_at"
            ]
        )

        if isinstance(
            observed_at,
            datetime,
        ):

            readable_time = (
                observed_at.strftime(
                    "%I:%M %p"
                )
            )

        else:

            readable_time = (
                str(
                    observed_at
                )
            )

        return (
            "MemorAi recorded a camera-confirmed "
            f"medication event for {description} "
            f"at {readable_time} today."
        )

    # ========================================================
    # SAFE ANSWER: LAST CONFIRMATION
    # ========================================================

    def build_last_confirmation_answer(
        self,
        medicine_name: str,
    ) -> str:

        medicine = (
            get_medicine_by_name(
                medicine_name
            )
        )

        if medicine is None:

            return (
                "I do not have that medicine "
                "registered in MemorAi."
            )

        latest = (
            self.get_last_confirmation(
                int(
                    medicine[
                        "id"
                    ]
                )
            )
        )

        description = (
            medicine[
                "medicine_name"
            ]
        )

        if (
            medicine[
                "dosage"
            ]
        ):

            description += (
                f" {medicine['dosage']}"
            )

        if latest is None:

            return (
                "MemorAi has no camera-confirmed "
                f"medication event for "
                f"{description}."
            )

        observed_at = (
            latest[
                "observed_at"
            ]
        )

        if isinstance(
            observed_at,
            datetime,
        ):

            readable_time = (
                observed_at.strftime(
                    "%d %b %Y at %I:%M %p"
                )
            )

        else:

            readable_time = (
                str(
                    observed_at
                )
            )

        return (
            "MemorAi last recorded a "
            "camera-confirmed medication event "
            f"for {description} on "
            f"{readable_time}."
        )

    # ========================================================
    # MARK UNCONFIRMED
    # ========================================================

    def record_unconfirmed_schedule(
        self,
        medicine_id: int,
        scheduled_for: datetime,
        observed_at: datetime | None = None,
    ) -> int:

        if observed_at is None:

            observed_at = (
                datetime.now()
            )

        event_id = (
            save_medicine_event(
                medicine_id=(
                    medicine_id
                ),
                observed_at=(
                    observed_at
                ),
                scheduled_for=(
                    scheduled_for
                ),
                status=(
                    STATUS_UNCONFIRMED
                ),
                confidence=None,
                medicine_similarity=None,
                person_similarity=None,
                action_confidence=None,
            )
        )

        self._index_medicine_event(
            event_id
        )

        return event_id

    # ========================================================
    # MISSED STATE
    # ========================================================

    def record_missed_schedule(
        self,
        medicine_id: int,
        scheduled_for: datetime,
        observed_at: datetime | None = None,
    ) -> int:

        if observed_at is None:

            observed_at = (
                datetime.now()
            )

        event_id = (
            save_medicine_event(
                medicine_id=(
                    medicine_id
                ),
                observed_at=(
                    observed_at
                ),
                scheduled_for=(
                    scheduled_for
                ),
                status=(
                    STATUS_MISSED
                ),
                confidence=None,
                medicine_similarity=None,
                person_similarity=None,
                action_confidence=None,
            )
        )

        self._index_medicine_event(
            event_id
        )

        return event_id