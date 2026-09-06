from __future__ import annotations

from datetime import (
    datetime,
    time,
)

from pathlib import Path

from uuid import uuid4

import numpy as np

from database.db_manager import (
    execute,
    initialize_database,
)

from database.medicine_repository import (
    STATUS_CAMERA_CONFIRMED,
    create_medicine,
    get_medicine_event_by_id,
)

from database.person_repository import (
    create_person,
)

from medicine.medicine_action_tracker import (
    ConfirmedMedicationAction,
    MedicationActionObservation,
    MedicationActionState,
)

from medicine.medicine_camera_pipeline import (
    ActivePersonContext,
    MedicineCameraPipeline,
)

from medicine.medicine_monitor import (
    ConfirmedMedicineRecognition,
    MedicineObservation,
)

from medicine.medicine_service import (
    MedicineService,
)


# ============================================================
# TEST OUTPUT
# ============================================================

def print_test_result(
    test_name: str,
    passed: bool,
) -> None:

    status = (
        "PASS"
        if passed
        else "FAIL"
    )

    print(
        f"{status}: {test_name}"
    )


# ============================================================
# FAKE MEDICINE MONITOR
# ============================================================

class FakeMedicineMonitor:

    def __init__(
        self,
        medicine_id: int,
        medicine_name: str,
    ) -> None:

        self.medicine_id = (
            medicine_id
        )

        self.medicine_name = (
            medicine_name
        )

        self.call_count = 0

    def process_image(
        self,
        image: np.ndarray,
    ) -> tuple[
        MedicineObservation,
        ConfirmedMedicineRecognition | None,
    ]:

        self.call_count += 1

        now = (
            datetime.now()
        )

        observation = (
            MedicineObservation(
                medicine_id=(
                    self.medicine_id
                ),
                medicine_name=(
                    self.medicine_name
                ),
                dosage="500 mg",
                similarity=0.92,
                is_known=True,
                observed_at=now,
            )
        ) # type: ignore

        confirmed = (
            ConfirmedMedicineRecognition(
                medicine_id=(
                    self.medicine_id
                ),
                medicine_name=(
                    self.medicine_name
                ),
                dosage="500 mg",
                similarity=0.92,
                observed_at=now,
                confirmation_frames=3,
            )
        )

        return (
            observation,
            confirmed,
        )

    def reset(
        self,
    ) -> None:

        pass


# ============================================================
# FAKE ACTION TRACKER
# ============================================================

class FakeActionTracker:

    def __init__(
        self,
    ) -> None:

        self.armed = False

        self.medicine_id: (
            int
            | None
        ) = None

        self.medicine_name: (
            str
            | None
        ) = None

        self.person_id: (
            int
            | None
        ) = None

        self.person_name: (
            str
            | None
        ) = None

        self.person_similarity: (
            float
            | None
        ) = None

        self.medicine_similarity: (
            float
            | None
        ) = None

    @property
    def is_active(
        self,
    ) -> bool:

        return (
            self.armed
        )

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

        self.armed = True

        self.medicine_id = (
            medicine_id
        )

        self.medicine_name = (
            medicine_name
        )

        self.person_id = (
            person_id
        )

        self.person_name = (
            person_name
        )

        self.person_similarity = (
            person_similarity
        )

        self.medicine_similarity = (
            medicine_similarity
        )

        _ = (
            dosage,
            armed_at,
        )

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

        assert (
            self.armed
        )

        assert (
            self.medicine_id
            is not None
        )

        assert (
            self.medicine_name
            is not None
        )

        assert (
            self.person_id
            is not None
        )

        assert (
            self.person_similarity
            is not None
        )

        assert (
            self.medicine_similarity
            is not None
        )

        if observed_at is None:

            observed_at = (
                datetime.now()
            )

        observation = (
            MedicationActionObservation(
                state=(
                    MedicationActionState
                    .CONFIRMED
                ),
                hand_detected=True,
                normalized_hand_mouth_distance=0.10,
                near_mouth=False,
                retreat_detected=True,
                observed_at=observed_at,
            )
        )

        confirmed = (
            ConfirmedMedicationAction(
                medicine_id=(
                    self.medicine_id
                ),
                medicine_name=(
                    self.medicine_name
                ),
                dosage="500 mg",
                medicine_similarity=(
                    self.medicine_similarity
                ),
                person_id=(
                    self.person_id
                ),
                person_name=(
                    self.person_name
                ),
                person_similarity=(
                    self.person_similarity
                ),
                action_confidence=0.90,
                closest_normalized_distance=0.10,
                confirmed_at=observed_at,
            )
        )

        _ = (
            frame,
            face_bbox,
        )

        return (
            observation,
            confirmed,
        )

    def reset(
        self,
    ) -> None:

        self.armed = False


# ============================================================
# TEST FRAME
# ============================================================

def create_test_frame() -> np.ndarray:

    return np.zeros(
        (
            480,
            640,
            3,
        ),
        dtype=np.uint8,
    )


# ============================================================
# CLEANUP
# ============================================================

def cleanup(
    medicine_id: int | None,
    person_id: int | None,
) -> None:

    if medicine_id is not None:

        execute(
            """
            DELETE FROM medicines
            WHERE id = %s
            """,
            (
                medicine_id,
            ),
        )

    if person_id is not None:

        execute(
            """
            DELETE FROM person_events
            WHERE person_id = %s
            """,
            (
                person_id,
            ),
        )

        execute(
            """
            DELETE FROM person_face_embeddings
            WHERE person_id = %s
            """,
            (
                person_id,
            ),
        )

        execute(
            """
            DELETE FROM registered_people
            WHERE id = %s
            """,
            (
                person_id,
            ),
        )


# ============================================================
# TEST SUITE
# ============================================================

def test_medicine_camera_pipeline() -> None:

    print(
        "\n"
        "MemorAi Medicine Camera Pipeline "
        "Test Suite\n"
    )

    initialize_database()

    medicine_id: int | None = None

    person_id: int | None = None

    evidence_path: (
        Path
        | None
    ) = None

    suffix = (
        uuid4()
        .hex[:8]
    )

    medicine_name = (
        "Camera Test Medicine "
        f"{suffix}"
    )

    person_name = (
        "camera_test_person_"
        f"{suffix}"
    )

    try:

        # ====================================================
        # CREATE TEST MEDICINE
        # ====================================================

        medicine_id = (
            create_medicine(
                medicine_name=medicine_name,
                dosage="500 mg",
                schedule_time=time(
                    8,
                    0,
                ),
                schedule_label="Morning",
                instructions=(
                    "Camera pipeline test."
                ),
            )
        )

        assert (
            medicine_id
            > 0
        )

        print_test_result(
            "Temporary medicine",
            True,
        )

        # ====================================================
        # CREATE TEST PERSON
        # ====================================================

        person_id = (
            create_person(
                name=person_name,
                display_name=(
                    "Camera Test Person"
                ),
                relation="Test",
            )
        )

        assert (
            person_id
            > 0
        )

        print_test_result(
            "Temporary registered person",
            True,
        )

        # ====================================================
        # PIPELINE WITH FAKE VISION COMPONENTS
        # ====================================================

        medicine_monitor = (
            FakeMedicineMonitor(
                medicine_id=(
                    medicine_id
                ),
                medicine_name=(
                    medicine_name
                ),
            )
        )

        action_tracker = (
            FakeActionTracker()
        )

        service = (
            MedicineService()
        )

        pipeline = (
            MedicineCameraPipeline(
                medicine_monitor=(
                    medicine_monitor
                ),  # type: ignore[arg-type]
                action_tracker=(
                    action_tracker
                ),  # type: ignore[arg-type]
                medicine_service=(
                    service
                ),
            )
        )

        # ====================================================
        # PERSON CONTEXT
        # ====================================================

        person = (
            ActivePersonContext(
                person_id=(
                    person_id
                ),
                person_name=(
                    person_name
                ),
                display_name=(
                    "Camera Test Person"
                ),
                similarity=0.85,
                bbox=(
                    220,
                    80,
                    200,
                    240,
                ),
                observed_at=(
                    datetime.now()
                ),
            )
        )

        frame = (
            create_test_frame()
        )

        # ====================================================
        # CONFIRM MEDICINE + ARM ACTION TRACKER
        # ====================================================

        (
            observation,
            confirmed_medicine,
        ) = (
            pipeline
            .process_medicine_image(
                medicine_image=frame,
                person=person,
            )
        )

        assert (
            observation.is_known
        )

        assert (
            confirmed_medicine
            is not None
        )

        assert (
            pipeline.action_active
        )

        assert (
            pipeline.status.phase
            == "watching_action"
        )

        print_test_result(
            "Medicine + person arms action tracker",
            True,
        )

        # ====================================================
        # ACTION -> DATABASE EVENT
        # ====================================================

        result = (
            pipeline
            .process_action_frame(
                frame=frame,
                person=person,
            )
        )

        assert (
            result
            is not None
        )

        evidence_path = Path(
            result.evidence_image_path
        )

        print_test_result(
            "Medication action creates event",
            True,
        )

        # ====================================================
        # CORRECT PERSON
        # ====================================================

        assert (
            result.person_id
            == person_id
        )

        assert (
            result.person_name
            == "Camera Test Person"
        )

        print_test_result(
            "Correct person associated",
            True,
        )

        # ====================================================
        # CORRECT MEDICINE
        # ====================================================

        assert (
            result.medicine_id
            == medicine_id
        )

        assert (
            result.medicine_name
            == medicine_name
        )

        print_test_result(
            "Correct medicine associated",
            True,
        )

        # ====================================================
        # CONFIDENCE
        # ====================================================

        expected_overall = (
            (
                0.40
                * 0.92
            )
            +
            (
                0.40
                * 0.90
            )
            +
            (
                0.20
                * 0.85
            )
        )

        assert abs(
            result.overall_confidence
            - expected_overall
        ) < 0.0001

        assert (
            result.medicine_similarity
            == 0.92
        )

        assert (
            result.person_similarity
            == 0.85
        )

        assert (
            result.action_confidence
            == 0.90
        )

        print_test_result(
            "Combined visual confidence",
            True,
        )

        # ====================================================
        # STATUS
        # ====================================================

        assert (
            result.status
            == STATUS_CAMERA_CONFIRMED
        )

        print_test_result(
            "Camera-confirmed status",
            True,
        )

        # ====================================================
        # DATABASE ROW
        # ====================================================

        stored = (
            get_medicine_event_by_id(
                result.event_id
            )
        )

        assert (
            stored
            is not None
        )

        assert (
            int(
                stored[
                    "medicine_id"
                ]
            )
            == medicine_id
        )

        assert (
            int(
                stored[
                    "person_id"
                ]
            )
            == person_id
        )

        assert (
            stored[
                "status"
            ]
            == STATUS_CAMERA_CONFIRMED
        )

        print_test_result(
            "Medicine event stored in MySQL",
            True,
        )

        # ====================================================
        # SAFE MEMORY TEXT
        # ====================================================

        memory_text = str(
            stored[
                "memory_text"
            ]
            or ""
        )

        assert (
            "camera-confirmed"
            in memory_text.lower()
        )

        assert (
            "swallowed"
            not in memory_text.lower()
        )

        assert (
            "definitely took"
            not in memory_text.lower()
        )

        print_test_result(
            "Safe medicine memory text",
            True,
        )

        # ====================================================
        # EVIDENCE IMAGE
        # ====================================================

        assert (
            evidence_path.exists()
        )

        assert (
            evidence_path.is_file()
        )

        print_test_result(
            "Medicine evidence image",
            True,
        )

        # ====================================================
        # PIPELINE RESET AFTER EVENT
        # ====================================================

        assert not (
            pipeline.action_active
        )

        print_test_result(
            "Pipeline resets after stored event",
            True,
        )

        print(
            "\n"
            "All medicine camera pipeline "
            "tests passed."
            "\n"
        )

    finally:

        cleanup(
            medicine_id=medicine_id,
            person_id=person_id,
        )

        if (
            evidence_path is not None
            and evidence_path.exists()
        ):

            try:

                evidence_path.unlink()

            except OSError:

                pass


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    test_medicine_camera_pipeline()