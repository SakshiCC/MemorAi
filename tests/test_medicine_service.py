from __future__ import annotations

from datetime import (
    date,
    datetime,
    time,
)
from uuid import uuid4

from database.db_manager import (
    execute,
    initialize_database,
)

from database.medicine_repository import (
    STATUS_CAMERA_CONFIRMED,
    STATUS_LOW_CONFIDENCE,
    get_medicine_by_id,
    get_medicine_events_for_date,
)

from medicine.medicine_service import (
    MedicineService,
)


# ============================================================
# TEST RESULT
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
# CLEANUP
# ============================================================

def cleanup_test_medicine(
    medicine_id: int | None,
) -> None:

    if medicine_id is None:

        return

    execute(
        """
        DELETE FROM medicines
        WHERE id = %s
        """,
        (
            medicine_id,
        ),
    )


# ============================================================
# TEST SUITE
# ============================================================

def test_medicine_service() -> None:

    print(
        "\n"
        "MemorAi Medicine Service Test Suite\n"
    )

    initialize_database()

    service = (
        MedicineService()
    )

    medicine_id: int | None = None

    unique_suffix = (
        uuid4()
        .hex[:8]
    )

    medicine_name = (
        "MemorAi Service Test "
        f"{unique_suffix}"
    )

    try:

        # ====================================================
        # REGISTER
        # ====================================================

        medicine_id = (
            service.register_medicine(
                medicine_name=medicine_name,
                dosage="500 mg",
                schedule_time=time(
                    8,
                    0,
                ),
                schedule_label="Morning",
                instructions="After breakfast",
            )
        )

        assert (
            medicine_id
            > 0
        )

        print_test_result(
            "Register medicine",
            True,
        )

        # ====================================================
        # DUPLICATE PROTECTION
        # ====================================================

        duplicate_blocked = False

        try:

            service.register_medicine(
                medicine_name=medicine_name,
                dosage="500 mg",
                schedule_time=time(
                    8,
                    0,
                ),
            )

        except ValueError:

            duplicate_blocked = True

        assert (
            duplicate_blocked
        )

        print_test_result(
            "Duplicate medicine protection",
            True,
        )

        # ====================================================
        # ADD SECOND SCHEDULE
        # ====================================================

        second_schedule_id = (
            service.add_schedule(
                medicine_id=medicine_id,
                schedule_time=time(
                    20,
                    0,
                ),
                schedule_label="Evening",
            )
        )

        assert (
            second_schedule_id
            > 0
        )

        print_test_result(
            "Add second schedule",
            True,
        )

        # ====================================================
        # HIGH CONFIDENCE
        # ====================================================

        status = (
            service.determine_camera_status(
                confidence=0.90,
                medicine_similarity=0.92,
                person_similarity=0.80,
                action_confidence=0.88,
            )
        )

        assert (
            status
            == STATUS_CAMERA_CONFIRMED
        )

        print_test_result(
            "High-confidence camera status",
            True,
        )

        # ====================================================
        # LOW MEDICINE SIMILARITY
        # ====================================================

        status = (
            service.determine_camera_status(
                confidence=0.90,
                medicine_similarity=0.40,
                person_similarity=0.80,
                action_confidence=0.88,
            )
        )

        assert (
            status
            == STATUS_LOW_CONFIDENCE
        )

        print_test_result(
            "Reject weak medicine match",
            True,
        )

        # ====================================================
        # LOW ACTION CONFIDENCE
        # ====================================================

        status = (
            service.determine_camera_status(
                confidence=0.90,
                medicine_similarity=0.90,
                person_similarity=0.80,
                action_confidence=0.30,
            )
        )

        assert (
            status
            == STATUS_LOW_CONFIDENCE
        )

        print_test_result(
            "Reject weak medication action",
            True,
        )

        # ====================================================
        # RECORD CAMERA EVENT
        # ====================================================

        observed_at = (
            datetime.now()
        )

        event_id = (
            service.record_camera_observation(
                medicine_id=medicine_id,
                observed_at=observed_at,
                confidence=0.93,
                medicine_similarity=0.95,
                action_confidence=0.91,
                evidence_image_path=(
                    "evidence/"
                    "medicine_service_test.jpg"
                ),
            )
        )

        assert (
            event_id
            > 0
        )

        print_test_result(
            "Record camera observation",
            True,
        )

        # ====================================================
        # CONFIRMED TODAY
        # ====================================================

        confirmed_today = (
            service.was_medicine_confirmed_today(
                medicine_id
            )
        )

        assert (
            confirmed_today
        )

        print_test_result(
            "Medicine confirmed today",
            True,
        )

        # ====================================================
        # DATE HISTORY
        # ====================================================

        today_events = (
            get_medicine_events_for_date(
                target_date=date.today(),
                medicine_id=medicine_id,
                confirmed_only=True,
            )
        )

        assert (
            len(
                today_events
            )
            >= 1
        )

        print_test_result(
            "Today medicine history",
            True,
        )

        # ====================================================
        # LAST CONFIRMATION
        # ====================================================

        latest = (
            service.get_last_confirmation(
                medicine_id
            )
        )

        assert (
            latest
            is not None
        )

        assert (
            latest[
                "status"
            ]
            == STATUS_CAMERA_CONFIRMED
        )

        print_test_result(
            "Last medicine confirmation",
            True,
        )

        # ====================================================
        # SAFE TODAY ANSWER
        # ====================================================

        answer = (
            service.build_today_confirmation_answer(
                medicine_name
            )
        )

        assert (
            "camera-confirmed"
            in answer.lower()
        )

        assert (
            medicine_name.lower()
            in answer.lower()
        )

        # The service must not overclaim swallowing.

        assert (
            "definitely took"
            not in answer.lower()
        )

        assert (
            "swallowed"
            not in answer.lower()
        )

        print_test_result(
            "Safe today confirmation answer",
            True,
        )

        # ====================================================
        # SAFE LAST ANSWER
        # ====================================================

        answer = (
            service.build_last_confirmation_answer(
                medicine_name
            )
        )

        assert (
            "camera-confirmed"
            in answer.lower()
        )

        assert (
            "swallowed"
            not in answer.lower()
        )

        print_test_result(
            "Safe last confirmation answer",
            True,
        )

        # ====================================================
        # ACTIVE SCHEDULE LIST
        # ====================================================

        todays_schedule = (
            service.get_todays_schedule()
        )

        matching = [
            item
            for item
            in todays_schedule
            if int(
                item[
                    "medicine_id"
                ]
            )
            == medicine_id
        ]

        assert (
            len(
                matching
            )
            >= 2
        )

        print_test_result(
            "Today's medicine schedules",
            True,
        )

        # ====================================================
        # TODAY STATUS
        # ====================================================

        today_status = (
            service.get_todays_medicine_status()
        )

        matching_status = [
            item
            for item
            in today_status
            if int(
                item[
                    "medicine_id"
                ]
            )
            == medicine_id
        ]

        assert (
            len(
                matching_status
            )
            >= 2
        )

        assert all(
            item[
                "confirmation_status"
            ]
            == STATUS_CAMERA_CONFIRMED
            for item
            in matching_status
        )

        print_test_result(
            "Today's medicine status",
            True,
        )

        # ====================================================
        # DEACTIVATION
        # ====================================================

        service.deactivate(
            medicine_id
        )

        medicine = (
            get_medicine_by_id(
                medicine_id
            )
        )

        assert (
            medicine
            is not None
        )

        assert not bool(
            medicine[
                "is_active"
            ]
        )

        print_test_result(
            "Deactivate medicine",
            True,
        )

        # ====================================================
        # INACTIVE MEDICINE CANNOT RECEIVE CAMERA CONFIRMATION
        # ====================================================

        blocked = False

        try:

            service.record_camera_observation(
                medicine_id=medicine_id,
                observed_at=datetime.now(),
                confidence=0.90,
                medicine_similarity=0.90,
                action_confidence=0.90,
            )

        except ValueError:

            blocked = True

        assert (
            blocked
        )

        print_test_result(
            "Block inactive medicine confirmation",
            True,
        )

        # ====================================================
        # REACTIVATE
        # ====================================================

        service.reactivate(
            medicine_id
        )

        medicine = (
            get_medicine_by_id(
                medicine_id
            )
        )

        assert (
            medicine
            is not None
        )

        assert bool(
            medicine[
                "is_active"
            ]
        )

        print_test_result(
            "Reactivate medicine",
            True,
        )

        print(
            "\n"
            "All medicine service tests passed."
            "\n"
        )

    finally:

        cleanup_test_medicine(
            medicine_id
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    test_medicine_service()