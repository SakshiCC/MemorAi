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
    add_medicine_reference_image,
    add_medicine_schedule,
    create_medicine,
    deactivate_medicine,
    deactivate_medicine_schedule,
    get_all_medicines,
    get_latest_medicine_event,
    get_medicine_by_id,
    get_medicine_by_name,
    get_medicine_events_for_date,
    get_medicine_reference_images,
    get_medicine_schedules,
    reactivate_medicine,
    reactivate_medicine_schedule,
    save_medicine_event,
    set_primary_reference_image,
    update_medicine,
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

    # All dependent medicine records use
    # ON DELETE CASCADE.

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

def test_medicine_repository() -> None:

    print(
        "\n"
        "MemorAi Medicine Repository Test Suite\n"
    )

    initialize_database()

    medicine_id: int | None = None

    unique_suffix = (
        uuid4()
        .hex[:8]
    )

    medicine_name = (
        "MemorAi Test Medicine "
        f"{unique_suffix}"
    )

    try:

        # ====================================================
        # CREATE
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
                instructions="Test instruction",
            )
        )

        assert (
            medicine_id
            > 0
        )

        print_test_result(
            "Create medicine",
            True,
        )

        # ====================================================
        # GET BY ID
        # ====================================================

        medicine = (
            get_medicine_by_id(
                medicine_id
            )
        )

        assert (
            medicine
            is not None
        )

        assert (
            medicine[
                "medicine_name"
            ]
            == medicine_name
        )

        assert (
            medicine[
                "dosage"
            ]
            == "500 mg"
        )

        print_test_result(
            "Get medicine by ID",
            True,
        )

        # ====================================================
        # GET BY NAME
        # ====================================================

        medicine_by_name = (
            get_medicine_by_name(
                medicine_name
            )
        )

        assert (
            medicine_by_name
            is not None
        )

        assert (
            int(
                medicine_by_name[
                    "id"
                ]
            )
            == medicine_id
        )

        print_test_result(
            "Get medicine by name",
            True,
        )

        # ====================================================
        # PRIMARY SCHEDULE CREATED AUTOMATICALLY
        # ====================================================

        schedules = (
            get_medicine_schedules(
                medicine_id,
                active_only=False,
            )
        )

        assert (
            len(
                schedules
            )
            >= 1
        )

        print_test_result(
            "Primary medicine schedule",
            True,
        )

        # ====================================================
        # SECOND SCHEDULE
        # ====================================================

        evening_schedule_id = (
            add_medicine_schedule(
                medicine_id=medicine_id,
                schedule_time=time(
                    20,
                    0,
                ),
                schedule_label="Evening",
            )
        )

        assert (
            evening_schedule_id
            > 0
        )

        schedules = (
            get_medicine_schedules(
                medicine_id,
                active_only=True,
            )
        )

        assert (
            len(
                schedules
            )
            >= 2
        )

        print_test_result(
            "Add multiple schedules",
            True,
        )

        # ====================================================
        # SCHEDULE ACTIVE / INACTIVE
        # ====================================================

        deactivate_medicine_schedule(
            evening_schedule_id
        )

        active_schedules = (
            get_medicine_schedules(
                medicine_id,
                active_only=True,
            )
        )

        assert all(
            int(
                schedule[
                    "id"
                ]
            )
            != evening_schedule_id
            for schedule
            in active_schedules
        )

        reactivate_medicine_schedule(
            evening_schedule_id
        )

        active_schedules = (
            get_medicine_schedules(
                medicine_id,
                active_only=True,
            )
        )

        assert any(
            int(
                schedule[
                    "id"
                ]
            )
            == evening_schedule_id
            for schedule
            in active_schedules
        )

        print_test_result(
            "Schedule activate/deactivate",
            True,
        )

        # ====================================================
        # UPDATE
        # ====================================================

        update_medicine(
            medicine_id=medicine_id,
            dosage="750 mg",
            instructions=(
                "Updated test instruction"
            ),
        )

        updated = (
            get_medicine_by_id(
                medicine_id
            )
        )

        assert (
            updated
            is not None
        )

        assert (
            updated[
                "dosage"
            ]
            == "750 mg"
        )

        print_test_result(
            "Update medicine",
            True,
        )

        # ====================================================
        # REFERENCE IMAGES
        # ====================================================

        image_1 = (
            add_medicine_reference_image(
                medicine_id=medicine_id,
                image_path=(
                    "test_data/"
                    "medicine_sample_01.jpg"
                ),
                embedding=b"test_embedding_1",
                embedding_model=(
                    "test_model"
                ),
                is_primary=False,
            )
        )

        image_2 = (
            add_medicine_reference_image(
                medicine_id=medicine_id,
                image_path=(
                    "test_data/"
                    "medicine_sample_02.jpg"
                ),
                embedding=b"test_embedding_2",
                embedding_model=(
                    "test_model"
                ),
                is_primary=False,
            )
        )

        assert (
            image_1
            > 0
        )

        assert (
            image_2
            > 0
        )

        set_primary_reference_image(
            medicine_id=medicine_id,
            reference_image_id=image_2,
        )

        images = (
            get_medicine_reference_images(
                medicine_id
            )
        )

        assert (
            len(
                images
            )
            == 2
        )

        primary_images = [
            row
            for row
            in images
            if bool(
                row[
                    "is_primary"
                ]
            )
        ]

        assert (
            len(
                primary_images
            )
            == 1
        )

        assert (
            int(
                primary_images[
                    0
                ][
                    "id"
                ]
            )
            == image_2
        )

        print_test_result(
            "Medicine reference images",
            True,
        )

        # ====================================================
        # SAVE CAMERA CONFIRMATION
        # ====================================================

        observed_at = (
            datetime.now()
        )

        event_id = (
            save_medicine_event(
                medicine_id=medicine_id,
                observed_at=observed_at,
                status=(
                    STATUS_CAMERA_CONFIRMED
                ),
                confidence=0.92,
                medicine_similarity=0.94,
                action_confidence=0.88,
                evidence_image_path=(
                    "evidence/test_medicine.jpg"
                ),
            )
        )

        assert (
            event_id
            > 0
        )

        print_test_result(
            "Save medicine event",
            True,
        )

        # ====================================================
        # LATEST EVENT
        # ====================================================

        latest = (
            get_latest_medicine_event(
                medicine_id,
                confirmed_only=True,
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

        assert (
            latest[
                "medicine_name"
            ]
            == medicine_name
        )

        print_test_result(
            "Latest medicine event",
            True,
        )

        # ====================================================
        # DATE HISTORY
        # ====================================================

        events_today = (
            get_medicine_events_for_date(
                target_date=date.today(),
                medicine_id=medicine_id,
                confirmed_only=True,
            )
        )

        assert (
            len(
                events_today
            )
            >= 1
        )

        print_test_result(
            "Medicine events by date",
            True,
        )

        # ====================================================
        # ACTIVE / INACTIVE
        # ====================================================

        deactivate_medicine(
            medicine_id
        )

        inactive = (
            get_medicine_by_id(
                medicine_id
            )
        )

        assert (
            inactive
            is not None
        )

        assert not bool(
            inactive[
                "is_active"
            ]
        )

        reactivate_medicine(
            medicine_id
        )

        active = (
            get_medicine_by_id(
                medicine_id
            )
        )

        assert (
            active
            is not None
        )

        assert bool(
            active[
                "is_active"
            ]
        )

        print_test_result(
            "Medicine activate/deactivate",
            True,
        )

        # ====================================================
        # LIST MEDICINES
        # ====================================================

        medicines = (
            get_all_medicines(
                active_only=False
            )
        )

        assert any(
            int(
                item[
                    "id"
                ]
            )
            == medicine_id
            for item
            in medicines
        )

        print_test_result(
            "List medicines",
            True,
        )

        print(
            "\n"
            "All medicine repository tests passed."
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

    test_medicine_repository()