from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import cv2

from database.person_repository import (
    deactivate_person,
    get_latest_person_event,
)
from face.face_recognition import (
    FaceRecognitionEngine,
)
from face.person_monitor import (
    PersonMonitor,
)
from face.person_registration import (
    PersonRegistrationService,
)


# ============================================================
# TEST PATHS
# ============================================================

TEST_DIR = (
    Path("data")
    / "people"
    / "registration_test"
)

REGISTRATION_IMAGES = [
    TEST_DIR / "sample1.jpeg",
    TEST_DIR / "sample2.jpeg",
    TEST_DIR / "sample3.jpeg",
    TEST_DIR / "sample4.jpeg",
    TEST_DIR / "sample5.jpeg",
]

LIVE_FRAME_IMAGE = (
    TEST_DIR
    / "sample6.jpeg"
)


# ============================================================
# HELPERS
# ============================================================

def print_pass(
    name: str,
) -> None:

    print(
        f"PASS: {name}"
    )


def load_image(
    path: Path,
):

    image = cv2.imread(
        str(
            path
        )
    )

    if image is None:

        raise RuntimeError(
            f"Could not load: {path}"
        )

    return image


def validate_files() -> None:

    required = (
        REGISTRATION_IMAGES
        + [
            LIVE_FRAME_IMAGE,
        ]
    )

    missing = [
        path
        for path in required
        if not path.exists()
    ]

    if missing:

        raise FileNotFoundError(
            "\n".join(
                str(path)
                for path in missing
            )
        )

    print_pass(
        "Person monitor test images available"
    )


# ============================================================
# FULL TEST
# ============================================================

def test_person_monitor() -> None:

    print()
    print(
        "MemorAi Person Monitor Test Suite"
    )

    print(
        "=" * 50
    )

    validate_files()

    # --------------------------------------------------------
    # FACE ENGINE
    # --------------------------------------------------------

    engine = (
        FaceRecognitionEngine()
    )

    print_pass(
        "Face engine initialization"
    )

    # --------------------------------------------------------
    # TEMPORARY REGISTERED PERSON
    # --------------------------------------------------------

    suffix = (
        uuid4().hex[:8]
    )

    internal_name = (
        f"monitor_test_{suffix}"
    )

    display_name = (
        "Monitor Test Person"
    )

    registration_service = (
        PersonRegistrationService(
            face_engine=engine,
            required_samples=5,
        )
    )

    registration_result = (
        registration_service.register_person(
            name=internal_name,
            display_name=display_name,
            sample_paths=(
                REGISTRATION_IMAGES
            ),
            notes=(
                "Temporary person-monitor test identity."
            ),
        )
    )

    person_id = (
        registration_result.person_id
    )

    print_pass(
        "Temporary monitor identity registered"
    )

    # --------------------------------------------------------
    # MONITOR
    # --------------------------------------------------------

    monitor = (
        PersonMonitor(
            face_engine=engine,
            confirmation_frames=3,
            cooldown_seconds=120,
        )
    )

    print_pass(
        "PersonMonitor initialization"
    )

    frame = load_image(
        LIVE_FRAME_IMAGE
    )

    try:

        # ----------------------------------------------------
        # FRAME 1
        # ----------------------------------------------------

        observations_1, events_1 = (
            monitor.process_frame(
                frame
            )
        )

        assert observations_1

        assert not events_1

        print_pass(
            "Frame 1 recognized but not confirmed"
        )

        # ----------------------------------------------------
        # FRAME 2
        # ----------------------------------------------------

        observations_2, events_2 = (
            monitor.process_frame(
                frame
            )
        )

        assert observations_2

        assert not events_2

        print_pass(
            "Frame 2 recognized but not confirmed"
        )

        # ----------------------------------------------------
        # FRAME 3
        # ----------------------------------------------------

        observations_3, events_3 = (
            monitor.process_frame(
                frame
            )
        )

        assert observations_3

        assert len(
            events_3
        ) == 1

        encounter = (
            events_3[0]
        )

        assert (
            encounter.person_id
            == person_id
        )

        assert (
            encounter.display_name
            == display_name
        )

        assert (
            encounter.similarity
            >= 0.50
        )

        assert Path(
            encounter.image_path
        ).exists()

        assert (
            encounter.event_id
            > 0
        )

        print(
            f"Confirmed person: "
            f"{encounter.display_name}"
        )

        print(
            f"Similarity: "
            f"{encounter.similarity:.6f}"
        )

        print(
            f"Event ID: "
            f"{encounter.event_id}"
        )

        print(
            f"Evidence: "
            f"{encounter.image_path}"
        )

        print_pass(
            "Frame 3 confirmed and stored encounter"
        )

        # ----------------------------------------------------
        # DATABASE CHECK
        # ----------------------------------------------------

        latest = (
            get_latest_person_event(
                person_id
            )
        )

        assert latest is not None

        assert int(
            latest[
                "person_id"
            ]
        ) == person_id

        assert (
            latest[
                "person_name"
            ]
            == display_name
        )

        print_pass(
            "Person event stored in MySQL"
        )

        # ----------------------------------------------------
        # COOLDOWN CHECK
        # ----------------------------------------------------

        _, cooldown_events_1 = (
            monitor.process_frame(
                frame
            )
        )

        _, cooldown_events_2 = (
            monitor.process_frame(
                frame
            )
        )

        _, cooldown_events_3 = (
            monitor.process_frame(
                frame
            )
        )

        assert not (
            cooldown_events_1
            or cooldown_events_2
            or cooldown_events_3
        )

        print_pass(
            "Cooldown prevents duplicate encounter"
        )

    finally:

        # ----------------------------------------------------
        # DEACTIVATE TEST IDENTITY
        # ----------------------------------------------------

        deactivate_person(
            person_id
        )

        print_pass(
            "Temporary monitor identity deactivated"
        )

    print()
    print(
        "All MemorAi person monitor tests passed."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    test_person_monitor()