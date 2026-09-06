from __future__ import annotations

import sys
import time

from pathlib import Path

import cv2
import numpy as np

from database.medicine_repository import (
    get_all_medicine_reference_images,
    get_all_medicines,
)

from database.person_repository import (
    get_all_people,
)

from app.services.camera_service import (
    CameraService,
    CameraState,
)


# ============================================================
# CONFIG
# ============================================================

TEST_TIMEOUT_SECONDS = 120

WINDOW_NAME = (
    "MemorAi Live Medicine Camera Test"
)


# ============================================================
# SEPARATOR
# ============================================================

def separator() -> None:

    print(
        "\n"
        + "=" * 70
    )


# ============================================================
# PRECHECK
# ============================================================

def precheck() -> bool:

    separator()

    print(
        "MemorAi Live Medicine Camera Test"
    )

    separator()

    medicines = (
        get_all_medicines(
            active_only=True
        )
    )

    references = (
        get_all_medicine_reference_images(
            active_medicines_only=True
        )
    )

    people = (
        get_all_people(
            active_only=True
        )
    )

    valid_references = [
        row
        for row in references
        if row.get(
            "embedding"
        )
        is not None
    ]

    print(
        "\nActive registered people:",
        len(
            people
        ),
    )

    print(
        "Active registered medicines:",
        len(
            medicines
        ),
    )

    print(
        "Medicine visual references:",
        len(
            valid_references
        ),
    )

    if not people:

        print(
            "\nFAIL: No active person registered."
        )

        return False

    if not medicines:

        print(
            "\nFAIL: No active medicine registered."
        )

        return False

    if not valid_references:

        print(
            "\nFAIL: No medicine visual references."
        )

        return False

    print(
        "\nRegistered medicines:"
    )

    for medicine in medicines:

        print(
            "  -",
            medicine.get(
                "medicine_name"
            ),
            medicine.get(
                "dosage"
            )
            or "",
        )

    print(
        "\nRegistered people:"
    )

    for person in people:

        print(
            "  -",
            (
                person.get(
                    "display_name"
                )
                or person.get(
                    "name"
                )
            ),
        )

    return True


# ============================================================
# DRAW TEXT
# ============================================================

def draw_text(
    frame: np.ndarray,
    text: str,
    position: tuple[
        int,
        int,
    ],
    scale: float = 0.55,
    thickness: int = 2,
) -> None:

    x, y = (
        position
    )

    cv2.putText(
        frame,
        text,
        (
            x + 1,
            y + 1,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        thickness + 2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        text,
        (
            x,
            y,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


# ============================================================
# DRAW MEDICINE REGION
# ============================================================

def draw_medicine_scan_region(
    frame: np.ndarray,
) -> None:

    height, width = (
        frame.shape[:2]
    )

    (
        x1,
        y1,
        x2,
        y2,
    ) = (
        CameraService
        .get_medicine_scan_bounds(
            frame_width=width,
            frame_height=height,
        )
    )

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (255, 255, 255),
        2,
    )

    draw_text(
        frame,
        (
            "MEDICINE ONLY - "
            "FILL MOST OF THIS BOX"
        ),
        (
            x1,
            max(
                25,
                y1 - 10,
            ),
        ),
        scale=0.48,
    )


# ============================================================
# LIVE STATUS
# ============================================================

def draw_live_status(
    frame: np.ndarray,
    camera: CameraService,
    elapsed_seconds: float,
) -> np.ndarray:

    display = (
        frame.copy()
    )

    status = (
        camera.get_medicine_live_status()
    )

    draw_medicine_scan_region(
        display
    )

    y = 28

    draw_text(
        display,
        "MemorAi Medicine Test",
        (15, y),
        scale=0.65,
    )

    y += 30

    draw_text(
        display,
        (
            f"Time: {elapsed_seconds:.1f}s "
            f"/ {TEST_TIMEOUT_SECONDS}s"
        ),
        (15, y),
    )

    y += 25

    draw_text(
        display,
        (
            "Phase: "
            f"{status.phase}"
        ),
        (15, y),
    )

    y += 25

    # --------------------------------------------------------
    # PERSON
    # --------------------------------------------------------

    if status.person_name:

        person_text = (
            "Person: "
            f"{status.person_name}"
        )

        if (
            status.person_similarity
            is not None
        ):

            person_text += (
                "  "
                f"{status.person_similarity:.3f}"
            )

    else:

        person_text = (
            "Person: waiting..."
        )

    draw_text(
        display,
        person_text,
        (15, y),
    )

    y += 25

    # --------------------------------------------------------
    # BEST CANDIDATE
    # --------------------------------------------------------

    if status.medicine_name:

        candidate_text = (
            "Best candidate: "
            f"{status.medicine_name}"
        )

    else:

        candidate_text = (
            "Best candidate: none"
        )

    draw_text(
        display,
        candidate_text,
        (15, y),
    )

    y += 25

    # --------------------------------------------------------
    # SIMILARITY
    # --------------------------------------------------------

    if (
        status.medicine_similarity
        is not None
    ):

        similarity_text = (
            "Similarity: "
            f"{status.medicine_similarity:.3f}"
        )

    else:

        similarity_text = (
            "Similarity: --"
        )

    draw_text(
        display,
        similarity_text,
        (15, y),
    )

    y += 25

    # --------------------------------------------------------
    # THRESHOLD
    # --------------------------------------------------------

    if (
        status.medicine_threshold
        is not None
    ):

        threshold_text = (
            "Required: "
            f"{status.medicine_threshold:.3f}"
        )

    else:

        threshold_text = (
            "Required: --"
        )

    draw_text(
        display,
        threshold_text,
        (15, y),
    )

    y += 25

    # --------------------------------------------------------
    # ACCEPTED
    # --------------------------------------------------------

    accepted_text = (
        "Match: "
        + (
            "YES"
            if status.medicine_is_known
            else "NO"
        )
    )

    draw_text(
        display,
        accepted_text,
        (15, y),
    )

    y += 25

    # --------------------------------------------------------
    # CONFIRMATION FRAMES
    # --------------------------------------------------------

    if (
        status.confirmation_required
        > 0
    ):

        frames_text = (
            "Confirmation: "
            f"{status.confirmation_count}"
            f"/{status.confirmation_required}"
        )

    else:

        frames_text = (
            "Confirmation: 0/0"
        )

    draw_text(
        display,
        frames_text,
        (15, y),
    )

    # ========================================================
    # BOTTOM MESSAGE
    # ========================================================

    height, _width = (
        display.shape[:2]
    )

    bottom_y = (
        height - 75
    )

    if status.phase == "waiting_for_person":

        instruction = (
            "LOOK AT THE CAMERA"
        )

    elif (
        status.phase
        == "scanning_medicine"
    ):

        if status.medicine_is_known:

            instruction = (
                "MATCH FOUND - KEEP PACKAGE STILL"
            )

        else:

            instruction = (
                "FILL THE MEDICINE BOX "
                "WITH THE REGISTERED PACKAGE"
            )

    elif (
        status.phase
        == "watching_action"
    ):

        instruction = (
            "MEDICINE CONFIRMED - "
            "PERFORM HAND TO MOUTH THEN MOVE AWAY"
        )

    elif (
        status.phase
        == "event_saved"
    ):

        instruction = (
            "CAMERA-CONFIRMED EVENT SAVED"
        )

    else:

        instruction = (
            status.message
        )

    draw_text(
        display,
        instruction,
        (
            15,
            bottom_y,
        ),
        scale=0.55,
    )

    draw_text(
        display,
        "Press Q to stop",
        (
            15,
            bottom_y + 30,
        ),
        scale=0.48,
    )

    return display


# ============================================================
# PRINT STATUS
# ============================================================

def print_status_if_changed(
    camera: CameraService,
    previous_signature,
):

    status = (
        camera.get_medicine_live_status()
    )

    signature = (
        status.phase,
        status.message,
        status.medicine_name,
        (
            round(
                status.medicine_similarity,
                3,
            )
            if status.medicine_similarity
            is not None
            else None
        ),
        status.medicine_is_known,
        status.confirmation_count,
    )

    if signature == previous_signature:

        return previous_signature

    print(
        "[Medicine]"
    )

    print(
        "  Phase:",
        status.phase,
    )

    print(
        "  Message:",
        status.message,
    )

    if status.person_name:

        print(
            "  Person:",
            status.person_name,
        )

    if (
        status.person_similarity
        is not None
    ):

        print(
            (
                "  Person similarity: "
                f"{status.person_similarity:.4f}"
            )
        )

    if status.medicine_name:

        print(
            "  Best candidate:",
            status.medicine_name,
        )

    if (
        status.medicine_similarity
        is not None
    ):

        print(
            (
                "  Similarity: "
                f"{status.medicine_similarity:.4f}"
            )
        )

    if (
        status.medicine_threshold
        is not None
    ):

        print(
            (
                "  Required threshold: "
                f"{status.medicine_threshold:.4f}"
            )
        )

    print(
        "  Accepted:",
        status.medicine_is_known,
    )

    print(
        (
            "  Confirmation: "
            f"{status.confirmation_count}"
            f"/{status.confirmation_required}"
        )
    )

    print()

    return signature


# ============================================================
# SUCCESS
# ============================================================

def print_success(
    camera: CameraService,
) -> None:

    result = (
        camera.get_latest_medicine()
    )

    assert result is not None

    separator()

    print(
        "PASS: LIVE MEDICINE EVENT CREATED"
    )

    separator()

    print(
        "Event ID:",
        result.event_id,
    )

    print(
        "Medicine:",
        result.medicine_name,
    )

    print(
        "Dosage:",
        result.dosage
        or "—",
    )

    print(
        "Person:",
        result.person_name,
    )

    print(
        (
            "Medicine similarity: "
            f"{result.medicine_similarity:.4f}"
        )
    )

    print(
        (
            "Person similarity: "
            f"{result.person_similarity:.4f}"
        )
    )

    print(
        (
            "Action confidence: "
            f"{result.action_confidence:.4f}"
        )
    )

    print(
        (
            "Overall confidence: "
            f"{result.overall_confidence:.4f}"
        )
    )

    print(
        "Status:",
        result.status,
    )

    print(
        "Evidence:",
        result.image_path,
    )

    assert (
        result.event_id > 0
    )

    assert (
        Path(
            result.image_path
        ).exists()
    )

    separator()

    print(
        "Live Medicine Camera test passed."
    )


# ============================================================
# TEST
# ============================================================

def test_live_medicine_camera() -> None:

    if not precheck():

        sys.exit(
            1
        )

    separator()

    print(
        "Starting webcam..."
    )

    print()

    print(
        (
            "Keep your face visible. "
            "Place the registered medicine "
            "inside the smaller medicine box."
        )
    )

    print(
        (
            "The package should fill most of the box."
        )
    )

    print(
        (
            "Watch BEST CANDIDATE, SIMILARITY "
            "and REQUIRED threshold."
        )
    )

    print(
        (
            "Only perform the hand-to-mouth action "
            "after Phase becomes watching_action."
        )
    )

    separator()

    camera = CameraService(
        capture_fps=30,

        stream_fps=10,

        face_inference_fps=4,

        medicine_scan_fps=2,

        medicine_action_fps=8,

        enable_object_monitoring=False,

        enable_person_monitoring=True,

        enable_medicine_monitoring=True,
    )

    initial_version = (
        camera.get_medicine_version()
    )

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL,
    )

    cv2.resizeWindow(
        WINDOW_NAME,
        960,
        720,
    )

    try:

        camera.start()

        if (
            camera.state
            == CameraState.ERROR
        ):

            raise RuntimeError(
                camera.last_error
                or
                "Camera failed to start."
            )

        started = (
            time.monotonic()
        )

        previous_signature = None

        while (
            time.monotonic()
            - started
            < TEST_TIMEOUT_SECONDS
        ):

            elapsed = (
                time.monotonic()
                - started
            )

            frame = (
                camera.get_latest_frame_copy()
            )

            if frame is not None:

                display = (
                    draw_live_status(
                        frame=frame,
                        camera=camera,
                        elapsed_seconds=elapsed,
                    )
                )

                cv2.imshow(
                    WINDOW_NAME,
                    display,
                )

            key = (
                cv2.waitKey(
                    1
                )
                & 0xFF
            )

            if key in (
                ord(
                    "q"
                ),
                ord(
                    "Q"
                ),
            ):

                print(
                    "\nTest stopped by user."
                )

                return

            previous_signature = (
                print_status_if_changed(
                    camera=(
                        camera
                    ),
                    previous_signature=(
                        previous_signature
                    ),
                )
            )

            if (
                camera.get_medicine_version()
                > initial_version
            ):

                print_success(
                    camera
                )

                return

            time.sleep(
                0.03
            )

        separator()

        print(
            "FAIL: Test timed out."
        )

        status = (
            camera.get_medicine_live_status()
        )

        print(
            "Final phase:",
            status.phase,
        )

        print(
            "Best candidate:",
            status.medicine_name,
        )

        print(
            "Similarity:",
            status.medicine_similarity,
        )

        print(
            "Required:",
            status.medicine_threshold,
        )

        print(
            "Accepted:",
            status.medicine_is_known,
        )

        print(
            (
                "Confirmation: "
                f"{status.confirmation_count}"
                f"/{status.confirmation_required}"
            )
        )

        if camera.last_face_error:

            print(
                "Face error:",
                camera.last_face_error,
            )

        if camera.last_medicine_error:

            print(
                "Medicine error:",
                camera.last_medicine_error,
            )

        raise AssertionError(
            "Live medicine confirmation timed out."
        )

    finally:

        print(
            "\nStopping webcam..."
        )

        camera.stop()

        cv2.destroyAllWindows()

        print(
            "Webcam stopped."
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    test_live_medicine_camera()