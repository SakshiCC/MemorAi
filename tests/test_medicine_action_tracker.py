from __future__ import annotations

from datetime import datetime

import numpy as np

from medicine.hand_landmarker import (
    HandObservation,
    HandPoint,
)

from medicine.medicine_action_tracker import (
    MedicationActionState,
    MedicationActionTracker,
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
# FAKE HAND DETECTOR
# ============================================================

class FakeHandDetector:
    """
    Return predefined hand focus points.

    Each item is:

        (x, y)

    in normalized image coordinates.

    None means no hand was detected.
    """

    def __init__(
        self,
        positions: list[
            tuple[
                float,
                float,
            ]
            | None
        ],
    ) -> None:

        self.positions = list(
            positions
        )

        self.index = 0

    def detect(
        self,
        frame: np.ndarray,
    ) -> list[
        HandObservation
    ]:

        if not self.positions:

            return []

        if (
            self.index
            >= len(
                self.positions
            )
        ):

            position = (
                self.positions[
                    -1
                ]
            )

        else:

            position = (
                self.positions[
                    self.index
                ]
            )

            self.index += 1

        if position is None:

            return []

        (
            x,
            y,
        ) = position

        landmarks = [
            HandPoint(
                x=x,
                y=y,
                z=0.0,
            )
            for _ in range(
                21
            )
        ]

        return [
            HandObservation(
                handedness="Right",
                handedness_score=0.99,
                landmarks=landmarks,
                focus_x=x,
                focus_y=y,
            )
        ]


# ============================================================
# TEST FRAME
# ============================================================

def test_frame() -> np.ndarray:

    return np.zeros(
        (
            480,
            640,
            3,
        ),
        dtype=np.uint8,
    )


# ============================================================
# FACE BBOX
# ============================================================

def test_face_bbox() -> tuple[
    int,
    int,
    int,
    int,
]:
    """
    Face:

        x = 220
        y = 80
        width = 200
        height = 240

    Approx mouth center becomes roughly:

        x = 320 / 640 = 0.50
        y = 252.8 / 480 ≈ 0.527
    """

    return (
        220,
        80,
        200,
        240,
    )


# ============================================================
# ARM
# ============================================================

def arm_tracker(
    tracker: MedicationActionTracker,
) -> None:

    tracker.arm(
        medicine_id=1,
        medicine_name="Metformin",
        dosage="500 mg",
        medicine_similarity=0.91,
        person_id=10,
        person_name="Test Person",
        person_similarity=0.82,
        armed_at=datetime.now(),
    )


# ============================================================
# TEST 1
# INITIAL STATE
# ============================================================

def test_arm_state() -> None:

    detector = (
        FakeHandDetector(
            [
                (
                    0.10,
                    0.10,
                )
            ]
        )
    )

    tracker = (
        MedicationActionTracker(
            hand_detector=detector,  # type: ignore[arg-type]
            required_near_frames=2,
            required_retreat_frames=2,
        )
    )

    assert (
        tracker.state
        == MedicationActionState.IDLE
    )

    arm_tracker(
        tracker
    )

    assert (
        tracker.state
        == MedicationActionState
        .WAITING_FOR_APPROACH
    )

    assert (
        tracker.is_active
    )

    print_test_result(
        "Tracker arm state",
        True,
    )


# ============================================================
# TEST 2
# HAND FAR DOES NOT CONFIRM
# ============================================================

def test_far_hand_does_not_confirm() -> None:

    detector = (
        FakeHandDetector(
            [
                (
                    0.10,
                    0.10,
                ),
                (
                    0.12,
                    0.12,
                ),
                (
                    0.15,
                    0.15,
                ),
            ]
        )
    )

    tracker = (
        MedicationActionTracker(
            hand_detector=detector,  # type: ignore[arg-type]
            required_near_frames=2,
            required_retreat_frames=2,
        )
    )

    arm_tracker(
        tracker
    )

    confirmations = []

    for _ in range(
        3
    ):

        _observation, confirmed = (
            tracker.process_frame(
                frame=test_frame(),
                face_bbox=test_face_bbox(),
            )
        )

        confirmations.append(
            confirmed
        )

    assert all(
        item is None
        for item
        in confirmations
    )

    assert (
        tracker.state
        == MedicationActionState
        .WAITING_FOR_APPROACH
    )

    print_test_result(
        "Far hand does not confirm action",
        True,
    )


# ============================================================
# TEST 3
# SINGLE NEAR FRAME NOT ENOUGH
# ============================================================

def test_single_near_frame_not_enough() -> None:

    detector = (
        FakeHandDetector(
            [
                (
                    0.50,
                    0.53,
                ),
                (
                    0.10,
                    0.10,
                ),
            ]
        )
    )

    tracker = (
        MedicationActionTracker(
            hand_detector=detector,  # type: ignore[arg-type]
            required_near_frames=2,
            required_retreat_frames=2,
        )
    )

    arm_tracker(
        tracker
    )

    tracker.process_frame(
        test_frame(),
        test_face_bbox(),
    )

    _observation, confirmed = (
        tracker.process_frame(
            test_frame(),
            test_face_bbox(),
        )
    )

    assert (
        confirmed is None
    )

    assert (
        tracker.state
        == MedicationActionState
        .WAITING_FOR_APPROACH
    )

    print_test_result(
        "Single near-mouth frame rejected",
        True,
    )


# ============================================================
# TEST 4
# APPROACH WITHOUT RETREAT DOES NOT CONFIRM
# ============================================================

def test_approach_without_retreat() -> None:

    detector = (
        FakeHandDetector(
            [
                (
                    0.50,
                    0.53,
                ),
                (
                    0.50,
                    0.52,
                ),
                (
                    0.51,
                    0.52,
                ),
                (
                    0.50,
                    0.53,
                ),
            ]
        )
    )

    tracker = (
        MedicationActionTracker(
            hand_detector=detector,  # type: ignore[arg-type]
            required_near_frames=2,
            required_retreat_frames=2,
        )
    )

    arm_tracker(
        tracker
    )

    confirmation = None

    for _ in range(
        4
    ):

        _observation, confirmation = (
            tracker.process_frame(
                test_frame(),
                test_face_bbox(),
            )
        )

    assert (
        confirmation is None
    )

    assert (
        tracker.state
        == MedicationActionState
        .WAITING_FOR_RETREAT
    )

    print_test_result(
        "Near-mouth action requires retreat",
        True,
    )


# ============================================================
# TEST 5
# COMPLETE ACTION
# ============================================================

def test_complete_action_confirmation() -> None:

    detector = (
        FakeHandDetector(
            [
                # --------------------------------------------
                # FAR
                # --------------------------------------------

                (
                    0.10,
                    0.10,
                ),

                # --------------------------------------------
                # APPROACH / NEAR
                # --------------------------------------------

                (
                    0.49,
                    0.53,
                ),

                (
                    0.50,
                    0.52,
                ),

                # --------------------------------------------
                # RETREAT
                # --------------------------------------------

                (
                    0.10,
                    0.10,
                ),

                (
                    0.10,
                    0.12,
                ),
            ]
        )
    )

    tracker = (
        MedicationActionTracker(
            hand_detector=detector,  # type: ignore[arg-type]
            required_near_frames=2,
            required_retreat_frames=2,
            action_window_seconds=20.0,
        )
    )

    arm_tracker(
        tracker
    )

    confirmed_action = None

    for _ in range(
        5
    ):

        _observation, confirmed = (
            tracker.process_frame(
                test_frame(),
                test_face_bbox(),
            )
        )

        if confirmed is not None:

            confirmed_action = (
                confirmed
            )

    assert (
        confirmed_action
        is not None
    )

    assert (
        confirmed_action.medicine_id
        == 1
    )

    assert (
        confirmed_action.medicine_name
        == "Metformin"
    )

    assert (
        confirmed_action.person_id
        == 10
    )

    assert (
        0.0
        <= confirmed_action.action_confidence
        <= 1.0
    )

    assert (
        confirmed_action.action_confidence
        > 0.40
    )

    assert (
        tracker.state
        == MedicationActionState.CONFIRMED
    )

    print_test_result(
        "Complete medication gesture sequence",
        True,
    )


# ============================================================
# TEST 6
# HAND DISAPPEARING IS NOT RETREAT
# ============================================================

def test_missing_hand_not_retreat() -> None:

    detector = (
        FakeHandDetector(
            [
                (
                    0.50,
                    0.53,
                ),
                (
                    0.50,
                    0.52,
                ),
                None,
                None,
            ]
        )
    )

    tracker = (
        MedicationActionTracker(
            hand_detector=detector,  # type: ignore[arg-type]
            required_near_frames=2,
            required_retreat_frames=2,
        )
    )

    arm_tracker(
        tracker
    )

    confirmation = None

    for _ in range(
        4
    ):

        _observation, confirmation = (
            tracker.process_frame(
                test_frame(),
                test_face_bbox(),
            )
        )

    assert (
        confirmation is None
    )

    assert (
        tracker.state
        == MedicationActionState
        .WAITING_FOR_RETREAT
    )

    print_test_result(
        "Occluded hand is not treated as retreat",
        True,
    )


# ============================================================
# TEST 7
# RESET
# ============================================================

def test_reset() -> None:

    detector = (
        FakeHandDetector(
            [
                (
                    0.50,
                    0.53,
                )
            ]
        )
    )

    tracker = (
        MedicationActionTracker(
            hand_detector=detector,  # type: ignore[arg-type]
        )
    )

    arm_tracker(
        tracker
    )

    tracker.reset()

    assert (
        tracker.state
        == MedicationActionState.IDLE
    )

    assert not (
        tracker.is_active
    )

    print_test_result(
        "Medication action tracker reset",
        True,
    )


# ============================================================
# FULL TEST SUITE
# ============================================================

def test_medicine_action_tracker() -> None:

    print(
        "\n"
        "MemorAi Medication Action Tracker "
        "Test Suite\n"
    )

    test_arm_state()

    test_far_hand_does_not_confirm()

    test_single_near_frame_not_enough()

    test_approach_without_retreat()

    test_complete_action_confirmation()

    test_missing_hand_not_retreat()

    test_reset()

    print(
        "\n"
        "All medication action tracker tests passed."
        "\n"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    test_medicine_action_tracker()