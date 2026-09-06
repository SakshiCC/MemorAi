from __future__ import annotations

from dataclasses import replace

import time

import numpy as np

from medicine.medicine_monitor import (
    MedicineMonitor,
)

from medicine.medicine_recognizer import (
    MedicineMatch,
)


# ============================================================
# TEST IMAGE
# ============================================================

TEST_IMAGE = np.zeros(
    (
        100,
        100,
        3,
    ),
    dtype=np.uint8,
)


# ============================================================
# FAKE RECOGNIZER
# ============================================================

class FakeMedicineRecognizer:

    def __init__(
        self,
        matches: list[
            MedicineMatch
        ],
    ) -> None:

        self.matches = list(
            matches
        )

        self.index = 0


    def recognize_image(
        self,
        _image: np.ndarray,
    ) -> MedicineMatch:

        if not self.matches:

            raise RuntimeError(
                "Fake recognizer has no matches."
            )

        if (
            self.index
            >= len(
                self.matches
            )
        ):

            return (
                self.matches[
                    -1
                ]
            )

        result = (
            self.matches[
                self.index
            ]
        )

        self.index += 1

        return result


# ============================================================
# MATCH HELPERS
# ============================================================

def known_match(
    medicine_id: int = 1,
    medicine_name: str = "Metformin",
    similarity: float = 0.90,
) -> MedicineMatch:

    return MedicineMatch(
        medicine_id=medicine_id,

        medicine_name=medicine_name,

        dosage="500mg",

        similarity=similarity,

        is_known=True,

        threshold=0.60,

        feature_verified=True,

        feature_score=0.70,

        feature_good_matches=12,

        feature_inliers=8,

        feature_inlier_ratio=(
            8 / 12
        ),
    )


def rejected_match(
    medicine_id: int = 1,
    medicine_name: str = "Metformin",
    similarity: float = 0.68,
) -> MedicineMatch:

    return MedicineMatch(
        medicine_id=medicine_id,

        medicine_name=medicine_name,

        dosage="500mg",

        similarity=similarity,

        is_known=False,

        threshold=0.60,

        feature_verified=False,

        feature_score=0.25,

        feature_good_matches=5,

        feature_inliers=0,

        feature_inlier_ratio=0.0,
    )


# ============================================================
# PRINT RESULT
# ============================================================

def print_pass(
    name: str,
) -> None:

    print(
        f"PASS: {name}"
    )


# ============================================================
# TEST 1
# SINGLE POSITIVE
# ============================================================

def test_one_positive_not_confirmed() -> None:

    recognizer = (
        FakeMedicineRecognizer(
            [
                known_match(),
            ]
        )
    )

    monitor = (
        MedicineMonitor(
            recognizer=recognizer,
            confirmation_frames=3,
            evidence_window_size=5,
            cooldown_seconds=0,
        )
    )

    observation, confirmed = (
        monitor.process_image(
            TEST_IMAGE
        )
    )

    assert confirmed is None

    assert (
        observation.confirmation_count
        == 1
    )

    assert (
        observation.confirmation_required
        == 3
    )

    print_pass(
        "One positive does not confirm medicine"
    )


# ============================================================
# TEST 2
# THREE POSITIVES CONFIRM
# ============================================================

def test_three_positives_confirm() -> None:

    recognizer = (
        FakeMedicineRecognizer(
            [
                known_match(
                    similarity=0.80
                ),

                known_match(
                    similarity=0.82
                ),

                known_match(
                    similarity=0.84
                ),
            ]
        )
    )

    monitor = (
        MedicineMonitor(
            recognizer=recognizer,
            confirmation_frames=3,
            evidence_window_size=5,
            cooldown_seconds=0,
        )
    )

    first, result1 = (
        monitor.process_image(
            TEST_IMAGE
        )
    )

    second, result2 = (
        monitor.process_image(
            TEST_IMAGE
        )
    )

    third, result3 = (
        monitor.process_image(
            TEST_IMAGE
        )
    )

    assert result1 is None

    assert result2 is None

    assert result3 is not None

    assert (
        first.confirmation_count
        == 1
    )

    assert (
        second.confirmation_count
        == 2
    )

    assert (
        third.confirmation_count
        == 3
    )

    assert (
        abs(
            result3.similarity
            - 0.82
        )
        < 0.0001
    )

    print_pass(
        "Three verified observations confirm medicine"
    )


# ============================================================
# TEST 3
# ONE BAD FRAME DOES NOT RESET
# ============================================================

def test_bad_frame_does_not_reset() -> None:

    recognizer = (
        FakeMedicineRecognizer(
            [
                known_match(
                    similarity=0.80
                ),

                rejected_match(
                    similarity=0.70
                ),

                known_match(
                    similarity=0.82
                ),

                known_match(
                    similarity=0.84
                ),
            ]
        )
    )

    monitor = (
        MedicineMonitor(
            recognizer=recognizer,
            confirmation_frames=3,
            evidence_window_size=5,
            cooldown_seconds=0,
        )
    )

    obs1, result1 = (
        monitor.process_image(
            TEST_IMAGE
        )
    )

    obs2, result2 = (
        monitor.process_image(
            TEST_IMAGE
        )
    )

    obs3, result3 = (
        monitor.process_image(
            TEST_IMAGE
        )
    )

    obs4, result4 = (
        monitor.process_image(
            TEST_IMAGE
        )
    )

    assert result1 is None

    assert result2 is None

    assert result3 is None

    assert result4 is not None

    assert (
        obs1.confirmation_count
        == 1
    )

    # IMPORTANT:
    # failed frame no longer resets evidence to zero
    assert (
        obs2.confirmation_count
        == 1
    )

    assert (
        obs3.confirmation_count
        == 2
    )

    assert (
        obs4.confirmation_count
        == 3
    )

    print_pass(
        "Bad frame does not destroy valid recent evidence"
    )


# ============================================================
# TEST 4
# THREE OF FIVE
# ============================================================

def test_three_of_five_confirmation() -> None:

    recognizer = (
        FakeMedicineRecognizer(
            [
                known_match(
                    similarity=0.80
                ),

                rejected_match(
                    similarity=0.68
                ),

                known_match(
                    similarity=0.81
                ),

                rejected_match(
                    similarity=0.69
                ),

                known_match(
                    similarity=0.82
                ),
            ]
        )
    )

    monitor = (
        MedicineMonitor(
            recognizer=recognizer,
            confirmation_frames=3,
            evidence_window_size=5,
            cooldown_seconds=0,
        )
    )

    final_result = None

    final_observation = None

    for _ in range(
        5
    ):

        (
            final_observation,
            result,
        ) = (
            monitor.process_image(
                TEST_IMAGE
            )
        )

        if result is not None:

            final_result = (
                result
            )

    assert (
        final_result
        is not None
    )

    assert (
        final_observation
        is not None
    )

    assert (
        final_observation.confirmation_count
        == 3
    )

    print_pass(
        "Three of five recent observations confirm medicine"
    )


# ============================================================
# TEST 5
# TWO OF FIVE DOES NOT CONFIRM
# ============================================================

def test_two_of_five_not_confirmed() -> None:

    recognizer = (
        FakeMedicineRecognizer(
            [
                known_match(),

                rejected_match(),

                known_match(),

                rejected_match(),

                rejected_match(),
            ]
        )
    )

    monitor = (
        MedicineMonitor(
            recognizer=recognizer,
            confirmation_frames=3,
            evidence_window_size=5,
            cooldown_seconds=0,
        )
    )

    result = None

    observation = None

    for _ in range(
        5
    ):

        observation, result = (
            monitor.process_image(
                TEST_IMAGE
            )
        )

        assert (
            result is None
        )

    assert (
        observation is not None
    )

    assert (
        observation.confirmation_count
        == 2
    )

    print_pass(
        "Two of five observations do not confirm medicine"
    )


# ============================================================
# TEST 6
# DIFFERENT MEDICINE CLEARS WINDOW
# ============================================================

def test_different_medicine_clears_evidence() -> None:

    recognizer = (
        FakeMedicineRecognizer(
            [
                known_match(
                    medicine_id=1,
                    medicine_name="Metformin",
                ),

                known_match(
                    medicine_id=1,
                    medicine_name="Metformin",
                ),

                known_match(
                    medicine_id=2,
                    medicine_name="Amlodipine",
                ),

                known_match(
                    medicine_id=2,
                    medicine_name="Amlodipine",
                ),

                known_match(
                    medicine_id=2,
                    medicine_name="Amlodipine",
                ),
            ]
        )
    )

    monitor = (
        MedicineMonitor(
            recognizer=recognizer,
            confirmation_frames=3,
            evidence_window_size=5,
            cooldown_seconds=0,
        )
    )

    first, first_result = (
        monitor.process_image(
            TEST_IMAGE
        )
    )

    second, second_result = (
        monitor.process_image(
            TEST_IMAGE
        )
    )

    third, third_result = (
        monitor.process_image(
            TEST_IMAGE
        )
    )

    fourth, fourth_result = (
        monitor.process_image(
            TEST_IMAGE
        )
    )

    fifth, fifth_result = (
        monitor.process_image(
            TEST_IMAGE
        )
    )

    assert first_result is None

    assert second_result is None

    # New medicine starts from one evidence item.
    assert third_result is None

    assert (
        third.confirmation_count
        == 1
    )

    assert fourth_result is None

    assert (
        fourth.confirmation_count
        == 2
    )

    assert fifth_result is not None

    assert (
        fifth_result.medicine_id
        == 2
    )

    assert (
        fifth_result.medicine_name
        == "Amlodipine"
    )

    print_pass(
        "Different medicine clears previous evidence"
    )


# ============================================================
# TEST 7
# EVIDENCE WINDOW DROPS OLD POSITIVES
# ============================================================

def test_window_discards_old_evidence() -> None:

    recognizer = (
        FakeMedicineRecognizer(
            [
                known_match(),

                known_match(),

                rejected_match(),

                rejected_match(),

                rejected_match(),

                known_match(),
            ]
        )
    )

    monitor = (
        MedicineMonitor(
            recognizer=recognizer,
            confirmation_frames=3,
            evidence_window_size=5,
            cooldown_seconds=0,
        )
    )

    result = None

    observation = None

    for _ in range(
        6
    ):

        observation, result = (
            monitor.process_image(
                TEST_IMAGE
            )
        )

        assert (
            result is None
        )

    # At this point the first PASS has fallen out of the
    # five-observation rolling window.
    #
    # Window:
    #
    # PASS
    # FAIL
    # FAIL
    # FAIL
    # PASS
    #
    # => 2 verified observations
    assert (
        observation is not None
    )

    assert (
        observation.confirmation_count
        == 2
    )

    print_pass(
        "Rolling window discards old evidence"
    )


# ============================================================
# TEST 8
# TIMEOUT
# ============================================================

def test_old_evidence_expires() -> None:

    recognizer = (
        FakeMedicineRecognizer(
            [
                known_match(),

                known_match(),

                known_match(),
            ]
        )
    )

    monitor = (
        MedicineMonitor(
            recognizer=recognizer,
            confirmation_frames=3,
            evidence_window_size=5,
            evidence_timeout_seconds=0.10,
            cooldown_seconds=0,
        )
    )

    observation1, result1 = (
        monitor.process_image(
            TEST_IMAGE
        )
    )

    observation2, result2 = (
        monitor.process_image(
            TEST_IMAGE
        )
    )

    assert result1 is None

    assert result2 is None

    assert (
        observation2.confirmation_count
        == 2
    )

    # Allow the two old observations to expire.
    time.sleep(
        0.15
    )

    observation3, result3 = (
        monitor.process_image(
            TEST_IMAGE
        )
    )

    assert (
        result3 is None
    )

    assert (
        observation3.confirmation_count
        == 1
    )

    print_pass(
        "Expired medicine evidence is not reused"
    )


# ============================================================
# TEST 9
# COOLDOWN
# ============================================================

def test_confirmation_cooldown() -> None:

    recognizer = (
        FakeMedicineRecognizer(
            [
                known_match(),
                known_match(),
                known_match(),

                known_match(),
                known_match(),
                known_match(),
            ]
        )
    )

    monitor = (
        MedicineMonitor(
            recognizer=recognizer,
            confirmation_frames=3,
            evidence_window_size=5,
            cooldown_seconds=60,
        )
    )

    first_result = None

    for _ in range(
        3
    ):

        _observation, result = (
            monitor.process_image(
                TEST_IMAGE
            )
        )

        if result is not None:

            first_result = (
                result
            )

    assert (
        first_result
        is not None
    )

    # New evidence may build again, but confirmation result
    # is suppressed during cooldown.
    second_result = None

    for _ in range(
        3
    ):

        _observation, result = (
            monitor.process_image(
                TEST_IMAGE
            )
        )

        if result is not None:

            second_result = (
                result
            )

    assert (
        second_result
        is None
    )

    print_pass(
        "Medicine confirmation cooldown"
    )


# ============================================================
# TEST 10
# RESET
# ============================================================

def test_monitor_reset() -> None:

    recognizer = (
        FakeMedicineRecognizer(
            [
                known_match(),
                known_match(),
                known_match(),
                known_match(),
            ]
        )
    )

    monitor = (
        MedicineMonitor(
            recognizer=recognizer,
            confirmation_frames=3,
            evidence_window_size=5,
            cooldown_seconds=0,
        )
    )

    first, first_result = (
        monitor.process_image(
            TEST_IMAGE
        )
    )

    second, second_result = (
        monitor.process_image(
            TEST_IMAGE
        )
    )

    assert first_result is None

    assert second_result is None

    assert (
        second.confirmation_count
        == 2
    )

    monitor.reset()

    after_reset, reset_result = (
        monitor.process_image(
            TEST_IMAGE
        )
    )

    assert (
        reset_result is None
    )

    assert (
        after_reset.confirmation_count
        == 1
    )

    print_pass(
        "Medicine monitor reset"
    )


# ============================================================
# TEST SUITE
# ============================================================

def test_medicine_monitor() -> None:

    print(
        "\n"
        "MemorAi Medicine Monitor Test Suite"
        "\n"
    )

    test_one_positive_not_confirmed()

    test_three_positives_confirm()

    test_bad_frame_does_not_reset()

    test_three_of_five_confirmation()

    test_two_of_five_not_confirmed()

    test_different_medicine_clears_evidence()

    test_window_discards_old_evidence()

    test_old_evidence_expires()

    test_confirmation_cooldown()

    test_monitor_reset()

    print(
        "\n"
        "All medicine monitor tests passed."
        "\n"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    test_medicine_monitor()