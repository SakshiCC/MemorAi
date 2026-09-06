from __future__ import annotations

from conversation.metadata_extractor import (
    ConversationMetadataExtractor,
)


# ============================================================
# HELPER
# ============================================================

def print_test_result(
    test_name: str,
) -> None:

    print(
        f"PASS: {test_name}"
    )


# ============================================================
# TEST
# ============================================================

def test_conversation_metadata() -> None:

    print(
        (
            "\nMemorAi Conversation "
            "Metadata Test Suite\n"
        )
    )

    extractor = (
        ConversationMetadataExtractor()
    )

    # ========================================================
    # APPOINTMENT
    # ========================================================

    appointment = (
        extractor.extract(
            transcript=(
                "Saksi told me that my doctor "
                "appointment is on Monday at "
                "10 a.m. I need to bring my "
                "medical reports and arrive "
                "15 minutes early."
            ),

            matched_categories=(
                "appointment",
                "schedule",
                "task",
            ),
        )
    )

    assert (
        appointment.title
        == "Medical Appointment"
    )

    assert (
        appointment.participant_text
        == "Saksi"
    )

    assert (
        "appointment"
        in appointment.summary.lower()
    )

    assert (
        "medical reports"
        in appointment.summary.lower()
    )

    assert (
        "15 minutes early"
        in appointment.summary.lower()
    )

    print_test_result(
        "Appointment metadata extraction"
    )

    # ========================================================
    # MEDICINE
    # ========================================================

    medicine = (
        extractor.extract(
            transcript=(
                "Rahul reminded me to take "
                "my medicine after dinner tonight "
                "and call the doctor tomorrow."
            ),

            matched_categories=(
                "medicine",
                "reminder",
                "task",
                "schedule",
            ),
        )
    )

    assert (
        medicine.title
        == "Medical Appointment"
        or medicine.title
        == "Medicine Reminder"
    )

    assert (
        medicine.participant_text
        == "Rahul"
    )

    assert (
        "medicine"
        in medicine.summary.lower()
    )

    print_test_result(
        "Medicine metadata extraction"
    )

    # ========================================================
    # SHOPPING
    # ========================================================

    shopping = (
        extractor.extract(
            transcript=(
                "Neha reminded me to buy milk "
                "and bread this evening before "
                "the shops close."
            ),

            matched_categories=(
                "reminder",
                "task",
                "schedule",
            ),
        )
    )

    assert (
        shopping.title
        == "Shopping Reminder"
    )

    assert (
        shopping.participant_text
        == "Neha"
    )

    assert (
        "milk"
        in shopping.summary.lower()
    )

    print_test_result(
        "Shopping metadata extraction"
    )

    # ========================================================
    # EXPLICIT PARTICIPANT OVERRIDE
    # ========================================================

    explicit_participant = (
        extractor.extract(
            transcript=(
                "Saksi told me that my doctor "
                "appointment is on Monday."
            ),

            provided_participant=(
                "Sakshi"
            ),
        )
    )

    assert (
        explicit_participant
        .participant_text
        == "Sakshi"
    )

    print_test_result(
        "Explicit participant overrides transcription"
    )

    # ========================================================
    # EXPLICIT TITLE OVERRIDE
    # ========================================================

    explicit_title = (
        extractor.extract(
            transcript=(
                "Rahul reminded me to call "
                "the hospital tomorrow."
            ),

            provided_title=(
                "Hospital Follow-up"
            ),
        )
    )

    assert (
        explicit_title.title
        == "Hospital Follow-up"
    )

    print_test_result(
        "Explicit title overrides automatic title"
    )

    # ========================================================
    # UNKNOWN PARTICIPANT
    # ========================================================

    unknown_participant = (
        extractor.extract(
            transcript=(
                "My appointment is tomorrow "
                "at ten in the morning."
            )
        )
    )

    assert (
        unknown_participant
        .participant_text
        is None
    )

    print_test_result(
        "Unknown participant remains unset"
    )

    # ========================================================
    # EMPTY TRANSCRIPT
    # ========================================================

    try:

        extractor.extract(
            transcript=""
        )

    except ValueError:

        print_test_result(
            "Empty metadata transcript protection"
        )

    else:

        raise AssertionError(
            (
                "Empty transcript was incorrectly "
                "accepted."
            )
        )

    print(
        (
            "\nAll MemorAi Conversation "
            "Metadata tests passed."
        )
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    test_conversation_metadata()