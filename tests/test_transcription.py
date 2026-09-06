from speech.transcription import (
    TranscriptionCleaner,
)


def test_transcription_cleaner():

    print(
        "\nMemorAi Transcription Test Suite\n"
    )

    cleaner = (
        TranscriptionCleaner()
    )

    # -------------------------------------------------
    # Whitespace
    # -------------------------------------------------

    result = (
        cleaner.validate(
            "   Where is my cellphone?   "
        )
    )

    assert (
        result.is_valid
        is True
    )

    assert (
        result.text
        == "Where is my cellphone?"
    )

    print(
        "PASS: Surrounding whitespace cleaned"
    )

    # -------------------------------------------------
    # Repeated whitespace
    # -------------------------------------------------

    result = (
        cleaner.validate(
            "Where    are    my    glasses?"
        )
    )

    assert (
        result.text
        == "Where are my glasses?"
    )

    print(
        "PASS: Repeated whitespace cleaned"
    )

    # -------------------------------------------------
    # Newlines
    # -------------------------------------------------

    result = (
        cleaner.validate(
            "When did I\n"
            "take my medicine?"
        )
    )

    assert (
        result.text
        == "When did I take my medicine?"
    )

    print(
        "PASS: Newlines cleaned"
    )

    # -------------------------------------------------
    # None
    # -------------------------------------------------

    result = (
        cleaner.validate(
            None
        )
    )

    assert (
        result.is_valid
        is False
    )

    print(
        "PASS: None transcription rejected"
    )

    # -------------------------------------------------
    # Empty
    # -------------------------------------------------

    result = (
        cleaner.validate(
            ""
        )
    )

    assert (
        result.is_valid
        is False
    )

    print(
        "PASS: Empty transcription rejected"
    )

    # -------------------------------------------------
    # Silence labels
    # -------------------------------------------------

    for noise_text in [
        "...",
        "[Music]",
        "(noise)",
        "[Silence]",
        "[Background Noise]",
        "[Laughter]",
    ]:

        result = (
            cleaner.validate(
                noise_text
            )
        )

        assert (
            result.is_valid
            is False
        )

    print(
        "PASS: Common non-speech labels rejected"
    )

    # -------------------------------------------------
    # Valid question
    # -------------------------------------------------

    result = (
        cleaner.validate(
            "Where is my cell phone?"
        )
    )

    assert (
        result.is_valid
        is True
    )

    print(
        "PASS: Valid memory question accepted"
    )

    # -------------------------------------------------
    # Meaning preservation
    # -------------------------------------------------

    result = (
        cleaner.validate(
            "That is my cell phone."
        )
    )

    assert (
        result.text
        == "That is my cell phone."
    )

    print(
        "PASS: User meaning preserved"
    )

    # -------------------------------------------------
    # Repetition hallucination
    # -------------------------------------------------

    result = (
        cleaner.validate(
            "the the the the the "
            "the the the the"
        )
    )

    assert (
        result.is_valid
        is False
    )

    print(
        "PASS: Excessive repeated speech rejected"
    )

    print(
        "\nAll MemorAi transcription tests passed."
    )


if __name__ == "__main__":

    test_transcription_cleaner()