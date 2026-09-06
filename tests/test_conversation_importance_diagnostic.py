from __future__ import annotations

from conversation.importance_analyzer import (
    ConversationImportanceAnalyzer,
)


# ============================================================
# GLOBAL RESULTS
# ============================================================

failures: list[str] = []


# ============================================================
# DISPLAY HELPER
# ============================================================

def show_result(
    name: str,
    transcript: str,
    expected_important: bool,
) -> None:

    analyzer = (
        ConversationImportanceAnalyzer(
            threshold=0.60
        )
    )

    result = (
        analyzer.analyze(
            transcript
        )
    )

    passed = (
        result.is_important
        == expected_important
    )

    status = (
        "PASS"
        if passed
        else "FAIL"
    )

    expected_text = (
        "IMPORTANT"
        if expected_important
        else "NOT IMPORTANT"
    )

    actual_text = (
        "IMPORTANT"
        if result.is_important
        else "NOT IMPORTANT"
    )

    print(
        "\n"
        + "-" * 70
    )

    print(
        f"{status}: {name}"
    )

    print(
        f"Transcript: {transcript}"
    )

    print(
        f"Expected:   {expected_text}"
    )

    print(
        f"Actual:     {actual_text}"
    )

    print(
        f"Score:      {result.score:.3f}"
    )

    print(
        (
            "Categories: "
            f"{result.matched_categories}"
        )
    )

    print(
        "Reasons:"
    )

    for reason in (
        result.reasons
    ):

        print(
            f"  - {reason}"
        )

    if not passed:

        failures.append(
            (
                f"{name}: expected "
                f"{expected_text}, "
                f"got {actual_text} "
                f"(score={result.score:.3f})"
            )
        )


# ============================================================
# IMPORTANT MEMORIES
# ============================================================

def test_important_memories() -> None:

    show_result(
        name="Doctor appointment",

        transcript=(
            "My doctor appointment "
            "is Monday at 10 AM."
        ),

        expected_important=True,
    )

    show_result(
        name="Reminder",

        transcript=(
            "Remind me to call Rahul "
            "tomorrow."
        ),

        expected_important=True,
    )

    show_result(
        name="Short shopping task",

        transcript=(
            "Buy milk tomorrow."
        ),

        expected_important=True,
    )

    show_result(
        name="Passport location",

        transcript=(
            "My passport is "
            "in the drawer."
        ),

        expected_important=True,
    )

    show_result(
        name="Keys location",

        transcript=(
            "I kept my keys "
            "in the cupboard."
        ),

        expected_important=True,
    )

    show_result(
        name="Medicine instruction",

        transcript=(
            "Remember to take your "
            "medicine at 8 PM."
        ),

        expected_important=True,
    )

    show_result(
        name="Bill payment",

        transcript=(
            "Pay the electricity bill "
            "tomorrow."
        ),

        expected_important=True,
    )

    show_result(
        name="Visit plan",

        transcript=(
            "Rahul will visit tomorrow."
        ),

        expected_important=True,
    )


# ============================================================
# CASUAL / UNIMPORTANT
# ============================================================

def test_unimportant_memories() -> None:

    show_result(
        name="Hello",
        transcript="Hello.",
        expected_important=False,
    )

    show_result(
        name="How are you",
        transcript="How are you?",
        expected_important=False,
    )

    show_result(
        name="Fine response",
        transcript="I am fine.",
        expected_important=False,
    )

    show_result(
        name="Good morning",
        transcript="Good morning.",
        expected_important=False,
    )

    show_result(
        name="Thanks",
        transcript="Thank you.",
        expected_important=False,
    )

    show_result(
        name="Okay",
        transcript="Okay.",
        expected_important=False,
    )

    show_result(
        name="Small talk",
        transcript=(
            "The weather is nice today."
        ),
        expected_important=False,
    )

    show_result(
        name="Casual opinion",
        transcript=(
            "I really liked that movie."
        ),
        expected_important=False,
    )


# ============================================================
# MIXED
# ============================================================

def test_mixed_conversation() -> None:

    show_result(
        name=(
            "Casual conversation with "
            "important reminder"
        ),

        transcript=(
            "Hello, how are you? "
            "I am fine. "
            "Remind me to call the doctor "
            "tomorrow at 10 AM."
        ),

        expected_important=True,
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print(
        "\n"
        + "=" * 70
    )

    print(
        (
            "MemorAi Conversation "
            "Importance Diagnostic"
        )
    )

    print(
        "=" * 70
    )

    print(
        (
            "\nTesting Conversation Importance "
            "without microphone, database, "
            "Whisper or ChromaDB writes."
        )
    )

    test_important_memories()

    test_unimportant_memories()

    test_mixed_conversation()

    print(
        "\n"
        + "=" * 70
    )

    if failures:

        print(
            (
                f"DIAGNOSTIC FAILED: "
                f"{len(failures)} mismatch(es)"
            )
        )

        print(
            "=" * 70
        )

        for failure in failures:

            print(
                f"- {failure}"
            )

        raise AssertionError(
            (
                f"{len(failures)} conversation "
                "importance diagnostic case(s) failed."
            )
        )

    print(
        "ALL DIAGNOSTIC TESTS PASSED"
    )

    print(
        "=" * 70
        + "\n"
    )


if __name__ == "__main__":

    main()