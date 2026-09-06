from __future__ import annotations

from conversation.conversation_settings import (
    get_conversation_settings,
    get_importance_threshold,
    is_conversation_memory_enabled,
    require_conversation_memory_permission,
    set_conversation_memory_enabled,
    set_importance_threshold,
)

from conversation.importance_analyzer import (
    ConversationImportanceAnalyzer,
)

from database.db_manager import (
    initialize_database,
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

def test_conversation_privacy() -> None:

    print(
        (
            "\nMemorAi Conversation Privacy "
            "and Importance Test Suite\n"
        )
    )

    # ========================================================
    # DATABASE
    # ========================================================

    initialize_database()

    settings = (
        get_conversation_settings()
    )

    assert (
        "memory_enabled"
        in settings
    )

    assert (
        "importance_threshold"
        in settings
    )

    print_test_result(
        "Conversation privacy settings available"
    )

    # Preserve the user's actual settings.

    original_enabled = (
        is_conversation_memory_enabled()
    )

    original_threshold = (
        get_importance_threshold()
    )

    try:

        # ====================================================
        # PRIVACY OFF
        # ====================================================

        set_conversation_memory_enabled(
            False
        )

        assert (
            is_conversation_memory_enabled()
            is False
        )

        try:

            require_conversation_memory_permission()

        except PermissionError:

            print_test_result(
                "Conversation recording blocked when disabled"
            )

        else:

            raise AssertionError(
                (
                    "Conversation Memory permission "
                    "was not enforced."
                )
            )

        # ====================================================
        # PRIVACY ON
        # ====================================================

        set_conversation_memory_enabled(
            True
        )

        assert (
            is_conversation_memory_enabled()
            is True
        )

        require_conversation_memory_permission()

        print_test_result(
            "Conversation recording allowed when enabled"
        )

        # ====================================================
        # THRESHOLD
        # ====================================================

        set_importance_threshold(
            0.60
        )

        assert abs(
            get_importance_threshold()
            - 0.60
        ) < 0.001

        print_test_result(
            "Conversation importance threshold persistence"
        )

        # ====================================================
        # CASUAL CONVERSATION
        # ====================================================

        analyzer = (
            ConversationImportanceAnalyzer(
                threshold=0.60
            )
        )

        casual = (
            analyzer.analyze(
                "Hello. How are you?"
            )
        )

        assert (
            casual.is_important
            is False
        )

        print_test_result(
            "Casual conversation rejected"
        )

        # ====================================================
        # APPOINTMENT
        # ====================================================

        appointment = (
            analyzer.analyze(
                (
                    "Rahul said my doctor appointment "
                    "is Monday at 10 AM and he will "
                    "pick me up at 9:30 AM."
                )
            )
        )

        assert (
            appointment.is_important
            is True
        )

        assert (
            appointment.score
            >= 0.60
        )

        print_test_result(
            "Important appointment conversation accepted"
        )

        # ====================================================
        # MEDICINE
        # ====================================================

        medicine = (
            analyzer.analyze(
                (
                    "Remember to take the medicine "
                    "after dinner tonight and call "
                    "the doctor tomorrow morning."
                )
            )
        )

        assert (
            medicine.is_important
            is True
        )

        print_test_result(
            "Important medicine conversation accepted"
        )

        # ====================================================
        # OBJECT LOCATION
        # ====================================================

        location = (
            analyzer.analyze(
                (
                    "I kept your passport in the "
                    "second drawer. Remember to take "
                    "it tomorrow morning."
                )
            )
        )

        assert (
            location.is_important
            is True
        )

        print_test_result(
            "Important location conversation accepted"
        )

        # ====================================================
        # LOW INFORMATION
        # ====================================================

        low_information = (
            analyzer.analyze(
                (
                    "We talked for a while about "
                    "a movie and it was funny."
                )
            )
        )

        assert (
            low_information.is_important
            is False
        )

        print_test_result(
            "Low-value conversation rejected"
        )

        # ====================================================
        # INVALID THRESHOLD
        # ====================================================

        try:

            set_importance_threshold(
                1.20
            )

        except ValueError:

            print_test_result(
                "Invalid importance threshold protection"
            )

        else:

            raise AssertionError(
                (
                    "Invalid threshold was "
                    "incorrectly accepted."
                )
            )

    finally:

        # ====================================================
        # RESTORE REAL USER SETTINGS
        # ====================================================

        set_conversation_memory_enabled(
            original_enabled
        )

        set_importance_threshold(
            original_threshold
        )

    print(
        (
            "\nAll MemorAi Conversation Privacy "
            "and Importance tests passed."
        )
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    test_conversation_privacy()