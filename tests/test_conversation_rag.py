from __future__ import annotations

from datetime import (
    datetime,
    timedelta,
)

from conversation.conversation_repository import (
    add_conversation_segment,
    create_conversation_session,
    delete_conversation,
    finalize_conversation,
    get_conversation_session,
)

from database.conversation_vector_memory import (
    ConversationVectorMemory,
)

from database.db_manager import (
    initialize_database,
)

from llm.chat import (
    FALLBACK_ANSWER,
    MemorAiChat,
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
# ASSERT CONTAINS ANY
# ============================================================

def assert_contains_any(
    text: str,
    values: tuple[str, ...],
) -> None:

    lowered = (
        str(
            text
            or ""
        )
        .lower()
    )

    assert any(
        value.lower()
        in lowered

        for value in values
    ), (
        f"Expected one of {values} in answer:\n"
        f"{text}"
    )


# ============================================================
# MAIN TEST
# ============================================================

def test_conversation_rag() -> None:

    print(
        "\n"
        + "=" * 70
    )

    print(
        (
            "MemorAi Conversation RAG "
            "End-to-End Test Suite"
        )
    )

    print(
        "=" * 70
    )

    # ========================================================
    # DATABASE
    # ========================================================

    initialize_database()

    # ========================================================
    # VECTOR MEMORY
    # ========================================================

    vector_memory = (
        ConversationVectorMemory()
    )

    # ========================================================
    # TEST IDS
    # ========================================================

    appointment_id: (
        int
        | None
    ) = None

    shopping_id: (
        int
        | None
    ) = None

    location_id: (
        int
        | None
    ) = None

    visit_id: (
        int
        | None
    ) = None

    created_ids: list[int] = []

    try:

        now = (
            datetime.now()
            .replace(
                microsecond=0
            )
        )

        # ====================================================
        # CONVERSATION 1
        # DOCTOR APPOINTMENT
        # ====================================================

        appointment_transcript = (
            "Meera told me that my doctor "
            "appointment is on Wednesday at 3 PM. "
            "She reminded me to bring my medical "
            "reports and arrive ten minutes early."
        )

        appointment_id = (
            create_conversation_session(
                started_at=(
                    now
                ),

                title=(
                    "Medical Appointment"
                ),

                participant_text=(
                    "Meera"
                ),

                status="processing",
            )
        )

        created_ids.append(
            appointment_id
        )

        add_conversation_segment(
            conversation_id=(
                appointment_id
            ),

            sequence_no=1,

            transcript=(
                appointment_transcript
            ),

            started_at=(
                now
            ),

            ended_at=(
                now
                + timedelta(
                    seconds=12
                )
            ),
        )

        finalize_conversation(
            conversation_id=(
                appointment_id
            ),

            ended_at=(
                now
                + timedelta(
                    seconds=12
                )
            ),

            full_transcript=(
                appointment_transcript
            ),

            summary=(
                "Doctor appointment is Wednesday "
                "at 3 PM. Bring medical reports "
                "and arrive ten minutes early."
            ),

            title=(
                "Medical Appointment"
            ),

            participant_text=(
                "Meera"
            ),
        )

        appointment_record = (
            get_conversation_session(
                appointment_id
            )
        )

        assert (
            appointment_record
            is not None
        )

        vector_memory.add_conversation(
            conversation=(
                appointment_record
            ),

            importance_score=(
                1.0
            ),

            matched_categories=(
                "appointment",
                "reminder",
                "schedule",
                "task",
            ),
        )

        print_test_result(
            (
                "Appointment conversation "
                "stored and indexed"
            )
        )

        # ====================================================
        # CONVERSATION 2
        # SHOPPING TASK
        # ====================================================

        shopping_started = (
            now
            + timedelta(
                minutes=2
            )
        )

        shopping_transcript = (
            "Rahul reminded me to buy milk "
            "and bread tomorrow."
        )

        shopping_id = (
            create_conversation_session(
                started_at=(
                    shopping_started
                ),

                title=(
                    "Shopping Reminder"
                ),

                participant_text=(
                    "Rahul"
                ),

                status="processing",
            )
        )

        created_ids.append(
            shopping_id
        )

        add_conversation_segment(
            conversation_id=(
                shopping_id
            ),

            sequence_no=1,

            transcript=(
                shopping_transcript
            ),

            started_at=(
                shopping_started
            ),

            ended_at=(
                shopping_started
                + timedelta(
                    seconds=8
                )
            ),
        )

        finalize_conversation(
            conversation_id=(
                shopping_id
            ),

            ended_at=(
                shopping_started
                + timedelta(
                    seconds=8
                )
            ),

            full_transcript=(
                shopping_transcript
            ),

            summary=(
                "Buy milk and bread tomorrow."
            ),

            title=(
                "Shopping Reminder"
            ),

            participant_text=(
                "Rahul"
            ),
        )

        shopping_record = (
            get_conversation_session(
                shopping_id
            )
        )

        assert (
            shopping_record
            is not None
        )

        vector_memory.add_conversation(
            conversation=(
                shopping_record
            ),

            importance_score=(
                0.90
            ),

            matched_categories=(
                "reminder",
                "task",
                "schedule",
            ),
        )

        print_test_result(
            (
                "Shopping conversation "
                "stored and indexed"
            )
        )

        # ====================================================
        # CONVERSATION 3
        # LOCATION MEMORY
        # ====================================================

        location_started = (
            now
            + timedelta(
                minutes=4
            )
        )

        location_transcript = (
            "Sakshi told me that my passport "
            "is in the second drawer."
        )

        location_id = (
            create_conversation_session(
                started_at=(
                    location_started
                ),

                title=(
                    "Important Location Reminder"
                ),

                participant_text=(
                    "Sakshi"
                ),

                status="processing",
            )
        )

        created_ids.append(
            location_id
        )

        add_conversation_segment(
            conversation_id=(
                location_id
            ),

            sequence_no=1,

            transcript=(
                location_transcript
            ),

            started_at=(
                location_started
            ),

            ended_at=(
                location_started
                + timedelta(
                    seconds=7
                )
            ),
        )

        finalize_conversation(
            conversation_id=(
                location_id
            ),

            ended_at=(
                location_started
                + timedelta(
                    seconds=7
                )
            ),

            full_transcript=(
                location_transcript
            ),

            summary=(
                "Passport is in the second drawer."
            ),

            title=(
                "Important Location Reminder"
            ),

            participant_text=(
                "Sakshi"
            ),
        )

        location_record = (
            get_conversation_session(
                location_id
            )
        )

        assert (
            location_record
            is not None
        )

        vector_memory.add_conversation(
            conversation=(
                location_record
            ),

            importance_score=(
                0.65
            ),

            matched_categories=(
                "location_memory",
            ),
        )

        print_test_result(
            (
                "Location conversation "
                "stored and indexed"
            )
        )

        # ====================================================
        # CONVERSATION 4
        # VISIT PLAN
        # ====================================================

        visit_started = (
            now
            + timedelta(
                minutes=6
            )
        )

        visit_transcript = (
            "Rahul said that he will visit "
            "tomorrow evening."
        )

        visit_id = (
            create_conversation_session(
                started_at=(
                    visit_started
                ),

                title=(
                    "Visit Plan"
                ),

                participant_text=(
                    "Rahul"
                ),

                status="processing",
            )
        )

        created_ids.append(
            visit_id
        )

        add_conversation_segment(
            conversation_id=(
                visit_id
            ),

            sequence_no=1,

            transcript=(
                visit_transcript
            ),

            started_at=(
                visit_started
            ),

            ended_at=(
                visit_started
                + timedelta(
                    seconds=7
                )
            ),
        )

        finalize_conversation(
            conversation_id=(
                visit_id
            ),

            ended_at=(
                visit_started
                + timedelta(
                    seconds=7
                )
            ),

            full_transcript=(
                visit_transcript
            ),

            summary=(
                "Rahul will visit tomorrow evening."
            ),

            title=(
                "Visit Plan"
            ),

            participant_text=(
                "Rahul"
            ),
        )

        visit_record = (
            get_conversation_session(
                visit_id
            )
        )

        assert (
            visit_record
            is not None
        )

        vector_memory.add_conversation(
            conversation=(
                visit_record
            ),

            importance_score=(
                0.65
            ),

            matched_categories=(
                "commitment",
                "schedule",
            ),
        )

        print_test_result(
            (
                "Visit conversation "
                "stored and indexed"
            )
        )

        # ====================================================
        # CHAT
        # ====================================================

        chat = (
            MemorAiChat()
        )

        # ====================================================
        # TEST 1
        # PARTICIPANT CONVERSATION
        # ====================================================

        question = (
            "What did Meera tell me "
            "about my doctor appointment?"
        )

        answer = (
            chat.answer(
                question
            )
        )

        print(
            "\nQuestion:",
            question,
        )

        print(
            "Answer:",
            answer,
        )

        assert (
            answer
            != FALLBACK_ANSWER
        )

        assert_contains_any(
            answer,
            (
                "wednesday",
                "3 pm",
                "medical reports",
                "ten minutes",
            ),
        )

        print_test_result(
            (
                "Conversation RAG participant "
                "appointment answer"
            )
        )

        # ====================================================
        # TEST 2
        # WHAT TO BRING
        # ====================================================

        question = (
            "What should I bring to "
            "my doctor appointment?"
        )

        answer = (
            chat.answer(
                question
            )
        )

        print(
            "\nQuestion:",
            question,
        )

        print(
            "Answer:",
            answer,
        )

        assert (
            answer
            != FALLBACK_ANSWER
        )

        assert_contains_any(
            answer,
            (
                "medical reports",
                "reports",
            ),
        )

        print_test_result(
            (
                "Conversation RAG semantic "
                "appointment detail"
            )
        )

        # ====================================================
        # TEST 3
        # SHOPPING TASK
        # ====================================================

        question = (
            "What did Rahul remind "
            "me to buy?"
        )

        answer = (
            chat.answer(
                question
            )
        )

        print(
            "\nQuestion:",
            question,
        )

        print(
            "Answer:",
            answer,
        )

        assert (
            answer
            != FALLBACK_ANSWER
        )

        assert_contains_any(
            answer,
            (
                "milk",
                "bread",
            ),
        )

        print_test_result(
            (
                "Conversation RAG shopping "
                "reminder answer"
            )
        )

        # ====================================================
        # TEST 4
        # LOCATION MEMORY
        # ====================================================

        question = (
            "What did Sakshi tell me "
            "about my passport?"
        )

        answer = (
            chat.answer(
                question
            )
        )

        print(
            "\nQuestion:",
            question,
        )

        print(
            "Answer:",
            answer,
        )

        assert (
            answer
            != FALLBACK_ANSWER
        )

        assert_contains_any(
            answer,
            (
                "second drawer",
                "drawer",
                "passport",
            ),
        )

        print_test_result(
            (
                "Conversation RAG location "
                "memory answer"
            )
        )

        # ====================================================
        # TEST 5
        # VISIT PLAN
        # ====================================================

        question = (
            "What did Rahul say "
            "about visiting?"
        )

        answer = (
            chat.answer(
                question
            )
        )

        print(
            "\nQuestion:",
            question,
        )

        print(
            "Answer:",
            answer,
        )

        assert (
            answer
            != FALLBACK_ANSWER
        )

        assert_contains_any(
            answer,
            (
                "visit",
                "tomorrow",
                "evening",
            ),
        )

        print_test_result(
            (
                "Conversation RAG visit "
                "plan answer"
            )
        )

        # ====================================================
        # TEST 6
        # TODAY
        # ====================================================

        question = (
            "What did Meera tell "
            "me today?"
        )

        answer = (
            chat.answer(
                question
            )
        )

        print(
            "\nQuestion:",
            question,
        )

        print(
            "Answer:",
            answer,
        )

        assert (
            answer
            != FALLBACK_ANSWER
        )

        assert_contains_any(
            answer,
            (
                "appointment",
                "medical reports",
                "wednesday",
            ),
        )

        print_test_result(
            (
                "Conversation RAG today "
                "answer"
            )
        )

        # ====================================================
        # TEST 7
        # RECENT CONVERSATION
        # ====================================================

        question = (
            "What was my latest "
            "conversation with Rahul?"
        )

        answer = (
            chat.answer(
                question
            )
        )

        print(
            "\nQuestion:",
            question,
        )

        print(
            "Answer:",
            answer,
        )

        assert (
            answer
            != FALLBACK_ANSWER
        )

        assert_contains_any(
            answer,
            (
                "visit",
                "tomorrow",
                "evening",
            ),
        )

        print_test_result(
            (
                "Conversation RAG latest "
                "conversation answer"
            )
        )

        # ====================================================
        # TEST 8
        # UNKNOWN CONVERSATION
        # ====================================================

        question = (
            "What did Alexander tell me "
            "about my helicopter?"
        )

        answer = (
            chat.answer(
                question
            )
        )

        print(
            "\nQuestion:",
            question,
        )

        print(
            "Answer:",
            answer,
        )

        # ----------------------------------------------------
        # Unknown information must never be invented.
        # ----------------------------------------------------

        assert (
            "helicopter"
            not in answer.lower()
            or answer
            == FALLBACK_ANSWER
        )

        print_test_result(
            (
                "Unknown conversation does "
                "not invent information"
            )
        )

    finally:

        # ====================================================
        # CLEANUP
        # ====================================================

        for conversation_id in (
            created_ids
        ):

            try:

                vector_memory.delete_conversation(
                    conversation_id
                )

            except Exception:

                pass

            try:

                delete_conversation(
                    conversation_id
                )

            except Exception:

                pass

    print(
        "\n"
        + "=" * 70
    )

    print(
        (
            "ALL MEMORAI CONVERSATION "
            "RAG TESTS PASSED"
        )
    )

    print(
        "=" * 70
        + "\n"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    test_conversation_rag()