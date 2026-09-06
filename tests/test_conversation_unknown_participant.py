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

from memory.conversation_retriever import (
    ConversationMemoryRetriever,
)


# ============================================================
# TEST
# ============================================================

def test_unknown_conversation_participant() -> None:

    print(
        "\n"
        + "=" * 70
    )

    print(
        "MemorAi Unknown Conversation Participant Safety Test"
    )

    print(
        "=" * 70
    )

    initialize_database()

    vector_memory = (
        ConversationVectorMemory()
    )

    conversation_id: (
        int
        | None
    ) = None

    try:

        # ====================================================
        # CREATE KNOWN CONVERSATION
        # ====================================================

        started_at = (
            datetime.now()
            .replace(
                microsecond=0
            )
        )

        transcript = (
            "Meera told me that my doctor "
            "appointment is Wednesday at 3 PM."
        )

        conversation_id = (
            create_conversation_session(
                started_at=(
                    started_at
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

        add_conversation_segment(
            conversation_id=(
                conversation_id
            ),

            sequence_no=1,

            transcript=(
                transcript
            ),

            started_at=(
                started_at
            ),

            ended_at=(
                started_at
                + timedelta(
                    seconds=5
                )
            ),
        )

        finalize_conversation(
            conversation_id=(
                conversation_id
            ),

            ended_at=(
                started_at
                + timedelta(
                    seconds=5
                )
            ),

            full_transcript=(
                transcript
            ),

            summary=(
                "Doctor appointment is "
                "Wednesday at 3 PM."
            ),

            title=(
                "Medical Appointment"
            ),

            participant_text=(
                "Meera"
            ),
        )

        conversation = (
            get_conversation_session(
                conversation_id
            )
        )

        assert (
            conversation
            is not None
        )

        vector_memory.add_conversation(
            conversation=(
                conversation
            ),

            importance_score=(
                1.0
            ),

            matched_categories=(
                "appointment",
                "schedule",
            ),
        )

        print(
            "PASS: Known conversation created"
        )

        # ====================================================
        # RETRIEVER SAFETY
        # ====================================================

        retriever = (
            ConversationMemoryRetriever(
                vector_memory=(
                    vector_memory
                )
            )
        )

        question = (
            "What did Alexander tell me "
            "about my helicopter?"
        )

        referenced = (
            retriever
            .extract_referenced_participant(
                question
            )
        )

        assert (
            referenced
            == "Alexander"
        )

        print(
            "PASS: Explicit unknown participant detected"
        )

        resolved = (
            retriever.resolve_participant(
                question
            )
        )

        assert (
            resolved
            is None
        )

        print(
            "PASS: Unknown participant not falsely resolved"
        )

        memories = (
            retriever.retrieve(
                question,
                top_k=5,
            )
        )

        assert (
            memories
            == []
        )

        print(
            (
                "PASS: Unknown participant does not "
                "retrieve unrelated conversation"
            )
        )

        # ====================================================
        # CHAT SAFETY
        # ====================================================

        chat = (
            MemorAiChat()
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
            == FALLBACK_ANSWER
        )

        print(
            (
                "PASS: MemorAiChat safely returns "
                "fallback for unknown participant"
            )
        )

    finally:

        if (
            conversation_id
            is not None
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
            "ALL UNKNOWN CONVERSATION "
            "PARTICIPANT SAFETY TESTS PASSED"
        )
    )

    print(
        "=" * 70
        + "\n"
    )


if __name__ == "__main__":

    test_unknown_conversation_participant()