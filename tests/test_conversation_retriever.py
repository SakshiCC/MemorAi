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
)

from database.conversation_vector_memory import (
    ConversationVectorMemory,
)

from database.db_manager import (
    initialize_database,
)

from memory.conversation_retriever import (
    ConversationMemoryRetriever,
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

def test_conversation_retriever() -> None:

    print(
        (
            "\nMemorAi Conversation "
            "Retriever Test Suite\n"
        )
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
        # TEMPORARY CONVERSATION
        # ====================================================

        started_at = (
            datetime.now()
            .replace(
                microsecond=0
            )
        )

        ended_at = (
            started_at
            + timedelta(
                seconds=15
            )
        )

        transcript = (
            "Meera told me that my physiotherapy "
            "appointment is on Wednesday at 3 PM. "
            "She reminded me to bring my exercise "
            "sheet and arrive ten minutes early."
        )

        conversation_id = (
            create_conversation_session(
                started_at=(
                    started_at
                ),

                title=(
                    "Physiotherapy appointment"
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
                ended_at
            ),
        )

        finalize_conversation(
            conversation_id=(
                conversation_id
            ),

            ended_at=(
                ended_at
            ),

            full_transcript=(
                transcript
            ),

            title=(
                "Physiotherapy appointment"
            ),

            participant_text=(
                "Meera"
            ),
        )

        from conversation.conversation_repository import (
            get_conversation_session,
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
                "reminder",
                "schedule",
                "task",
            ),
        )

        print_test_result(
            "Temporary conversation indexed"
        )

        # ====================================================
        # RETRIEVER
        # ====================================================

        retriever = (
            ConversationMemoryRetriever(
                vector_memory=(
                    vector_memory
                )
            )
        )

        # ====================================================
        # INTENT
        # ====================================================

        assert (
            retriever.is_conversation_question(
                (
                    "What did Meera tell me "
                    "about my appointment?"
                )
            )
            is True
        )

        assert (
            retriever.is_conversation_question(
                "Where is my phone?"
            )
            is False
        )

        print_test_result(
            "Conversation question detection"
        )

        # ====================================================
        # PARTICIPANT
        # ====================================================

        participant = (
            retriever.resolve_participant(
                (
                    "What did Meera tell me "
                    "about physiotherapy?"
                )
            )
        )

        assert (
            participant
            == "Meera"
        )

        print_test_result(
            "Conversation participant resolution"
        )

        # ====================================================
        # SEMANTIC RETRIEVAL
        # ====================================================

        memories = (
            retriever.retrieve(
                (
                    "What should I bring to my "
                    "physiotherapy appointment?"
                ),
                top_k=5,
            )
        )

        assert (
            memories
        )

        matching = [
            memory

            for memory
            in memories

            if (
                memory.conversation_id
                == conversation_id
            )
        ]

        assert (
            matching
        )

        result = (
            matching[0]
        )

        assert (
            "exercise sheet"
            in str(
                result.metadata.get(
                    "full_transcript"
                )
            ).lower()
        )

        print_test_result(
            "Semantic conversation retrieval"
        )

        # ====================================================
        # PERSON + SEMANTIC
        # ====================================================

        meera_memories = (
            retriever.retrieve(
                (
                    "What did Meera tell me "
                    "to bring?"
                ),
                top_k=5,
            )
        )

        assert (
            any(
                memory.conversation_id
                == conversation_id

                for memory
                in meera_memories
            )
        )

        print_test_result(
            "Participant conversation retrieval"
        )

        # ====================================================
        # TODAY
        # ====================================================

        today_memories = (
            retriever.retrieve(
                (
                    "What did Meera tell me today?"
                ),
                top_k=10,
            )
        )

        assert (
            any(
                memory.conversation_id
                == conversation_id

                for memory
                in today_memories
            )
        )

        print_test_result(
            "Today's conversation retrieval"
        )

        # ====================================================
        # RECENT
        # ====================================================

        recent_memories = (
            retriever.retrieve(
                (
                    "What was my latest "
                    "conversation with Meera?"
                ),
                top_k=5,
            )
        )

        assert (
            any(
                memory.conversation_id
                == conversation_id

                for memory
                in recent_memories
            )
        )

        print_test_result(
            "Recent conversation retrieval"
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
        (
            "\nAll MemorAi Conversation "
            "Retriever tests passed."
        )
    )


if __name__ == "__main__":

    test_conversation_retriever()