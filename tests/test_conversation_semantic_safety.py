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

from memory.conversation_retriever import (
    ConversationMemoryRetriever,
)


# ============================================================
# TEST
# ============================================================

def test_conversation_semantic_safety() -> None:

    print(
        "\n"
        + "=" * 70
    )

    print(
        "MemorAi Conversation Semantic Safety Test"
    )

    print(
        "=" * 70
    )

    initialize_database()

    vector_memory = (
        ConversationVectorMemory()
    )

    appointment_id: (
        int
        | None
    ) = None

    shopping_id: (
        int
        | None
    ) = None

    try:

        now = (
            datetime.now()
            .replace(
                microsecond=0
            )
        )

        # ====================================================
        # APPOINTMENT MEMORY
        # ====================================================

        appointment_text = (
            "I will have a doctor appointment "
            "tomorrow at 10 AM."
        )

        appointment_id = (
            create_conversation_session(
                started_at=(
                    now
                ),

                title=(
                    "Medical Appointment"
                ),

                participant_text=None,

                status="processing",
            )
        )

        add_conversation_segment(
            conversation_id=(
                appointment_id
            ),

            sequence_no=1,

            transcript=(
                appointment_text
            ),

            started_at=(
                now
            ),

            ended_at=(
                now
                + timedelta(
                    seconds=5
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
                    seconds=5
                )
            ),

            full_transcript=(
                appointment_text
            ),

            summary=(
                appointment_text
            ),

            title=(
                "Medical Appointment"
            ),
        )

        appointment = (
            get_conversation_session(
                appointment_id
            )
        )

        assert (
            appointment
            is not None
        )

        vector_memory.add_conversation(
            appointment,
            importance_score=0.80,
            matched_categories=(
                "appointment",
                "schedule",
            ),
        )

        print(
            "PASS: Appointment memory created"
        )

        # ====================================================
        # SHOPPING MEMORY
        # ====================================================

        shopping_started = (
            now
            + timedelta(
                minutes=2
            )
        )

        shopping_text = (
            "Remember to buy milk and bread "
            "tomorrow."
        )

        shopping_id = (
            create_conversation_session(
                started_at=(
                    shopping_started
                ),

                title=(
                    "Shopping Reminder"
                ),

                participant_text=None,

                status="processing",
            )
        )

        add_conversation_segment(
            conversation_id=(
                shopping_id
            ),

            sequence_no=1,

            transcript=(
                shopping_text
            ),

            started_at=(
                shopping_started
            ),

            ended_at=(
                shopping_started
                + timedelta(
                    seconds=5
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
                    seconds=5
                )
            ),

            full_transcript=(
                shopping_text
            ),

            summary=(
                "Buy milk and bread tomorrow."
            ),

            title=(
                "Shopping Reminder"
            ),
        )

        shopping = (
            get_conversation_session(
                shopping_id
            )
        )

        assert (
            shopping
            is not None
        )

        vector_memory.add_conversation(
            shopping,
            importance_score=0.90,
            matched_categories=(
                "reminder",
                "task",
                "schedule",
            ),
        )

        print(
            "PASS: Shopping memory created"
        )

        # ====================================================
        # RETRIEVE BUY QUESTION
        # ====================================================

        retriever = (
            ConversationMemoryRetriever(
                vector_memory=(
                    vector_memory
                )
            )
        )

        question = (
            "What do I need to buy tomorrow?"
        )

        memories = (
            retriever.retrieve(
                question,
                top_k=5,
            )
        )

        assert (
            memories
        )

        ids = {
            memory.conversation_id

            for memory
            in memories
        }

        assert (
            shopping_id
            in ids
        )

        assert (
            appointment_id
            not in ids
        )

        print(
            (
                "PASS: Buy question retrieves "
                "shopping memory"
            )
        )

        print(
            (
                "PASS: Buy question rejects "
                "unrelated appointment memory"
            )
        )

        # ====================================================
        # REMOVE SHOPPING VECTOR
        #
        # Now only the appointment remains semantically
        # available.
        # ====================================================

        vector_memory.delete_conversation(
            shopping_id
        )

        delete_conversation(
            shopping_id
        )

        shopping_id = None

        memories_without_shopping = (
            retriever.retrieve(
                question,
                top_k=5,
            )
        )

        # ----------------------------------------------------
        # There is no valid BUY evidence anymore.
        # ----------------------------------------------------

        assert (
            memories_without_shopping
            == []
        )

        print(
            (
                "PASS: No shopping memory returns "
                "no unrelated conversation"
            )
        )

    finally:

        if (
            appointment_id
            is not None
        ):

            try:

                vector_memory.delete_conversation(
                    appointment_id
                )

            except Exception:

                pass

            try:

                delete_conversation(
                    appointment_id
                )

            except Exception:

                pass

        if (
            shopping_id
            is not None
        ):

            try:

                vector_memory.delete_conversation(
                    shopping_id
                )

            except Exception:

                pass

            try:

                delete_conversation(
                    shopping_id
                )

            except Exception:

                pass

    print(
        "\n"
        + "=" * 70
    )

    print(
        (
            "ALL CONVERSATION SEMANTIC "
            "SAFETY TESTS PASSED"
        )
    )

    print(
        "=" * 70
        + "\n"
    )


if __name__ == "__main__":

    test_conversation_semantic_safety()