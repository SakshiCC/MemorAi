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

def test_conversation_participant_alias() -> None:

    print(
        "\n"
        + "=" * 70
    )

    print(
        "MemorAi Conversation Participant Alias Test"
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
        # SIMULATE OLD WHISPER SPELLING
        # ====================================================

        started_at = (
            datetime.now()
            .replace(
                microsecond=0
            )
        )

        transcript = (
            "Saksi told me that my doctor "
            "appointment is on Monday at 10 AM. "
            "I need to bring my medical reports."
        )

        conversation_id = (
            create_conversation_session(
                started_at=(
                    started_at
                ),

                title=(
                    "Medical Appointment"
                ),

                # IMPORTANT:
                # Deliberately use Whisper spelling.
                participant_text=(
                    "Saksi"
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
                    seconds=8
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
                    seconds=8
                )
            ),

            full_transcript=(
                transcript
            ),

            summary=(
                "Doctor appointment is Monday "
                "at 10 AM. Bring medical reports."
            ),

            title=(
                "Medical Appointment"
            ),

            participant_text=(
                "Saksi"
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
                "task",
            ),
        )

        print(
            "PASS: Whisper-style participant record created"
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
        # CANONICAL DISPLAY
        # ====================================================

        canonical = (
            retriever.canonicalize_participant(
                "Saksi"
            )
        )

        print(
            (
                "Stored participant: "
                "Saksi"
            )
        )

        print(
            (
                "Canonical participant: "
                f"{canonical}"
            )
        )

        assert (
            canonical is not None
        )

        # ----------------------------------------------------
        # This assertion expects the registered_people table
        # to contain Sakshi.
        # ----------------------------------------------------

        assert (
            canonical.lower()
            == "sakshi"
        )

        print(
            "PASS: Saksi safely canonicalized to Sakshi"
        )

        # ====================================================
        # ASK USING CORRECT SPELLING
        # ====================================================

        question = (
            "What did Sakshi tell me "
            "about my appointment?"
        )

        resolved = (
            retriever.resolve_participant(
                question
            )
        )

        print(
            (
                "Resolved storage participant: "
                f"{resolved}"
            )
        )

        assert (
            resolved
            == "Saksi"
        )

        print(
            (
                "PASS: Correct spelling resolves "
                "old Whisper spelling"
            )
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

        matching = [
            memory

            for memory in memories

            if (
                memory.conversation_id
                == conversation_id
            )
        ]

        assert (
            matching
        )

        result = (
            matching[
                0
            ]
        )

        display_participant = (
            str(
                result.metadata.get(
                    "participant_text"
                )
                or ""
            )
        )

        print(
            (
                "Participant displayed to user: "
                f"{display_participant}"
            )
        )

        assert (
            display_participant.lower()
            == "sakshi"
        )

        print(
            (
                "PASS: Retrieved memory displays "
                "canonical participant name"
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
            "ALL CONVERSATION PARTICIPANT "
            "ALIAS TESTS PASSED"
        )
    )

    print(
        "=" * 70
        + "\n"
    )


if __name__ == "__main__":

    test_conversation_participant_alias()