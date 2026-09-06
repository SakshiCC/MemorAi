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

from database.index_existing_conversations import (
    index_conversation_if_needed,
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

def test_index_existing_conversations() -> None:

    print(
        (
            "\nMemorAi Existing Conversation "
            "Indexer Test Suite\n"
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
        # CREATE TEMP MYSQL CONVERSATION
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
                seconds=10
            )
        )

        transcript = (
            "Riya reminded me that my eye "
            "appointment is on Friday at 4 PM "
            "and I should bring my old prescription."
        )

        conversation_id = (
            create_conversation_session(
                started_at=started_at,
                title=(
                    "Eye appointment"
                ),
                participant_text=(
                    "Riya"
                ),
                status="processing",
            )
        )

        assert (
            conversation_id
            > 0
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
                "Eye appointment"
            ),
            participant_text=(
                "Riya"
            ),
        )

        record = (
            get_conversation_session(
                conversation_id
            )
        )

        assert (
            record
            is not None
        )

        assert (
            record[
                "status"
            ]
            == "completed"
        )

        print_test_result(
            (
                "Temporary completed "
                "conversation created"
            )
        )

        # ====================================================
        # ENSURE VECTOR DOES NOT EXIST FIRST
        # ====================================================

        vector_memory.delete_conversation(
            conversation_id
        )

        assert (
            vector_memory.get_conversation(
                conversation_id
            )
            is None
        )

        print_test_result(
            (
                "Conversation starts "
                "without vector memory"
            )
        )

        # ====================================================
        # INDEX
        # ====================================================

        status = (
            index_conversation_if_needed(
                conversation=record,
                vector_memory=vector_memory,
                force=False,
            )
        )

        assert (
            status
            == "indexed"
        )

        print_test_result(
            "Existing conversation indexed"
        )

        # ====================================================
        # VERIFY CHROMA
        # ====================================================

        vector_record = (
            vector_memory
            .get_conversation(
                conversation_id
            )
        )

        assert (
            vector_record
            is not None
        )

        assert (
            "eye appointment"
            in vector_record[
                "text"
            ].lower()
        )

        assert (
            vector_record[
                "metadata"
            ][
                "participant_text"
            ]
            == "Riya"
        )

        print_test_result(
            (
                "Indexed conversation "
                "available in ChromaDB"
            )
        )

        # ====================================================
        # DO NOT DUPLICATE
        # ====================================================

        second_status = (
            index_conversation_if_needed(
                conversation=record,
                vector_memory=vector_memory,
                force=False,
            )
        )

        assert (
            second_status
            == "already_indexed"
        )

        print_test_result(
            (
                "Already indexed conversation "
                "is detected"
            )
        )

        # ====================================================
        # FORCE REINDEX
        # ====================================================

        forced_status = (
            index_conversation_if_needed(
                conversation=record,
                vector_memory=vector_memory,
                force=True,
            )
        )

        assert (
            forced_status
            == "indexed"
        )

        print_test_result(
            "Force re-index works"
        )

        # ====================================================
        # SEMANTIC SEARCH
        # ====================================================

        search_result = (
            vector_memory.search(
                query=(
                    "What should I take to "
                    "my eye doctor?"
                ),
                n_results=5,
            )
        )

        ids = (
            search_result.get(
                "ids",
                [[]],
            )[0]
        )

        assert (
            (
                "conversation_"
                f"{conversation_id}"
            )
            in ids
        )

        print_test_result(
            (
                "Re-indexed conversation "
                "is semantically searchable"
            )
        )

    finally:

        # ====================================================
        # CLEANUP CHROMA
        # ====================================================

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

            # =================================================
            # CLEANUP MYSQL
            # =================================================

            try:

                delete_conversation(
                    conversation_id
                )

            except Exception:

                pass

    print(
        (
            "\nAll MemorAi Existing Conversation "
            "Indexer tests passed."
        )
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    test_index_existing_conversations()