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

from database.backfill_conversation_metadata import (
    backfill_conversation_metadata,
    conversation_needs_metadata_backfill,
)

from database.conversation_vector_memory import (
    ConversationVectorMemory,
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

def test_backfill_conversation_metadata() -> None:

    print(
        (
            "\nMemorAi Conversation Metadata "
            "Backfill Test Suite\n"
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

    preserve_id: (
        int
        | None
    ) = None

    try:

        # ====================================================
        # TEST 1
        # CREATE OLD-STYLE CONVERSATION
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
                seconds=12
            )
        )

        transcript = (
            "Riya told me that my doctor "
            "appointment is on Thursday at "
            "2 PM. I need to bring my medical "
            "reports and arrive ten minutes early."
        )

        conversation_id = (
            create_conversation_session(
                started_at=(
                    started_at
                ),

                title=None,

                participant_text=None,

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

            title=None,

            participant_text=None,

            summary=None,
        )

        old_record = (
            get_conversation_session(
                conversation_id
            )
        )

        assert (
            old_record
            is not None
        )

        assert (
            old_record[
                "title"
            ]
            is None
        )

        assert (
            old_record[
                "participant_text"
            ]
            is None
        )

        assert (
            old_record[
                "summary"
            ]
            is None
        )

        print_test_result(
            "Old-style conversation created"
        )

        # ====================================================
        # TEST 2
        # DETECT MISSING METADATA
        # ====================================================

        assert (
            conversation_needs_metadata_backfill(
                old_record
            )
            is True
        )

        print_test_result(
            "Missing metadata detected"
        )

        # ====================================================
        # TEST 3
        # BACKFILL
        # ====================================================

        status = (
            backfill_conversation_metadata(
                conversation=(
                    old_record
                ),

                vector_memory=(
                    vector_memory
                ),

                reindex=True,
            )
        )

        assert (
            status
            == "updated"
        )

        print_test_result(
            "Conversation metadata backfilled"
        )

        # ====================================================
        # TEST 4
        # VERIFY MYSQL
        # ====================================================

        updated = (
            get_conversation_session(
                conversation_id
            )
        )

        assert (
            updated
            is not None
        )

        assert (
            updated[
                "title"
            ]
            == "Medical Appointment"
        )

        assert (
            updated[
                "participant_text"
            ]
            == "Riya"
        )

        summary = (
            str(
                updated[
                    "summary"
                ]
                or ""
            )
            .lower()
        )

        assert (
            "appointment"
            in summary
        )

        assert (
            "medical reports"
            in summary
        )

        assert (
            "ten minutes early"
            in summary
        )

        print_test_result(
            (
                "Generated metadata saved "
                "to MySQL"
            )
        )

        # ====================================================
        # TEST 5
        # VERIFY CHROMA REINDEX
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

        vector_metadata = (
            vector_record[
                "metadata"
            ]
        )

        assert (
            vector_metadata[
                "title"
            ]
            == "Medical Appointment"
        )

        assert (
            vector_metadata[
                "participant_text"
            ]
            == "Riya"
        )

        assert (
            "medical reports"
            in vector_record[
                "text"
            ].lower()
        )

        print_test_result(
            (
                "Backfilled metadata "
                "re-indexed into ChromaDB"
            )
        )

        # ====================================================
        # TEST 6
        # ALREADY COMPLETE
        # ====================================================

        updated_again = (
            get_conversation_session(
                conversation_id
            )
        )

        second_status = (
            backfill_conversation_metadata(
                conversation=(
                    updated_again
                ), # type: ignore

                vector_memory=(
                    vector_memory
                ),

                reindex=True,
            )
        )

        assert (
            second_status
            == "already_complete"
        )

        print_test_result(
            (
                "Complete conversation "
                "is not modified again"
            )
        )

        # ====================================================
        # TEST 7
        # PRESERVE EXISTING VALUES
        # ====================================================

        preserve_started = (
            datetime.now()
            .replace(
                microsecond=0
            )
            + timedelta(
                minutes=2
            )
        )

        preserve_ended = (
            preserve_started
            + timedelta(
                seconds=8
            )
        )

        preserve_transcript = (
            "Rahul reminded me that my "
            "medicine should be taken after "
            "dinner tonight."
        )

        preserve_id = (
            create_conversation_session(
                started_at=(
                    preserve_started
                ),

                title=(
                    "My Custom Title"
                ),

                participant_text=(
                    "Rahul Kumar"
                ),

                status="processing",
            )
        )

        add_conversation_segment(
            conversation_id=(
                preserve_id
            ),

            sequence_no=1,

            transcript=(
                preserve_transcript
            ),

            started_at=(
                preserve_started
            ),

            ended_at=(
                preserve_ended
            ),
        )

        finalize_conversation(
            conversation_id=(
                preserve_id
            ),

            ended_at=(
                preserve_ended
            ),

            full_transcript=(
                preserve_transcript
            ),

            title=(
                "My Custom Title"
            ),

            participant_text=(
                "Rahul Kumar"
            ),

            summary=None,
        )

        preserve_record = (
            get_conversation_session(
                preserve_id
            )
        )

        assert (
            preserve_record
            is not None
        )

        preserve_status = (
            backfill_conversation_metadata(
                conversation=(
                    preserve_record
                ),

                vector_memory=(
                    vector_memory
                ),

                reindex=True,
            )
        )

        assert (
            preserve_status
            == "updated"
        )

        preserved = (
            get_conversation_session(
                preserve_id
            )
        )

        assert (
            preserved
            is not None
        )

        assert (
            preserved[
                "title"
            ]
            == "My Custom Title"
        )

        assert (
            preserved[
                "participant_text"
            ]
            == "Rahul Kumar"
        )

        assert (
            preserved[
                "summary"
            ]
            is not None
        )

        print_test_result(
            "Existing metadata values preserved"
        )

        # ====================================================
        # TEST 8
        # SEMANTIC SEARCH AFTER BACKFILL
        # ====================================================

        search_result = (
            vector_memory.search(
                query=(
                    "What reports should I bring "
                    "to my doctor appointment?"
                ),
                n_results=10,
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
                "Backfilled conversation remains "
                "semantically searchable"
            )
        )

    finally:

        # ====================================================
        # CLEANUP FIRST CONVERSATION
        # ====================================================

        if (
            conversation_id
            is not None
        ):

            try:

                vector_memory
                .delete_conversation(
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

        # ====================================================
        # CLEANUP PRESERVATION CONVERSATION
        # ====================================================

        if (
            preserve_id
            is not None
        ):

            try:

                vector_memory
                .delete_conversation(
                    preserve_id
                )

            except Exception:

                pass

            try:

                delete_conversation(
                    preserve_id
                )

            except Exception:

                pass

    print(
        (
            "\nAll MemorAi Conversation Metadata "
            "Backfill tests passed."
        )
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    test_backfill_conversation_metadata()