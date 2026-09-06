from __future__ import annotations

from datetime import (
    datetime,
    timedelta,
)

from conversation.conversation_repository import (
    add_conversation_segment,
    build_full_transcript,
    create_conversation_session,
    delete_conversation,
    finalize_conversation,
    get_conversation_segments,
    get_conversation_session,
    get_conversations_for_date,
    get_next_segment_number,
    search_conversations,
    update_conversation_status,
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
# MAIN TEST
# ============================================================

def test_conversation_repository() -> None:

    print(
        "\nMemorAi Conversation Repository Test Suite\n"
    )

    # ========================================================
    # INITIALIZE DATABASE
    # ========================================================

    initialize_database()

    print_test_result(
        "Conversation database tables"
    )

    conversation_id: int | None = None

    try:

        # ====================================================
        # CREATE SESSION
        # ====================================================

        started_at = (
            datetime.now()
            .replace(
                microsecond=0
            )
        )

        conversation_id = (
            create_conversation_session(
                started_at=started_at,
                title=(
                    "Temporary conversation test"
                ),
                participant_text=(
                    "Rahul"
                ),
            )
        )

        assert (
            conversation_id
            > 0
        )

        session = (
            get_conversation_session(
                conversation_id
            )
        )

        assert (
            session
            is not None
        )

        assert (
            session[
                "status"
            ]
            == "recording"
        )

        assert (
            session[
                "participant_text"
            ]
            == "Rahul"
        )

        print_test_result(
            "Create conversation session"
        )

        # ====================================================
        # FIRST SEGMENT
        # ====================================================

        segment_1_id = (
            add_conversation_segment(
                conversation_id=(
                    conversation_id
                ),
                sequence_no=1,
                transcript=(
                    "Rahul told me that my "
                    "doctor appointment is Monday."
                ),
                started_at=started_at,
                ended_at=(
                    started_at
                    + timedelta(
                        seconds=4
                    )
                ),
            )
        )

        assert (
            segment_1_id
            > 0
        )

        print_test_result(
            "Store first conversation segment"
        )

        # ====================================================
        # NEXT SEGMENT NUMBER
        # ====================================================

        next_sequence = (
            get_next_segment_number(
                conversation_id
            )
        )

        assert (
            next_sequence
            == 2
        )

        print_test_result(
            "Conversation segment sequencing"
        )

        # ====================================================
        # SECOND SEGMENT
        # ====================================================

        segment_2_id = (
            add_conversation_segment(
                conversation_id=(
                    conversation_id
                ),
                sequence_no=(
                    next_sequence
                ),
                transcript=(
                    "He said that he will pick "
                    "me up at nine thirty."
                ),
                started_at=(
                    started_at
                    + timedelta(
                        seconds=5
                    )
                ),
                ended_at=(
                    started_at
                    + timedelta(
                        seconds=9
                    )
                ),
            )
        )

        assert (
            segment_2_id
            > 0
        )

        print_test_result(
            "Store second conversation segment"
        )

        # ====================================================
        # RETRIEVE SEGMENTS
        # ====================================================

        segments = (
            get_conversation_segments(
                conversation_id
            )
        )

        assert (
            len(
                segments
            )
            == 2
        )

        assert (
            segments[0][
                "sequence_no"
            ]
            == 1
        )

        assert (
            segments[1][
                "sequence_no"
            ]
            == 2
        )

        print_test_result(
            "Retrieve ordered conversation segments"
        )

        # ====================================================
        # BUILD ORIGINAL TRANSCRIPT
        # ====================================================

        transcript = (
            build_full_transcript(
                conversation_id
            )
        )

        expected_text_1 = (
            "doctor appointment is Monday"
        )

        expected_text_2 = (
            "pick me up at nine thirty"
        )

        assert (
            expected_text_1
            in transcript
        )

        assert (
            expected_text_2
            in transcript
        )

        print_test_result(
            "Build full conversation transcript"
        )

        # ====================================================
        # PROCESSING STATUS
        # ====================================================

        update_conversation_status(
            conversation_id,
            "processing",
        )

        session = (
            get_conversation_session(
                conversation_id
            )
        )

        assert (
            session
            is not None
        )

        assert (
            session[
                "status"
            ]
            == "processing"
        )

        print_test_result(
            "Update conversation status"
        )

        # ====================================================
        # FINALIZE
        # ====================================================

        ended_at = (
            started_at
            + timedelta(
                seconds=10
            )
        )

        summary = (
            "Rahul reminded the user about "
            "a doctor appointment on Monday "
            "and said he would pick them up "
            "at 9:30."
        )

        finalize_conversation(
            conversation_id=(
                conversation_id
            ),
            ended_at=(
                ended_at
            ),
            summary=(
                summary
            ),
        )

        session = (
            get_conversation_session(
                conversation_id
            )
        )

        assert (
            session
            is not None
        )

        assert (
            session[
                "status"
            ]
            == "completed"
        )

        assert (
            session[
                "full_transcript"
            ]
        )

        assert (
            expected_text_1
            in session[
                "full_transcript"
            ]
        )

        assert (
            session[
                "summary"
            ]
            == summary
        )

        assert (
            session[
                "ended_at"
            ]
            is not None
        )

        print_test_result(
            "Finalize conversation session"
        )

        # ====================================================
        # DATE RETRIEVAL
        # ====================================================

        date_results = (
            get_conversations_for_date(
                started_at.date(),
                completed_only=True,
            )
        )

        matching_ids = {
            int(
                item[
                    "id"
                ]
            )
            for item
            in date_results
        }

        assert (
            conversation_id
            in matching_ids
        )

        print_test_result(
            "Conversation date retrieval"
        )

        # ====================================================
        # TEXT SEARCH
        # ====================================================

        search_results = (
            search_conversations(
                "doctor appointment"
            )
        )

        matching_ids = {
            int(
                item[
                    "id"
                ]
            )
            for item
            in search_results
        }

        assert (
            conversation_id
            in matching_ids
        )

        print_test_result(
            "Conversation text search"
        )

        # ====================================================
        # SUMMARY SEARCH
        # ====================================================

        summary_results = (
            search_conversations(
                "9:30"
            )
        )

        matching_ids = {
            int(
                item[
                    "id"
                ]
            )
            for item
            in summary_results
        }

        assert (
            conversation_id
            in matching_ids
        )

        print_test_result(
            "Conversation summary search"
        )

    finally:

        # ====================================================
        # CLEANUP
        # ====================================================

        if (
            conversation_id
            is not None
        ):

            delete_conversation(
                conversation_id
            )

            deleted_session = (
                get_conversation_session(
                    conversation_id
                )
            )

            assert (
                deleted_session
                is None
            )

            deleted_segments = (
                get_conversation_segments(
                    conversation_id
                )
            )

            assert (
                deleted_segments
                == []
            )

            print_test_result(
                "Conversation cascading cleanup"
            )

    print(
        "\nAll MemorAi Conversation "
        "Repository tests passed."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    test_conversation_repository()