from __future__ import annotations

from datetime import (
    datetime,
    timedelta,
)

from pathlib import (
    Path,
)

from conversation.conversation_repository import (
    delete_conversation,
    get_conversation_session,
)

from conversation.conversation_service import (
    ConversationService,
)

from conversation.conversation_settings import (
    get_importance_threshold,
    is_conversation_memory_enabled,
    set_conversation_memory_enabled,
    set_importance_threshold,
)

from database.conversation_vector_memory import (
    ConversationVectorMemory,
)

from database.db_manager import (
    initialize_database,
)


# ============================================================
# FAKE WHISPER
# ============================================================

class FakeConversationTranscriber:

    def __init__(
        self,
        transcript: str,
    ) -> None:

        self.transcript = (
            transcript
        )

        self.calls = 0


    def transcribe(
        self,
        audio_path: str,
    ) -> str:

        self.calls += 1

        return (
            self.transcript
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

def test_conversation_vector_integration() -> None:

    print(
        (
            "\nMemorAi Conversation Vector "
            "Integration Test Suite\n"
        )
    )

    initialize_database()

    # ========================================================
    # PRESERVE REAL SETTINGS
    # ========================================================

    original_enabled = (
        is_conversation_memory_enabled()
    )

    original_threshold = (
        get_importance_threshold()
    )

    # ========================================================
    # TEMP AUDIO
    # ========================================================

    recordings_dir = (
        Path(
            "recordings"
        )
    )

    recordings_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_audio = (
        recordings_dir
        / "temporary_conversation_vector_integration.webm"
    )

    temp_audio.write_bytes(
        b"temporary fake conversation audio"
    )

    conversation_id: (
        int
        | None
    ) = None

    vector_memory = (
        ConversationVectorMemory()
    )

    try:

        # ====================================================
        # ENABLE FOR TEST
        # ====================================================

        set_conversation_memory_enabled(
            True
        )

        set_importance_threshold(
            0.60
        )

        # ====================================================
        # IMPORTANT TRANSCRIPT
        # ====================================================

        transcript = (
            "Neha reminded me that my dentist "
            "appointment is on Tuesday at 11 AM. "
            "I need to bring my previous reports "
            "and arrive fifteen minutes early."
        )

        fake_transcriber = (
            FakeConversationTranscriber(
                transcript=(
                    transcript
                )
            )
        )

        service = (
            ConversationService(
                transcriber=(
                    fake_transcriber
                ),
                vector_memory=(
                    vector_memory
                ),
            )
        )

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

        # ====================================================
        # PROCESS
        # ====================================================

        result = (
            service.process_complete_audio(
                audio_path=(
                    temp_audio
                ),

                title=(
                    "Dentist appointment"
                ),

                participant_text=(
                    "Neha"
                ),

                started_at=(
                    started_at
                ),

                ended_at=(
                    ended_at
                ),
            )
        )

        # ====================================================
        # WHISPER CALLED
        # ====================================================

        assert (
            fake_transcriber.calls
            == 1
        )

        print_test_result(
            "Conversation transcription pipeline"
        )

        # ====================================================
        # IMPORTANT / STORED
        # ====================================================

        assert (
            result.stored
            is True
        )

        assert (
            result.conversation_id
            is not None
        )

        assert (
            result.importance_score
            >= 0.60
        )

        conversation_id = int(
            result.conversation_id
        )

        print_test_result(
            "Important conversation stored"
        )

        # ====================================================
        # MYSQL
        # ====================================================

        mysql_record = (
            get_conversation_session(
                conversation_id
            )
        )

        assert (
            mysql_record
            is not None
        )

        assert (
            mysql_record[
                "status"
            ]
            == "completed"
        )

        assert (
            mysql_record[
                "participant_text"
            ]
            == "Neha"
        )

        assert (
            transcript
            == mysql_record[
                "full_transcript"
            ]
        )

        print_test_result(
            "Important conversation preserved in MySQL"
        )

        # ====================================================
        # VECTOR INDEX STATUS
        # ====================================================

        assert (
            result.vector_indexed
            is True
        )

        print_test_result(
            "Conversation automatically vector indexed"
        )

        # ====================================================
        # EXACT VECTOR MEMORY
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
            (
                "dentist appointment"
                in vector_record[
                    "text"
                ].lower()
            )
        )

        assert (
            vector_record[
                "metadata"
            ][
                "memory_type"
            ]
            == "conversation"
        )

        assert (
            vector_record[
                "metadata"
            ][
                "participant_text"
            ]
            == "Neha"
        )

        print_test_result(
            "Conversation exists in ChromaDB"
        )

        # ====================================================
        # SEMANTIC SEARCH
        # ====================================================

        semantic_result = (
            vector_memory.search(
                query=(
                    "What do I need to bring "
                    "to my dentist visit?"
                ),
                n_results=5,
            )
        )

        semantic_ids = (
            semantic_result.get(
                "ids",
                [[]],
            )[0]
        )

        expected_memory_id = (
            "conversation_"
            f"{conversation_id}"
        )

        assert (
            expected_memory_id
            in semantic_ids
        )

        print_test_result(
            "Live conversation semantic retrieval"
        )

        # ====================================================
        # PARTICIPANT FILTER
        # ====================================================

        participant_result = (
            vector_memory.search(
                query=(
                    "What appointment did "
                    "we discuss?"
                ),
                n_results=5,
                participant_text=(
                    "Neha"
                ),
            )
        )

        participant_ids = (
            participant_result.get(
                "ids",
                [[]],
            )[0]
        )

        assert (
            expected_memory_id
            in participant_ids
        )

        print_test_result(
            "Live conversation participant retrieval"
        )

        # ====================================================
        # MANUAL REINDEX
        # ====================================================

        reindexed = (
            service.index_conversation(
                conversation_id=(
                    conversation_id
                ),
                importance_score=(
                    result.importance_score
                ),
                matched_categories=(
                    result.matched_categories
                ),
            )
        )

        assert (
            reindexed
            is True
        )

        second_vector_record = (
            vector_memory
            .get_conversation(
                conversation_id
            )
        )

        assert (
            second_vector_record
            is not None
        )

        print_test_result(
            "Existing conversation can be safely re-indexed"
        )

    finally:

        # ====================================================
        # DELETE CHROMA TEST MEMORY
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
            # DELETE MYSQL TEST MEMORY
            # =================================================

            try:

                delete_conversation(
                    conversation_id
                )

            except Exception:

                pass

        # ====================================================
        # DELETE TEMP AUDIO
        # ====================================================

        if temp_audio.exists():

            temp_audio.unlink()

        # ====================================================
        # RESTORE USER SETTINGS
        # ====================================================

        set_conversation_memory_enabled(
            original_enabled
        )

        set_importance_threshold(
            original_threshold
        )

    print(
        (
            "\nAll MemorAi Conversation Vector "
            "Integration tests passed."
        )
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    test_conversation_vector_integration()