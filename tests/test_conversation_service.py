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
    get_conversation_segments,
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

from database.db_manager import (
    initialize_database,
)


# ============================================================
# FAKE TRANSCRIBER
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

def test_conversation_service() -> None:

    print(
        (
            "\nMemorAi Privacy-Aware "
            "Conversation Service Test Suite\n"
        )
    )

    # ========================================================
    # DATABASE
    # ========================================================

    initialize_database()

    print_test_result(
        "Conversation database available"
    )

    # ========================================================
    # PRESERVE USER SETTINGS
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

    recording_dir = (
        Path(
            "recordings"
        )
    )

    recording_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_audio = (
        recording_dir
        / "temporary_privacy_service_test.webm"
    )

    temp_audio.write_bytes(
        b"temporary fake conversation audio"
    )

    created_conversation_ids: list[int] = []

    try:

        # ====================================================
        # STANDARD TEST THRESHOLD
        # ====================================================

        set_importance_threshold(
            0.60
        )

        # ====================================================
        # PRIVACY DISABLED
        # ====================================================

        set_conversation_memory_enabled(
            False
        )

        blocked_transcriber = (
            FakeConversationTranscriber(
                transcript=(
                    "Rahul said my doctor appointment "
                    "is Monday at 10 AM."
                )
            )
        )

        blocked_service = (
            ConversationService(
                transcriber=(
                    blocked_transcriber
                )
            )
        )

        try:

            blocked_service.process_complete_audio(
                temp_audio
            )

        except PermissionError:

            pass

        else:

            raise AssertionError(
                (
                    "Conversation processing was "
                    "not blocked while disabled."
                )
            )

        assert (
            blocked_transcriber.calls
            == 0
        )

        print_test_result(
            (
                "Disabled Conversation Memory "
                "blocks processing before Whisper"
            )
        )

        # ====================================================
        # ENABLE
        # ====================================================

        set_conversation_memory_enabled(
            True
        )

        # ====================================================
        # CASUAL CONVERSATION
        # ====================================================

        casual_transcriber = (
            FakeConversationTranscriber(
                transcript=(
                    "Hello. How are you?"
                )
            )
        )

        casual_service = (
            ConversationService(
                transcriber=(
                    casual_transcriber
                )
            )
        )

        casual_result = (
            casual_service
            .process_complete_audio(
                audio_path=(
                    temp_audio
                ),
                title=(
                    "Temporary casual conversation"
                ),
            )
        )

        assert (
            casual_transcriber.calls
            == 1
        )

        assert (
            casual_result.stored
            is False
        )

        assert (
            casual_result.conversation_id
            is None
        )

        assert (
            casual_result.importance_score
            < 0.60
        )

        print_test_result(
            (
                "Casual conversation is "
                "discarded without permanent storage"
            )
        )

        # ====================================================
        # IMPORTANT APPOINTMENT CONVERSATION
        # ====================================================

        important_transcript = (
            "Rahul said my doctor appointment "
            "is Monday at 10 AM and he will "
            "pick me up at 9:30 AM."
        )

        important_transcriber = (
            FakeConversationTranscriber(
                transcript=(
                    important_transcript
                )
            )
        )

        important_service = (
            ConversationService(
                transcriber=(
                    important_transcriber
                )
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
                seconds=12
            )
        )

        important_result = (
            important_service
            .process_complete_audio(
                audio_path=(
                    temp_audio
                ),
                title=(
                    "Doctor appointment"
                ),
                participant_text=(
                    "Rahul"
                ),
                started_at=(
                    started_at
                ),
                ended_at=(
                    ended_at
                ),
            )
        )

        assert (
            important_result.stored
            is True
        )

        assert (
            important_result.conversation_id
            is not None
        )

        assert (
            important_result.importance_score
            >= 0.60
        )

        conversation_id = int(
            important_result.conversation_id
        )

        created_conversation_ids.append(
            conversation_id
        )

        print_test_result(
            "Important conversation accepted"
        )

        # ====================================================
        # VERIFY DATABASE SESSION
        # ====================================================

        stored_session = (
            get_conversation_session(
                conversation_id
            )
        )

        assert (
            stored_session
            is not None
        )

        assert (
            stored_session[
                "status"
            ]
            == "completed"
        )

        assert (
            stored_session[
                "participant_text"
            ]
            == "Rahul"
        )

        assert (
            stored_session[
                "title"
            ]
            == "Doctor appointment"
        )

        assert (
            important_transcript
            == stored_session[
                "full_transcript"
            ]
        )

        print_test_result(
            (
                "Important conversation stored "
                "in conversation_sessions"
            )
        )

        # ====================================================
        # VERIFY ORIGINAL TRANSCRIPT SEGMENT
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
            == 1
        )

        assert (
            segments[0][
                "sequence_no"
            ]
            == 1
        )

        assert (
            segments[0][
                "transcript"
            ]
            == important_transcript
        )

        print_test_result(
            "Original important transcript preserved"
        )

        # ====================================================
        # MEDICINE CONVERSATION
        # ====================================================

        medicine_transcriber = (
            FakeConversationTranscriber(
                transcript=(
                    "Remember to take your medicine "
                    "after dinner tonight and call "
                    "the doctor tomorrow morning."
                )
            )
        )

        medicine_service = (
            ConversationService(
                transcriber=(
                    medicine_transcriber
                )
            )
        )

        medicine_result = (
            medicine_service
            .process_complete_audio(
                audio_path=(
                    temp_audio
                ),
                title=(
                    "Medicine reminder"
                ),
            )
        )

        assert (
            medicine_result.stored
            is True
        )

        assert (
            medicine_result.conversation_id
            is not None
        )

        created_conversation_ids.append(
            int(
                medicine_result.conversation_id
            )
        )

        print_test_result(
            "Important medicine conversation stored"
        )

        # ====================================================
        # LOCATION MEMORY
        # ====================================================

        location_transcriber = (
            FakeConversationTranscriber(
                transcript=(
                    "I kept your passport in the "
                    "second drawer. Remember to take "
                    "it tomorrow morning."
                )
            )
        )

        location_service = (
            ConversationService(
                transcriber=(
                    location_transcriber
                )
            )
        )

        location_result = (
            location_service
            .process_complete_audio(
                audio_path=(
                    temp_audio
                ),
            )
        )

        assert (
            location_result.stored
            is True
        )

        assert (
            location_result.conversation_id
            is not None
        )

        created_conversation_ids.append(
            int(
                location_result.conversation_id
            )
        )

        print_test_result(
            "Important object-location conversation stored"
        )

        # ====================================================
        # EMPTY TRANSCRIPTION
        # ====================================================

        empty_transcriber = (
            FakeConversationTranscriber(
                transcript=""
            )
        )

        empty_service = (
            ConversationService(
                transcriber=(
                    empty_transcriber
                )
            )
        )

        try:

            empty_service.process_complete_audio(
                temp_audio
            )

        except ValueError:

            print_test_result(
                "Empty transcription safely rejected"
            )

        else:

            raise AssertionError(
                (
                    "Empty conversation transcription "
                    "was incorrectly accepted."
                )
            )

        # ====================================================
        # MISSING AUDIO FILE
        # ====================================================

        try:

            important_service.process_complete_audio(
                (
                    "recordings/"
                    "missing_conversation_audio.webm"
                )
            )

        except FileNotFoundError:

            print_test_result(
                "Missing temporary audio safely rejected"
            )

        else:

            raise AssertionError(
                (
                    "Missing conversation audio "
                    "was incorrectly accepted."
                )
            )

    finally:

        # ====================================================
        # DELETE TEST CONVERSATIONS
        # ====================================================

        for conversation_id in (
            created_conversation_ids
        ):

            try:

                delete_conversation(
                    conversation_id
                )

            except Exception:

                pass

        # ====================================================
        # DELETE TEMPORARY AUDIO
        # ====================================================

        if temp_audio.exists():

            temp_audio.unlink()

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
            "\nAll MemorAi Privacy-Aware "
            "Conversation Service tests passed."
        )
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    test_conversation_service()