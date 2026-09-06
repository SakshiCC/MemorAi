from __future__ import annotations

from datetime import (
    datetime,
    timedelta,
)

from pathlib import (
    Path,
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

from speech.voice_chat import (
    MemorAiVoiceChat,
)


# ============================================================
# FAKE TRANSCRIBER
# ============================================================

class FakeTranscriber:
    """
    Fake Whisper transcriber.

    This avoids:

        - microphone usage
        - real Whisper loading
        - speaking aloud

    while still testing:

        fake speech
            ↓
        MemorAiVoiceChat
            ↓
        real MemorAiChat
            ↓
        real Conversation RAG
    """

    def __init__(
        self,
        text: str,
    ) -> None:

        self.text = (
            text
        )

        self.call_count = 0

        self.last_audio_path = None


    def transcribe(
        self,
        audio_path,
    ) -> str:

        self.call_count += 1

        self.last_audio_path = (
            str(
                audio_path
            )
        )

        return (
            self.text
        )


# ============================================================
# HELPERS
# ============================================================

def print_test_result(
    test_name: str,
) -> None:

    print(
        f"PASS: {test_name}"
    )


def print_question_answer(
    question: str,
    answer: str,
) -> None:

    print(
        "\nQuestion:",
        question,
    )

    print(
        "Answer:",
        answer,
    )


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

        for value
        in values
    ), (
        f"Expected one of {values} "
        f"in answer:\n{text}"
    )


# ============================================================
# CREATE TEMPORARY CONVERSATION
# ============================================================

def create_test_conversation(
    vector_memory: ConversationVectorMemory,
    participant_text: str | None,
    title: str,
    transcript: str,
    summary: str,
    importance_score: float,
    matched_categories: tuple[str, ...],
    started_at: datetime,
) -> int:

    ended_at = (
        started_at
        + timedelta(
            seconds=8
        )
    )

    conversation_id = (
        create_conversation_session(
            started_at=(
                started_at
            ),

            title=(
                title
            ),

            participant_text=(
                participant_text
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

        summary=(
            summary
        ),

        title=(
            title
        ),

        participant_text=(
            participant_text
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
            importance_score
        ),

        matched_categories=(
            matched_categories
        ),
    )

    return (
        conversation_id
    )


# ============================================================
# ASK THROUGH FAKE VOICE
# ============================================================

def ask_through_voice(
    question_text: str,
) -> tuple[
    str,
    str,
    FakeTranscriber,
]:

    fake_transcriber = (
        FakeTranscriber(
            text=(
                question_text
            )
        )
    )

    chat = (
        MemorAiChat()
    )

    voice_chat = (
        MemorAiVoiceChat(
            chat=(
                chat
            ),

            transcriber=(
                fake_transcriber
            ),
        )
    )

    # --------------------------------------------------------
    # The fake transcriber does not inspect the audio contents.
    #
    # The file does not need to exist because FakeTranscriber
    # simply returns its configured text.
    # --------------------------------------------------------

    fake_audio_path = (
        Path(
            "recordings/"
            "fake_voice_conversation_test.wav"
        )
    )

    question, answer = (
        voice_chat.ask_from_audio(
            fake_audio_path
        )
    )

    return (
        question,
        answer,
        fake_transcriber,
    )


# ============================================================
# MAIN TEST
# ============================================================

def test_voice_conversation_rag() -> None:

    print(
        "\n"
        + "=" * 70
    )

    print(
        (
            "MemorAi Voice + Conversation "
            "RAG Integration Test Suite"
        )
    )

    print(
        "=" * 70
    )

    print(
        (
            "\nNo microphone is used. "
            "Real Whisper is not loaded."
        )
    )

    initialize_database()

    vector_memory = (
        ConversationVectorMemory()
    )

    created_ids: list[int] = []

    try:

        now = (
            datetime.now()
            .replace(
                microsecond=0
            )
        )

        # ====================================================
        # MEMORY 1
        # APPOINTMENT
        # ====================================================

        appointment_id = (
            create_test_conversation(
                vector_memory=(
                    vector_memory
                ),

                participant_text=(
                    "Meera"
                ),

                title=(
                    "Medical Appointment"
                ),

                transcript=(
                    "Meera told me that my doctor "
                    "appointment is on Friday at 4 PM. "
                    "She reminded me to bring my blood "
                    "test reports and identification card."
                ),

                summary=(
                    "Doctor appointment is Friday at "
                    "4 PM. Bring blood test reports "
                    "and identification card."
                ),

                importance_score=(
                    1.0
                ),

                matched_categories=(
                    "appointment",
                    "reminder",
                    "task",
                    "schedule",
                ),

                started_at=(
                    now
                ),
            )
        )

        created_ids.append(
            appointment_id
        )

        print_test_result(
            (
                "Appointment conversation "
                "stored and indexed"
            )
        )

        # ====================================================
        # MEMORY 2
        # SHOPPING
        # ====================================================

        shopping_id = (
            create_test_conversation(
                vector_memory=(
                    vector_memory
                ),

                participant_text=(
                    "Rahul"
                ),

                title=(
                    "Shopping Reminder"
                ),

                transcript=(
                    "Rahul reminded me to buy "
                    "rice and bananas tomorrow."
                ),

                summary=(
                    "Buy rice and bananas tomorrow."
                ),

                importance_score=(
                    0.90
                ),

                matched_categories=(
                    "reminder",
                    "task",
                    "schedule",
                ),

                started_at=(
                    now
                    + timedelta(
                        minutes=2
                    )
                ),
            )
        )

        created_ids.append(
            shopping_id
        )

        print_test_result(
            (
                "Shopping conversation "
                "stored and indexed"
            )
        )

        # ====================================================
        # TEST 1
        # EXPLICIT PARTICIPANT QUESTION
        # ====================================================

        question, answer, transcriber = (
            ask_through_voice(
                (
                    "What did Meera tell me "
                    "about my doctor appointment?"
                )
            )
        )

        print_question_answer(
            question,
            answer,
        )

        assert (
            transcriber.call_count
            == 1
        )

        assert (
            question
            == (
                "What did Meera tell me "
                "about my doctor appointment?"
            )
        )

        assert (
            answer
            != FALLBACK_ANSWER
        )

        assert_contains_any(
            answer,
            (
                "friday",
                "4 pm",
                "blood test reports",
            ),
        )

        print_test_result(
            (
                "Voice participant conversation "
                "retrieval"
            )
        )

        # ====================================================
        # TEST 2
        # APPOINTMENT DETAIL
        # ====================================================

        question, answer, _ = (
            ask_through_voice(
                (
                    "What should I bring "
                    "to my doctor appointment?"
                )
            )
        )

        print_question_answer(
            question,
            answer,
        )

        assert (
            answer
            != FALLBACK_ANSWER
        )

        assert_contains_any(
            answer,
            (
                "blood test reports",
                "identification card",
            ),
        )

        print_test_result(
            (
                "Voice semantic appointment "
                "detail retrieval"
            )
        )

        # ====================================================
        # TEST 3
        # SHOPPING QUESTION
        # ====================================================

        question, answer, _ = (
            ask_through_voice(
                (
                    "What do I need "
                    "to buy tomorrow?"
                )
            )
        )

        print_question_answer(
            question,
            answer,
        )

        assert (
            answer
            != FALLBACK_ANSWER
        )

        lowered = (
            answer.lower()
        )

        assert (
            "rice"
            in lowered
        )

        assert (
            "banana"
            in lowered
        )

        print_test_result(
            (
                "Voice shopping "
                "conversation retrieval"
            )
        )

        # ====================================================
        # TEST 4
        # UNKNOWN PARTICIPANT
        # ====================================================

        question, answer, _ = (
            ask_through_voice(
                (
                    "What did Alexander tell me "
                    "about my helicopter?"
                )
            )
        )

        print_question_answer(
            question,
            answer,
        )

        assert (
            answer
            == FALLBACK_ANSWER
        )

        print_test_result(
            (
                "Voice unknown participant "
                "safe fallback"
            )
        )

        # ====================================================
        # TEST 5
        # UNRELATED BUY QUESTION AFTER REMOVING SHOPPING
        # ====================================================

        vector_memory.delete_conversation(
            shopping_id
        )

        delete_conversation(
            shopping_id
        )

        created_ids.remove(
            shopping_id
        )

        question, answer, _ = (
            ask_through_voice(
                (
                    "What do I need "
                    "to buy tomorrow?"
                )
            )
        )

        print_question_answer(
            question,
            answer,
        )

        assert (
            answer
            == FALLBACK_ANSWER
        )

        print_test_result(
            (
                "Voice semantic safety rejects "
                "unrelated appointment memory"
            )
        )

        # ====================================================
        # TEST 6
        # REAL WHISPER NOT USED
        # ====================================================

        fake_transcriber = (
            FakeTranscriber(
                text=(
                    "When is my doctor appointment?"
                )
            )
        )

        voice_chat = (
            MemorAiVoiceChat(
                chat=(
                    MemorAiChat()
                ),

                transcriber=(
                    fake_transcriber
                ),
            )
        )

        assert (
            voice_chat.transcriber
            is fake_transcriber
        )

        assert isinstance(
            voice_chat.transcriber,
            FakeTranscriber,
        )

        print_test_result(
            (
                "Test uses fake Whisper "
                "without microphone"
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
            "ALL VOICE + CONVERSATION "
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

    test_voice_conversation_rag()