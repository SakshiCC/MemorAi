from __future__ import annotations

import uuid

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
    execute,
    initialize_database,
)

from database.event_repository import (
    get_latest_event_for_object,
)

from database.medicine_repository import (
    get_recent_medicine_events,
)

from database.person_repository import (
    create_person,
    save_person_event,
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

    Purpose:

        Controlled spoken question
                ↓
        MemorAiVoiceChat
                ↓
        real MemorAiChat
                ↓
        Unified RAG

    This avoids:

        - microphone usage
        - real Whisper loading
        - speaking aloud

    while still testing the complete Voice -> RAG bridge.
    """

    def __init__(
        self,
        text: str,
    ) -> None:

        self.text = (
            text
        )

        self.call_count = 0

        self.last_audio_path: (
            str
            | None
        ) = None


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
# OUTPUT HELPERS
# ============================================================

def print_header(
    title: str,
) -> None:

    print()

    print(
        "=" * 76
    )

    print(
        title
    )

    print(
        "=" * 76
    )


def print_pass(
    test_name: str,
) -> None:

    print(
        f"PASS: {test_name}"
    )


def print_question_answer(
    question: str,
    answer: str,
) -> None:

    print()

    print(
        f"Voice question: {question}"
    )

    print(
        f"MemorAi answer: {answer}"
    )


def assert_contains(
    text: str,
    expected: str,
) -> None:

    lowered = (
        str(
            text
            or ""
        )
        .lower()
    )

    assert (
        expected.lower()
        in lowered
    ), (
        f"Expected '{expected}' "
        f"in answer:\n{text}"
    )


def assert_contains_any(
    text: str,
    expected_values: tuple[
        str,
        ...,
    ],
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
        in expected_values
    ), (
        f"Expected one of "
        f"{expected_values} "
        f"in answer:\n{text}"
    )


# ============================================================
# TEST DATA
# ============================================================

class VoiceUnifiedRAGTestData:

    def __init__(
        self,
    ) -> None:

        # ----------------------------------------------------
        # UNIQUE SUFFIX
        # ----------------------------------------------------

        raw_suffix = (
            uuid.uuid4()
            .hex
        )

        letter_suffix = (
            "".join(
                character

                for character
                in raw_suffix

                if character.isalpha()
            )
            [:8]
        )

        if not letter_suffix:

            letter_suffix = (
                "abcdef"
            )

        # ====================================================
        # PERSON
        # ====================================================

        self.person_name = (
            f"voice_unified_rahul_"
            f"{raw_suffix[:8]}"
        )

        self.person_display_name = (
            f"Voice Unified Rahul "
            f"{letter_suffix}"
        )

        self.person_id: (
            int
            | None
        ) = None

        self.person_detected_at = (
            datetime.now()
            .replace(
                microsecond=0
            )
            - timedelta(
                minutes=3
            )
        )

        # ====================================================
        # CONVERSATION
        # ====================================================

        self.conversation_id: (
            int
            | None
        ) = None

        self.conversation_started_at = (
            datetime.now()
            .replace(
                microsecond=0
            )
        )

        self.conversation_transcript = (
            f"{self.person_display_name} told me "
            "that my physiotherapy appointment "
            "is on Thursday at 2 PM. "
            "I should bring my blue medical folder."
        )

        self.conversation_summary = (
            "Physiotherapy appointment is Thursday "
            "at 2 PM. Bring the blue medical folder."
        )

        # ====================================================
        # VECTOR MEMORY
        # ====================================================

        self.conversation_vector_memory = (
            ConversationVectorMemory()
        )


    # ========================================================
    # CREATE PERSON MEMORY
    # ========================================================

    def create_person_memory(
        self,
    ) -> None:

        self.person_id = (
            create_person(
                name=(
                    self.person_name
                ),

                display_name=(
                    self.person_display_name
                ),

                relation=(
                    "Son"
                ),

                notes=(
                    "Temporary Voice Unified RAG "
                    "regression test person."
                ),
            )
        )

        save_person_event(
            person_id=(
                self.person_id
            ),

            similarity=(
                0.93
            ),

            detected_at=(
                self.person_detected_at
            ),
        )


    # ========================================================
    # CREATE CONVERSATION MEMORY
    # ========================================================

    def create_conversation_memory(
        self,
    ) -> None:

        started_at = (
            self.conversation_started_at
        )

        ended_at = (
            started_at
            + timedelta(
                seconds=10
            )
        )

        self.conversation_id = (
            create_conversation_session(
                started_at=(
                    started_at
                ),

                title=(
                    "Voice Physiotherapy Appointment"
                ),

                participant_text=(
                    self.person_display_name
                ),

                status=(
                    "processing"
                ),
            )
        )

        add_conversation_segment(
            conversation_id=(
                self.conversation_id
            ),

            sequence_no=(
                1
            ),

            transcript=(
                self.conversation_transcript
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
                self.conversation_id
            ),

            ended_at=(
                ended_at
            ),

            full_transcript=(
                self.conversation_transcript
            ),

            summary=(
                self.conversation_summary
            ),

            title=(
                "Voice Physiotherapy Appointment"
            ),

            participant_text=(
                self.person_display_name
            ),
        )

        conversation = (
            get_conversation_session(
                self.conversation_id
            )
        )

        assert (
            conversation
            is not None
        )

        self.conversation_vector_memory.add_conversation(
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
                "reminder",
            ),
        )


    # ========================================================
    # CREATE ALL
    # ========================================================

    def create(
        self,
    ) -> None:

        self.create_person_memory()

        print_pass(
            "Temporary person memory created"
        )

        self.create_conversation_memory()

        print_pass(
            (
                "Temporary conversation memory "
                "created and indexed"
            )
        )


    # ========================================================
    # CLEAN PERSON
    # ========================================================

    @staticmethod
    def _cleanup_person(
        person_id: int | None,
    ) -> None:

        if person_id is None:

            return

        execute(
            """
            DELETE FROM person_events
            WHERE person_id = %s
            """,
            (
                person_id,
            ),
        )

        execute(
            """
            DELETE FROM person_face_embeddings
            WHERE person_id = %s
            """,
            (
                person_id,
            ),
        )

        execute(
            """
            DELETE FROM registered_people
            WHERE id = %s
            """,
            (
                person_id,
            ),
        )


    # ========================================================
    # CLEANUP
    # ========================================================

    def cleanup(
        self,
    ) -> None:

        # ----------------------------------------------------
        # CONVERSATION
        # ----------------------------------------------------

        if (
            self.conversation_id
            is not None
        ):

            try:

                self.conversation_vector_memory.delete_conversation(
                    self.conversation_id
                )

            except Exception as error:

                print(
                    (
                        "Cleanup warning: "
                        "conversation vector: "
                        f"{error}"
                    )
                )

            try:

                delete_conversation(
                    self.conversation_id
                )

            except Exception as error:

                print(
                    (
                        "Cleanup warning: "
                        "conversation record: "
                        f"{error}"
                    )
                )

        # ----------------------------------------------------
        # PERSON
        # ----------------------------------------------------

        try:

            self._cleanup_person(
                self.person_id
            )

        except Exception as error:

            print(
                (
                    "Cleanup warning: "
                    "temporary person: "
                    f"{error}"
                )
            )


# ============================================================
# REAL OBJECT MEMORY
# ============================================================

def get_real_object_memory(
) -> dict:

    memory = (
        get_latest_event_for_object(
            "cell phone"
        )
    )

    assert (
        memory is not None
    ), (
        "No real cell phone memory exists.\n"
        "Run the normal object camera pipeline "
        "and detect a cell phone first."
    )

    return (
        memory
    )


# ============================================================
# REAL MEDICINE MEMORY
# ============================================================

def get_real_camera_confirmed_medicine_event(
) -> dict:

    events = (
        get_recent_medicine_events(
            limit=100
        )
    )

    for event in events:

        status = (
            str(
                event.get(
                    "status"
                )
                or ""
            )
            .strip()
            .lower()
        )

        if (
            status
            != "camera_confirmed"
        ):

            continue

        medicine_name = (
            str(
                event.get(
                    "medicine_name"
                )
                or event.get(
                    "name"
                )
                or ""
            )
            .strip()
        )

        if medicine_name:

            return (
                event
            )

    raise AssertionError(
        (
            "No real camera-confirmed medicine "
            "event exists.\nRun the normal medicine "
            "camera pipeline first."
        )
    )


# ============================================================
# ASK THROUGH VOICE
# ============================================================

def ask_through_voice(
    chat: MemorAiChat,
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
    # FakeTranscriber does not read file contents.
    # --------------------------------------------------------

    fake_audio_path = (
        Path(
            "recordings/"
            "fake_voice_unified_rag_test.wav"
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
# OBJECT VOICE TEST
# ============================================================

def test_voice_object_memory(
    chat: MemorAiChat,
    real_object: dict,
) -> None:

    print_header(
        "1. VOICE -> OBJECT MEMORY"
    )

    question, answer, transcriber = (
        ask_through_voice(
            chat=(
                chat
            ),

            question_text=(
                "Where is my cell phone?"
            ),
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
        answer
        != FALLBACK_ANSWER
    )

    assert_contains(
        answer,
        "cell phone",
    )

    assert_contains(
        answer,
        "last observed",
    )

    expected_location = (
        str(
            real_object.get(
                "location"
            )
            or ""
        )
        .strip()
    )

    if expected_location:

        assert_contains(
            answer,
            expected_location,
        )

    assert not (
        answer
        .strip()
        .lower()
        .startswith(
            "your cell phone is"
        )
    )

    print_pass(
        "Voice Object RAG answer"
    )

    print_pass(
        "Object safety wording preserved"
    )


# ============================================================
# PERSON VOICE TEST
# ============================================================

def test_voice_person_memory(
    chat: MemorAiChat,
    data: VoiceUnifiedRAGTestData,
) -> None:

    print_header(
        "2. VOICE -> PERSON MEMORY"
    )

    question_text = (
        "When did I last see "
        f"{data.person_display_name}?"
    )

    question, answer, transcriber = (
        ask_through_voice(
            chat=(
                chat
            ),

            question_text=(
                question_text
            ),
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
        answer
        != FALLBACK_ANSWER
    )

    assert_contains(
        answer,
        data.person_display_name,
    )

    assert_contains(
        answer,
        "today",
    )

    print_pass(
        "Voice Person RAG answer"
    )


# ============================================================
# CONVERSATION VOICE TEST
# ============================================================

def test_voice_conversation_memory(
    chat: MemorAiChat,
    data: VoiceUnifiedRAGTestData,
) -> None:

    print_header(
        "3. VOICE -> CONVERSATION MEMORY"
    )

    # --------------------------------------------------------
    # PERSON-SPECIFIC CONVERSATION
    # --------------------------------------------------------

    question_text = (
        "What did "
        f"{data.person_display_name} "
        "tell me about my "
        "physiotherapy appointment?"
    )

    question, answer, transcriber = (
        ask_through_voice(
            chat=(
                chat
            ),

            question_text=(
                question_text
            ),
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
        answer
        != FALLBACK_ANSWER
    )

    assert_contains_any(
        answer,
        (
            "thursday",
            "2 pm",
            "blue medical folder",
        ),
    )

    print_pass(
        (
            "Voice Conversation RAG wins "
            "over Person Memory"
        )
    )

    # --------------------------------------------------------
    # NATURAL SEMANTIC QUESTION
    # --------------------------------------------------------

    question_text = (
        "What should I bring to "
        "my physiotherapy appointment?"
    )

    question, answer, _ = (
        ask_through_voice(
            chat=(
                chat
            ),

            question_text=(
                question_text
            ),
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

    assert_contains(
        answer,
        "blue medical folder",
    )

    print_pass(
        (
            "Voice natural semantic "
            "conversation retrieval"
        )
    )


# ============================================================
# MEDICINE VOICE TEST
# ============================================================

def test_voice_medicine_memory(
    chat: MemorAiChat,
    medicine_event: dict,
) -> None:

    print_header(
        "4. VOICE -> MEDICINE MEMORY"
    )

    medicine_name = (
        str(
            medicine_event.get(
                "medicine_name"
            )
            or medicine_event.get(
                "name"
            )
            or ""
        )
        .strip()
    )

    assert (
        medicine_name
    )

    question_text = (
        f"Did I take {medicine_name} today?"
    )

    question, answer, transcriber = (
        ask_through_voice(
            chat=(
                chat
            ),

            question_text=(
                question_text
            ),
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

    lowered = (
        answer.lower()
    )

    assert (
        medicine_name.lower()
        in lowered
    )

    assert (
        "camera-confirmed"
        in lowered

        or "camera confirmed"
        in lowered
    )

    unsafe_phrases = (
        f"you took {medicine_name.lower()}",
        f"you swallowed {medicine_name.lower()}",
        (
            "you definitely took "
            f"{medicine_name.lower()}"
        ),
    )

    for unsafe_phrase in (
        unsafe_phrases
    ):

        assert (
            unsafe_phrase
            not in lowered
        )

    print_pass(
        "Voice Medicine RAG answer"
    )

    print_pass(
        (
            "Medicine camera-confirmed "
            "safety wording preserved"
        )
    )


# ============================================================
# UNKNOWN VOICE MEMORY SAFETY
# ============================================================

def test_voice_unknown_memory_safety(
    chat: MemorAiChat,
) -> None:

    print_header(
        "5. VOICE UNKNOWN MEMORY SAFETY"
    )

    # --------------------------------------------------------
    # UNKNOWN OBJECT
    # --------------------------------------------------------

    question, answer, _ = (
        ask_through_voice(
            chat=(
                chat
            ),

            question_text=(
                "Where are my car keys?"
            ),
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

    print_pass(
        "Voice unknown object fallback"
    )

    # --------------------------------------------------------
    # UNKNOWN PERSON
    # --------------------------------------------------------

    question, answer, _ = (
        ask_through_voice(
            chat=(
                chat
            ),

            question_text=(
                "When did I last see "
                "PersonWhoDefinitelyDoesNotExist?"
            ),
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

    print_pass(
        "Voice unknown person fallback"
    )

    # --------------------------------------------------------
    # UNKNOWN CONVERSATION
    # --------------------------------------------------------

    question, answer, _ = (
        ask_through_voice(
            chat=(
                chat
            ),

            question_text=(
                "What did Alexander tell me "
                "about my helicopter?"
            ),
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

    print_pass(
        (
            "Voice unknown conversation "
            "fallback"
        )
    )

    # --------------------------------------------------------
    # UNKNOWN MEDICINE
    # --------------------------------------------------------

    fake_medicine = (
        "zzzxxyy_voice_unified_fake_medicine"
    )

    question, answer, _ = (
        ask_through_voice(
            chat=(
                chat
            ),

            question_text=(
                f"Did I take "
                f"{fake_medicine} today?"
            ),
        )
    )

    print_question_answer(
        question,
        answer,
    )

    lowered = (
        answer.lower()
    )

    assert (
        answer
        == FALLBACK_ANSWER

        or "no reliable"
        in lowered

        or "could not find"
        in lowered

        or "no camera-confirmed"
        in lowered

        or "no camera confirmed"
        in lowered
    )

    assert (
        (
            "camera-confirmed medication event for "
            f"{fake_medicine.lower()}"
        )
        not in lowered
    )

    print_pass(
        "Voice unknown medicine handled safely"
    )


# ============================================================
# CROSS-MODULE VOICE COLLISION TEST
# ============================================================

def test_voice_cross_module_collisions(
    chat: MemorAiChat,
    data: VoiceUnifiedRAGTestData,
    medicine_name: str,
) -> None:

    print_header(
        "6. VOICE CROSS-MODULE COLLISIONS"
    )

    # --------------------------------------------------------
    # PERSON
    # --------------------------------------------------------

    person_question, person_answer, _ = (
        ask_through_voice(
            chat=(
                chat
            ),

            question_text=(
                "When did I last see "
                f"{data.person_display_name}?"
            ),
        )
    )

    print_question_answer(
        person_question,
        person_answer,
    )

    # --------------------------------------------------------
    # CONVERSATION
    # --------------------------------------------------------

    conversation_question, conversation_answer, _ = (
        ask_through_voice(
            chat=(
                chat
            ),

            question_text=(
                "What did "
                f"{data.person_display_name} "
                "tell me about my "
                "physiotherapy appointment?"
            ),
        )
    )

    print_question_answer(
        conversation_question,
        conversation_answer,
    )

    assert_contains(
        person_answer,
        "last saw",
    )

    assert_contains_any(
        conversation_answer,
        (
            "physiotherapy",
            "thursday",
            "blue medical folder",
        ),
    )

    assert (
        "blue medical folder"
        not in person_answer.lower()
    )

    print_pass(
        (
            "Voice Person and Conversation "
            "memories remain separated"
        )
    )

    # --------------------------------------------------------
    # OBJECT
    # --------------------------------------------------------

    object_question, object_answer, _ = (
        ask_through_voice(
            chat=(
                chat
            ),

            question_text=(
                "Where is my cell phone?"
            ),
        )
    )

    print_question_answer(
        object_question,
        object_answer,
    )

    # --------------------------------------------------------
    # MEDICINE
    # --------------------------------------------------------

    medicine_question, medicine_answer, _ = (
        ask_through_voice(
            chat=(
                chat
            ),

            question_text=(
                f"Did I take "
                f"{medicine_name} today?"
            ),
        )
    )

    print_question_answer(
        medicine_question,
        medicine_answer,
    )

    assert (
        "camera-confirmed"
        not in object_answer.lower()
    )

    assert (
        "cell phone"
        not in medicine_answer.lower()
    )

    assert (
        "camera-confirmed"
        not in conversation_answer.lower()
    )

    assert (
        "physiotherapy"
        not in medicine_answer.lower()
    )

    print_pass(
        (
            "Voice Object, Medicine and "
            "Conversation memories remain separated"
        )
    )


# ============================================================
# INVALID TRANSCRIPTION TEST
# ============================================================

def test_invalid_voice_transcription(
    chat: MemorAiChat,
) -> None:

    print_header(
        "7. INVALID VOICE TRANSCRIPTION"
    )

    fake_transcriber = (
        FakeTranscriber(
            text=""
        )
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

    fake_audio_path = (
        Path(
            "recordings/"
            "fake_invalid_voice_test.wav"
        )
    )

    error_raised = (
        False
    )

    try:

        voice_chat.ask_from_audio(
            fake_audio_path
        )

    except ValueError:

        error_raised = (
            True
        )

    assert (
        error_raised
    ), (
        "Expected empty transcription "
        "to raise ValueError."
    )

    assert (
        fake_transcriber.call_count
        == 1
    )

    print_pass(
        (
            "Empty voice transcription "
            "is rejected safely"
        )
    )


# ============================================================
# FAKE WHISPER TEST
# ============================================================

def test_fake_whisper_is_used(
    chat: MemorAiChat,
) -> None:

    print_header(
        "8. TEST ENVIRONMENT SAFETY"
    )

    fake_transcriber = (
        FakeTranscriber(
            text=(
                "Where is my cell phone?"
            )
        )
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

    assert (
        voice_chat.transcriber
        is fake_transcriber
    )

    assert isinstance(
        voice_chat.transcriber,
        FakeTranscriber,
    )

    print_pass(
        (
            "Real Whisper is not loaded "
            "during automated test"
        )
    )

    print_pass(
        (
            "Microphone is not required "
            "during automated test"
        )
    )


# ============================================================
# MAIN TEST
# ============================================================

def test_voice_unified_rag(
) -> None:

    print_header(
        (
            "MemorAi Voice + Unified "
            "Multi-Memory RAG Regression Test"
        )
    )

    print()

    print(
        "This test validates:"
    )

    print(
        "  Voice -> Object Memory"
    )

    print(
        "  Voice -> Person Memory"
    )

    print(
        "  Voice -> Conversation Memory"
    )

    print(
        "  Voice -> Medicine Memory"
    )

    print()

    print(
        "It also validates:"
    )

    print(
        "  Cross-module routing"
    )

    print(
        "  Unknown-memory safety"
    )

    print(
        "  Invalid transcription handling"
    )

    print(
        "  No real microphone requirement"
    )

    print(
        "  No real Whisper loading"
    )

    # ========================================================
    # DATABASE
    # ========================================================

    initialize_database()

    # ========================================================
    # REAL OBJECT
    # ========================================================

    real_object = (
        get_real_object_memory()
    )

    print_pass(
        "Real object memory available"
    )

    # ========================================================
    # REAL MEDICINE
    # ========================================================

    medicine_event = (
        get_real_camera_confirmed_medicine_event()
    )

    medicine_name = (
        str(
            medicine_event.get(
                "medicine_name"
            )
            or medicine_event.get(
                "name"
            )
            or ""
        )
        .strip()
    )

    assert (
        medicine_name
    )

    print_pass(
        (
            "Real camera-confirmed "
            "medicine memory available"
        )
    )

    print()

    print(
        "Real object used:"
    )

    print(
        (
            "  Object: "
            f"{real_object.get('object_name')}"
        )
    )

    print(
        (
            "  Location: "
            f"{real_object.get('location')}"
        )
    )

    print()

    print(
        "Real medicine used:"
    )

    print(
        (
            "  Medicine: "
            f"{medicine_name}"
        )
    )

    print(
        (
            "  Status: "
            f"{medicine_event.get('status')}"
        )
    )

    # ========================================================
    # TEMPORARY TEST DATA
    # ========================================================

    data = (
        VoiceUnifiedRAGTestData()
    )

    try:

        data.create()

        # ====================================================
        # ONE SHARED REAL MEMORAI CHAT
        # ====================================================

        chat = (
            MemorAiChat()
        )

        # ====================================================
        # RUN TESTS
        # ====================================================

        test_voice_object_memory(
            chat=(
                chat
            ),

            real_object=(
                real_object
            ),
        )

        test_voice_person_memory(
            chat=(
                chat
            ),

            data=(
                data
            ),
        )

        test_voice_conversation_memory(
            chat=(
                chat
            ),

            data=(
                data
            ),
        )

        test_voice_medicine_memory(
            chat=(
                chat
            ),

            medicine_event=(
                medicine_event
            ),
        )

        test_voice_unknown_memory_safety(
            chat=(
                chat
            )
        )

        test_voice_cross_module_collisions(
            chat=(
                chat
            ),

            data=(
                data
            ),

            medicine_name=(
                medicine_name
            ),
        )

        test_invalid_voice_transcription(
            chat=(
                chat
            )
        )

        test_fake_whisper_is_used(
            chat=(
                chat
            )
        )

        # ====================================================
        # FINAL PASS
        # ====================================================

        print_header(
            (
                "PASS: MEMORAI VOICE + "
                "UNIFIED RAG REGRESSION SUITE"
            )
        )

        print()

        print(
            "Verified architecture:"
        )

        print()

        print(
            "Fake Voice / Transcription"
        )

        print(
            "          |"
        )

        print(
            "          v"
        )

        print(
            "MemorAiVoiceChat"
        )

        print(
            "          |"
        )

        print(
            "          v"
        )

        print(
            "MemorAiChat"
        )

        print(
            "          |"
        )

        print(
            "          +--> Object Memory"
        )

        print(
            "          +--> Person Memory"
        )

        print(
            "          +--> Conversation Memory"
        )

        print(
            "          +--> Medicine Memory"
        )

        print()

        print(
            (
                "Voice routing, grounding, "
                "fallback and safety checks passed."
            )
        )

    finally:

        data.cleanup()

        print()

        print(
            (
                "Temporary Voice Unified RAG "
                "Person and Conversation data cleaned."
            )
        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    test_voice_unified_rag()