from __future__ import annotations

import uuid

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


# ============================================================
# OUTPUT HELPERS
# ============================================================

def print_header(
    title: str,
) -> None:

    print()

    print(
        "=" * 74
    )

    print(
        title
    )

    print(
        "=" * 74
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
        f"Question: {question}"
    )

    print(
        f"Answer:   {answer}"
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
    expected_values: tuple[str, ...],
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

class UnifiedRAGTestData:

    def __init__(
        self,
    ) -> None:

        # ----------------------------------------------------
        # GENERATE UNIQUE TEST SUFFIX
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
            f"unified_rahul_{raw_suffix[:8]}"
        )

        self.person_display_name = (
            f"Unified Rahul {letter_suffix}"
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
        #
        # Same display name is intentionally used.
        #
        # Person:
        #     When did I last see Rahul?
        #
        # Conversation:
        #     What did Rahul tell me?
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
                    "Temporary unified RAG "
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
                    "Physiotherapy Appointment"
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
                "Physiotherapy Appointment"
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
    # CREATE ALL TEST DATA
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
        # CONVERSATION VECTOR + MYSQL
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
                        "Cleanup warning: could not "
                        "delete conversation vector: "
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
                        "Cleanup warning: could not "
                        "delete conversation record: "
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
                    "Cleanup warning: could not "
                    "delete temporary person: "
                    f"{error}"
                )
            )


# ============================================================
# REAL OBJECT MEMORY
# ============================================================

def get_real_object_memory() -> dict:

    memory = (
        get_latest_event_for_object(
            "cell phone"
        )
    )

    assert (
        memory is not None
    ), (
        "No cell phone memory exists. "
        "Run the normal object camera pipeline first."
    )

    return (
        memory
    )


# ============================================================
# REAL MEDICINE EVENT
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
            "event exists. Run the medicine camera "
            "pipeline before this unified test."
        )
    )


# ============================================================
# ROUTING CLASSIFICATION
# ============================================================

def test_routing_classification(
    chat: MemorAiChat,
    data: UnifiedRAGTestData,
    medicine_name: str,
) -> None:

    print_header(
        "1. ROUTING CLASSIFICATION"
    )

    # ========================================================
    # OBJECT
    # ========================================================

    object_question = (
        "Where is my cell phone?"
    )

    assert not (
        chat.conversation_retriever
        .is_conversation_question(
            object_question
        )
    )

    assert not (
        chat.medicine_retriever
        .is_medicine_question(
            object_question
        )
    )

    assert not (
        chat.person_retriever
        .is_person_question(
            object_question
        )
    )

    print_pass(
        (
            "Object question is not stolen "
            "by other modules"
        )
    )

    # ========================================================
    # PERSON
    # ========================================================

    person_question = (
        "When did I last see "
        f"{data.person_display_name}?"
    )

    assert not (
        chat.conversation_retriever
        .is_conversation_question(
            person_question
        )
    )

    assert not (
        chat.medicine_retriever
        .is_medicine_question(
            person_question
        )
    )

    assert (
        chat.person_retriever
        .is_person_question(
            person_question
        )
    )

    print_pass(
        (
            "Person encounter question routes "
            "to Person Memory"
        )
    )

    # ========================================================
    # CONVERSATION VS PERSON
    # ========================================================

    conversation_question = (
        "What did "
        f"{data.person_display_name} "
        "tell me about my physiotherapy appointment?"
    )

    assert (
        chat.conversation_retriever
        .is_conversation_question(
            conversation_question
        )
    )

    assert (
        chat.person_retriever
        .is_person_question(
            conversation_question
        )
    )

    print_pass(
        (
            "Conversation question remains "
            "conversation-routed even when "
            "participant is a registered person"
        )
    )

    # ========================================================
    # MEDICINE
    # ========================================================

    medicine_question = (
        f"Did I take {medicine_name} today?"
    )

    assert (
        chat.medicine_retriever
        .is_medicine_question(
            medicine_question
        )
    )

    print_pass(
        (
            "Medicine question recognized "
            "by Medicine Memory"
        )
    )


# ============================================================
# OBJECT RAG
# ============================================================

def test_object_rag(
    chat: MemorAiChat,
    real_object: dict,
) -> None:

    print_header(
        "2. OBJECT MEMORY"
    )

    question = (
        "Where is my cell phone?"
    )

    answer = (
        chat.answer(
            question
        )
    )

    print_question_answer(
        question,
        answer,
    )

    expected_location = (
        str(
            real_object.get(
                "location"
            )
            or ""
        )
        .strip()
        .lower()
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
        "Object RAG answer"
    )

    print_pass(
        (
            "Object answer preserves safe "
            "last-observed wording"
        )
    )


# ============================================================
# PERSON RAG
# ============================================================

def test_person_rag(
    chat: MemorAiChat,
    data: UnifiedRAGTestData,
) -> None:

    print_header(
        "3. PERSON MEMORY"
    )

    question = (
        "When did I last see "
        f"{data.person_display_name}?"
    )

    answer = (
        chat.answer(
            question
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
        data.person_display_name,
    )

    assert_contains(
        answer,
        "today",
    )

    print_pass(
        "Person RAG answer"
    )


# ============================================================
# CONVERSATION RAG
# ============================================================

def test_conversation_rag(
    chat: MemorAiChat,
    data: UnifiedRAGTestData,
) -> None:

    print_header(
        "4. CONVERSATION MEMORY"
    )

    # ========================================================
    # PERSON-SPECIFIC CONVERSATION
    # ========================================================

    question = (
        "What did "
        f"{data.person_display_name} "
        "tell me about my physiotherapy appointment?"
    )

    answer = (
        chat.answer(
            question
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
            "thursday",
            "2 pm",
            "blue medical folder",
        ),
    )

    print_pass(
        (
            "Conversation RAG wins "
            "over Person Memory"
        )
    )

    # ========================================================
    # NATURAL SEMANTIC QUESTION
    # ========================================================

    question = (
        "What should I bring to "
        "my physiotherapy appointment?"
    )

    answer = (
        chat.answer(
            question
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
            "Natural conversation question "
            "retrieval"
        )
    )


# ============================================================
# MEDICINE RAG
# ============================================================

def test_medicine_rag(
    chat: MemorAiChat,
    medicine_event: dict,
) -> None:

    print_header(
        "5. MEDICINE MEMORY"
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

    question = (
        f"Did I take {medicine_name} today?"
    )

    answer = (
        chat.answer(
            question
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
        "Medicine RAG answer"
    )

    print_pass(
        (
            "Medicine answer preserves "
            "camera-confirmed safety wording"
        )
    )


# ============================================================
# UNKNOWN MEMORY SAFETY
# ============================================================

def test_unknown_memory_safety(
    chat: MemorAiChat,
) -> None:

    print_header(
        "6. UNKNOWN MEMORY SAFETY"
    )

    # ========================================================
    # UNKNOWN OBJECT
    # ========================================================

    question = (
        "Where are my car keys?"
    )

    answer = (
        chat.answer(
            question
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
        "Unknown object fallback"
    )

    # ========================================================
    # UNKNOWN PERSON
    # ========================================================

    question = (
        "When did I last see "
        "PersonWhoDefinitelyDoesNotExist?"
    )

    answer = (
        chat.answer(
            question
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
        "Unknown person fallback"
    )

    # ========================================================
    # UNKNOWN CONVERSATION PERSON
    # ========================================================

    question = (
        "What did Alexander tell me "
        "about my helicopter?"
    )

    answer = (
        chat.answer(
            question
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
            "Unknown conversation participant "
            "fallback"
        )
    )

    # ========================================================
    # UNKNOWN MEDICINE
    # ========================================================

    fake_medicine = (
        "zzzxxyy_unified_fake_medicine"
    )

    question = (
        f"Did I take {fake_medicine} today?"
    )

    answer = (
        chat.answer(
            question
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
        "Unknown medicine handled safely"
    )


# ============================================================
# CROSS-MODULE COLLISIONS
# ============================================================

def test_cross_module_collisions(
    chat: MemorAiChat,
    data: UnifiedRAGTestData,
    medicine_name: str,
) -> None:

    print_header(
        "7. CROSS-MODULE COLLISION TESTS"
    )

    # ========================================================
    # PERSON VS CONVERSATION
    # ========================================================

    person_question = (
        "When did I last see "
        f"{data.person_display_name}?"
    )

    person_answer = (
        chat.answer(
            person_question
        )
    )

    conversation_question = (
        "What did "
        f"{data.person_display_name} "
        "tell me about my physiotherapy appointment?"
    )

    conversation_answer = (
        chat.answer(
            conversation_question
        )
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
            "Person and Conversation "
            "memories remain separated"
        )
    )

    # ========================================================
    # OBJECT VS MEDICINE
    # ========================================================

    object_question = (
        "Where is my cell phone?"
    )

    object_answer = (
        chat.answer(
            object_question
        )
    )

    medicine_question = (
        f"Did I take {medicine_name} today?"
    )

    medicine_answer = (
        chat.answer(
            medicine_question
        )
    )

    assert (
        "camera-confirmed"
        not in object_answer.lower()
    )

    assert (
        "cell phone"
        not in medicine_answer.lower()
    )

    print_pass(
        (
            "Object and Medicine "
            "memories remain separated"
        )
    )

    # ========================================================
    # CONVERSATION VS MEDICINE
    # ========================================================

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
            "Conversation and Medicine "
            "memories remain separated"
        )
    )


# ============================================================
# BASIC REGRESSION
# ============================================================

def test_basic_regression(
    chat: MemorAiChat,
) -> None:

    print_header(
        "8. BASIC REGRESSION"
    )

    answer = (
        chat.answer(
            ""
        )
    )

    assert (
        answer
        == (
            "Please ask a "
            "memory-related question."
        )
    )

    print_pass(
        "Empty question handling"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print_header(
        (
            "MemorAi Unified Multi-Memory "
            "RAG Regression Test"
        )
    )

    print(
        "\nThis test validates:"
    )

    print(
        "  Object Memory"
    )

    print(
        "  Person Memory"
    )

    print(
        "  Conversation Memory"
    )

    print(
        "  Medicine Memory"
    )

    print(
        (
            "\nIt also checks cross-module "
            "routing and safety."
        )
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
    # TEMP TEST DATA
    # ========================================================

    data = (
        UnifiedRAGTestData()
    )

    try:

        data.create()

        # ====================================================
        # ONE SHARED MEMORAI CHAT
        # ====================================================

        chat = (
            MemorAiChat()
        )

        # ====================================================
        # RUN
        # ====================================================

        test_routing_classification(
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

        test_object_rag(
            chat=(
                chat
            ),

            real_object=(
                real_object
            ),
        )

        test_person_rag(
            chat=(
                chat
            ),

            data=(
                data
            ),
        )

        test_conversation_rag(
            chat=(
                chat
            ),

            data=(
                data
            ),
        )

        test_medicine_rag(
            chat=(
                chat
            ),

            medicine_event=(
                medicine_event
            ),
        )

        test_unknown_memory_safety(
            chat=(
                chat
            ),
        )

        test_cross_module_collisions(
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

        test_basic_regression(
            chat=(
                chat
            ),
        )

        # ====================================================
        # PASS
        # ====================================================

        print_header(
            (
                "PASS: MEMORAI UNIFIED "
                "RAG REGRESSION SUITE"
            )
        )

        print()

        print(
            "Verified architecture:"
        )

        print()

        print(
            "Object Memory -----------\\"
        )

        print(
            "Person Memory ------------\\"
        )

        print(
            (
                "Conversation Memory ------- > "
                "MemorAiChat -> Answer"
            )
        )

        print(
            "Medicine Memory ----------/"
        )

        print()

        print(
            (
                "Cross-module routing, grounding, "
                "fallback and safety checks passed."
            )
        )

    finally:

        data.cleanup()

        print()

        print(
            (
                "Temporary Person and Conversation "
                "regression data cleaned."
            )
        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()