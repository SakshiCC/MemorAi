from __future__ import annotations

import uuid

from datetime import (
    datetime,
    timedelta,
)

from database.db_manager import (
    execute,
    initialize_database,
)

from database.person_repository import (
    create_person,
    get_recent_person_events,
    save_person_event,
)

from llm.chat import (
    FALLBACK_ANSWER,
    MemorAiChat,
)


# ============================================================
# OUTPUT
# ============================================================

def print_test_result(
    test_name: str,
) -> None:

    print(
        f"PASS: {test_name}"
    )


# ============================================================
# TEMPORARY TEST DATA
# ============================================================

class PersonRAGTestData:

    def __init__(
        self,
    ) -> None:

        suffix = (
            uuid.uuid4()
            .hex[:8]
        )

        self.rahul_name = (
            f"rag_chat_rahul_{suffix}"
        )

        self.rahul_display_name = (
            f"Rahul Chat {suffix}"
        )

        self.priya_name = (
            f"rag_chat_priya_{suffix}"
        )

        self.priya_display_name = (
            f"Priya Chat {suffix}"
        )

        self.rahul_id: (
            int
            | None
        ) = None

        self.priya_id: (
            int
            | None
        ) = None

        now = datetime.now()

        # Keep today's test encounters close to the current
        # time, while still maintaining a clear order.
        self.rahul_today = (
            now
            - timedelta(
                minutes=2
            )
        ).replace(
            microsecond=0
        )

        self.priya_today = (
            now
            - timedelta(
                minutes=10
            )
        ).replace(
            microsecond=0
        )

        yesterday = (
            now
            - timedelta(
                days=1
            )
        )

        self.rahul_yesterday = (
            yesterday.replace(
                hour=17,
                minute=15,
                second=0,
                microsecond=0,
            )
        )

    # ========================================================
    # CREATE
    # ========================================================

    def create(
        self,
    ) -> None:

        self.rahul_id = (
            create_person(
                name=(
                    self.rahul_name
                ),
                display_name=(
                    self.rahul_display_name
                ),
                relation="Son",
                notes=(
                    "Temporary person RAG test."
                ),
            )
        )

        self.priya_id = (
            create_person(
                name=(
                    self.priya_name
                ),
                display_name=(
                    self.priya_display_name
                ),
                relation="Friend",
                notes=(
                    "Temporary person RAG test."
                ),
            )
        )

        save_person_event(
            person_id=(
                self.rahul_id
            ),
            similarity=0.87,
            detected_at=(
                self.rahul_yesterday
            ),
        )

        save_person_event(
            person_id=(
                self.priya_id
            ),
            similarity=0.84,
            detected_at=(
                self.priya_today
            ),
        )

        save_person_event(
            person_id=(
                self.rahul_id
            ),
            similarity=0.92,
            detected_at=(
                self.rahul_today
            ),
        )

    # ========================================================
    # CLEANUP
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

    def cleanup(
        self,
    ) -> None:

        self._cleanup_person(
            self.rahul_id
        )

        self._cleanup_person(
            self.priya_id
        )


# ============================================================
# LAST SEEN BY NAME
# ============================================================

def test_last_seen_by_name(
    chat: MemorAiChat,
    data: PersonRAGTestData,
) -> None:

    answer = (
        chat.answer(
            (
                "When did I last see "
                f"{data.rahul_display_name}?"
            )
        )
    )

    lowered = (
        answer.lower()
    )

    assert (
        data.rahul_display_name.lower()
        in lowered
    )

    assert (
        "today"
        in lowered
    )

    print_test_result(
        "Last seen person by name"
    )


# ============================================================
# PERSON TODAY
# ============================================================

def test_person_today(
    chat: MemorAiChat,
    data: PersonRAGTestData,
) -> None:

    answer = (
        chat.answer(
            (
                "Did I see "
                f"{data.rahul_display_name} "
                "today?"
            )
        )
    )

    lowered = (
        answer.lower()
    )

    assert lowered.startswith(
        "yes."
    )

    assert (
        data.rahul_display_name.lower()
        in lowered
    )

    assert (
        "today"
        in lowered
    )

    print_test_result(
        "Person seen today"
    )


# ============================================================
# PERSON YESTERDAY
# ============================================================

def test_person_yesterday(
    chat: MemorAiChat,
    data: PersonRAGTestData,
) -> None:

    answer = (
        chat.answer(
            (
                "Did I see "
                f"{data.rahul_display_name} "
                "yesterday?"
            )
        )
    )

    lowered = (
        answer.lower()
    )

    assert lowered.startswith(
        "yes."
    )

    assert (
        data.rahul_display_name.lower()
        in lowered
    )

    assert (
        "yesterday"
        in lowered
    )

    assert (
        "5:15 pm"
        in lowered
    )

    print_test_result(
        "Person seen yesterday"
    )


# ============================================================
# RELATION TODAY
# ============================================================

def test_relation_today(
    chat: MemorAiChat,
    data: PersonRAGTestData,
) -> None:

    answer = (
        chat.answer(
            "Did I see my son today?"
        )
    )

    lowered = (
        answer.lower()
    )

    assert lowered.startswith(
        "yes."
    )

    assert (
        data.rahul_display_name.lower()
        in lowered
    )

    assert (
        "son"
        in lowered
    )

    print_test_result(
        "Relation-based person question"
    )


# ============================================================
# WHO TODAY
# ============================================================

def test_who_today(
    chat: MemorAiChat,
    data: PersonRAGTestData,
) -> None:

    answer = (
        chat.answer(
            "Who did I see today?"
        )
    )

    lowered = (
        answer.lower()
    )

    assert (
        data.rahul_display_name.lower()
        in lowered
    )

    assert (
        data.priya_display_name.lower()
        in lowered
    )

    assert (
        "today"
        in lowered
    )

    print_test_result(
        "Who did I see today"
    )


# ============================================================
# LAST PERSON
# ============================================================

def test_last_person(
    chat: MemorAiChat,
) -> None:

    # --------------------------------------------------------
    # GET THE ACTUAL LATEST DATABASE RECORD
    # --------------------------------------------------------

    latest_events = (
        get_recent_person_events(
            limit=1
        )
    )

    assert latest_events

    expected_latest = (
        latest_events[0]
    )

    expected_name = str(
        expected_latest.get(
            "person_name"
        )
        or ""
    ).strip()

    assert expected_name

    # --------------------------------------------------------
    # ASK MEMORAI
    # --------------------------------------------------------

    answer = (
        chat.answer(
            "Who was the last person I saw?"
        )
    )

    lowered = (
        answer.lower()
    )

    # --------------------------------------------------------
    # VERIFY CHAT AGREES WITH DATABASE
    # --------------------------------------------------------

    assert (
        expected_name.lower()
        in lowered
    )

    assert (
        "last person"
        in lowered
    )

    print_test_result(
        "Last person seen"
    )


# ============================================================
# UNKNOWN PERSON
# ============================================================

def test_unknown_person(
    chat: MemorAiChat,
) -> None:

    answer = (
        chat.answer(
            (
                "When did I last see "
                "PersonWhoDefinitelyDoesNotExist?"
            )
        )
    )

    assert (
        answer
        == FALLBACK_ANSWER
    )

    print_test_result(
        "Unknown person fallback"
    )


# ============================================================
# EMPTY QUESTION REGRESSION
# ============================================================

def test_empty_question(
    chat: MemorAiChat,
) -> None:

    answer = (
        chat.answer(
            ""
        )
    )

    assert answer == (
        "Please ask a memory-related question."
    )

    print_test_result(
        "Empty question regression"
    )


# ============================================================
# OBJECT QUESTION ROUTING REGRESSION
# ============================================================

def test_object_question_routing(
    chat: MemorAiChat,
) -> None:

    is_person = (
        chat.person_retriever
        .is_person_question(
            "Where is my cell phone?"
        )
    )

    assert not is_person

    print_test_result(
        "Object question remains object-routed"
    )


# ============================================================
# RUN
# ============================================================

def test_person_rag() -> None:

    print(
        "\nMemorAi Person RAG Test Suite\n"
    )

    initialize_database()

    data = (
        PersonRAGTestData()
    )

    chat = (
        MemorAiChat()
    )

    try:

        data.create()

        test_last_seen_by_name(
            chat,
            data,
        )

        test_person_today(
            chat,
            data,
        )

        test_person_yesterday(
            chat,
            data,
        )

        test_relation_today(
            chat,
            data,
        )

        test_who_today(
            chat,
            data,
        )

        test_last_person(
            chat
        )

        test_unknown_person(
            chat
        )

        test_empty_question(
            chat
        )

        test_object_question_routing(
            chat
        )

        print(
            "\nAll MemorAi Person RAG "
            "tests passed."
        )

    finally:

        data.cleanup()

        print(
            "\nTemporary person RAG "
            "test data cleaned."
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    test_person_rag()