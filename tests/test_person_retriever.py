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
    save_person_event,
)

from memory.person_retriever import (
    PersonMemoryRetriever,
)


# ============================================================
# TEST OUTPUT
# ============================================================

def print_test_result(
    test_name: str,
    passed: bool,
) -> None:

    status = (
        "PASS"
        if passed
        else "FAIL"
    )

    print(
        f"{status}: {test_name}"
    )


# ============================================================
# TEMP TEST DATA
# ============================================================

class PersonRetrieverTestData:

    def __init__(
        self,
    ) -> None:

        suffix = (
            uuid.uuid4()
            .hex[:8]
        )

        self.rahul_name = (
            f"rag_rahul_{suffix}"
        )

        self.rahul_display_name = (
            f"Rahul Rag {suffix}"
        )

        self.priya_name = (
            f"rag_priya_{suffix}"
        )

        self.priya_display_name = (
            f"Priya Rag {suffix}"
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

        # Keep events safely inside their intended dates.
        self.rahul_today = (
            now.replace(
                hour=10,
                minute=15,
                second=0,
                microsecond=0,
            )
        )

        self.priya_today = (
            now.replace(
                hour=9,
                minute=30,
                second=0,
                microsecond=0,
            )
        )

        yesterday = (
            now
            - timedelta(
                days=1
            )
        )

        self.rahul_yesterday = (
            yesterday.replace(
                hour=18,
                minute=45,
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
                    "Temporary person retriever test."
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
                    "Temporary person retriever test."
                ),
            )
        )

        save_person_event(
            person_id=(
                self.rahul_id
            ),
            similarity=0.88,
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
            similarity=0.91,
            detected_at=(
                self.rahul_today
            ),
        )

    # ========================================================
    # CLEANUP ONE PERSON
    # ========================================================

    @staticmethod
    def _cleanup_person(
        person_id: int | None,
    ) -> None:

        if person_id is None:

            return

        # Delete children first because of the foreign keys.

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

        self._cleanup_person(
            self.rahul_id
        )

        self._cleanup_person(
            self.priya_id
        )


# ============================================================
# EMPTY QUESTION
# ============================================================

def test_empty_question(
    retriever: PersonMemoryRetriever,
) -> None:

    memories = (
        retriever.retrieve(
            ""
        )
    )

    assert memories == []

    print_test_result(
        "Empty person question",
        True,
    )


# ============================================================
# PERSON QUESTION DETECTION
# ============================================================

def test_person_question_detection(
    retriever: PersonMemoryRetriever,
    data: PersonRetrieverTestData,
) -> None:

    assert retriever.is_person_question(
        (
            "When did I last see "
            f"{data.rahul_display_name}?"
        )
    )

    assert retriever.is_person_question(
        "Who did I see today?"
    )

    assert not retriever.is_person_question(
        "Where is my cell phone?"
    )

    print_test_result(
        "Person question detection",
        True,
    )


# ============================================================
# PERSON NAME RESOLUTION
# ============================================================

def test_name_resolution(
    retriever: PersonMemoryRetriever,
    data: PersonRetrieverTestData,
) -> None:

    people = (
        retriever.resolve_people(
            (
                "When did I last see "
                f"{data.rahul_display_name}?"
            )
        )
    )

    assert people

    person_ids = {
        int(
            person[
                "id"
            ]
        )
        for person in people
    }

    assert (
        data.rahul_id
        in person_ids
    )

    print_test_result(
        "Registered person name resolution",
        True,
    )


# ============================================================
# RELATION RESOLUTION
# ============================================================

def test_relation_resolution(
    retriever: PersonMemoryRetriever,
    data: PersonRetrieverTestData,
) -> None:

    people = (
        retriever.resolve_people(
            "Did I see my son today?"
        )
    )

    assert people

    person_ids = {
        int(
            person[
                "id"
            ]
        )
        for person in people
    }

    assert (
        data.rahul_id
        in person_ids
    )

    print_test_result(
        "Relation resolution",
        True,
    )


# ============================================================
# LATEST PERSON BY NAME
# ============================================================

def test_latest_person_by_name(
    retriever: PersonMemoryRetriever,
    data: PersonRetrieverTestData,
) -> None:

    memories = (
        retriever.retrieve(
            (
                "When did I last see "
                f"{data.rahul_display_name}?"
            )
        )
    )

    assert memories

    latest = memories[0]

    assert int(
        latest[
            "person_id"
        ]
    ) == data.rahul_id

    assert (
        latest[
            "detected_at"
        ].date()
        == datetime.now().date()
    )

    print_test_result(
        "Latest encounter by person name",
        True,
    )


# ============================================================
# PERSON TODAY
# ============================================================

def test_person_today(
    retriever: PersonMemoryRetriever,
    data: PersonRetrieverTestData,
) -> None:

    memories = (
        retriever.retrieve(
            (
                "Did I see "
                f"{data.rahul_display_name} "
                "today?"
            )
        )
    )

    assert memories

    assert all(
        int(
            memory[
                "person_id"
            ]
        )
        == data.rahul_id
        for memory in memories
    )

    assert all(
        memory[
            "detected_at"
        ].date()
        == datetime.now().date()
        for memory in memories
    )

    print_test_result(
        "Specific person today retrieval",
        True,
    )


# ============================================================
# PERSON YESTERDAY
# ============================================================

def test_person_yesterday(
    retriever: PersonMemoryRetriever,
    data: PersonRetrieverTestData,
) -> None:

    memories = (
        retriever.retrieve(
            (
                "Did I see "
                f"{data.rahul_display_name} "
                "yesterday?"
            )
        )
    )

    assert memories

    yesterday = (
        datetime.now().date()
        - timedelta(
            days=1
        )
    )

    assert all(
        int(
            memory[
                "person_id"
            ]
        )
        == data.rahul_id
        for memory in memories
    )

    assert all(
        memory[
            "detected_at"
        ].date()
        == yesterday
        for memory in memories
    )

    print_test_result(
        "Specific person yesterday retrieval",
        True,
    )


# ============================================================
# WHO TODAY
# ============================================================

def test_who_today(
    retriever: PersonMemoryRetriever,
    data: PersonRetrieverTestData,
) -> None:

    memories = (
        retriever.retrieve(
            "Who did I see today?",
            top_k=50,
        )
    )

    assert memories

    person_ids = {
        int(
            memory[
                "person_id"
            ]
        )
        for memory in memories
    }

    assert (
        data.rahul_id
        in person_ids
    )

    assert (
        data.priya_id
        in person_ids
    )

    print_test_result(
        "Who did I see today retrieval",
        True,
    )


# ============================================================
# RELATION TODAY
# ============================================================

def test_relation_today(
    retriever: PersonMemoryRetriever,
    data: PersonRetrieverTestData,
) -> None:

    memories = (
        retriever.retrieve(
            "Did I see my son today?",
            top_k=50,
        )
    )

    assert memories

    person_ids = {
        int(
            memory[
                "person_id"
            ]
        )
        for memory in memories
    }

    assert (
        data.rahul_id
        in person_ids
    )

    print_test_result(
        "Relation-based today retrieval",
        True,
    )


# ============================================================
# LAST PERSON
# ============================================================

def test_last_person(
    retriever: PersonMemoryRetriever,
) -> None:

    memories = (
        retriever.retrieve(
            "Who was the last person I saw?"
        )
    )

    assert memories

    assert (
        memories[0].get(
            "person_id"
        )
        is not None
    )

    assert (
        memories[0].get(
            "detected_at"
        )
        is not None
    )

    print_test_result(
        "Last person retrieval",
        True,
    )


# ============================================================
# UNKNOWN PERSON
# ============================================================

def test_unknown_person(
    retriever: PersonMemoryRetriever,
) -> None:

    memories = (
        retriever.retrieve(
            (
                "When did I last see "
                "PersonWhoDefinitelyDoesNotExist?"
            )
        )
    )

    assert memories == []

    print_test_result(
        "Unknown person fallback",
        True,
    )


# ============================================================
# RUN TEST SUITE
# ============================================================

def test_person_retriever() -> None:

    print(
        "\nMemorAi Person Retriever Test Suite\n"
    )

    initialize_database()

    data = (
        PersonRetrieverTestData()
    )

    retriever = (
        PersonMemoryRetriever()
    )

    try:

        data.create()

        test_empty_question(
            retriever
        )

        test_person_question_detection(
            retriever,
            data,
        )

        test_name_resolution(
            retriever,
            data,
        )

        test_relation_resolution(
            retriever,
            data,
        )

        test_latest_person_by_name(
            retriever,
            data,
        )

        test_person_today(
            retriever,
            data,
        )

        test_person_yesterday(
            retriever,
            data,
        )

        test_who_today(
            retriever,
            data,
        )

        test_relation_today(
            retriever,
            data,
        )

        test_last_person(
            retriever
        )

        test_unknown_person(
            retriever
        )

        print(
            "\nAll MemorAi person retriever "
            "tests passed."
        )

    finally:

        # This always runs, even if one assertion fails.
        # Therefore temporary test identities should not
        # remain in the actual Memor_ai database.

        data.cleanup()

        print(
            "\nTemporary person test data cleaned."
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    test_person_retriever()