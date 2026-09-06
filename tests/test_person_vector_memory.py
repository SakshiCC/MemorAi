from __future__ import annotations

import uuid

from datetime import (
    datetime,
    timedelta,
)

from database.person_vector_memory import (
    PersonVectorMemory,
)

from rag.person_memory_search import (
    PersonMemorySearch,
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
# TEST DATA
# ============================================================

class PersonVectorTestData:

    def __init__(
        self,
    ) -> None:

        suffix = (
            uuid.uuid4()
            .hex[:8]
        )

        numeric_suffix = int(
            suffix[:5],
            16,
        )

        self.person_id = (
            9_000_000
            + numeric_suffix
        )

        self.event_id_1 = (
            8_000_000
            + numeric_suffix
        )

        self.event_id_2 = (
            self.event_id_1
            + 1
        )

        self.person_name = (
            f"Semantic Rahul {suffix}"
        )

        self.relation = (
            "Son"
        )

        now = (
            datetime.now()
            .replace(
                microsecond=0
            )
        )

        self.older_time = (
            now
            - timedelta(
                hours=2
            )
        )

        self.newer_time = (
            now
            - timedelta(
                minutes=5
            )
        )

    # ========================================================
    # EVENT 1
    # ========================================================

    def event_1(
        self,
    ) -> dict:

        older_time_text = (
            self.older_time.strftime(
                "%d %b %Y at %I:%M %p"
            )
        )

        return {
            "id": (
                self.event_id_1
            ),

            "person_id": (
                self.person_id
            ),

            "person_name": (
                self.person_name
            ),

            "relation": (
                self.relation
            ),

            "similarity": (
                0.87
            ),

            "detected_at": (
                self.older_time
            ),

            "image_path": (
                None
            ),

            "memory_text": (
                f"MemorAi saw "
                f"{self.person_name}, "
                f"the user's {self.relation}, "
                f"on {older_time_text}."
            ),
        }

    # ========================================================
    # EVENT 2
    # ========================================================

    def event_2(
        self,
    ) -> dict:

        newer_time_text = (
            self.newer_time.strftime(
                "%d %b %Y at %I:%M %p"
            )
        )

        return {
            "id": (
                self.event_id_2
            ),

            "person_id": (
                self.person_id
            ),

            "person_name": (
                self.person_name
            ),

            "relation": (
                self.relation
            ),

            "similarity": (
                0.93
            ),

            "detected_at": (
                self.newer_time
            ),

            "image_path": (
                None
            ),

            "memory_text": (
                f"MemorAi saw "
                f"{self.person_name}, "
                f"the user's {self.relation}, "
                f"on {newer_time_text}."
            ),
        }


# ============================================================
# TEST ADD
# ============================================================

def test_add_person_memories(
    vector_memory: PersonVectorMemory,
    data: PersonVectorTestData,
) -> None:

    vector_memory.add_person_event(
        data.event_1()
    )

    vector_memory.add_person_event(
        data.event_2()
    )

    print_test_result(
        "Add person vector memories"
    )


# ============================================================
# TEST SEARCH BY PERSON
# ============================================================

def test_search_by_name(
    search: PersonMemorySearch,
    data: PersonVectorTestData,
) -> None:

    results = (
        search.search(
            query=(
                "When did I see "
                f"{data.person_name}?"
            ),
            top_k=5,
            person_id=(
                data.person_id
            ),
        )
    )

    assert results

    first = (
        results[0]
    )

    assert (
        int(
            first.metadata.get(
                "person_id"
            ) # type: ignore
        )
        == data.person_id
    )

    assert (
        first.metadata.get(
            "person_name"
        )
        == data.person_name
    )

    print_test_result(
        "Semantic search by person"
    )


# ============================================================
# TEST RELATION METADATA
# ============================================================

def test_relation_metadata(
    search: PersonMemorySearch,
    data: PersonVectorTestData,
) -> None:

    results = (
        search.search(
            query=(
                "When did I see my son?"
            ),
            top_k=5,
            person_id=(
                data.person_id
            ),
        )
    )

    assert results

    first = (
        results[0]
    )

    assert (
        first.metadata.get(
            "relation"
        )
        == data.relation
    )

    print_test_result(
        "Person relation metadata"
    )


# ============================================================
# TEST NEWEST FIRST
# ============================================================

def test_newest_first(
    search: PersonMemorySearch,
    data: PersonVectorTestData,
) -> None:

    results = (
        search.search(
            query=(
                f"{data.person_name} "
                "recent encounter"
            ),
            top_k=5,
            person_id=(
                data.person_id
            ),
        )
    )

    assert (
        len(
            results
        )
        >= 2
    )

    expected_newer = (
        data.newer_time
        .isoformat(
            timespec="seconds"
        )
    )

    actual_newer = (
        results[0]
        .metadata
        .get(
            "detected_at"
        )
    )

    assert (
        actual_newer
        == expected_newer
    )

    print_test_result(
        "Prefer newer person memory"
    )


# ============================================================
# TEST MEMORY TYPE
# ============================================================

def test_memory_type(
    search: PersonMemorySearch,
    data: PersonVectorTestData,
) -> None:

    results = (
        search.search(
            query=(
                data.person_name
            ),
            top_k=5,
            person_id=(
                data.person_id
            ),
        )
    )

    assert results

    first = (
        results[0]
    )

    assert (
        first.metadata.get(
            "memory_type"
        )
        == "person"
    )

    assert (
        first.metadata.get(
            "event_type"
        )
        == "person_encounter"
    )

    print_test_result(
        "Person memory metadata type"
    )


# ============================================================
# CLEANUP
# ============================================================

def cleanup(
    vector_memory: PersonVectorMemory,
    data: PersonVectorTestData,
) -> None:

    try:

        vector_memory.delete_person_event(
            data.event_id_1
        )

    except Exception:

        pass

    try:

        vector_memory.delete_person_event(
            data.event_id_2
        )

    except Exception:

        pass


# ============================================================
# RUN
# ============================================================

def test_person_vector_memory(
) -> None:

    print(
        "\nMemorAi Person Vector "
        "Memory Test Suite\n"
    )

    vector_memory = (
        PersonVectorMemory()
    )

    search = (
        PersonMemorySearch()
    )

    data = (
        PersonVectorTestData()
    )

    try:

        test_add_person_memories(
            vector_memory,
            data,
        )

        test_search_by_name(
            search,
            data,
        )

        test_relation_metadata(
            search,
            data,
        )

        test_newest_first(
            search,
            data,
        )

        test_memory_type(
            search,
            data,
        )

        print(
            "\nAll MemorAi person "
            "vector-memory tests passed."
        )

    finally:

        cleanup(
            vector_memory,
            data,
        )

        print(
            "\nTemporary person vector "
            "memories cleaned."
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    test_person_vector_memory()