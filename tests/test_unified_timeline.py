from __future__ import annotations

from datetime import (
    date,
)

from timeline.timeline_manager import (
    FILTER_TODAY,
    MEMORY_TYPE_ALL,
    MEMORY_TYPE_CONVERSATION,
    MEMORY_TYPE_OBJECT,
    MEMORY_TYPE_PERSON,
    count_timeline,
    normalize_memory_type,
    search_timeline,
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

def test_unified_timeline() -> None:

    print(
        (
            "\nMemorAi Unified "
            "Timeline Test Suite\n"
        )
    )

    # ========================================================
    # MEMORY TYPE NORMALIZATION
    # ========================================================

    assert (
        normalize_memory_type(
            None
        )
        == MEMORY_TYPE_ALL
    )

    assert (
        normalize_memory_type(
            "objects"
        )
        == MEMORY_TYPE_OBJECT
    )

    assert (
        normalize_memory_type(
            "people"
        )
        == MEMORY_TYPE_PERSON
    )

    assert (
        normalize_memory_type(
            "conversations"
        )
        == MEMORY_TYPE_CONVERSATION
    )

    print_test_result(
        "Timeline memory type normalization"
    )

    # ========================================================
    # TODAY
    # ========================================================

    memories = (
        search_timeline(
            period=(
                FILTER_TODAY
            ),

            selected_date=(
                date.today()
            ),

            search_text=None,

            limit=100,

            event_type=None,
        )
    )

    assert isinstance(
        memories,
        list,
    )

    print_test_result(
        "Unified timeline search"
    )

    # ========================================================
    # COMMON STRUCTURE
    # ========================================================

    for memory in memories:

        assert (
            memory.get(
                "memory_type"
            )
            in {
                MEMORY_TYPE_OBJECT,
                MEMORY_TYPE_PERSON,
                MEMORY_TYPE_CONVERSATION,
            }
        )

        assert (
            "memory_key"
            in memory
        )

        assert (
            "time"
            in memory
        )

        assert (
            "title"
            in memory
        )

        assert (
            "description"
            in memory
        )

    print_test_result(
        "Unified timeline common record structure"
    )

    # ========================================================
    # CHRONOLOGICAL ORDER
    # ========================================================

    dated_memories = [
        memory

        for memory
        in memories

        if memory.get(
            "time"
        )
        is not None
    ]

    times = [
        memory[
            "time"
        ]

        for memory
        in dated_memories
    ]

    assert (
        times
        == sorted(
            times,
            reverse=True,
        )
    )

    print_test_result(
        "Unified timeline chronological ordering"
    )

    # ========================================================
    # OBJECT FILTER
    # ========================================================

    object_memories = (
        search_timeline(
            period=(
                FILTER_TODAY
            ),

            limit=100,

            event_type=(
                MEMORY_TYPE_OBJECT
            ),
        )
    )

    assert all(
        memory.get(
            "memory_type"
        )
        == MEMORY_TYPE_OBJECT

        for memory
        in object_memories
    )

    print_test_result(
        "Object-only timeline filter"
    )

    # ========================================================
    # PERSON FILTER
    # ========================================================

    person_memories = (
        search_timeline(
            period=(
                FILTER_TODAY
            ),

            limit=100,

            event_type=(
                MEMORY_TYPE_PERSON
            ),
        )
    )

    assert all(
        memory.get(
            "memory_type"
        )
        == MEMORY_TYPE_PERSON

        for memory
        in person_memories
    )

    print_test_result(
        "Person-only timeline filter"
    )

    # ========================================================
    # CONVERSATION FILTER
    # ========================================================

    conversation_memories = (
        search_timeline(
            period=(
                FILTER_TODAY
            ),

            limit=100,

            event_type=(
                MEMORY_TYPE_CONVERSATION
            ),
        )
    )

    assert all(
        memory.get(
            "memory_type"
        )
        == MEMORY_TYPE_CONVERSATION

        for memory
        in conversation_memories
    )

    print_test_result(
        "Conversation-only timeline filter"
    )

    # ========================================================
    # COUNT
    # ========================================================

    total = (
        count_timeline(
            period=(
                FILTER_TODAY
            ),
            event_type=None,
        )
    )

    assert (
        total
        >= len(
            memories
        )
        or len(
            memories
        )
        == 100
    )

    print_test_result(
        "Unified timeline count"
    )

    # ========================================================
    # CONVERSATION SEARCH
    # ========================================================

    appointment_memories = (
        search_timeline(
            period=(
                FILTER_TODAY
            ),

            search_text=(
                "appointment"
            ),

            limit=100,

            event_type=(
                MEMORY_TYPE_CONVERSATION
            ),
        )
    )

    for memory in (
        appointment_memories
    ):

        searchable = (
            " ".join(
                [
                    str(
                        memory.get(
                            "title"
                        )
                        or ""
                    ),

                    str(
                        memory.get(
                            "description"
                        )
                        or ""
                    ),

                    str(
                        memory.get(
                            "full_transcript"
                        )
                        or ""
                    ),
                ]
            )
            .lower()
        )

        assert (
            "appointment"
            in searchable
        )

    print_test_result(
        "Conversation timeline text search"
    )

    print(
        (
            "\nAll MemorAi Unified "
            "Timeline tests passed."
        )
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    test_unified_timeline()