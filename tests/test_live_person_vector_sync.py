from __future__ import annotations

from datetime import datetime

from database.person_repository import (
    get_recent_person_events,
)

from database.person_vector_memory import (
    PersonVectorMemory,
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
# NORMALIZE DATETIME
# ============================================================

def normalize_datetime(
    value,
) -> str:

    if isinstance(
        value,
        datetime,
    ):

        return value.isoformat(
            timespec="seconds"
        )

    return str(
        value
        or ""
    )


# ============================================================
# MAIN TEST
# ============================================================

def test_live_person_vector_sync(
) -> None:

    print(
        "\nMemorAi Live Person "
        "Vector Sync Test\n"
    )

    # ========================================================
    # MYSQL
    # ========================================================

    events = (
        get_recent_person_events(
            limit=1
        )
    )

    assert events, (
        "No person encounter exists "
        "in MySQL."
    )

    mysql_event = (
        events[0]
    )

    event_id = int(
        mysql_event[
            "id"
        ]
    )

    person_id = int(
        mysql_event[
            "person_id"
        ]
    )

    person_name = str(
        mysql_event[
            "person_name"
        ]
    )

    print(
        "Latest MySQL encounter:"
    )

    print(
        f"  Event ID: {event_id}"
    )

    print(
        f"  Person ID: {person_id}"
    )

    print(
        f"  Person: {person_name}"
    )

    # ========================================================
    # CHROMA
    # ========================================================

    vector_memory = (
        PersonVectorMemory()
    )

    vector_record = (
        vector_memory
        .get_person_event(
            event_id
        )
    )

    assert vector_record is not None, (
        "The latest MySQL person event "
        f"{event_id} does not exist in "
        "person ChromaDB."
    )

    print_test_result(
        "Latest MySQL event exists in Chroma"
    )

    # ========================================================
    # EXPECTED VECTOR ID
    # ========================================================

    expected_memory_id = (
        f"person_event_{event_id}"
    )

    assert (
        vector_record[
            "memory_id"
        ]
        == expected_memory_id
    )

    print_test_result(
        "Person vector memory ID"
    )

    # ========================================================
    # METADATA
    # ========================================================

    metadata = (
        vector_record[
            "metadata"
        ]
    )

    # --------------------------------------------------------
    # EVENT ID
    # --------------------------------------------------------

    assert int(
        metadata[
            "person_event_id"
        ]
    ) == event_id

    print_test_result(
        "Person event ID synchronized"
    )

    # --------------------------------------------------------
    # PERSON ID
    # --------------------------------------------------------

    assert int(
        metadata[
            "person_id"
        ]
    ) == person_id

    print_test_result(
        "Person ID synchronized"
    )

    # --------------------------------------------------------
    # PERSON NAME
    # --------------------------------------------------------

    assert (
        str(
            metadata[
                "person_name"
            ]
        )
        == person_name
    )

    print_test_result(
        "Person name synchronized"
    )

    # --------------------------------------------------------
    # RELATION
    # --------------------------------------------------------

    mysql_relation = (
        mysql_event.get(
            "relation"
        )
    )

    chroma_relation = (
        metadata.get(
            "relation"
        )
    )

    if mysql_relation is None:

        assert (
            chroma_relation
            is None
            or str(
                chroma_relation
            ).strip()
            == ""
        )

    else:

        assert (
            str(
                chroma_relation
            )
            == str(
                mysql_relation
            )
        )

    print_test_result(
        "Person relation synchronized"
    )

    # --------------------------------------------------------
    # DETECTED AT
    # --------------------------------------------------------

    mysql_detected_at = (
        normalize_datetime(
            mysql_event.get(
                "detected_at"
            )
        )
    )

    chroma_detected_at = (
        normalize_datetime(
            metadata.get(
                "detected_at"
            )
        )
    )

    assert (
        mysql_detected_at
        == chroma_detected_at
    )

    print_test_result(
        "Encounter timestamp synchronized"
    )

    # --------------------------------------------------------
    # MEMORY TEXT
    # --------------------------------------------------------

    mysql_memory_text = str(
        mysql_event.get(
            "memory_text"
        )
        or ""
    ).strip()

    chroma_memory_text = str(
        vector_record.get(
            "text"
        )
        or ""
    ).strip()

    assert mysql_memory_text

    assert (
        chroma_memory_text
        == mysql_memory_text
    )

    print_test_result(
        "Person memory text synchronized"
    )

    # --------------------------------------------------------
    # MEMORY TYPE
    # --------------------------------------------------------

    assert (
        metadata.get(
            "memory_type"
        )
        == "person"
    )

    assert (
        metadata.get(
            "event_type"
        )
        == "person_encounter"
    )

    print_test_result(
        "Person semantic metadata type"
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    print(
        "\nMySQL and Chroma person "
        "memory are synchronized."
    )

    print(
        f"Verified event: "
        f"person_event_{event_id}"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    test_live_person_vector_sync()