from __future__ import annotations

from database.medicine_repository import (
    STATUS_CAMERA_CONFIRMED,
    get_recent_medicine_events,
)

from database.medicine_vector_memory import (
    MedicineVectorMemory,
)

from rag.medicine_memory_search import (
    MedicineMemorySearch,
)


# ============================================================
# PRINT TEST RESULT
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
# GET REAL CONFIRMED MEDICINE EVENT
# ============================================================

def get_test_event() -> dict:
    """
    Use a real existing camera-confirmed medicine event.

    We deliberately do NOT create a fake confirmation event
    because medicine_events should represent actual MemorAi
    observations.
    """

    events = (
        get_recent_medicine_events(
            limit=20,
            confirmed_only=True,
        )
    )

    if not events:

        raise RuntimeError(
            (
                "No camera-confirmed medicine event "
                "exists in MySQL.\n"
                "Please perform one successful medicine "
                "camera confirmation before running this "
                "test."
            )
        )

    return (
        events[
            0
        ]
    )


# ============================================================
# TEST 1
# REAL EVENT EXISTS
# ============================================================

def test_real_confirmed_event_exists(
) -> dict:

    event = (
        get_test_event()
    )

    assert (
        event.get(
            "id"
        )
        is not None
    )

    assert (
        event.get(
            "medicine_id"
        )
        is not None
    )

    assert (
        event.get(
            "medicine_name"
        )
    )

    assert (
        event.get(
            "status"
        )
        == STATUS_CAMERA_CONFIRMED
    )

    assert (
        event.get(
            "observed_at"
        )
        is not None
    )

    print_test_result(
        "Real camera-confirmed medicine event exists",
        True,
    )

    return (
        event
    )


# ============================================================
# TEST 2
# ADD EVENT TO VECTOR MEMORY
# ============================================================

def test_add_event(
    event: dict,
) -> MedicineVectorMemory:

    vector_memory = (
        MedicineVectorMemory()
    )

    before_count = (
        vector_memory.count()
    )

    vector_memory.add_event(
        event
    )

    after_count = (
        vector_memory.count()
    )

    # --------------------------------------------------------
    # UPSERT MEANS:
    #
    # If event is new:
    #     count may increase by one.
    #
    # If event was already indexed:
    #     count stays the same.
    #
    # Both are valid.
    # --------------------------------------------------------

    assert (
        after_count
        >= before_count
    )

    assert (
        after_count
        >= 1
    )

    print_test_result(
        "Medicine event indexed into ChromaDB",
        True,
    )

    return (
        vector_memory
    )


# ============================================================
# TEST 3
# SEARCH BY MEDICINE NAME
# ============================================================

def test_search_by_medicine_name(
    event: dict,
) -> None:

    search = (
        MedicineMemorySearch()
    )

    medicine_name = (
        str(
            event[
                "medicine_name"
            ]
        )
    )

    query = (
        f"When was {medicine_name} "
        "camera confirmed?"
    )

    results = (
        search.search(
            query=(
                query
            ),

            top_k=5,

            medicine_id=int(
                event[
                    "medicine_id"
                ]
            ),
        )
    )

    assert (
        results
    )

    matching = [
        result

        for result
        in results

        if (
            result.medicine_event_id
            == int(
                event[
                    "id"
                ]
            )
        )
    ]

    assert (
        matching
    )

    result = (
        matching[
            0
        ]
    )

    assert (
        result.medicine_name
        is not None
    )

    assert (
        result.medicine_name.lower()
        == medicine_name.lower()
    )

    assert (
        result.status
        == STATUS_CAMERA_CONFIRMED
    )

    print_test_result(
        "Semantic medicine search by medicine name",
        True,
    )


# ============================================================
# TEST 4
# SEARCH CONFIRMED STATUS
# ============================================================

def test_search_confirmed_status(
    event: dict,
) -> None:

    search = (
        MedicineMemorySearch()
    )

    results = (
        search.search(
            query=(
                "What medicine event was "
                "camera confirmed?"
            ),

            top_k=10,

            status=(
                STATUS_CAMERA_CONFIRMED
            ),
        )
    )

    assert (
        results
    )

    for result in results:

        assert (
            result.status
            == STATUS_CAMERA_CONFIRMED
        )

    matching_ids = {
        result.medicine_event_id

        for result
        in results
    }

    assert (
        int(
            event[
                "id"
            ]
        )
        in matching_ids
    )

    print_test_result(
        "Medicine vector status filter",
        True,
    )


# ============================================================
# TEST 5
# SEARCH NATURAL LANGUAGE
# ============================================================

def test_natural_language_search(
    event: dict,
) -> None:

    search = (
        MedicineMemorySearch()
    )

    medicine_name = (
        str(
            event[
                "medicine_name"
            ]
        )
    )

    dosage = (
        str(
            event.get(
                "dosage"
            )
            or ""
        )
        .strip()
    )

    if dosage:

        query = (
            f"What happened with my "
            f"{medicine_name} {dosage} medicine?"
        )

    else:

        query = (
            f"What happened with my "
            f"{medicine_name} medicine?"
        )

    results = (
        search.search(
            query=(
                query
            ),

            top_k=5,

            medicine_id=int(
                event[
                    "medicine_id"
                ]
            ),
        )
    )

    assert (
        results
    )

    result = (
        results[
            0
        ]
    )

    assert (
        result.text
    )

    assert (
        result.medicine_name
        is not None
    )

    print_test_result(
        "Natural-language medicine semantic retrieval",
        True,
    )


# ============================================================
# TEST 6
# MEMORY METADATA
# ============================================================

def test_memory_metadata(
    event: dict,
) -> None:

    search = (
        MedicineMemorySearch()
    )

    results = (
        search.search(
            query=(
                "camera confirmed medicine event"
            ),

            top_k=10,

            medicine_id=int(
                event[
                    "medicine_id"
                ]
            ),
        )
    )

    assert (
        results
    )

    matching = [
        result

        for result
        in results

        if (
            result.medicine_event_id
            == int(
                event[
                    "id"
                ]
            )
        )
    ]

    assert (
        matching
    )

    result = (
        matching[
            0
        ]
    )

    metadata = (
        result.metadata
    )

    assert (
        metadata.get(
            "memory_type"
        )
        == "medicine"
    )

    assert (
        int(
            metadata[
                "medicine_id"
            ]
        )
        == int(
            event[
                "medicine_id"
            ]
        )
    )

    assert (
        metadata.get(
            "medicine_name"
        )
    )

    assert (
        metadata.get(
            "status"
        )
        == STATUS_CAMERA_CONFIRMED
    )

    assert (
        metadata.get(
            "observed_at"
        )
    )

    print_test_result(
        "Medicine vector metadata",
        True,
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print()
    print(
        "MemorAi Medicine Vector Memory Test Suite"
    )
    print(
        "========================================"
    )
    print()

    # ========================================================
    # REAL MYSQL EVENT
    # ========================================================

    event = (
        test_real_confirmed_event_exists()
    )

    print()
    print(
        "Using medicine event:"
    )

    print(
        (
            f"Event ID:      "
            f"{event.get('id')}"
        )
    )

    print(
        (
            f"Medicine ID:   "
            f"{event.get('medicine_id')}"
        )
    )

    print(
        (
            f"Medicine:      "
            f"{event.get('medicine_name')}"
        )
    )

    print(
        (
            f"Dosage:        "
            f"{event.get('dosage')}"
        )
    )

    print(
        (
            f"Status:        "
            f"{event.get('status')}"
        )
    )

    print(
        (
            f"Observed At:   "
            f"{event.get('observed_at')}"
        )
    )

    print()

    # ========================================================
    # VECTOR MEMORY
    # ========================================================

    vector_memory = (
        test_add_event(
            event
        )
    )

    print(
        (
            "Medicine vector memory count: "
            f"{vector_memory.count()}"
        )
    )

    # ========================================================
    # SEARCH TESTS
    # ========================================================

    test_search_by_medicine_name(
        event
    )

    test_search_confirmed_status(
        event
    )

    test_natural_language_search(
        event
    )

    test_memory_metadata(
        event
    )

    print()
    print(
        "========================================"
    )

    print(
        "PASS: Medicine Vector Memory Test Suite"
    )

    print(
        "========================================"
    )

    print()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()