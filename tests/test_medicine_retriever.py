from __future__ import annotations

from datetime import (
    date,
)

from database.medicine_repository import (
    STATUS_CAMERA_CONFIRMED,
    get_all_medicines,
    get_recent_medicine_events,
)

from database.medicine_vector_memory import (
    MedicineVectorMemory,
)

from memory.medicine_retriever import (
    MedicineMemoryRetriever,
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
# GET REAL MEDICINE EVENT
# ============================================================

def get_real_confirmed_event(
) -> dict:

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
                "Perform one successful medicine "
                "camera event before running this test."
            )
        )

    return (
        events[
            0
        ]
    )


# ============================================================
# ENSURE EVENT IS INDEXED
# ============================================================

def ensure_event_is_indexed(
    event: dict,
) -> None:

    vector_memory = (
        MedicineVectorMemory()
    )

    vector_memory.add_event(
        event
    )


# ============================================================
# TEST 1
# MEDICINE QUESTION DETECTION
# ============================================================

def test_medicine_question_detection(
    retriever: MedicineMemoryRetriever,
    medicine_name: str,
) -> None:

    questions = (
        f"Did I take {medicine_name} today?",
        f"When was {medicine_name} last confirmed?",
        "Which medicine was confirmed today?",
        "What medication event happened recently?",
        "Was my medicine camera confirmed?",
    )

    for question in questions:

        assert (
            retriever
            .is_medicine_question(
                question
            )
        )

    print_test_result(
        "Medicine question detection",
        True,
    )


# ============================================================
# TEST 2
# NON MEDICINE QUESTION
# ============================================================

def test_non_medicine_question(
    retriever: MedicineMemoryRetriever,
) -> None:

    assert not (
        retriever
        .is_medicine_question(
            "Where is my cell phone?"
        )
    )

    assert not (
        retriever
        .is_medicine_question(
            "When did I last see Sakshi?"
        )
    )

    assert not (
        retriever
        .is_medicine_question(
            "What did Rahul tell me?"
        )
    )

    print_test_result(
        "Non-medicine questions stay out",
        True,
    )


# ============================================================
# TEST 3
# RESOLVE MEDICINE NAME
# ============================================================

def test_resolve_medicine_name(
    retriever: MedicineMemoryRetriever,
    event: dict,
) -> None:

    medicine_name = (
        str(
            event[
                "medicine_name"
            ]
        )
    )

    medicine = (
        retriever
        .resolve_medicine(
            (
                f"When was {medicine_name} "
                "last confirmed?"
            )
        )
    )

    assert (
        medicine
        is not None
    )

    assert (
        int(
            medicine[
                "id"
            ]
        )
        == int(
            event[
                "medicine_id"
            ]
        )
    )

    print_test_result(
        "Registered medicine name resolution",
        True,
    )


# ============================================================
# TEST 4
# RESOLVE OCR ALIAS
# ============================================================

def test_resolve_ocr_alias(
    retriever: MedicineMemoryRetriever,
    event: dict,
) -> None:

    medicines = (
        get_all_medicines(
            active_only=False
        )
    )

    medicine = next(
        (
            item
            for item in medicines
            if int(
                item[
                    "id"
                ]
            )
            == int(
                event[
                    "medicine_id"
                ]
            )
        ),
        None,
    )

    assert (
        medicine
        is not None
    )

    raw_keywords = (
        str(
            medicine.get(
                "ocr_keywords"
            )
            or ""
        )
    )

    aliases = [
        item.strip()
        for item in raw_keywords.split(
            ","
        )
        if item.strip()
    ]

    if not aliases:

        print(
            (
                "SKIP: OCR alias resolution "
                "(no OCR keyword stored)"
            )
        )

        return

    alias = (
        aliases[
            -1
        ]
    )

    resolved = (
        retriever
        .resolve_medicine(
            (
                f"Did I take my "
                f"{alias} medicine today?"
            )
        )
    )

    assert (
        resolved
        is not None
    )

    assert (
        int(
            resolved[
                "id"
            ]
        )
        == int(
            event[
                "medicine_id"
            ]
        )
    )

    print_test_result(
        "Medicine OCR alias resolution",
        True,
    )


# ============================================================
# TEST 5
# TODAY SPECIFIC MEDICINE
# ============================================================

def test_today_specific_medicine(
    retriever: MedicineMemoryRetriever,
    event: dict,
) -> None:

    medicine_name = (
        str(
            event[
                "medicine_name"
            ]
        )
    )

    results = (
        retriever
        .retrieve(
            (
                f"Did I take "
                f"{medicine_name} today?"
            ),
            top_k=20,
        )
    )

    assert (
        results
    )

    matching = [
        result
        for result in results
        if (
            int(
                result.get(
                    "medicine_id",
                    -1,
                )
            )
            == int(
                event[
                    "medicine_id"
                ]
            )
        )
    ]

    assert (
        matching
    )

    assert any(
        (
            result.get(
                "status"
            )
            == STATUS_CAMERA_CONFIRMED
        )
        for result in matching
    )

    assert all(
        (
            result.get(
                "observed_at"
            ).date()
            == date.today()
        )
        for result in matching
        if hasattr(
            result.get(
                "observed_at"
            ),
            "date",
        )
    )

    print_test_result(
        "Today's specific medicine retrieval",
        True,
    )


# ============================================================
# TEST 6
# LATEST CONFIRMED MEDICINE
# ============================================================

def test_latest_confirmation(
    retriever: MedicineMemoryRetriever,
    event: dict,
) -> None:

    medicine_name = (
        str(
            event[
                "medicine_name"
            ]
        )
    )

    results = (
        retriever
        .retrieve(
            (
                f"When was {medicine_name} "
                "last confirmed?"
            ),
            top_k=5,
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
        int(
            result[
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
        result[
            "status"
        ]
        == STATUS_CAMERA_CONFIRMED
    )

    print_test_result(
        "Latest confirmed medicine retrieval",
        True,
    )


# ============================================================
# TEST 7
# ALL MEDICINES TODAY
# ============================================================

def test_all_medicines_today(
    retriever: MedicineMemoryRetriever,
    event: dict,
) -> None:

    results = (
        retriever
        .retrieve(
            (
                "Which medicines were "
                "confirmed today?"
            ),
            top_k=50,
        )
    )

    assert (
        results
    )

    matching = [
        result
        for result in results
        if (
            int(
                result.get(
                    "id",
                    -1,
                )
            )
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

    print_test_result(
        "Today's medicine-event retrieval",
        True,
    )


# ============================================================
# TEST 8
# RECENT MEDICINE
# ============================================================

def test_recent_medicine(
    retriever: MedicineMemoryRetriever,
    event: dict,
) -> None:

    results = (
        retriever
        .retrieve(
            (
                "What medication events "
                "happened recently?"
            ),
            top_k=20,
        )
    )

    assert (
        results
    )

    matching_ids = {
        int(
            result[
                "id"
            ]
        )
        for result in results
        if (
            result.get(
                "id"
            )
            is not None
        )
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
        "Recent medicine retrieval",
        True,
    )


# ============================================================
# TEST 9
# SEMANTIC MEDICINE FALLBACK
# ============================================================

def test_semantic_medicine_fallback(
    retriever: MedicineMemoryRetriever,
    event: dict,
) -> None:

    medicine_name = (
        str(
            event[
                "medicine_name"
            ]
        )
    )

    results = (
        retriever
        .retrieve(
            (
                "What medication memory "
                "does MemorAi remember?"
            ),
            top_k=10,
        )
    )

    assert (
        results
    )

    names = {
        str(
            result.get(
                "medicine_name"
            )
            or ""
        )
        .strip()
        .lower()
        for result in results
    }

    assert (
        medicine_name.lower()
        in names
    )

    print_test_result(
        "Semantic medicine fallback",
        True,
    )


# ============================================================
# TEST 10
# EMPTY QUESTION
# ============================================================

def test_empty_question(
    retriever: MedicineMemoryRetriever,
) -> None:

    assert (
        retriever.retrieve(
            ""
        )
        == []
    )

    assert not (
        retriever
        .is_medicine_question(
            ""
        )
    )

    print_test_result(
        "Empty medicine question handling",
        True,
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print()
    print(
        "MemorAi Medicine Retriever Test Suite"
    )
    print(
        "====================================="
    )
    print()

    # ========================================================
    # REAL EVENT
    # ========================================================

    event = (
        get_real_confirmed_event()
    )

    print(
        "Using real medicine event:"
    )

    print(
        (
            f"Event ID:    "
            f"{event.get('id')}"
        )
    )

    print(
        (
            f"Medicine:    "
            f"{event.get('medicine_name')}"
        )
    )

    print(
        (
            f"Dosage:      "
            f"{event.get('dosage')}"
        )
    )

    print(
        (
            f"Status:      "
            f"{event.get('status')}"
        )
    )

    print(
        (
            f"Observed At: "
            f"{event.get('observed_at')}"
        )
    )

    print()

    # ========================================================
    # ENSURE SEMANTIC MEMORY EXISTS
    # ========================================================

    ensure_event_is_indexed(
        event
    )

    # ========================================================
    # RETRIEVER
    # ========================================================

    retriever = (
        MedicineMemoryRetriever()
    )

    medicine_name = (
        str(
            event[
                "medicine_name"
            ]
        )
    )

    # ========================================================
    # TESTS
    # ========================================================

    test_medicine_question_detection(
        retriever,
        medicine_name,
    )

    test_non_medicine_question(
        retriever
    )

    test_resolve_medicine_name(
        retriever,
        event,
    )

    test_resolve_ocr_alias(
        retriever,
        event,
    )

    test_today_specific_medicine(
        retriever,
        event,
    )

    test_latest_confirmation(
        retriever,
        event,
    )

    test_all_medicines_today(
        retriever,
        event,
    )

    test_recent_medicine(
        retriever,
        event,
    )

    test_semantic_medicine_fallback(
        retriever,
        event,
    )

    test_empty_question(
        retriever
    )

    print()
    print(
        "====================================="
    )

    print(
        "PASS: Medicine Retriever Test Suite"
    )

    print(
        "====================================="
    )

    print()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()