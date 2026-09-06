from __future__ import annotations

from datetime import datetime
from uuid import uuid4

import numpy as np

from database.person_repository import (
    bytes_to_embedding,
    count_face_embeddings,
    create_person,
    create_person_memory_text,
    deactivate_person,
    delete_face_embedding,
    delete_face_embeddings_for_person,
    embedding_to_bytes,
    get_all_face_embeddings,
    get_all_people,
    get_face_embeddings,
    get_latest_person_event,
    get_person_by_id,
    get_person_by_name,
    get_person_events_between,
    get_recent_person_events,
    reactivate_person,
    save_face_embedding,
    save_person_event,
    update_person,
)


# ============================================================
# TEST HELPERS
# ============================================================

def print_test_result(
    test_name: str,
) -> None:

    print(
        f"PASS: {test_name}"
    )


def unique_test_name() -> str:

    suffix = uuid4().hex[:8]

    return (
        f"memorai_test_person_{suffix}"
    )


# ============================================================
# TEST EMBEDDING CONVERSION
# ============================================================

def test_embedding_conversion() -> None:

    original = np.array(
        [
            0.10,
            0.20,
            -0.30,
            0.40,
            0.50,
        ],
        dtype=np.float32,
    )

    (
        binary,
        dimension,
        dtype,
    ) = embedding_to_bytes(
        original
    )

    restored = bytes_to_embedding(
        binary,
        dimension,
        dtype,
    )

    assert dimension == 5

    assert dtype == "float32"

    assert np.allclose(
        original,
        restored,
    )

    print_test_result(
        "Embedding serialization"
    )


# ============================================================
# TEST PERSON CREATION
# ============================================================

def test_person_creation(
    test_name: str,
) -> int:

    person_id = create_person(
        test_name,
        display_name="MemorAi Test Person",
        notes="Temporary automated test identity.",
    )

    assert person_id > 0

    person = get_person_by_id(
        person_id
    )

    assert person is not None

    assert (
        person["name"]
        == test_name.lower()
    )

    assert (
        person["display_name"]
        == "MemorAi Test Person"
    )

    print_test_result(
        "Person creation"
    )

    return person_id


# ============================================================
# TEST GET BY NAME
# ============================================================

def test_get_person_by_name(
    test_name: str,
    person_id: int,
) -> None:

    person = get_person_by_name(
        test_name
    )

    assert person is not None

    assert int(
        person["id"]
    ) == person_id

    print_test_result(
        "Get person by name"
    )


# ============================================================
# TEST UPDATE
# ============================================================

def test_person_update(
    person_id: int,
) -> None:

    update_person(
        person_id,
        display_name=(
            "Updated MemorAi Test Person"
        ),
        notes=(
            "Updated repository test."
        ),
    )

    person = get_person_by_id(
        person_id
    )

    assert person is not None

    assert (
        person["display_name"]
        == "Updated MemorAi Test Person"
    )

    assert (
        person["notes"]
        == "Updated repository test."
    )

    print_test_result(
        "Person update"
    )


# ============================================================
# TEST FACE EMBEDDING
# ============================================================

def test_face_embedding(
    person_id: int,
) -> int:

    fake_embedding = np.random.default_rng(
        42
    ).random(
        128,
        dtype=np.float32,
    )

    embedding_id = (
        save_face_embedding(
            person_id,
            fake_embedding,
            sample_image_path=(
                "test/sample_face.jpg"
            ),
        )
    )

    assert embedding_id > 0

    count = count_face_embeddings(
        person_id
    )

    assert count >= 1

    embeddings = get_face_embeddings(
        person_id
    )

    assert embeddings

    stored = embeddings[-1]

    assert isinstance(
        stored["embedding"],
        np.ndarray,
    )

    assert stored[
        "embedding"
    ].shape == (128,)

    assert np.allclose(
        stored["embedding"],
        fake_embedding,
    )

    print_test_result(
        "Face embedding storage"
    )

    return embedding_id


# ============================================================
# TEST ALL EMBEDDINGS
# ============================================================

def test_all_embeddings(
    person_id: int,
) -> None:

    embeddings = (
        get_all_face_embeddings()
    )

    found = any(
        int(
            row["person_id"]
        )
        == person_id
        for row in embeddings
    )

    assert found

    print_test_result(
        "All face embeddings retrieval"
    )


# ============================================================
# TEST PERSON EVENT
# ============================================================

def test_person_event(
    person_id: int,
) -> int:

    detected_at = datetime.now()

    event_id = save_person_event(
        person_id=person_id,
        similarity=0.87,
        detected_at=detected_at,
        image_path=(
            "evidence/people/"
            "test_person.jpg"
        ),
    )

    assert event_id > 0

    latest = (
        get_latest_person_event(
            person_id
        )
    )

    assert latest is not None

    assert int(
        latest["person_id"]
    ) == person_id

    assert float(
        latest["similarity"]
    ) == 0.87

    assert latest[
        "memory_text"
    ]

    print_test_result(
        "Person encounter event"
    )

    return event_id


# ============================================================
# TEST DATE RANGE
# ============================================================

def test_person_event_range(
    person_id: int,
) -> None:

    now = datetime.now()

    start = now.replace(
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
    )

    end = now.replace(
        hour=23,
        minute=59,
        second=59,
        microsecond=999999,
    )

    events = (
        get_person_events_between(
            start,
            end,
            person_id=person_id,
        )
    )

    assert events

    print_test_result(
        "Person event time-range retrieval"
    )


# ============================================================
# TEST RECENT EVENTS
# ============================================================

def test_recent_events(
    person_id: int,
) -> None:

    events = (
        get_recent_person_events(
            limit=100
        )
    )

    found = any(
        int(
            event["person_id"]
        )
        == person_id
        for event in events
    )

    assert found

    print_test_result(
        "Recent person events retrieval"
    )


# ============================================================
# TEST MEMORY TEXT
# ============================================================

def test_memory_text() -> None:

    detected_at = datetime(
        2026,
        8,
        13,
        20,
        42,
    )

    text = (
        create_person_memory_text(
            "Rahul",
            detected_at,
        )
    )

    assert "Rahul" in text

    assert "13 Aug 2026" in text

    assert "08:42 PM" in text

    print_test_result(
        "Person memory text generation"
    )


# ============================================================
# TEST ACTIVE / INACTIVE
# ============================================================

def test_activation(
    person_id: int,
) -> None:

    deactivate_person(
        person_id
    )

    person = get_person_by_id(
        person_id
    )

    assert person is not None

    assert not bool(
        person["is_active"]
    )

    active_people = (
        get_all_people(
            active_only=True
        )
    )

    active_ids = {
        int(
            person["id"]
        )
        for person in active_people
    }

    assert (
        person_id
        not in active_ids
    )

    reactivate_person(
        person_id
    )

    person = get_person_by_id(
        person_id
    )

    assert person is not None

    assert bool(
        person["is_active"]
    )

    print_test_result(
        "Person activation state"
    )


# ============================================================
# CLEANUP
# ============================================================

def cleanup_test_data(
    person_id: int,
    embedding_id: int,
) -> None:

    # We cannot delete the registered person
    # because person_events uses ON DELETE RESTRICT.
    #
    # Instead:
    # 1. remove our fake embedding
    # 2. deactivate the temporary identity

    delete_face_embedding(
        embedding_id
    )

    remaining = (
        count_face_embeddings(
            person_id
        )
    )

    assert remaining == 0

    deactivate_person(
        person_id
    )

    print_test_result(
        "Test embedding cleanup"
    )


# ============================================================
# FULL TEST
# ============================================================

def test_person_repository() -> None:

    print()
    print(
        "MemorAi Person Repository Test Suite"
    )
    print("=" * 50)

    test_embedding_conversion()

    test_name = (
        unique_test_name()
    )

    person_id = (
        test_person_creation(
            test_name
        )
    )

    test_get_person_by_name(
        test_name,
        person_id,
    )

    test_person_update(
        person_id
    )

    embedding_id = (
        test_face_embedding(
            person_id
        )
    )

    test_all_embeddings(
        person_id
    )

    test_person_event(
        person_id
    )

    test_person_event_range(
        person_id
    )

    test_recent_events(
        person_id
    )

    test_memory_text()

    test_activation(
        person_id
    )

    cleanup_test_data(
        person_id,
        embedding_id,
    )

    print()
    print(
        "All MemorAi person repository "
        "tests passed."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    test_person_repository()