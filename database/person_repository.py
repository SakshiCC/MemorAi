from __future__ import annotations

from datetime import (
    date,
    datetime,
    time,
)
from typing import Any

import numpy as np

from database.db_manager import (
    execute,
    fetch_all,
    fetch_one,
)


# ============================================================
# HELPERS
# ============================================================

def normalize_person_name(
    name: str,
) -> str:

    normalized = " ".join(
        name.strip().lower().split()
    )

    if not normalized:

        raise ValueError(
            "Person name cannot be empty."
        )

    return normalized


def normalize_display_name(
    name: str,
) -> str:

    display_name = " ".join(
        name.strip().split()
    )

    if not display_name:

        raise ValueError(
            "Display name cannot be empty."
        )

    return display_name


def normalize_relation(
    relation: str | None,
) -> str | None:

    if relation is None:

        return None

    cleaned = " ".join(
        relation.strip().split()
    )

    if not cleaned:

        return None

    return cleaned


# ============================================================
# REGISTERED PEOPLE
# ============================================================

def create_person(
    name: str,
    display_name: str | None = None,
    relation: str | None = None,
    notes: str | None = None,
    profile_image_path: str | None = None,
) -> int:

    normalized_name = (
        normalize_person_name(
            name
        )
    )

    if display_name is None:

        cleaned_display_name = (
            normalize_display_name(
                name
            )
        )

    else:

        cleaned_display_name = (
            normalize_display_name(
                display_name
            )
        )

    cleaned_relation = (
        normalize_relation(
            relation
        )
    )

    existing = get_person_by_name(
        normalized_name
    )

    if existing is not None:

        raise ValueError(
            "Person already registered: "
            f"{cleaned_display_name}"
        )

    query = """
    INSERT INTO registered_people (
        name,
        display_name,
        relation,
        profile_image_path,
        notes,
        is_active
    )
    VALUES (
        %s,
        %s,
        %s,
        %s,
        %s,
        TRUE
    )
    """

    person_id = execute(
        query,
        (
            normalized_name,
            cleaned_display_name,
            cleaned_relation,
            profile_image_path,
            notes,
        ),
    )

    if person_id <= 0:

        raise RuntimeError(
            "Failed to create person."
        )

    return person_id


def get_person_by_id(
    person_id: int,
) -> dict[str, Any] | None:

    query = """
    SELECT
        id,
        name,
        display_name,
        relation,
        profile_image_path,
        notes,
        is_active,
        created_at,
        updated_at
    FROM registered_people
    WHERE id = %s
    LIMIT 1
    """

    return fetch_one(
        query,
        (
            person_id,
        ),
    )


def get_person_by_name(
    name: str,
) -> dict[str, Any] | None:

    normalized_name = (
        normalize_person_name(
            name
        )
    )

    query = """
    SELECT
        id,
        name,
        display_name,
        relation,
        profile_image_path,
        notes,
        is_active,
        created_at,
        updated_at
    FROM registered_people
    WHERE name = %s
    LIMIT 1
    """

    return fetch_one(
        query,
        (
            normalized_name,
        ),
    )


def get_all_people(
    active_only: bool = True,
) -> list[dict[str, Any]]:

    if active_only:

        query = """
        SELECT
            id,
            name,
            display_name,
            relation,
            profile_image_path,
            notes,
            is_active,
            created_at,
            updated_at
        FROM registered_people
        WHERE is_active = TRUE
        ORDER BY display_name ASC
        """

    else:

        query = """
        SELECT
            id,
            name,
            display_name,
            relation,
            profile_image_path,
            notes,
            is_active,
            created_at,
            updated_at
        FROM registered_people
        ORDER BY display_name ASC
        """

    return fetch_all(
        query
    )


# ============================================================
# PEOPLE + LAST SEEN
# ============================================================

def get_people_with_last_seen(
    active_only: bool = False,
) -> list[dict[str, Any]]:

    if active_only:

        query = """
        SELECT
            p.id,
            p.name,
            p.display_name,
            p.relation,
            p.profile_image_path,
            p.notes,
            p.is_active,
            p.created_at,
            p.updated_at,

            MAX(
                pe.detected_at
            ) AS last_seen

        FROM registered_people AS p

        LEFT JOIN person_events AS pe
            ON pe.person_id = p.id

        WHERE p.is_active = TRUE

        GROUP BY
            p.id,
            p.name,
            p.display_name,
            p.relation,
            p.profile_image_path,
            p.notes,
            p.is_active,
            p.created_at,
            p.updated_at

        ORDER BY
            p.display_name ASC
        """

    else:

        query = """
        SELECT
            p.id,
            p.name,
            p.display_name,
            p.relation,
            p.profile_image_path,
            p.notes,
            p.is_active,
            p.created_at,
            p.updated_at,

            MAX(
                pe.detected_at
            ) AS last_seen

        FROM registered_people AS p

        LEFT JOIN person_events AS pe
            ON pe.person_id = p.id

        GROUP BY
            p.id,
            p.name,
            p.display_name,
            p.relation,
            p.profile_image_path,
            p.notes,
            p.is_active,
            p.created_at,
            p.updated_at

        ORDER BY
            p.display_name ASC
        """

    return fetch_all(
        query
    )


# ============================================================
# UPDATE PERSON
# ============================================================

def update_person(
    person_id: int,
    display_name: str | None = None,
    relation: str | None = None,
    notes: str | None = None,
) -> None:

    person = get_person_by_id(
        person_id
    )

    if person is None:

        raise ValueError(
            "Person ID does not exist: "
            f"{person_id}"
        )

    new_display_name = (
        normalize_display_name(
            display_name
        )
        if display_name is not None
        else person["display_name"]
    )

    new_relation = (
        normalize_relation(
            relation
        )
        if relation is not None
        else person["relation"]
    )

    new_notes = (
        notes
        if notes is not None
        else person["notes"]
    )

    query = """
    UPDATE registered_people
    SET
        display_name = %s,
        relation = %s,
        notes = %s
    WHERE id = %s
    """

    execute(
        query,
        (
            new_display_name,
            new_relation,
            new_notes,
            person_id,
        ),
    )


# ============================================================
# UPDATE PROFILE PHOTO
# ============================================================

def update_profile_image_path(
    person_id: int,
    profile_image_path: str | None,
) -> None:

    person = get_person_by_id(
        person_id
    )

    if person is None:

        raise ValueError(
            "Person ID does not exist: "
            f"{person_id}"
        )

    execute(
        """
        UPDATE registered_people
        SET profile_image_path = %s
        WHERE id = %s
        """,
        (
            profile_image_path,
            person_id,
        ),
    )


# ============================================================
# ACTIVE / INACTIVE
# ============================================================

def deactivate_person(
    person_id: int,
) -> None:

    person = get_person_by_id(
        person_id
    )

    if person is None:

        raise ValueError(
            "Person ID does not exist: "
            f"{person_id}"
        )

    execute(
        """
        UPDATE registered_people
        SET is_active = FALSE
        WHERE id = %s
        """,
        (
            person_id,
        ),
    )


def reactivate_person(
    person_id: int,
) -> None:

    person = get_person_by_id(
        person_id
    )

    if person is None:

        raise ValueError(
            "Person ID does not exist: "
            f"{person_id}"
        )

    execute(
        """
        UPDATE registered_people
        SET is_active = TRUE
        WHERE id = %s
        """,
        (
            person_id,
        ),
    )


# ============================================================
# FACE EMBEDDINGS
# ============================================================

def embedding_to_bytes(
    embedding: np.ndarray,
) -> tuple[bytes, int, str]:

    if embedding is None:

        raise ValueError(
            "Embedding cannot be None."
        )

    embedding_array = np.asarray(
        embedding,
        dtype=np.float32,
    ).reshape(-1)

    if embedding_array.size == 0:

        raise ValueError(
            "Embedding cannot be empty."
        )

    return (
        embedding_array.tobytes(),
        int(
            embedding_array.size
        ),
        "float32",
    )


def bytes_to_embedding(
    embedding_bytes: bytes,
    embedding_dim: int,
    embedding_dtype: str = "float32",
) -> np.ndarray:

    if not embedding_bytes:

        raise ValueError(
            "Embedding data is empty."
        )

    if embedding_dtype != "float32":

        raise ValueError(
            "Unsupported embedding dtype: "
            f"{embedding_dtype}"
        )

    embedding = np.frombuffer(
        embedding_bytes,
        dtype=np.float32,
    ).copy()

    if embedding.size != embedding_dim:

        raise ValueError(
            "Stored embedding dimension does "
            "not match embedding data."
        )

    return embedding


def save_face_embedding(
    person_id: int,
    embedding: np.ndarray,
    sample_image_path: str | None = None,
) -> int:

    person = get_person_by_id(
        person_id
    )

    if person is None:

        raise ValueError(
            "Person ID does not exist: "
            f"{person_id}"
        )

    (
        embedding_bytes,
        embedding_dim,
        embedding_dtype,
    ) = embedding_to_bytes(
        embedding
    )

    query = """
    INSERT INTO person_face_embeddings (
        person_id,
        embedding,
        embedding_dim,
        embedding_dtype,
        sample_image_path
    )
    VALUES (
        %s,
        %s,
        %s,
        %s,
        %s
    )
    """

    embedding_id = execute(
        query,
        (
            person_id,
            embedding_bytes,
            embedding_dim,
            embedding_dtype,
            sample_image_path,
        ),
    )

    if embedding_id <= 0:

        raise RuntimeError(
            "Failed to save face embedding."
        )

    return embedding_id


def get_face_embeddings(
    person_id: int,
) -> list[dict[str, Any]]:

    rows = fetch_all(
        """
        SELECT
            id,
            person_id,
            embedding,
            embedding_dim,
            embedding_dtype,
            sample_image_path,
            created_at
        FROM person_face_embeddings
        WHERE person_id = %s
        ORDER BY id ASC
        """,
        (
            person_id,
        ),
    )

    results: list[
        dict[str, Any]
    ] = []

    for row in rows:

        row["embedding"] = (
            bytes_to_embedding(
                row["embedding"],
                int(
                    row["embedding_dim"]
                ),
                row[
                    "embedding_dtype"
                ],
            )
        )

        results.append(
            row
        )

    return results


def get_all_face_embeddings(
    active_people_only: bool = True,
) -> list[dict[str, Any]]:

    if active_people_only:

        query = """
        SELECT
            e.id,
            e.person_id,
            e.embedding,
            e.embedding_dim,
            e.embedding_dtype,
            e.sample_image_path,
            e.created_at,

            p.name,
            p.display_name,
            p.relation

        FROM person_face_embeddings AS e

        INNER JOIN registered_people AS p
            ON p.id = e.person_id

        WHERE p.is_active = TRUE

        ORDER BY
            p.id ASC,
            e.id ASC
        """

    else:

        query = """
        SELECT
            e.id,
            e.person_id,
            e.embedding,
            e.embedding_dim,
            e.embedding_dtype,
            e.sample_image_path,
            e.created_at,

            p.name,
            p.display_name,
            p.relation

        FROM person_face_embeddings AS e

        INNER JOIN registered_people AS p
            ON p.id = e.person_id

        ORDER BY
            p.id ASC,
            e.id ASC
        """

    rows = fetch_all(
        query
    )

    for row in rows:

        row["embedding"] = (
            bytes_to_embedding(
                row["embedding"],
                int(
                    row["embedding_dim"]
                ),
                row[
                    "embedding_dtype"
                ],
            )
        )

    return rows


def delete_face_embedding(
    embedding_id: int,
) -> None:

    execute(
        """
        DELETE FROM person_face_embeddings
        WHERE id = %s
        """,
        (
            embedding_id,
        ),
    )


def delete_face_embeddings_for_person(
    person_id: int,
) -> None:

    execute(
        """
        DELETE FROM person_face_embeddings
        WHERE person_id = %s
        """,
        (
            person_id,
        ),
    )


def count_face_embeddings(
    person_id: int,
) -> int:

    result = fetch_one(
        """
        SELECT COUNT(*) AS count
        FROM person_face_embeddings
        WHERE person_id = %s
        """,
        (
            person_id,
        ),
    )

    if result is None:

        return 0

    return int(
        result["count"]
    )


# ============================================================
# PERSON MEMORY TEXT
# ============================================================

def create_person_memory_text(
    person_name: str,
    detected_at: datetime,
    relation: str | None = None,
) -> str:

    readable_time = (
        detected_at.strftime(
            "%d %b %Y at %I:%M %p"
        )
    )

    if relation:

        return (
            f"MemorAi saw {person_name}, "
            f"the user's {relation}, "
            f"on {readable_time}."
        )

    return (
        f"MemorAi saw {person_name} "
        f"on {readable_time}."
    )


# ============================================================
# SAVE PERSON EVENT
# ============================================================

def save_person_event(
    person_id: int,
    similarity: float,
    detected_at: datetime,
    image_path: str | None = None,
    memory_text: str | None = None,
) -> int:

    person = get_person_by_id(
        person_id
    )

    if person is None:

        raise ValueError(
            "Person ID does not exist: "
            f"{person_id}"
        )

    display_name = (
        person["display_name"]
        or person["name"]
    )

    relation = (
        person["relation"]
    )

    similarity_value = float(
        similarity
    )

    if not (
        0.0
        <= similarity_value
        <= 1.0
    ):

        raise ValueError(
            "Similarity must be between "
            "0.0 and 1.0."
        )

    if memory_text is None:

        memory_text = (
            create_person_memory_text(
                person_name=(
                    display_name
                ),
                detected_at=(
                    detected_at
                ),
                relation=(
                    relation
                ),
            )
        )

    query = """
    INSERT INTO person_events (
        person_id,
        person_name,
        similarity,
        image_path,
        detected_at,
        memory_text
    )
    VALUES (
        %s,
        %s,
        %s,
        %s,
        %s,
        %s
    )
    """

    event_id = execute(
        query,
        (
            person_id,
            display_name,
            similarity_value,
            image_path,
            detected_at,
            memory_text,
        ),
    )

    if event_id <= 0:

        raise RuntimeError(
            "Failed to save person event."
        )

    return event_id


# ============================================================
# PERSON EVENT BY ID
# ============================================================

def get_person_event_by_id(
    event_id: int,
) -> dict[str, Any] | None:
    """
    Return one stored person encounter.

    The registered_people join provides relation and profile
    information without duplicating those values in
    person_events.
    """

    return fetch_one(
        """
        SELECT
            pe.id,
            pe.person_id,
            pe.person_name,
            pe.similarity,
            pe.image_path,
            pe.detected_at,
            pe.memory_text,
            pe.created_at,

            p.relation,
            p.profile_image_path

        FROM person_events AS pe

        INNER JOIN registered_people AS p
            ON p.id = pe.person_id

        WHERE pe.id = %s

        LIMIT 1
        """,
        (
            event_id,
        ),
    )


# ============================================================
# LATEST PERSON EVENT
# ============================================================

def get_latest_person_event(
    person_id: int,
) -> dict[str, Any] | None:

    return fetch_one(
        """
        SELECT
            pe.id,
            pe.person_id,
            pe.person_name,
            pe.similarity,
            pe.image_path,
            pe.detected_at,
            pe.memory_text,
            pe.created_at,

            p.relation,
            p.profile_image_path

        FROM person_events AS pe

        INNER JOIN registered_people AS p
            ON p.id = pe.person_id

        WHERE pe.person_id = %s

        ORDER BY
            pe.detected_at DESC

        LIMIT 1
        """,
        (
            person_id,
        ),
    )


def get_latest_person_event_by_name(
    name: str,
) -> dict[str, Any] | None:

    person = get_person_by_name(
        name
    )

    if person is None:

        return None

    return get_latest_person_event(
        int(
            person["id"]
        )
    )


# ============================================================
# RECENT PERSON EVENTS
# ============================================================

def get_recent_person_events(
    limit: int = 20,
) -> list[dict[str, Any]]:

    safe_limit = max(
        1,
        int(
            limit
        ),
    )

    return fetch_all(
        """
        SELECT
            pe.id,
            pe.person_id,
            pe.person_name,
            pe.similarity,
            pe.image_path,
            pe.detected_at,
            pe.memory_text,
            pe.created_at,

            p.relation,
            p.profile_image_path

        FROM person_events AS pe

        INNER JOIN registered_people AS p
            ON p.id = pe.person_id

        ORDER BY
            pe.detected_at DESC

        LIMIT %s
        """,
        (
            safe_limit,
        ),
    )


# ============================================================
# PERSON EVENTS BETWEEN TIMES
# ============================================================

def get_person_events_between(
    start_time: datetime,
    end_time: datetime,
    person_id: int | None = None,
) -> list[dict[str, Any]]:

    if start_time > end_time:

        raise ValueError(
            "start_time cannot be after "
            "end_time."
        )

    if person_id is None:

        return fetch_all(
            """
            SELECT
                pe.id,
                pe.person_id,
                pe.person_name,
                pe.similarity,
                pe.image_path,
                pe.detected_at,
                pe.memory_text,
                pe.created_at,

                p.relation,
                p.profile_image_path

            FROM person_events AS pe

            INNER JOIN registered_people AS p
                ON p.id = pe.person_id

            WHERE pe.detected_at >= %s
              AND pe.detected_at <= %s

            ORDER BY
                pe.detected_at DESC
            """,
            (
                start_time,
                end_time,
            ),
        )

    return fetch_all(
        """
        SELECT
            pe.id,
            pe.person_id,
            pe.person_name,
            pe.similarity,
            pe.image_path,
            pe.detected_at,
            pe.memory_text,
            pe.created_at,

            p.relation,
            p.profile_image_path

        FROM person_events AS pe

        INNER JOIN registered_people AS p
            ON p.id = pe.person_id

        WHERE pe.person_id = %s
          AND pe.detected_at >= %s
          AND pe.detected_at <= %s

        ORDER BY
            pe.detected_at DESC
        """,
        (
            person_id,
            start_time,
            end_time,
        ),
    )


# ============================================================
# PERSON EVENTS FOR ONE DATE
# ============================================================

def get_person_events_for_date(
    target_date: date,
    limit: int | None = None,
) -> list[dict[str, Any]]:

    start_time = datetime.combine(
        target_date,
        time.min,
    )

    end_time = datetime.combine(
        target_date,
        time.max,
    )

    if limit is None:

        return get_person_events_between(
            start_time=(
                start_time
            ),
            end_time=(
                end_time
            ),
        )

    safe_limit = max(
        1,
        int(
            limit
        ),
    )

    return fetch_all(
        """
        SELECT
            pe.id,
            pe.person_id,
            pe.person_name,
            pe.similarity,
            pe.image_path,
            pe.detected_at,
            pe.memory_text,
            pe.created_at,

            p.relation,
            p.profile_image_path

        FROM person_events AS pe

        INNER JOIN registered_people AS p
            ON p.id = pe.person_id

        WHERE pe.detected_at >= %s
          AND pe.detected_at <= %s

        ORDER BY
            pe.detected_at DESC

        LIMIT %s
        """,
        (
            start_time,
            end_time,
            safe_limit,
        ),
    )