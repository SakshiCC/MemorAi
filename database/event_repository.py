from __future__ import annotations

from datetime import (
    date,
    datetime,
    time,
)

from typing import Any

from database.db_manager import (
    execute,
    fetch_all,
    fetch_one,
)

from utils.constants import (
    EVENT_TYPE_OBJECT_DETECTED,
)

from utils.helpers import (
    normalize_object_name,
)


# ============================================================
# HIGH-LEVEL MEMORY EVENTS
# ============================================================


def save_event(
    object_name: str,
    confidence: float | None,
    location: str | None,
    detected_at: datetime,
    image_path: str | None = None,
    memory_text: str | None = None,
    event_type: str = EVENT_TYPE_OBJECT_DETECTED,
) -> int:
    """
    Save a high-level memory event.

    This table is mainly used by:

        - MemoryRetriever
        - Timeline
        - RAG
        - Chat
        - NiceGUI
    """

    query = """
        INSERT INTO events (
            event_type,
            object_name,
            confidence,
            location,
            detected_at,
            image_path,
            memory_text
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """

    return execute(
        query,
        (
            event_type,
            normalize_object_name(
                object_name
            ),
            confidence,
            location,
            detected_at,
            image_path,
            memory_text,
        ),
    )


# ============================================================
# RAW / DETAILED OBJECT DETECTION EVENTS
# ============================================================


def save_object_event(
    object_name: str,
    confidence: float,
    bbox: tuple[int, int, int, int],
    region: str | None,
    detected_at: datetime,
    image_path: str | None = None,
    memory_text: str | None = None,
) -> int:
    """
    Save a confirmed YOLO object detection
    into object_events.
    """

    x1, y1, x2, y2 = (
        bbox
    )

    center_x = int(
        (x1 + x2) / 2
    )

    center_y = int(
        (y1 + y2) / 2
    )

    query = """
        INSERT INTO object_events (
            object_name,
            confidence,
            x1,
            y1,
            x2,
            y2,
            center_x,
            center_y,
            region,
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
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s
        )
    """

    return execute(
        query,
        (
            normalize_object_name(
                object_name
            ),
            confidence,
            x1,
            y1,
            x2,
            y2,
            center_x,
            center_y,
            region,
            image_path,
            detected_at,
            memory_text,
        ),
    )


# ============================================================
# GET EVENT BY ID
# ============================================================


def get_event_by_id(
    event_id: int,
) -> dict[str, Any] | None:

    return fetch_one(
        """
        SELECT *
        FROM events
        WHERE id = %s
        """,
        (
            event_id,
        ),
    )


# ============================================================
# GET OBJECT EVENT BY ID
# ============================================================


def get_object_event_by_id(
    object_event_id: int,
) -> dict[str, Any] | None:

    return fetch_one(
        """
        SELECT *
        FROM object_events
        WHERE id = %s
        """,
        (
            object_event_id,
        ),
    )


# ============================================================
# RECENT HIGH-LEVEL EVENTS
# ============================================================


def get_recent_events(
    limit: int = 20,
) -> list[dict[str, Any]]:

    safe_limit = max(
        1,
        min(
            int(limit),
            10000,
        ),
    )

    return fetch_all(
        f"""
        SELECT *
        FROM events
        ORDER BY detected_at DESC
        LIMIT {safe_limit}
        """
    )


# ============================================================
# RECENT OBJECT EVENTS
# ============================================================


def get_recent_object_events(
    limit: int = 20,
) -> list[dict[str, Any]]:

    safe_limit = max(
        1,
        min(
            int(limit),
            10000,
        ),
    )

    return fetch_all(
        f"""
        SELECT *
        FROM object_events
        ORDER BY detected_at DESC
        LIMIT {safe_limit}
        """
    )


# ============================================================
# SEARCH EVENTS BY EXACT OBJECT
# ============================================================


def get_events_by_object(
    object_name: str,
    limit: int = 20,
) -> list[dict[str, Any]]:

    safe_limit = max(
        1,
        min(
            int(limit),
            10000,
        ),
    )

    return fetch_all(
        f"""
        SELECT *
        FROM events
        WHERE object_name = %s
        ORDER BY detected_at DESC
        LIMIT {safe_limit}
        """,
        (
            normalize_object_name(
                object_name
            ),
        ),
    )


# ============================================================
# SEARCH OBJECT_EVENTS BY OBJECT
# ============================================================


def get_object_events_by_object(
    object_name: str,
    limit: int = 20,
) -> list[dict[str, Any]]:

    safe_limit = max(
        1,
        min(
            int(limit),
            10000,
        ),
    )

    return fetch_all(
        f"""
        SELECT *
        FROM object_events
        WHERE object_name = %s
        ORDER BY detected_at DESC
        LIMIT {safe_limit}
        """,
        (
            normalize_object_name(
                object_name
            ),
        ),
    )


# ============================================================
# LATEST MEMORY EVENT FOR OBJECT
# ============================================================


def get_latest_event_for_object(
    object_name: str,
) -> dict[str, Any] | None:

    return fetch_one(
        """
        SELECT *
        FROM events
        WHERE object_name = %s
        ORDER BY detected_at DESC
        LIMIT 1
        """,
        (
            normalize_object_name(
                object_name
            ),
        ),
    )


# ============================================================
# LATEST DETAILED OBJECT DETECTION
# ============================================================


def get_latest_object_event(
    object_name: str,
) -> dict[str, Any] | None:

    return fetch_one(
        """
        SELECT *
        FROM object_events
        WHERE object_name = %s
        ORDER BY detected_at DESC
        LIMIT 1
        """,
        (
            normalize_object_name(
                object_name
            ),
        ),
    )


# ============================================================
# EVENTS BETWEEN TWO TIMES
# ============================================================


def get_events_between(
    start: datetime,
    end: datetime,
    object_name: str | None = None,
) -> list[dict[str, Any]]:

    if object_name:

        return fetch_all(
            """
            SELECT *
            FROM events
            WHERE object_name = %s
              AND detected_at BETWEEN %s AND %s
            ORDER BY detected_at DESC
            """,
            (
                normalize_object_name(
                    object_name
                ),
                start,
                end,
            ),
        )

    return fetch_all(
        """
        SELECT *
        FROM events
        WHERE detected_at BETWEEN %s AND %s
        ORDER BY detected_at DESC
        """,
        (
            start,
            end,
        ),
    )


# ============================================================
# EVENTS FOR PARTICULAR DATE
# ============================================================


def get_events_for_date(
    target_date: date,
    object_name: str | None = None,
) -> list[dict[str, Any]]:

    start = datetime.combine(
        target_date,
        time.min,
    )

    end = datetime.combine(
        target_date,
        time.max,
    )

    return get_events_between(
        start,
        end,
        object_name,
    )


# ============================================================
# FLEXIBLE MEMORY SEARCH
# ============================================================


def search_events(
    start: datetime | None = None,
    end: datetime | None = None,
    search_text: str | None = None,
    event_type: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """
    Flexible search for the Memory Timeline.

    Filtering happens directly in MySQL BEFORE
    the result limit is applied.

    This solves the previous problem where:

        latest 25 records
              ↓
        Python searched those 25
              ↓
        older book records disappeared

    Now:

        MySQL searches for book first
              ↓
        matching records
              ↓
        limit is applied

    Search checks:

        object_name
        location
        memory_text

    event_type is included for future memories:

        object_detected
        person_detected
        medicine_taken
        medicine_reminder
        conversation
    """

    safe_limit = max(
        1,
        min(
            int(limit),
            10000,
        ),
    )

    conditions: list[str] = []

    parameters: list[Any] = []

    # --------------------------------------------------------
    # START DATE/TIME
    # --------------------------------------------------------

    if start is not None:

        conditions.append(
            "detected_at >= %s"
        )

        parameters.append(
            start
        )

    # --------------------------------------------------------
    # END DATE/TIME
    # --------------------------------------------------------

    if end is not None:

        conditions.append(
            "detected_at <= %s"
        )

        parameters.append(
            end
        )

    # --------------------------------------------------------
    # SEARCH TEXT
    # --------------------------------------------------------

    cleaned_search = (
        str(
            search_text
            or ""
        )
        .strip()
        .lower()
    )

    if cleaned_search:

        normalized_search = (
            normalize_object_name(
                cleaned_search
            )
        )

        normalized_pattern = (
            "%"
            + str(
                normalized_search
            ).lower()
            + "%"
        )

        original_pattern = (
            "%"
            + cleaned_search
            + "%"
        )

        # REPLACE handles either:
        #
        # cell_phone
        #
        # or:
        #
        # cell phone

        conditions.append(
            """
            (
                REPLACE(
                    LOWER(
                        COALESCE(
                            object_name,
                            ''
                        )
                    ),
                    '_',
                    ' '
                ) LIKE %s

                OR

                LOWER(
                    COALESCE(
                        object_name,
                        ''
                    )
                ) LIKE %s

                OR

                LOWER(
                    COALESCE(
                        location,
                        ''
                    )
                ) LIKE %s

                OR

                LOWER(
                    COALESCE(
                        memory_text,
                        ''
                    )
                ) LIKE %s
            )
            """
        )

        parameters.extend(
            [
                normalized_pattern,
                original_pattern,
                original_pattern,
                original_pattern,
            ]
        )

    # --------------------------------------------------------
    # EVENT TYPE
    # --------------------------------------------------------

    cleaned_event_type = (
        str(
            event_type
            or ""
        )
        .strip()
    )

    if cleaned_event_type:

        conditions.append(
            "event_type = %s"
        )

        parameters.append(
            cleaned_event_type
        )

    # --------------------------------------------------------
    # WHERE
    # --------------------------------------------------------

    if conditions:

        where_clause = (
            "WHERE "
            + " AND ".join(
                conditions
            )
        )

    else:

        where_clause = ""

    # --------------------------------------------------------
    # QUERY
    # --------------------------------------------------------

    query = f"""
        SELECT *
        FROM events
        {where_clause}
        ORDER BY detected_at DESC
        LIMIT {safe_limit}
    """

    return fetch_all(
        query,
        tuple(
            parameters
        ),
    )


# ============================================================
# COUNT FLEXIBLE MEMORY SEARCH
# ============================================================


def count_events(
    start: datetime | None = None,
    end: datetime | None = None,
    search_text: str | None = None,
    event_type: str | None = None,
) -> int:
    """
    Count all matching records without applying
    the table display limit.
    """

    conditions: list[str] = []

    parameters: list[Any] = []

    # --------------------------------------------------------
    # START
    # --------------------------------------------------------

    if start is not None:

        conditions.append(
            "detected_at >= %s"
        )

        parameters.append(
            start
        )

    # --------------------------------------------------------
    # END
    # --------------------------------------------------------

    if end is not None:

        conditions.append(
            "detected_at <= %s"
        )

        parameters.append(
            end
        )

    # --------------------------------------------------------
    # SEARCH
    # --------------------------------------------------------

    cleaned_search = (
        str(
            search_text
            or ""
        )
        .strip()
        .lower()
    )

    if cleaned_search:

        normalized_search = (
            normalize_object_name(
                cleaned_search
            )
        )

        normalized_pattern = (
            "%"
            + str(
                normalized_search
            ).lower()
            + "%"
        )

        original_pattern = (
            "%"
            + cleaned_search
            + "%"
        )

        conditions.append(
            """
            (
                REPLACE(
                    LOWER(
                        COALESCE(
                            object_name,
                            ''
                        )
                    ),
                    '_',
                    ' '
                ) LIKE %s

                OR

                LOWER(
                    COALESCE(
                        object_name,
                        ''
                    )
                ) LIKE %s

                OR

                LOWER(
                    COALESCE(
                        location,
                        ''
                    )
                ) LIKE %s

                OR

                LOWER(
                    COALESCE(
                        memory_text,
                        ''
                    )
                ) LIKE %s
            )
            """
        )

        parameters.extend(
            [
                normalized_pattern,
                original_pattern,
                original_pattern,
                original_pattern,
            ]
        )

    # --------------------------------------------------------
    # EVENT TYPE
    # --------------------------------------------------------

    cleaned_event_type = (
        str(
            event_type
            or ""
        )
        .strip()
    )

    if cleaned_event_type:

        conditions.append(
            "event_type = %s"
        )

        parameters.append(
            cleaned_event_type
        )

    # --------------------------------------------------------
    # WHERE
    # --------------------------------------------------------

    if conditions:

        where_clause = (
            "WHERE "
            + " AND ".join(
                conditions
            )
        )

    else:

        where_clause = ""

    row = fetch_one(
        f"""
        SELECT COUNT(*) AS total
        FROM events
        {where_clause}
        """,
        tuple(
            parameters
        ),
    )

    if not row:

        return 0

    return int(
        row.get(
            "total",
            0,
        )
        or 0
    )


# ============================================================
# DELETE OLD MEMORY EVENTS
# ============================================================


def delete_old_events(
    before: datetime,
) -> int:

    return execute(
        """
        DELETE FROM events
        WHERE detected_at < %s
        """,
        (
            before,
        ),
    )