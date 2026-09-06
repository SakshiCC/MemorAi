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


# ============================================================
# VALID STATUS VALUES
# ============================================================

VALID_CONVERSATION_STATUSES = {
    "recording",
    "processing",
    "completed",
    "cancelled",
    "failed",
}


# ============================================================
# HELPERS
# ============================================================

def _clean_optional_text(
    value: str | None,
) -> str | None:

    if value is None:
        return None

    cleaned = str(value).strip()

    return cleaned if cleaned else None


def _validate_status(
    status: str,
) -> str:

    cleaned = str(status).strip().lower()

    if cleaned not in VALID_CONVERSATION_STATUSES:
        raise ValueError(
            "Invalid conversation status: "
            f"{status}"
        )

    return cleaned


# ============================================================
# CREATE CONVERSATION SESSION
# ============================================================

def create_conversation_session(
    started_at: datetime,
    title: str | None = None,
    participant_text: str | None = None,
    status: str = "recording",
) -> int:
    """
    Create a new conversation session.
    """

    if not isinstance(
        started_at,
        datetime,
    ):
        raise TypeError(
            "started_at must be a datetime."
        )

    clean_status = _validate_status(
        status
    )

    query = """
        INSERT INTO conversation_sessions (
            title,
            participant_text,
            status,
            started_at
        )
        VALUES (
            %s,
            %s,
            %s,
            %s
        )
    """

    return execute(
        query,
        (
            _clean_optional_text(
                title
            ),
            _clean_optional_text(
                participant_text
            ),
            clean_status,
            started_at,
        ),
    )


# ============================================================
# ADD CONVERSATION SEGMENT
# ============================================================

def add_conversation_segment(
    conversation_id: int,
    sequence_no: int,
    transcript: str,
    started_at: datetime | None = None,
    ended_at: datetime | None = None,
) -> int:
    """
    Store one transcript segment belonging to a conversation.
    """

    if int(conversation_id) <= 0:
        raise ValueError(
            "conversation_id must be positive."
        )

    if int(sequence_no) <= 0:
        raise ValueError(
            "sequence_no must be positive."
        )

    cleaned_transcript = (
        str(transcript).strip()
    )

    if not cleaned_transcript:
        raise ValueError(
            "Conversation transcript cannot be empty."
        )

    query = """
        INSERT INTO conversation_segments (
            conversation_id,
            sequence_no,
            transcript,
            started_at,
            ended_at
        )
        VALUES (
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
            int(conversation_id),
            int(sequence_no),
            cleaned_transcript,
            started_at,
            ended_at,
        ),
    )


# ============================================================
# GET CONVERSATION SESSION
# ============================================================

def get_conversation_session(
    conversation_id: int,
) -> dict[str, Any] | None:

    return fetch_one(
        """
        SELECT *
        FROM conversation_sessions
        WHERE id = %s
        """,
        (
            int(conversation_id),
        ),
    )


# ============================================================
# GET CONVERSATION SEGMENTS
# ============================================================

def get_conversation_segments(
    conversation_id: int,
) -> list[dict[str, Any]]:

    return fetch_all(
        """
        SELECT *
        FROM conversation_segments
        WHERE conversation_id = %s
        ORDER BY sequence_no ASC
        """,
        (
            int(conversation_id),
        ),
    )


# ============================================================
# GET NEXT SEGMENT NUMBER
# ============================================================

def get_next_segment_number(
    conversation_id: int,
) -> int:

    row = fetch_one(
        """
        SELECT
            COALESCE(
                MAX(sequence_no),
                0
            ) AS last_sequence
        FROM conversation_segments
        WHERE conversation_id = %s
        """,
        (
            int(conversation_id),
        ),
    )

    if not row:
        return 1

    return (
        int(
            row.get(
                "last_sequence",
                0,
            )
            or 0
        )
        + 1
    )


# ============================================================
# BUILD FULL TRANSCRIPT
# ============================================================

def build_full_transcript(
    conversation_id: int,
) -> str:
    """
    Rebuild the complete transcript from all stored segments.
    """

    segments = get_conversation_segments(
        conversation_id
    )

    parts: list[str] = []

    for segment in segments:

        transcript = str(
            segment.get(
                "transcript"
            )
            or ""
        ).strip()

        if transcript:
            parts.append(
                transcript
            )

    return " ".join(
        parts
    ).strip()


# ============================================================
# UPDATE STATUS
# ============================================================

def update_conversation_status(
    conversation_id: int,
    status: str,
) -> None:

    clean_status = _validate_status(
        status
    )

    execute(
        """
        UPDATE conversation_sessions
        SET status = %s
        WHERE id = %s
        """,
        (
            clean_status,
            int(conversation_id),
        ),
    )


# ============================================================
# UPDATE CONVERSATION DETAILS
# ============================================================

def update_conversation_details(
    conversation_id: int,
    title: str | None = None,
    participant_text: str | None = None,
    summary: str | None = None,
) -> None:

    execute(
        """
        UPDATE conversation_sessions
        SET
            title = %s,
            participant_text = %s,
            summary = %s
        WHERE id = %s
        """,
        (
            _clean_optional_text(
                title
            ),
            _clean_optional_text(
                participant_text
            ),
            _clean_optional_text(
                summary
            ),
            int(conversation_id),
        ),
    )


# ============================================================
# FINALIZE CONVERSATION
# ============================================================

def finalize_conversation(
    conversation_id: int,
    ended_at: datetime,
    full_transcript: str | None = None,
    summary: str | None = None,
    title: str | None = None,
    participant_text: str | None = None,
) -> None:
    """
    Complete a conversation and preserve the original
    transcript separately from its summary.
    """

    if not isinstance(
        ended_at,
        datetime,
    ):
        raise TypeError(
            "ended_at must be a datetime."
        )

    session = get_conversation_session(
        conversation_id
    )

    if not session:
        raise ValueError(
            "Conversation session "
            f"{conversation_id} does not exist."
        )

    transcript = _clean_optional_text(
        full_transcript
    )

    if transcript is None:
        transcript = build_full_transcript(
            conversation_id
        )

    if not transcript:
        raise ValueError(
            "Cannot finalize a conversation "
            "without a transcript."
        )

    started_at = session.get(
        "started_at"
    )

    if (
        isinstance(
            started_at,
            datetime,
        )
        and ended_at < started_at
    ):
        raise ValueError(
            "Conversation ended_at cannot "
            "be earlier than started_at."
        )

    final_title = _clean_optional_text(
        title
    )

    if final_title is None:
        final_title = _clean_optional_text(
            session.get(
                "title"
            )
        )

    final_participants = (
        _clean_optional_text(
            participant_text
        )
    )

    if final_participants is None:
        final_participants = (
            _clean_optional_text(
                session.get(
                    "participant_text"
                )
            )
        )

    execute(
        """
        UPDATE conversation_sessions
        SET
            title = %s,
            participant_text = %s,
            full_transcript = %s,
            summary = %s,
            status = 'completed',
            ended_at = %s
        WHERE id = %s
        """,
        (
            final_title,
            final_participants,
            transcript,
            _clean_optional_text(
                summary
            ),
            ended_at,
            int(conversation_id),
        ),
    )


# ============================================================
# CANCEL CONVERSATION
# ============================================================

def cancel_conversation(
    conversation_id: int,
    ended_at: datetime | None = None,
) -> None:

    final_ended_at = (
        ended_at
        or datetime.now()
    )

    execute(
        """
        UPDATE conversation_sessions
        SET
            status = 'cancelled',
            ended_at = %s
        WHERE id = %s
        """,
        (
            final_ended_at,
            int(conversation_id),
        ),
    )


# ============================================================
# RECENT CONVERSATIONS
# ============================================================

def get_recent_conversations(
    limit: int = 20,
    completed_only: bool = False,
) -> list[dict[str, Any]]:

    safe_limit = max(
        1,
        min(
            int(limit),
            1000,
        ),
    )

    if completed_only:

        return fetch_all(
            f"""
            SELECT *
            FROM conversation_sessions
            WHERE status = 'completed'
            ORDER BY started_at DESC
            LIMIT {safe_limit}
            """
        )

    return fetch_all(
        f"""
        SELECT *
        FROM conversation_sessions
        ORDER BY started_at DESC
        LIMIT {safe_limit}
        """
    )


# ============================================================
# CONVERSATIONS BETWEEN DATES
# ============================================================

def get_conversations_between(
    start: datetime,
    end: datetime,
    completed_only: bool = False,
) -> list[dict[str, Any]]:

    if completed_only:

        return fetch_all(
            """
            SELECT *
            FROM conversation_sessions
            WHERE started_at BETWEEN %s AND %s
              AND status = 'completed'
            ORDER BY started_at DESC
            """,
            (
                start,
                end,
            ),
        )

    return fetch_all(
        """
        SELECT *
        FROM conversation_sessions
        WHERE started_at BETWEEN %s AND %s
        ORDER BY started_at DESC
        """,
        (
            start,
            end,
        ),
    )


# ============================================================
# CONVERSATIONS FOR DATE
# ============================================================

def get_conversations_for_date(
    target_date: date,
    completed_only: bool = False,
) -> list[dict[str, Any]]:

    start = datetime.combine(
        target_date,
        time.min,
    )

    end = datetime.combine(
        target_date,
        time.max,
    )

    return get_conversations_between(
        start=start,
        end=end,
        completed_only=completed_only,
    )


# ============================================================
# SEARCH CONVERSATIONS
# ============================================================

def search_conversations(
    search_text: str,
    limit: int = 50,
) -> list[dict[str, Any]]:

    cleaned = str(
        search_text
    ).strip().lower()

    if not cleaned:
        return []

    safe_limit = max(
        1,
        min(
            int(limit),
            1000,
        ),
    )

    pattern = (
        "%"
        + cleaned
        + "%"
    )

    return fetch_all(
        f"""
        SELECT *
        FROM conversation_sessions

        WHERE
            LOWER(
                COALESCE(
                    title,
                    ''
                )
            ) LIKE %s

            OR LOWER(
                COALESCE(
                    participant_text,
                    ''
                )
            ) LIKE %s

            OR LOWER(
                COALESCE(
                    full_transcript,
                    ''
                )
            ) LIKE %s

            OR LOWER(
                COALESCE(
                    summary,
                    ''
                )
            ) LIKE %s

        ORDER BY started_at DESC

        LIMIT {safe_limit}
        """,
        (
            pattern,
            pattern,
            pattern,
            pattern,
        ),
    )


# ============================================================
# DELETE CONVERSATION
# ============================================================

def delete_conversation(
    conversation_id: int,
) -> None:
    """
    Delete a conversation.

    Its segments are automatically removed through
    the ON DELETE CASCADE foreign key.
    """

    execute(
        """
        DELETE FROM conversation_sessions
        WHERE id = %s
        """,
        (
            int(conversation_id),
        ),
    )