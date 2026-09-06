from __future__ import annotations

from typing import Any

from database.db_manager import (
    execute,
    fetch_one,
)


# ============================================================
# DEFAULTS
# ============================================================

DEFAULT_IMPORTANCE_THRESHOLD = 0.60

MIN_IMPORTANCE_THRESHOLD = 0.30

MAX_IMPORTANCE_THRESHOLD = 0.90


# ============================================================
# GET SETTINGS
# ============================================================

def get_conversation_settings() -> dict[str, Any]:
    """
    Return the persistent Conversation Memory privacy settings.

    Conversation Memory must always default to disabled if the
    row cannot be found for any reason.
    """

    row = fetch_one(
        """
        SELECT
            memory_enabled,
            importance_threshold,
            updated_at
        FROM conversation_settings
        WHERE id = 1
        """
    )

    if not row:

        return {
            "memory_enabled": False,
            "importance_threshold": (
                DEFAULT_IMPORTANCE_THRESHOLD
            ),
            "updated_at": None,
        }

    return {
        "memory_enabled": bool(
            row.get(
                "memory_enabled",
                False,
            )
        ),

        "importance_threshold": float(
            row.get(
                "importance_threshold",
                DEFAULT_IMPORTANCE_THRESHOLD,
            )
            or DEFAULT_IMPORTANCE_THRESHOLD
        ),

        "updated_at": row.get(
            "updated_at"
        ),
    }


# ============================================================
# CHECK ENABLED
# ============================================================

def is_conversation_memory_enabled() -> bool:
    """
    Return True only when the user has explicitly enabled
    Conversation Memory.
    """

    settings = (
        get_conversation_settings()
    )

    return bool(
        settings[
            "memory_enabled"
        ]
    )


# ============================================================
# ENABLE / DISABLE
# ============================================================

def set_conversation_memory_enabled(
    enabled: bool,
) -> None:
    """
    Persist the user's Conversation Memory permission.
    """

    execute(
        """
        UPDATE conversation_settings
        SET memory_enabled = %s
        WHERE id = 1
        """,
        (
            bool(
                enabled
            ),
        ),
    )


# ============================================================
# IMPORTANCE THRESHOLD
# ============================================================

def get_importance_threshold() -> float:

    settings = (
        get_conversation_settings()
    )

    return float(
        settings[
            "importance_threshold"
        ]
    )


def set_importance_threshold(
    threshold: float,
) -> None:
    """
    Set how selective MemorAi should be.

    Lower threshold:
        remembers more conversations

    Higher threshold:
        remembers only stronger / clearer memories
    """

    value = float(
        threshold
    )

    if not (
        MIN_IMPORTANCE_THRESHOLD
        <= value
        <= MAX_IMPORTANCE_THRESHOLD
    ):

        raise ValueError(
            (
                "Conversation importance threshold "
                f"must be between "
                f"{MIN_IMPORTANCE_THRESHOLD:.2f} "
                f"and "
                f"{MAX_IMPORTANCE_THRESHOLD:.2f}."
            )
        )

    execute(
        """
        UPDATE conversation_settings
        SET importance_threshold = %s
        WHERE id = 1
        """,
        (
            value,
        ),
    )


# ============================================================
# PRIVACY GUARD
# ============================================================

def require_conversation_memory_permission() -> None:
    """
    Backend privacy guard.

    Any recording/processing entry point should call this
    before accepting conversation audio.
    """

    if (
        not is_conversation_memory_enabled()
    ):

        raise PermissionError(
            (
                "Conversation Memory is disabled. "
                "Enable it in Settings before starting "
                "a conversation recording."
            )
        )