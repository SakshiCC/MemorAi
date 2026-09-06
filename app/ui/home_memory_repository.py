from __future__ import annotations

from typing import Any

from database.event_repository import (
    get_recent_events,
)


class HomeMemoryRepository:
    """
    Lightweight repository for the Home-page
    Recent Memories section.

    Reads directly from the existing MySQL events table.
    """

    def get_recent_memories(
        self,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Return newest high-level memory events.
        """

        safe_limit = max(
            1,
            min(
                int(limit),
                20,
            ),
        )

        return get_recent_events(
            limit=safe_limit
        )