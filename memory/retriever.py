from __future__ import annotations

import re
from datetime import date, timedelta
from typing import Any

from database.event_repository import (
    get_events_for_date,
    get_latest_event_for_object,
    get_recent_events,
)

from rag.memory_search import (
    MemorySearch,
    MemorySearchResult,
)

from utils.helpers import (
    normalize_object_name,
)


class MemoryRetriever:
    """
    Hybrid memory retriever.

    MySQL is used for:
    - latest object lookup
    - today
    - yesterday
    - recent events
    - exact time/date queries

    ChromaDB is used for:
    - semantic questions
    - less structured natural-language queries
    """

    def __init__(self) -> None:

        self.semantic_search = MemorySearch()

    # =========================================================
    # EXTRACT OBJECT NAME
    # =========================================================

    @staticmethod
    def _extract_known_object(
        question: str,
    ) -> str | None:

        normalized = question.lower()

        aliases = {

            "cell phone": [
                "cell phone",
                "phone",
                "mobile",
                "mobile phone",
            ],

            "bottle": [
                "bottle",
            ],

            "cup": [
                "cup",
                "mug",
            ],

            "book": [
                "book",
            ],

            "remote": [
                "remote",
                "remote control",
            ],

            "handbag": [
                "handbag",
                "purse",
            ],

            "backpack": [
                "backpack",
                "bag",
            ],

            "glasses": [
                "glasses",
                "spectacles",
            ],

            "keys": [
                "keys",
                "key",
            ],

            "scissors": [
                "scissors",
            ],
        }

        for canonical_name, words in aliases.items():

            for word in words:

                if re.search(
                    rf"\b{re.escape(word)}\b",
                    normalized,
                ):

                    return normalize_object_name(
                        canonical_name
                    )

        return None

    # =========================================================
    # MAIN RETRIEVAL
    # =========================================================

    def retrieve(
        self,
        question: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:

        cleaned = question.strip()

        if not cleaned:

            return []

        lowered = cleaned.lower()

        object_name = self._extract_known_object(
            cleaned
        )

        # =====================================================
        # TODAY
        # =====================================================

        if "today" in lowered:

            target_date = date.today()

            return get_events_for_date(
                target_date,
                object_name,
            )

        # =====================================================
        # YESTERDAY
        # =====================================================

        if "yesterday" in lowered:

            target_date = (
                date.today()
                - timedelta(days=1)
            )

            return get_events_for_date(
                target_date,
                object_name,
            )

        # =====================================================
        # RECENT / RECENTLY
        # =====================================================

        recent_phrases = (
            "recent",
            "recently",
            "latest objects",
            "recent objects",
            "what objects were detected",
            "what was detected",
            "what happened recently",
        )

        if any(
            phrase in lowered
            for phrase in recent_phrases
        ):

            events = get_recent_events(
                limit=max(
                    top_k,
                    10,
                )
            )

            if object_name:

                return [
                    event
                    for event in events
                    if event.get(
                        "object_name"
                    )
                    == object_name
                ]

            return events

        # =====================================================
        # WHERE / LAST / LATEST
        # =====================================================

        latest_phrases = (
            "where is",
            "where are",
            "where was",
            "where were",
            "where did i keep",
            "where did i put",
            "where did i leave",
            "last",
            "latest",
            "most recent",
        )

        if (
            object_name
            and any(
                phrase in lowered
                for phrase in latest_phrases
            )
        ):

            event = (
                get_latest_event_for_object(
                    object_name
                )
            )

            return (
                [event]
                if event
                else []
            )

        # =====================================================
        # WHEN QUESTIONS
        # =====================================================

        time_phrases = (
            "when did",
            "what time",
            "when was",
            "when were",
        )

        if (
            object_name
            and any(
                phrase in lowered
                for phrase in time_phrases
            )
        ):

            event = (
                get_latest_event_for_object(
                    object_name
                )
            )

            return (
                [event]
                if event
                else []
            )

        # =====================================================
        # SEMANTIC SEARCH
        # =====================================================

        memories = (
            self.semantic_search.search(
                query=cleaned,
                top_k=top_k,
                object_name=object_name,
            )
        )

        return [
            self._semantic_to_dict(
                memory
            )
            for memory in memories
        ]

    # =========================================================
    # CONVERT SEMANTIC MEMORY OBJECT
    # =========================================================

    @staticmethod
    def _semantic_to_dict(
        memory: MemorySearchResult,
    ) -> dict[str, Any]:

        result = dict(
            memory.metadata
        )

        result[
            "memory_id"
        ] = memory.memory_id

        result[
            "memory_text"
        ] = memory.text

        result[
            "distance"
        ] = memory.distance

        return result