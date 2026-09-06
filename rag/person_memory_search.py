from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from config import (
    DEFAULT_TOP_K,
    MAX_DISTANCE,
)

from database.person_vector_memory import (
    PersonVectorMemory,
)


# ============================================================
# RESULT
# ============================================================

@dataclass
class PersonMemorySearchResult:

    memory_id: str

    text: str

    distance: (
        float
        | None
    )

    metadata: dict[
        str,
        Any,
    ]

    # ========================================================
    # DETECTED AT
    # ========================================================

    @property
    def detected_at(
        self,
    ) -> str | None:

        value = (
            self.metadata.get(
                "detected_at"
            )
        )

        return (
            str(
                value
            )
            if value
            else None
        )

    # ========================================================
    # PERSON NAME
    # ========================================================

    @property
    def person_name(
        self,
    ) -> str | None:

        value = (
            self.metadata.get(
                "person_name"
            )
        )

        return (
            str(
                value
            )
            if value
            else None
        )

    # ========================================================
    # RELATION
    # ========================================================

    @property
    def relation(
        self,
    ) -> str | None:

        value = (
            self.metadata.get(
                "relation"
            )
        )

        return (
            str(
                value
            )
            if value
            else None
        )


# ============================================================
# SEARCH SERVICE
# ============================================================

class PersonMemorySearch:

    def __init__(
        self,
    ) -> None:

        self.vector_memory = (
            PersonVectorMemory()
        )

    # ========================================================
    # SEARCH
    # ========================================================

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        person_id: int | None = None,
        relation: str | None = None,
    ) -> list[
        PersonMemorySearchResult
    ]:

        where: (
            dict[
                str,
                Any,
            ]
            | None
        ) = None

        # ----------------------------------------------------
        # PERSON FILTER
        # ----------------------------------------------------

        if person_id is not None:

            where = {
                "person_id": int(
                    person_id
                )
            }

        # ----------------------------------------------------
        # RELATION FILTER
        # ----------------------------------------------------

        elif relation:

            where = {
                "relation": str(
                    relation
                )
            }

        raw = (
            self.vector_memory
            .search(
                query=query,
                n_results=top_k,
                where=where,
            )
        )

        ids = (
            raw.get(
                "ids",
                [[]],
            )[0]
        )

        documents = (
            raw.get(
                "documents",
                [[]],
            )[0]
        )

        metadatas = (
            raw.get(
                "metadatas",
                [[]],
            )[0]
        )

        distances = (
            raw.get(
                "distances",
                [[]],
            )[0]
        )

        results: list[
            PersonMemorySearchResult
        ] = []

        for (
            index,
            memory_id,
        ) in enumerate(
            ids
        ):

            distance = (
                float(
                    distances[
                        index
                    ]
                )
                if (
                    index
                    < len(
                        distances
                    )
                    and distances[
                        index
                    ]
                    is not None
                )
                else None
            )

            if (
                distance
                is not None
                and distance
                > MAX_DISTANCE
            ):

                continue

            metadata = (
                metadatas[
                    index
                ]
                if (
                    index
                    < len(
                        metadatas
                    )
                    and metadatas[
                        index
                    ]
                )
                else {}
            )

            text = (
                str(
                    documents[
                        index
                    ]
                )
                if (
                    index
                    < len(
                        documents
                    )
                )
                else ""
            )

            results.append(
                PersonMemorySearchResult(
                    memory_id=str(
                        memory_id
                    ),
                    text=text,
                    distance=(
                        distance
                    ),
                    metadata=dict(
                        metadata
                    ),
                )
            )

        return (
            self._prefer_newer(
                results
            )
        )

    # ========================================================
    # PREFER NEWER
    # ========================================================

    @staticmethod
    def _prefer_newer(
        memories: list[
            PersonMemorySearchResult
        ],
    ) -> list[
        PersonMemorySearchResult
    ]:

        def timestamp(
            memory: PersonMemorySearchResult,
        ) -> datetime:

            raw = (
                memory.detected_at
            )

            if not raw:

                return (
                    datetime.min
                )

            try:

                normalized = (
                    raw.replace(
                        "Z",
                        "+00:00",
                    )
                )

                parsed = (
                    datetime.fromisoformat(
                        normalized
                    )
                )

                return (
                    parsed.replace(
                        tzinfo=None
                    )
                )

            except ValueError:

                return (
                    datetime.min
                )

        return sorted(
            memories,
            key=timestamp,
            reverse=True,
        )