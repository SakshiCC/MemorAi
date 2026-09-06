from __future__ import annotations

from dataclasses import (
    dataclass,
)

from datetime import (
    datetime,
)

from typing import (
    Any,
)

from config import (
    DEFAULT_TOP_K,
    MAX_DISTANCE,
)

from database.medicine_vector_memory import (
    MedicineVectorMemory,
)


# ============================================================
# RESULT
# ============================================================

@dataclass
class MedicineMemorySearchResult:
    """
    One semantic medicine-memory result returned from
    ChromaDB.
    """

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
    # MEDICINE EVENT ID
    # ========================================================

    @property
    def medicine_event_id(
        self,
    ) -> int | None:

        value = (
            self.metadata.get(
                "medicine_event_id"
            )
        )

        if value is None:

            return None

        try:

            return int(
                value
            )

        except (
            TypeError,
            ValueError,
        ):

            return None

    # ========================================================
    # MEDICINE ID
    # ========================================================

    @property
    def medicine_id(
        self,
    ) -> int | None:

        value = (
            self.metadata.get(
                "medicine_id"
            )
        )

        if value is None:

            return None

        try:

            return int(
                value
            )

        except (
            TypeError,
            ValueError,
        ):

            return None

    # ========================================================
    # MEDICINE NAME
    # ========================================================

    @property
    def medicine_name(
        self,
    ) -> str | None:

        value = (
            self.metadata.get(
                "medicine_name"
            )
        )

        if value is None:

            return None

        cleaned = (
            str(
                value
            )
            .strip()
        )

        return (
            cleaned
            if cleaned
            else None
        )

    # ========================================================
    # DOSAGE
    # ========================================================

    @property
    def dosage(
        self,
    ) -> str | None:

        value = (
            self.metadata.get(
                "dosage"
            )
        )

        if value is None:

            return None

        cleaned = (
            str(
                value
            )
            .strip()
        )

        return (
            cleaned
            if cleaned
            else None
        )

    # ========================================================
    # STATUS
    # ========================================================

    @property
    def status(
        self,
    ) -> str | None:

        value = (
            self.metadata.get(
                "status"
            )
        )

        if value is None:

            return None

        cleaned = (
            str(
                value
            )
            .strip()
        )

        return (
            cleaned
            if cleaned
            else None
        )

    # ========================================================
    # OBSERVED AT
    # ========================================================

    @property
    def observed_at(
        self,
    ) -> str | None:

        value = (
            self.metadata.get(
                "observed_at"
            )
        )

        if value is None:

            return None

        return str(
            value
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

        if value is None:

            return None

        cleaned = (
            str(
                value
            )
            .strip()
        )

        return (
            cleaned
            if cleaned
            else None
        )


# ============================================================
# MEDICINE MEMORY SEARCH
# ============================================================

class MedicineMemorySearch:
    """
    Semantic search wrapper for medicine memories.

    MySQL remains authoritative.

    This class is only responsible for searching medicine
    memories already indexed into MedicineVectorMemory.

    Typical semantic questions:

        What medicine was confirmed recently?

        Which medication did MemorAi observe?

        Was there a medicine event this morning?

        What medication event happened after breakfast?

        What medicine did the camera recognize recently?
    """

    # ========================================================
    # INITIALIZE
    # ========================================================

    def __init__(
        self,
    ) -> None:

        self.vector_memory = (
            MedicineVectorMemory()
        )

    # ========================================================
    # SEARCH
    # ========================================================

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        medicine_id: int | None = None,
        status: str | None = None,
    ) -> list[
        MedicineMemorySearchResult
    ]:

        cleaned_query = (
            str(
                query
                or ""
            )
            .strip()
        )

        if not cleaned_query:

            return []

        raw = (
            self.vector_memory
            .search(
                query=(
                    cleaned_query
                ),

                n_results=(
                    top_k
                ),

                medicine_id=(
                    medicine_id
                ),

                status=(
                    status
                ),
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
            MedicineMemorySearchResult
        ] = []

        for (
            index,
            memory_id,
        ) in enumerate(
            ids
        ):

            # =================================================
            # DISTANCE
            # =================================================

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

                    and

                    distances[
                        index
                    ]
                    is not None
                )

                else None
            )

            # =================================================
            # DISTANCE FILTER
            # =================================================

            if (
                distance
                is not None

                and

                distance
                > MAX_DISTANCE
            ):

                continue

            # =================================================
            # METADATA
            # =================================================

            metadata = (
                metadatas[
                    index
                ]

                if (
                    index
                    < len(
                        metadatas
                    )

                    and

                    metadatas[
                        index
                    ]
                )

                else {}
            )

            # =================================================
            # DOCUMENT
            # =================================================

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
                MedicineMemorySearchResult(
                    memory_id=(
                        str(
                            memory_id
                        )
                    ),

                    text=(
                        text
                    ),

                    distance=(
                        distance
                    ),

                    metadata=dict(
                        metadata
                    ),
                )
            )

        # ====================================================
        # PREFER NEWER
        # ====================================================

        return (
            self._prefer_newer(
                results
            )
        )

    # ========================================================
    # DATETIME HELPER
    # ========================================================

    @staticmethod
    def _parse_datetime(
        value: str | None,
    ) -> datetime:

        if not value:

            return (
                datetime.min
            )

        try:

            normalized = (
                str(
                    value
                )
                .replace(
                    "Z",
                    "+00:00",
                )
            )

            parsed = (
                datetime.fromisoformat(
                    normalized
                )
            )

            if (
                parsed.tzinfo
                is not None
            ):

                parsed = (
                    parsed.replace(
                        tzinfo=None
                    )
                )

            return (
                parsed
            )

        except (
            TypeError,
            ValueError,
        ):

            return (
                datetime.min
            )

    # ========================================================
    # PREFER NEWER
    # ========================================================

    @classmethod
    def _prefer_newer(
        cls,
        memories: list[
            MedicineMemorySearchResult
        ],
    ) -> list[
        MedicineMemorySearchResult
    ]:
        """
        Chroma provides semantic relevance.

        If multiple relevant medicine memories are returned,
        prefer newer observations for the same general level
        of semantic relevance.

        This matches the behavior already used by the object
        and person memory search layers.
        """

        return sorted(
            memories,

            key=lambda memory: (
                cls._parse_datetime(
                    memory.observed_at
                )
            ),

            reverse=True,
        )