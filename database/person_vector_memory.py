from __future__ import annotations

from datetime import datetime
from typing import Any

import chromadb

from config import (
    CHROMA_PATH,
)

from rag.embeddings import (
    create_embedding,
)

from utils.logger import (
    get_logger,
)


# ============================================================
# LOGGER
# ============================================================

logger = get_logger(
    "database.person_vector_memory",
    "database.log",
)


# ============================================================
# COLLECTION
# ============================================================

PERSON_CHROMA_COLLECTION_NAME = (
    "memorai_person_memories"
)


# ============================================================
# PERSON VECTOR MEMORY
# ============================================================

class PersonVectorMemory:
    """
    ChromaDB storage for person encounter memories.

    MySQL remains authoritative for exact factual history.

    This collection supports semantic retrieval and provides
    simple helpers for verifying indexed person events.
    """

    def __init__(
        self,
    ) -> None:

        self.client = (
            chromadb.PersistentClient(
                path=str(
                    CHROMA_PATH
                )
            )
        )

        self.collection = (
            self.client
            .get_or_create_collection(
                name=(
                    PERSON_CHROMA_COLLECTION_NAME
                ),
                metadata={
                    "hnsw:space": "cosine",
                },
            )
        )

    # ========================================================
    # CLEAN METADATA
    # ========================================================

    @staticmethod
    def _clean_metadata(
        metadata: dict[str, Any],
    ) -> dict[str, Any]:

        cleaned: dict[
            str,
            Any,
        ] = {}

        for (
            key,
            value,
        ) in metadata.items():

            if value is None:

                continue

            if isinstance(
                value,
                datetime,
            ):

                cleaned[
                    key
                ] = value.isoformat(
                    timespec="seconds"
                )

            elif isinstance(
                value,
                (
                    str,
                    int,
                    float,
                    bool,
                ),
            ):

                cleaned[
                    key
                ] = value

            else:

                cleaned[
                    key
                ] = str(
                    value
                )

        return cleaned

    # ========================================================
    # ADD MEMORY
    # ========================================================

    def add_memory(
        self,
        memory_id: str,
        text: str,
        metadata: dict[str, Any],
    ) -> None:

        cleaned_text = (
            text.strip()
        )

        if not cleaned_text:

            raise ValueError(
                "Person memory text "
                "cannot be empty."
            )

        vector = (
            create_embedding(
                cleaned_text
            )
        )

        self.collection.upsert(
            ids=[
                str(
                    memory_id
                )
            ],
            documents=[
                cleaned_text
            ],
            embeddings=[
                vector
            ],
            metadatas=[
                self._clean_metadata(
                    metadata
                )
            ],
        )

    # ========================================================
    # ADD PERSON EVENT
    # ========================================================

    def add_person_event(
        self,
        event: dict[str, Any],
    ) -> None:

        event_id = (
            event.get(
                "id"
            )
        )

        if event_id is None:

            raise ValueError(
                "Person event must "
                "contain an id."
            )

        person_id = (
            event.get(
                "person_id"
            )
        )

        if person_id is None:

            raise ValueError(
                "Person event must "
                "contain a person_id."
            )

        person_name = str(
            event.get(
                "person_name"
            )
            or "unknown person"
        )

        relation = (
            event.get(
                "relation"
            )
        )

        similarity = (
            event.get(
                "similarity"
            )
        )

        detected_at = (
            event.get(
                "detected_at"
            )
        )

        memory_text = (
            event.get(
                "memory_text"
            )
        )

        # ----------------------------------------------------
        # FALLBACK MEMORY TEXT
        # ----------------------------------------------------

        if memory_text:

            text = str(
                memory_text
            ).strip()

        else:

            if isinstance(
                detected_at,
                datetime,
            ):

                time_text = (
                    detected_at.strftime(
                        "%d %b %Y "
                        "at %I:%M %p"
                    )
                )

            else:

                time_text = str(
                    detected_at
                    or "an unknown time"
                )

            if relation:

                text = (
                    f"MemorAi saw "
                    f"{person_name}, "
                    f"the user's "
                    f"{relation}, "
                    f"on {time_text}."
                )

            else:

                text = (
                    f"MemorAi saw "
                    f"{person_name} "
                    f"on {time_text}."
                )

        metadata = {
            "person_event_id": int(
                event_id
            ),

            "person_id": int(
                person_id
            ),

            "person_name": (
                person_name
            ),

            "relation": (
                relation
            ),

            "similarity": (
                float(
                    similarity
                )
                if similarity
                is not None
                else None
            ),

            "detected_at": (
                detected_at
            ),

            "image_path": (
                event.get(
                    "image_path"
                )
            ),

            "memory_type": (
                "person"
            ),

            "event_type": (
                "person_encounter"
            ),
        }

        self.add_memory(
            memory_id=(
                f"person_event_"
                f"{event_id}"
            ),
            text=text,
            metadata=metadata,
        )

    # ========================================================
    # GET MEMORY BY ID
    # ========================================================

    def get_memory(
        self,
        memory_id: str,
    ) -> dict[str, Any] | None:
        """
        Return one exact Chroma memory by ID.
        """

        result = (
            self.collection.get(
                ids=[
                    str(
                        memory_id
                    )
                ],
                include=[
                    "documents",
                    "metadatas",
                ],
            )
        )

        ids = (
            result.get(
                "ids",
                [],
            )
        )

        if not ids:

            return None

        documents = (
            result.get(
                "documents",
                [],
            )
        )

        metadatas = (
            result.get(
                "metadatas",
                [],
            )
        )

        document = (
            documents[0]
            if documents
            else ""
        )

        metadata = (
            metadatas[0]
            if (
                metadatas
                and metadatas[0]
            )
            else {}
        )

        return {
            "memory_id": str(
                ids[0]
            ),
            "text": str(
                document
                or ""
            ),
            "metadata": dict(
                metadata
            ),
        }

    # ========================================================
    # GET PERSON EVENT
    # ========================================================

    def get_person_event(
        self,
        event_id: int,
    ) -> dict[str, Any] | None:

        return self.get_memory(
            (
                f"person_event_"
                f"{int(event_id)}"
            )
        )

    # ========================================================
    # SEARCH
    # ========================================================

    def search(
        self,
        query: str,
        n_results: int = 5,
        where: dict[str, Any] | None = None,
    ) -> dict[str, Any]:

        cleaned_query = (
            query.strip()
        )

        if not cleaned_query:

            raise ValueError(
                "Search query "
                "cannot be empty."
            )

        count = (
            self.collection.count()
        )

        if count == 0:

            return {
                "ids": [[]],
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]],
            }

        top_k = min(
            max(
                1,
                int(
                    n_results
                ),
            ),
            count,
        )

        kwargs: dict[
            str,
            Any,
        ] = {
            "query_embeddings": [
                create_embedding(
                    cleaned_query
                )
            ],

            "n_results": (
                top_k
            ),

            "include": [
                "documents",
                "metadatas",
                "distances",
            ],
        }

        if where:

            kwargs[
                "where"
            ] = where

        return (
            self.collection
            .query(
                **kwargs
            )
        ) # type: ignore

    # ========================================================
    # COUNT
    # ========================================================

    def count(
        self,
    ) -> int:

        return (
            self.collection.count()
        )

    # ========================================================
    # DELETE MEMORY
    # ========================================================

    def delete_memory(
        self,
        memory_id: str,
    ) -> None:

        self.collection.delete(
            ids=[
                str(
                    memory_id
                )
            ]
        )

    # ========================================================
    # DELETE PERSON EVENT
    # ========================================================

    def delete_person_event(
        self,
        event_id: int,
    ) -> None:

        self.delete_memory(
            (
                f"person_event_"
                f"{event_id}"
            )
        )

    # ========================================================
    # DELETE PERSON MEMORIES
    # ========================================================

    def delete_person_memories(
        self,
        person_id: int,
    ) -> None:

        self.collection.delete(
            where={
                "person_id": int(
                    person_id
                )
            }
        )