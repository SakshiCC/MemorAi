from __future__ import annotations

from datetime import datetime
from typing import Any

import chromadb

from config import CHROMA_PATH

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
    "database.medicine_vector_memory",
    "database.log",
)


# ============================================================
# CONSTANTS
# ============================================================

MEDICINE_COLLECTION_NAME = (
    "medicine_memories"
)


# ============================================================
# MEDICINE VECTOR MEMORY
# ============================================================

class MedicineVectorMemory:
    """
    ChromaDB vector memory for medicine events.

    MySQL remains the authoritative source for medicine
    records and medicine events.

    This collection is used only for semantic retrieval.

    Example semantic questions:

        What medicine was confirmed recently?

        Did I remember to take my stomach medicine?

        What medication event happened this morning?

        Was any medicine observed after breakfast?

    Important safety rule:

    A camera-confirmed medicine event means MemorAi observed
    sufficient visual evidence of a medication event.

    It does NOT prove that the medicine was swallowed.
    """

    # ========================================================
    # INITIALIZE
    # ========================================================

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
                    MEDICINE_COLLECTION_NAME
                ),
                metadata={
                    "hnsw:space": (
                        "cosine"
                    )
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
            str(
                text
                or ""
            )
            .strip()
        )

        if not cleaned_text:

            raise ValueError(
                "Medicine memory text "
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
    # ADD MEDICINE EVENT
    # ========================================================

    def add_event(
        self,
        event: dict[str, Any],
    ) -> None:
        """
        Index one authoritative MySQL medicine event.

        Expected event structure comes from:

            get_medicine_event_by_id()

        in database.medicine_repository.
        """

        event_id = (
            event.get(
                "id"
            )
        )

        if event_id is None:

            raise ValueError(
                "Medicine event must contain "
                "an id."
            )

        medicine_id = (
            event.get(
                "medicine_id"
            )
        )

        if medicine_id is None:

            raise ValueError(
                "Medicine event must contain "
                "a medicine_id."
            )

        medicine_name = (
            str(
                event.get(
                    "medicine_name"
                )
                or "medicine"
            )
            .strip()
        )

        dosage = (
            str(
                event.get(
                    "dosage"
                )
                or ""
            )
            .strip()
        )

        status = (
            str(
                event.get(
                    "status"
                )
                or ""
            )
            .strip()
        )

        observed_at = (
            event.get(
                "observed_at"
            )
        )

        # ----------------------------------------------------
        # MEMORY TEXT
        # ----------------------------------------------------

        memory_text = (
            str(
                event.get(
                    "memory_text"
                )
                or ""
            )
            .strip()
        )

        if not memory_text:

            description = (
                medicine_name
            )

            if dosage:

                description += (
                    f" {dosage}"
                )

            if isinstance(
                observed_at,
                datetime,
            ):

                readable_time = (
                    observed_at.strftime(
                        "%d %b %Y "
                        "at %I:%M %p"
                    )
                )

            else:

                readable_time = (
                    str(
                        observed_at
                        or "unknown time"
                    )
                )

            memory_text = (
                "MemorAi observed a medication "
                f"event involving {description} "
                f"on {readable_time}. "
                f"The event status was {status}."
            )

        # ----------------------------------------------------
        # METADATA
        # ----------------------------------------------------

        metadata = {

            "medicine_event_id": int(
                event_id
            ),

            "medicine_id": int(
                medicine_id
            ),

            "medicine_name": (
                medicine_name
            ),

            "dosage": (
                dosage
            ),

            "status": (
                status
            ),

            "event_type": (
                event.get(
                    "event_type"
                )
            ),

            "observed_at": (
                observed_at
            ),

            "scheduled_for": (
                event.get(
                    "scheduled_for"
                )
            ),

            "confirmation_method": (
                event.get(
                    "confirmation_method"
                )
            ),

            "confidence": (
                event.get(
                    "confidence"
                )
            ),

            "medicine_similarity": (
                event.get(
                    "medicine_similarity"
                )
            ),

            "person_similarity": (
                event.get(
                    "person_similarity"
                )
            ),

            "action_confidence": (
                event.get(
                    "action_confidence"
                )
            ),

            "evidence_image_path": (
                event.get(
                    "evidence_image_path"
                )
            ),

            "person_id": (
                event.get(
                    "person_id"
                )
            ),

            "person_name": (
                event.get(
                    "person_name"
                )
            ),

            "person_relation": (
                event.get(
                    "person_relation"
                )
            ),

            "memory_type": (
                "medicine"
            ),
        }

        self.add_memory(
            memory_id=(
                f"medicine_event_"
                f"{event_id}"
            ),
            text=memory_text,
            metadata=metadata,
        )

        logger.info(
            (
                "Medicine memory indexed. "
                "event_id=%s medicine_id=%s "
                "medicine=%s status=%s"
            ),
            event_id,
            medicine_id,
            medicine_name,
            status,
        )

    # ========================================================
    # SEARCH
    # ========================================================

    def search(
        self,
        query: str,
        n_results: int = 5,
        medicine_id: int | None = None,
        status: str | None = None,
    ) -> dict[str, Any]:

        cleaned_query = (
            str(
                query
                or ""
            )
            .strip()
        )

        if not cleaned_query:

            raise ValueError(
                "Medicine search query "
                "cannot be empty."
            )

        count = (
            self.collection.count()
        )

        if count == 0:

            return {
                "ids": [
                    []
                ],
                "documents": [
                    []
                ],
                "metadatas": [
                    []
                ],
                "distances": [
                    []
                ],
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

        # ----------------------------------------------------
        # FILTER
        # ----------------------------------------------------

        where: (
            dict[str, Any]
            | None
        ) = None

        if (
            medicine_id is not None
            and status
        ):

            where = {
                "$and": [
                    {
                        "medicine_id": int(
                            medicine_id
                        )
                    },
                    {
                        "status": str(
                            status
                        )
                    },
                ]
            }

        elif medicine_id is not None:

            where = {
                "medicine_id": int(
                    medicine_id
                )
            }

        elif status:

            where = {
                "status": str(
                    status
                )
            }

        # ----------------------------------------------------
        # QUERY
        # ----------------------------------------------------

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
            self.collection.query(
                **kwargs
            )
        )

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
    # DELETE EVENT
    # ========================================================

    def delete_event(
        self,
        event_id: int,
    ) -> None:

        self.delete_memory(
            memory_id=(
                f"medicine_event_"
                f"{event_id}"
            )
        )