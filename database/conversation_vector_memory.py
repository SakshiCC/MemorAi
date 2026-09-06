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
    "database.conversation_vector_memory",
    "database.log",
)


# ============================================================
# COLLECTION
# ============================================================

CONVERSATION_CHROMA_COLLECTION_NAME = (
    "memorai_conversation_memories"
)


# ============================================================
# CONVERSATION VECTOR MEMORY
# ============================================================

class ConversationVectorMemory:
    """
    ChromaDB storage for important conversation memories.

    Privacy rule:

        Only conversations that have already passed the
        ConversationImportanceAnalyzer should be added here.

    MySQL remains authoritative for:

        - original transcript
        - exact conversation time
        - participant
        - title
        - conversation status

    ChromaDB is used for:

        - semantic retrieval
        - flexible natural-language questions
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
                    CONVERSATION_CHROMA_COLLECTION_NAME
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
    # BUILD MEMORY TEXT
    # ========================================================

    @staticmethod
    def build_memory_text(
        conversation: dict[str, Any],
    ) -> str:
        """
        Build semantic text from a completed conversation.

        We include useful context such as:

            participant
            title
            date/time
            transcript
            summary

        The original transcript remains preserved in MySQL.
        """

        conversation_id = (
            conversation.get(
                "id"
            )
        )

        if conversation_id is None:

            raise ValueError(
                (
                    "Conversation must contain "
                    "an id."
                )
            )

        status = str(
            conversation.get(
                "status"
            )
            or ""
        ).strip().lower()

        if (
            status
            and status != "completed"
        ):

            raise ValueError(
                (
                    "Only completed conversations "
                    "can be indexed."
                )
            )

        transcript = str(
            conversation.get(
                "full_transcript"
            )
            or ""
        ).strip()

        if not transcript:

            raise ValueError(
                (
                    "Conversation must contain "
                    "a full transcript."
                )
            )

        title = str(
            conversation.get(
                "title"
            )
            or ""
        ).strip()

        participant = str(
            conversation.get(
                "participant_text"
            )
            or ""
        ).strip()

        summary = str(
            conversation.get(
                "summary"
            )
            or ""
        ).strip()

        started_at = (
            conversation.get(
                "started_at"
            )
        )

        # ----------------------------------------------------
        # TIME TEXT
        # ----------------------------------------------------

        if isinstance(
            started_at,
            datetime,
        ):

            date_text = (
                started_at.strftime(
                    "%d %B %Y at %I:%M %p"
                )
            )

        else:

            date_text = str(
                started_at
                or ""
            ).strip()

        # ----------------------------------------------------
        # MEMORY PARTS
        # ----------------------------------------------------

        parts: list[str] = []

        if participant:

            parts.append(
                (
                    "MemorAi remembers an important "
                    f"conversation with {participant}."
                )
            )

        else:

            parts.append(
                (
                    "MemorAi remembers an important "
                    "conversation."
                )
            )

        if date_text:

            parts.append(
                (
                    "The conversation occurred on "
                    f"{date_text}."
                )
            )

        if title:

            parts.append(
                (
                    "Conversation topic: "
                    f"{title}."
                )
            )

        if summary:

            parts.append(
                (
                    "Conversation summary: "
                    f"{summary}"
                )
            )

        parts.append(
            (
                "Conversation transcript: "
                f"{transcript}"
            )
        )

        return (
            " ".join(
                parts
            )
            .strip()
        )

    # ========================================================
    # ADD GENERIC MEMORY
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
            )
            .strip()
        )

        if not cleaned_text:

            raise ValueError(
                (
                    "Conversation memory text "
                    "cannot be empty."
                )
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
    # ADD CONVERSATION
    # ========================================================

    def add_conversation(
        self,
        conversation: dict[str, Any],
        importance_score: float | None = None,
        matched_categories: (
            tuple[str, ...]
            | list[str]
            | None
        ) = None,
    ) -> None:
        """
        Add one completed important conversation to ChromaDB.

        IMPORTANT:

        The caller is responsible for ensuring the
        conversation already passed the importance filter.
        """

        conversation_id = (
            conversation.get(
                "id"
            )
        )

        if conversation_id is None:

            raise ValueError(
                (
                    "Conversation must "
                    "contain an id."
                )
            )

        status = str(
            conversation.get(
                "status"
            )
            or ""
        ).strip().lower()

        if status != "completed":

            raise ValueError(
                (
                    "Only completed conversations "
                    "can become vector memories."
                )
            )

        transcript = str(
            conversation.get(
                "full_transcript"
            )
            or ""
        ).strip()

        if not transcript:

            raise ValueError(
                (
                    "Completed conversation "
                    "does not contain a transcript."
                )
            )

        # ----------------------------------------------------
        # MEMORY TEXT
        # ----------------------------------------------------

        text = (
            self.build_memory_text(
                conversation
            )
        )

        # ----------------------------------------------------
        # CATEGORY TEXT
        # ----------------------------------------------------

        category_text = ""

        if matched_categories:

            cleaned_categories = [
                str(
                    category
                ).strip()

                for category
                in matched_categories

                if str(
                    category
                ).strip()
            ]

            category_text = (
                ", ".join(
                    cleaned_categories
                )
            )

        # ----------------------------------------------------
        # METADATA
        # ----------------------------------------------------

        metadata = {
            "conversation_id": int(
                conversation_id
            ),

            "title": (
                conversation.get(
                    "title"
                )
            ),

            "participant_text": (
                conversation.get(
                    "participant_text"
                )
            ),

            "started_at": (
                conversation.get(
                    "started_at"
                )
            ),

            "ended_at": (
                conversation.get(
                    "ended_at"
                )
            ),

            "status": (
                status
            ),

            "importance_score": (
                float(
                    importance_score
                )
                if importance_score
                is not None
                else None
            ),

            "importance_categories": (
                category_text
                or None
            ),

            "memory_type": (
                "conversation"
            ),

            "event_type": (
                "important_conversation"
            ),
        }

        # ----------------------------------------------------
        # UPSERT
        # ----------------------------------------------------

        self.add_memory(
            memory_id=(
                "conversation_"
                f"{conversation_id}"
            ),
            text=text,
            metadata=metadata,
        )

        logger.info(
            (
                "Conversation %s indexed into "
                "conversation vector memory."
            ),
            conversation_id,
        )

    # ========================================================
    # GET MEMORY
    # ========================================================

    def get_memory(
        self,
        memory_id: str,
    ) -> dict[str, Any] | None:

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
    # GET CONVERSATION MEMORY
    # ========================================================

    def get_conversation(
        self,
        conversation_id: int,
    ) -> dict[str, Any] | None:

        return (
            self.get_memory(
                (
                    "conversation_"
                    f"{int(conversation_id)}"
                )
            )
        )

    # ========================================================
    # SEARCH
    # ========================================================

    def search(
        self,
        query: str,
        n_results: int = 5,
        participant_text: str | None = None,
    ) -> dict[str, Any]:

        cleaned_query = (
            str(
                query
            )
            .strip()
        )

        if not cleaned_query:

            raise ValueError(
                (
                    "Conversation search "
                    "query cannot be empty."
                )
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

        if participant_text:

            kwargs[
                "where"
            ] = {
                "participant_text": str(
                    participant_text
                ).strip()
            }

        return (
            self.collection.query(
                **kwargs
            )
        )  # type: ignore

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
    # DELETE CONVERSATION
    # ========================================================

    def delete_conversation(
        self,
        conversation_id: int,
    ) -> None:

        self.delete_memory(
            (
                "conversation_"
                f"{int(conversation_id)}"
            )
        )