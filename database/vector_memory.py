from __future__ import annotations

from datetime import datetime
from typing import Any

import chromadb

from config import CHROMA_COLLECTION_NAME, CHROMA_PATH
from rag.embeddings import create_embedding
from utils.logger import get_logger

logger = get_logger("database.vector_memory", "database.log")


class VectorMemory:
    def __init__(self) -> None:
        self.client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        self.collection = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    @staticmethod
    def _clean_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        cleaned: dict[str, Any] = {}

        for key, value in metadata.items():
            if value is None:
                continue

            if isinstance(value, datetime):
                cleaned[key] = value.isoformat(timespec="seconds")
            elif isinstance(value, (str, int, float, bool)):
                cleaned[key] = value
            else:
                cleaned[key] = str(value)

        return cleaned

    def add_memory(
        self,
        memory_id: str,
        text: str,
        metadata: dict[str, Any],
    ) -> None:
        vector = create_embedding(text)

        self.collection.upsert(
            ids=[str(memory_id)],
            documents=[text],
            embeddings=[vector],
            metadatas=[self._clean_metadata(metadata)],
        )

    def add_event(self, event: dict[str, Any]) -> None:
        event_id = event.get("id")

        if event_id is None:
            raise ValueError("Event must contain an id.")

        object_name = str(event.get("object_name") or "object")
        location = str(event.get("location") or "unknown location")
        detected_at = event.get("detected_at")

        if isinstance(detected_at, datetime):
            time_text = detected_at.strftime("%Y-%m-%d %H:%M:%S")
        else:
            time_text = str(detected_at or "unknown time")

        text = (
            str(event.get("memory_text")).strip()
            if event.get("memory_text")
            else (
                f"MemorAi observed {object_name} at {location} "
                f"on {time_text}."
            )
        )

        metadata = {
            "event_id": int(event_id),
            "object_name": object_name,
            "location": location,
            "detected_at": detected_at,
            "image_path": event.get("image_path"),
            "event_type": event.get("event_type"),
        }

        self.add_memory(
            memory_id=f"event_{event_id}",
            text=text,
            metadata=metadata,
        )

    def search(
        self,
        query: str,
        n_results: int = 5,
        where: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not query.strip():
            raise ValueError("Search query cannot be empty.")

        count = self.collection.count()

        if count == 0:
            return {
                "ids": [[]],
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]],
            }

        top_k = min(max(1, int(n_results)), count)

        kwargs: dict[str, Any] = {
            "query_embeddings": [create_embedding(query)],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }

        if where:
            kwargs["where"] = where

        return self.collection.query(**kwargs)

    def count(self) -> int:
        return self.collection.count()

    def delete_memory(self, memory_id: str) -> None:
        self.collection.delete(ids=[str(memory_id)])
