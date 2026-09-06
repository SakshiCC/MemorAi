from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from config import DEFAULT_TOP_K, MAX_DISTANCE
from database.vector_memory import VectorMemory


@dataclass
class MemorySearchResult:
    memory_id: str
    text: str
    distance: float | None
    metadata: dict[str, Any]

    @property
    def detected_at(self) -> str | None:
        value = self.metadata.get("detected_at")
        return str(value) if value else None


class MemorySearch:
    def __init__(self) -> None:
        self.vector_memory = VectorMemory()

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        object_name: str | None = None,
    ) -> list[MemorySearchResult]:
        where = {"object_name": object_name} if object_name else None

        raw = self.vector_memory.search(
            query=query,
            n_results=top_k,
            where=where,
        )

        ids = raw.get("ids", [[]])[0]
        documents = raw.get("documents", [[]])[0]
        metadatas = raw.get("metadatas", [[]])[0]
        distances = raw.get("distances", [[]])[0]

        results: list[MemorySearchResult] = []

        for index, memory_id in enumerate(ids):
            distance = (
                float(distances[index])
                if index < len(distances)
                and distances[index] is not None
                else None
            )

            if distance is not None and distance > MAX_DISTANCE:
                continue

            metadata = (
                metadatas[index]
                if index < len(metadatas)
                and metadatas[index]
                else {}
            )

            text = (
                str(documents[index])
                if index < len(documents)
                else ""
            )

            results.append(
                MemorySearchResult(
                    memory_id=str(memory_id),
                    text=text,
                    distance=distance,
                    metadata=dict(metadata),
                )
            )

        return self._prefer_newer(results)

    @staticmethod
    def _prefer_newer(
        memories: list[MemorySearchResult],
    ) -> list[MemorySearchResult]:
        def timestamp(memory: MemorySearchResult) -> datetime:
            raw = memory.detected_at

            if not raw:
                return datetime.min

            try:
                normalized = raw.replace("Z", "+00:00")
                parsed = datetime.fromisoformat(normalized)
                return parsed.replace(tzinfo=None)
            except ValueError:
                return datetime.min

        return sorted(memories, key=timestamp, reverse=True)
