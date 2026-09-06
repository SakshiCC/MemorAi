from __future__ import annotations

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL_NAME


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def create_embedding(text: str) -> list[float]:
    cleaned = text.strip()

    if not cleaned:
        raise ValueError("Text cannot be empty.")

    vector = get_embedding_model().encode(
        cleaned,
        normalize_embeddings=True,
    )
    return vector.tolist()


def create_embeddings(texts: list[str]) -> list[list[float]]:
    cleaned = [text.strip() for text in texts if text.strip()]

    if not cleaned:
        return []

    vectors = get_embedding_model().encode(
        cleaned,
        normalize_embeddings=True,
    )
    return vectors.tolist()
