from __future__ import annotations

from llm.chat import MemorAiChat


class HomeRagAdapter:
    """
    Streamlit adapter around the existing MemorAiChat backend.

    Existing backend:
        llm/chat.py
            ↓
        MemoryRetriever
            ↓
        MySQL + ChromaDB
            ↓
        deterministic answers / Ollama
    """

    def __init__(self) -> None:
        self.chat = MemorAiChat()

    def answer_question(
        self,
        question: str,
    ) -> str:
        """
        Home.py expects answer_question().
        Existing backend exposes answer().

        This adapter simply connects the two interfaces.
        """

        return self.chat.answer(
            question
        )