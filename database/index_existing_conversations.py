from __future__ import annotations

from typing import Any

from conversation.conversation_repository import (
    get_recent_conversations,
)

from database.conversation_vector_memory import (
    ConversationVectorMemory,
)

from utils.logger import (
    get_logger,
)


# ============================================================
# LOGGER
# ============================================================

logger = get_logger(
    "database.index_existing_conversations",
    "database.log",
)


# ============================================================
# RESULT
# ============================================================

def _empty_result() -> dict[str, int]:

    return {
        "checked": 0,
        "indexed": 0,
        "already_indexed": 0,
        "skipped": 0,
        "failed": 0,
    }


# ============================================================
# INDEX ONE CONVERSATION
# ============================================================

def index_conversation_if_needed(
    conversation: dict[str, Any],
    vector_memory: ConversationVectorMemory,
    force: bool = False,
) -> str:
    """
    Index one completed conversation.

    Returns one of:

        indexed
        already_indexed
        skipped
        failed
    """

    conversation_id = (
        conversation.get(
            "id"
        )
    )

    if conversation_id is None:

        logger.warning(
            "Skipping conversation without id."
        )

        return "skipped"

    conversation_id = int(
        conversation_id
    )

    # ========================================================
    # ONLY COMPLETED MEMORIES
    # ========================================================

    status = (
        str(
            conversation.get(
                "status"
            )
            or ""
        )
        .strip()
        .lower()
    )

    if status != "completed":

        return "skipped"

    # ========================================================
    # MUST HAVE TRANSCRIPT
    # ========================================================

    transcript = (
        str(
            conversation.get(
                "full_transcript"
            )
            or ""
        )
        .strip()
    )

    if not transcript:

        logger.warning(
            (
                "Skipping conversation %s "
                "because it has no transcript."
            ),
            conversation_id,
        )

        return "skipped"

    # ========================================================
    # CHECK EXISTING VECTOR
    # ========================================================

    existing = (
        vector_memory
        .get_conversation(
            conversation_id
        )
    )

    if (
        existing is not None
        and not force
    ):

        return "already_indexed"

    # ========================================================
    # INDEX
    # ========================================================

    try:

        vector_memory.add_conversation(
            conversation=conversation,
        )

        logger.info(
            (
                "Existing conversation indexed. "
                "conversation_id=%s"
            ),
            conversation_id,
        )

        return "indexed"

    except Exception:

        logger.exception(
            (
                "Could not index existing "
                "conversation %s."
            ),
            conversation_id,
        )

        return "failed"


# ============================================================
# INDEX EXISTING CONVERSATIONS
# ============================================================

def index_existing_conversations(
    limit: int = 10000,
    force: bool = False,
    vector_memory: (
        ConversationVectorMemory
        | None
    ) = None,
) -> dict[str, int]:
    """
    Index completed MySQL conversation memories into
    the dedicated Conversation ChromaDB collection.

    Important:

        - MySQL remains authoritative.
        - Existing Chroma memories are not duplicated.
        - Chroma uses upsert, so force=True is safe.
        - Incomplete / empty conversations are ignored.
    """

    result = (
        _empty_result()
    )

    if vector_memory is None:

        vector_memory = (
            ConversationVectorMemory()
        )

    conversations = (
        get_recent_conversations(
            limit=limit,
            completed_only=True,
        )
    )

    for conversation in conversations:

        result[
            "checked"
        ] += 1

        status = (
            index_conversation_if_needed(
                conversation=conversation,
                vector_memory=vector_memory,
                force=force,
            )
        )

        if (
            status
            in result
        ):

            result[
                status
            ] += 1

        else:

            result[
                "failed"
            ] += 1

    return result


# ============================================================
# TERMINAL
# ============================================================

def main() -> None:

    print(
        "\n"
        + "=" * 60
    )

    print(
        "MemorAi Existing Conversation Indexer"
    )

    print(
        "=" * 60
    )

    print(
        (
            "\nScanning completed "
            "conversation memories..."
        )
    )

    result = (
        index_existing_conversations()
    )

    print(
        "\nIndexing complete.\n"
    )

    print(
        (
            "Checked: "
            f"{result['checked']}"
        )
    )

    print(
        (
            "Indexed: "
            f"{result['indexed']}"
        )
    )

    print(
        (
            "Already indexed: "
            f"{result['already_indexed']}"
        )
    )

    print(
        (
            "Skipped: "
            f"{result['skipped']}"
        )
    )

    print(
        (
            "Failed: "
            f"{result['failed']}"
        )
    )


if __name__ == "__main__":

    main()