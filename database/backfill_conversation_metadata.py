from __future__ import annotations

from typing import Any

from conversation.conversation_repository import (
    get_conversation_session,
    get_recent_conversations,
    update_conversation_details,
)

from conversation.metadata_extractor import (
    ConversationMetadataExtractor,
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
    "database.backfill_conversation_metadata",
    "database.log",
)


# ============================================================
# RESULT HELPERS
# ============================================================

def _empty_result() -> dict[str, int]:

    return {
        "checked": 0,
        "updated": 0,
        "already_complete": 0,
        "skipped": 0,
        "reindexed": 0,
        "reindex_failed": 0,
        "failed": 0,
    }


# ============================================================
# CLEAN OPTIONAL TEXT
# ============================================================

def _clean_optional_text(
    value: Any,
) -> str | None:

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
# CHECK WHETHER METADATA IS MISSING
# ============================================================

def conversation_needs_metadata_backfill(
    conversation: dict[str, Any],
) -> bool:
    """
    Return True when at least one generated metadata field
    is missing.

    Existing user-entered values are never considered
    missing and will not be overwritten.
    """

    title = (
        _clean_optional_text(
            conversation.get(
                "title"
            )
        )
    )

    participant = (
        _clean_optional_text(
            conversation.get(
                "participant_text"
            )
        )
    )

    summary = (
        _clean_optional_text(
            conversation.get(
                "summary"
            )
        )
    )

    return (
        title is None
        or participant is None
        or summary is None
    )


# ============================================================
# BACKFILL ONE CONVERSATION
# ============================================================

def backfill_conversation_metadata(
    conversation: dict[str, Any],
    extractor: (
        ConversationMetadataExtractor
        | None
    ) = None,
    vector_memory: (
        ConversationVectorMemory
        | None
    ) = None,
    reindex: bool = True,
) -> str:
    """
    Fill missing metadata for one completed conversation.

    Returns:

        updated
        already_complete
        skipped
        failed

    Important:

    - Existing title is preserved.
    - Existing participant is preserved.
    - Existing summary is preserved.
    - Only missing values are generated.
    """

    conversation_id = (
        conversation.get(
            "id"
        )
    )

    if conversation_id is None:

        logger.warning(
            (
                "Skipping conversation without "
                "an id."
            )
        )

        return (
            "skipped"
        )

    conversation_id = int(
        conversation_id
    )

    # ========================================================
    # COMPLETED ONLY
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

        logger.info(
            (
                "Skipping conversation %s "
                "because status is %s."
            ),
            conversation_id,
            status,
        )

        return (
            "skipped"
        )

    # ========================================================
    # TRANSCRIPT REQUIRED
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

        return (
            "skipped"
        )

    # ========================================================
    # ALREADY COMPLETE
    # ========================================================

    if not conversation_needs_metadata_backfill(
        conversation
    ):

        return (
            "already_complete"
        )

    # ========================================================
    # CURRENT VALUES
    # ========================================================

    current_title = (
        _clean_optional_text(
            conversation.get(
                "title"
            )
        )
    )

    current_participant = (
        _clean_optional_text(
            conversation.get(
                "participant_text"
            )
        )
    )

    current_summary = (
        _clean_optional_text(
            conversation.get(
                "summary"
            )
        )
    )

    # ========================================================
    # EXTRACT
    # ========================================================

    if extractor is None:

        extractor = (
            ConversationMetadataExtractor()
        )

    try:

        metadata = (
            extractor.extract(
                transcript=(
                    transcript
                ),

                # Existing values deliberately become
                # overrides so they are preserved.

                provided_title=(
                    current_title
                ),

                provided_participant=(
                    current_participant
                ),
            )
        )

        # ----------------------------------------------------
        # PRESERVE EXISTING VALUES
        # ----------------------------------------------------

        final_title = (
            current_title
            or metadata.title
        )

        final_participant = (
            current_participant
            or metadata.participant_text
        )

        final_summary = (
            current_summary
            or metadata.summary
        )

        # ----------------------------------------------------
        # UPDATE MYSQL
        # ----------------------------------------------------

        update_conversation_details(
            conversation_id=(
                conversation_id
            ),

            title=(
                final_title
            ),

            participant_text=(
                final_participant
            ),

            summary=(
                final_summary
            ),
        )

        logger.info(
            (
                "Conversation metadata backfilled. "
                "conversation_id=%s "
                "title=%s participant=%s"
            ),
            conversation_id,
            final_title,
            final_participant,
        )

    except Exception:

        logger.exception(
            (
                "Conversation metadata backfill "
                "failed. conversation_id=%s"
            ),
            conversation_id,
        )

        return (
            "failed"
        )

    # ========================================================
    # REINDEX UPDATED RECORD
    # ========================================================

    if reindex:

        if vector_memory is None:

            vector_memory = (
                ConversationVectorMemory()
            )

        try:

            updated_conversation = (
                get_conversation_session(
                    conversation_id
                )
            )

            if updated_conversation:

                vector_memory.add_conversation(
                    conversation=(
                        updated_conversation
                    )
                )

                logger.info(
                    (
                        "Backfilled conversation "
                        "re-indexed. conversation_id=%s"
                    ),
                    conversation_id,
                )

        except Exception:

            # ------------------------------------------------
            # MYSQL UPDATE MUST REMAIN EVEN IF CHROMA FAILS
            # ------------------------------------------------

            logger.exception(
                (
                    "Conversation metadata was updated "
                    "but ChromaDB re-indexing failed. "
                    "conversation_id=%s"
                ),
                conversation_id,
            )

    return (
        "updated"
    )


# ============================================================
# BACKFILL ALL EXISTING CONVERSATIONS
# ============================================================

def backfill_existing_conversation_metadata(
    limit: int = 10000,
    reindex: bool = True,
) -> dict[str, int]:
    """
    Backfill missing metadata across existing completed
    conversation memories.

    This operation does NOT:

        - record audio
        - run Whisper
        - create new conversation memories
        - require Conversation Memory permission

    It only enriches records that already exist.
    """

    result = (
        _empty_result()
    )

    extractor = (
        ConversationMetadataExtractor()
    )

    vector_memory = (
        ConversationVectorMemory()
        if reindex
        else None
    )

    conversations = (
        get_recent_conversations(
            limit=(
                limit
            ),
            completed_only=True,
        )
    )

    for conversation in conversations:

        result[
            "checked"
        ] += 1

        needs_backfill = (
            conversation_needs_metadata_backfill(
                conversation
            )
        )

        status = (
            backfill_conversation_metadata(
                conversation=(
                    conversation
                ),

                extractor=(
                    extractor
                ),

                vector_memory=(
                    vector_memory
                ),

                reindex=(
                    reindex
                ),
            )
        )

        if status == "updated":

            result[
                "updated"
            ] += 1

            if reindex:

                updated_record = (
                    get_conversation_session(
                        int(
                            conversation[
                                "id"
                            ]
                        )
                    )
                )

                if (
                    updated_record
                    is not None
                ):

                    try:

                        indexed = (
                            vector_memory
                            .get_conversation(
                                int(
                                    conversation[
                                        "id"
                                    ]
                                )
                            )
                            if vector_memory
                            is not None
                            else None
                        )

                        if indexed:

                            result[
                                "reindexed"
                            ] += 1

                        else:

                            result[
                                "reindex_failed"
                            ] += 1

                    except Exception:

                        result[
                            "reindex_failed"
                        ] += 1

        elif (
            status
            == "already_complete"
        ):

            result[
                "already_complete"
            ] += 1

        elif (
            status
            == "skipped"
        ):

            result[
                "skipped"
            ] += 1

        else:

            result[
                "failed"
            ] += 1

        # This variable intentionally documents why an
        # already-complete result was not counted as updated.
        _ = needs_backfill

    return (
        result
    )


# ============================================================
# COMMAND LINE
# ============================================================

def main() -> None:

    print(
        "\n"
        + "=" * 65
    )

    print(
        "MemorAi Conversation Metadata Backfill"
    )

    print(
        "=" * 65
    )

    print(
        (
            "\nScanning completed conversation "
            "memories..."
        )
    )

    result = (
        backfill_existing_conversation_metadata(
            reindex=True
        )
    )

    print(
        "\nBackfill complete.\n"
    )

    print(
        (
            "Checked: "
            f"{result['checked']}"
        )
    )

    print(
        (
            "Updated: "
            f"{result['updated']}"
        )
    )

    print(
        (
            "Already complete: "
            f"{result['already_complete']}"
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
            "Re-indexed: "
            f"{result['reindexed']}"
        )
    )

    print(
        (
            "Re-index failed: "
            f"{result['reindex_failed']}"
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