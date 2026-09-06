from __future__ import annotations

from dataclasses import dataclass

from datetime import (
    datetime,
)

from pathlib import (
    Path,
)

from typing import (
    Any,
)

from conversation.conversation_repository import (
    add_conversation_segment,
    create_conversation_session,
    finalize_conversation,
    get_conversation_session,
)

from conversation.conversation_settings import (
    get_importance_threshold,
    require_conversation_memory_permission,
)

from conversation.importance_analyzer import (
    ConversationImportanceAnalyzer,
    ConversationImportanceResult,
)

from conversation.metadata_extractor import (
    ConversationMetadataExtractor,
    ConversationMetadataResult,
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
    "conversation.service",
    "app.log",
)


# ============================================================
# PROCESS RESULT
# ============================================================

@dataclass
class ConversationProcessResult:

    stored: bool

    transcript: str

    importance_score: float

    importance_reasons: tuple[str, ...]

    matched_categories: tuple[str, ...]

    conversation_id: int | None = None

    vector_indexed: bool = False

    started_at: datetime | None = None

    ended_at: datetime | None = None

    title: str | None = None

    participant_text: str | None = None

    summary: str | None = None


    def to_dict(
        self,
    ) -> dict[str, Any]:

        return {
            "stored": (
                self.stored
            ),

            "vector_indexed": (
                self.vector_indexed
            ),

            "conversation_id": (
                self.conversation_id
            ),

            "transcript": (
                self.transcript
            ),

            "importance_score": (
                self.importance_score
            ),

            "importance_reasons": list(
                self.importance_reasons
            ),

            "matched_categories": list(
                self.matched_categories
            ),

            "started_at": (
                self.started_at
            ),

            "ended_at": (
                self.ended_at
            ),

            "title": (
                self.title
            ),

            "participant_text": (
                self.participant_text
            ),

            "summary": (
                self.summary
            ),
        }


# ============================================================
# CONVERSATION SERVICE
# ============================================================

class ConversationService:
    """
    Privacy-aware Conversation Memory service.

    Pipeline:

        permission
            ↓
        temporary audio
            ↓
        Whisper
            ↓
        importance
            ↓
        unimportant → discard

        important
            ↓
        metadata extraction
            ↓
        MySQL
            ↓
        BGE
            ↓
        ChromaDB

    Privacy:

        Metadata extraction and permanent storage occur only
        after the transcript passes the importance filter.
    """

    def __init__(
        self,
        transcriber=None,
        importance_analyzer=None,
        metadata_extractor=None,
        vector_memory=None,
    ) -> None:

        # ----------------------------------------------------
        # TRANSCRIBER
        # ----------------------------------------------------

        if transcriber is not None:

            self.transcriber = (
                transcriber
            )

        else:

            from speech.whisper_module import (
                WhisperTranscriber,
            )

            self.transcriber = (
                WhisperTranscriber(
                    model_size="base.en",
                    device="cpu",
                    compute_type="int8",
                )
            )

        # ----------------------------------------------------
        # IMPORTANCE
        # ----------------------------------------------------

        self.importance_analyzer = (
            importance_analyzer
        )

        # ----------------------------------------------------
        # METADATA
        # ----------------------------------------------------

        self.metadata_extractor = (
            metadata_extractor
            or ConversationMetadataExtractor()
        )

        # ----------------------------------------------------
        # VECTOR MEMORY
        # ----------------------------------------------------

        self.vector_memory = (
            vector_memory
            or ConversationVectorMemory()
        )


    # ========================================================
    # PERMISSION
    # ========================================================

    @staticmethod
    def require_permission() -> None:

        require_conversation_memory_permission()


    # ========================================================
    # AUDIO VALIDATION
    # ========================================================

    @staticmethod
    def validate_audio_file(
        audio_path: str | Path,
    ) -> Path:

        path = (
            Path(
                audio_path
            )
        )

        if not path.exists():

            raise FileNotFoundError(
                (
                    "Conversation audio file "
                    "was not found: "
                    f"{path}"
                )
            )

        if not path.is_file():

            raise ValueError(
                (
                    "Conversation audio path "
                    "is not a file: "
                    f"{path}"
                )
            )

        if (
            path.stat().st_size
            <= 0
        ):

            raise ValueError(
                (
                    "Conversation audio "
                    "file is empty."
                )
            )

        return (
            path
        )


    # ========================================================
    # TRANSCRIBE
    # ========================================================

    def transcribe_audio(
        self,
        audio_path: str | Path,
    ) -> str:

        self.require_permission()

        path = (
            self.validate_audio_file(
                audio_path
            )
        )

        logger.info(
            (
                "Processing permitted "
                "conversation audio: %s"
            ),
            path,
        )

        try:

            transcript = (
                self.transcriber
                .transcribe(
                    str(
                        path
                    )
                )
            )

        except Exception as exc:

            logger.exception(
                (
                    "Conversation "
                    "transcription failed."
                )
            )

            raise RuntimeError(
                (
                    "MemorAi could not "
                    "transcribe the "
                    "conversation audio."
                )
            ) from exc

        cleaned = (
            str(
                transcript
                or ""
            )
            .strip()
        )

        if not cleaned:

            raise ValueError(
                (
                    "No understandable speech "
                    "was detected in the "
                    "conversation recording."
                )
            )

        return (
            cleaned
        )


    # ========================================================
    # IMPORTANCE ANALYZER
    # ========================================================

    def _get_importance_analyzer(
        self,
    ) -> ConversationImportanceAnalyzer:

        if (
            self.importance_analyzer
            is not None
        ):

            return (
                self.importance_analyzer
            )

        return (
            ConversationImportanceAnalyzer(
                threshold=(
                    get_importance_threshold()
                )
            )
        )


    # ========================================================
    # ANALYZE IMPORTANCE
    # ========================================================

    def analyze_transcript(
        self,
        transcript: str,
    ) -> ConversationImportanceResult:

        cleaned = (
            str(
                transcript
                or ""
            )
            .strip()
        )

        if not cleaned:

            raise ValueError(
                (
                    "Cannot analyze an empty "
                    "conversation transcript."
                )
            )

        result = (
            self
            ._get_importance_analyzer()
            .analyze(
                cleaned
            )
        )

        logger.info(
            (
                "Conversation importance analyzed. "
                "important=%s score=%.3f "
                "categories=%s"
            ),
            result.is_important,
            result.score,
            result.matched_categories,
        )

        return (
            result
        )


    # ========================================================
    # METADATA EXTRACTION
    # ========================================================

    def extract_metadata(
        self,
        transcript: str,
        importance: ConversationImportanceResult,
        title: str | None = None,
        participant_text: str | None = None,
    ) -> ConversationMetadataResult:
        """
        Extract metadata only for a conversation that has
        already passed the importance filter.
        """

        if not importance.is_important:

            raise ValueError(
                (
                    "Metadata extraction is only "
                    "allowed for important conversations."
                )
            )

        result = (
            self.metadata_extractor
            .extract(
                transcript=(
                    transcript
                ),

                provided_title=(
                    title
                ),

                provided_participant=(
                    participant_text
                ),

                matched_categories=(
                    importance.matched_categories
                ),
            )
        )

        logger.info(
            (
                "Conversation metadata extracted. "
                "title=%s participant=%s"
            ),
            result.title,
            result.participant_text,
        )

        return (
            result
        )


    # ========================================================
    # STORE IMPORTANT CONVERSATION
    # ========================================================

    def store_important_conversation(
        self,
        transcript: str,
        importance: ConversationImportanceResult,
        title: str | None = None,
        participant_text: str | None = None,
        started_at: datetime | None = None,
        ended_at: datetime | None = None,
        summary: str | None = None,
    ) -> int:

        self.require_permission()

        if not importance.is_important:

            raise ValueError(
                (
                    "An unimportant conversation "
                    "cannot be stored."
                )
            )

        cleaned_transcript = (
            str(
                transcript
                or ""
            )
            .strip()
        )

        if not cleaned_transcript:

            raise ValueError(
                (
                    "Conversation transcript "
                    "cannot be empty."
                )
            )

        actual_started_at = (
            started_at
            or datetime.now()
        )

        actual_ended_at = (
            ended_at
            or datetime.now()
        )

        if (
            actual_ended_at
            < actual_started_at
        ):

            raise ValueError(
                (
                    "Conversation ended_at cannot "
                    "be earlier than started_at."
                )
            )

        conversation_id = (
            create_conversation_session(
                started_at=(
                    actual_started_at
                ),

                title=(
                    title
                ),

                participant_text=(
                    participant_text
                ),

                status="processing",
            )
        )

        if conversation_id <= 0:

            raise RuntimeError(
                (
                    "MemorAi could not create "
                    "the conversation memory."
                )
            )

        try:

            segment_id = (
                add_conversation_segment(
                    conversation_id=(
                        conversation_id
                    ),

                    sequence_no=1,

                    transcript=(
                        cleaned_transcript
                    ),

                    started_at=(
                        actual_started_at
                    ),

                    ended_at=(
                        actual_ended_at
                    ),
                )
            )

            if segment_id <= 0:

                raise RuntimeError(
                    (
                        "MemorAi could not store "
                        "the conversation transcript."
                    )
                )

            finalize_conversation(
                conversation_id=(
                    conversation_id
                ),

                ended_at=(
                    actual_ended_at
                ),

                full_transcript=(
                    cleaned_transcript
                ),

                summary=(
                    summary
                ),

                title=(
                    title
                ),

                participant_text=(
                    participant_text
                ),
            )

        except Exception:

            logger.exception(
                (
                    "Failed while storing important "
                    "conversation. "
                    "conversation_id=%s"
                ),
                conversation_id,
            )

            raise

        logger.info(
            (
                "Important conversation stored "
                "in MySQL. conversation_id=%s "
                "importance_score=%.3f"
            ),
            conversation_id,
            importance.score,
        )

        return (
            conversation_id
        )


    # ========================================================
    # INDEX EXISTING CONVERSATION
    # ========================================================

    def index_conversation(
        self,
        conversation_id: int,
        importance_score: float | None = None,
        matched_categories: (
            tuple[str, ...]
            | list[str]
            | None
        ) = None,
    ) -> bool:

        conversation = (
            get_conversation_session(
                conversation_id
            )
        )

        if not conversation:

            raise ValueError(
                (
                    "Conversation session "
                    f"{conversation_id} "
                    "does not exist."
                )
            )

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

            raise ValueError(
                (
                    "Only completed conversations "
                    "can be indexed."
                )
            )

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

            raise ValueError(
                (
                    "Conversation has no "
                    "transcript to index."
                )
            )

        self.vector_memory.add_conversation(
            conversation=(
                conversation
            ),

            importance_score=(
                importance_score
            ),

            matched_categories=(
                matched_categories
            ),
        )

        logger.info(
            (
                "Conversation vector indexed. "
                "conversation_id=%s"
            ),
            conversation_id,
        )

        return (
            True
        )


    # ========================================================
    # SAFE INDEXING
    # ========================================================

    def _try_index_conversation(
        self,
        conversation_id: int,
        importance: ConversationImportanceResult,
    ) -> bool:

        try:

            return (
                self.index_conversation(
                    conversation_id=(
                        conversation_id
                    ),

                    importance_score=(
                        importance.score
                    ),

                    matched_categories=(
                        importance.matched_categories
                    ),
                )
            )

        except Exception:

            logger.exception(
                (
                    "Conversation was stored in MySQL "
                    "but vector indexing failed. "
                    "conversation_id=%s"
                ),
                conversation_id,
            )

            return (
                False
            )


    # ========================================================
    # PROCESS COMPLETE AUDIO
    # ========================================================

    def process_complete_audio(
        self,
        audio_path: str | Path,
        title: str | None = None,
        participant_text: str | None = None,
        started_at: datetime | None = None,
        ended_at: datetime | None = None,
        summary: str | None = None,
    ) -> ConversationProcessResult:

        # ----------------------------------------------------
        # PRIVACY
        # ----------------------------------------------------

        self.require_permission()

        actual_started_at = (
            started_at
            or datetime.now()
        )

        # ----------------------------------------------------
        # WHISPER
        # ----------------------------------------------------

        transcript = (
            self.transcribe_audio(
                audio_path
            )
        )

        # ----------------------------------------------------
        # IMPORTANCE
        # ----------------------------------------------------

        importance = (
            self.analyze_transcript(
                transcript
            )
        )

        # ====================================================
        # UNIMPORTANT
        # ====================================================

        if not importance.is_important:

            logger.info(
                (
                    "Conversation discarded. "
                    "score=%.3f"
                ),
                importance.score,
            )

            return (
                ConversationProcessResult(
                    stored=False,

                    vector_indexed=False,

                    conversation_id=None,

                    transcript=(
                        transcript
                    ),

                    importance_score=(
                        importance.score
                    ),

                    importance_reasons=(
                        importance.reasons
                    ),

                    matched_categories=(
                        importance.matched_categories
                    ),

                    started_at=(
                        actual_started_at
                    ),

                    ended_at=(
                        ended_at
                        or datetime.now()
                    ),

                    title=(
                        title
                    ),

                    participant_text=(
                        participant_text
                    ),

                    summary=None,
                )
            )

        # ====================================================
        # IMPORTANT
        # ====================================================

        actual_ended_at = (
            ended_at
            or datetime.now()
        )

        # ----------------------------------------------------
        # METADATA
        # ----------------------------------------------------

        metadata = (
            self.extract_metadata(
                transcript=(
                    transcript
                ),

                importance=(
                    importance
                ),

                title=(
                    title
                ),

                participant_text=(
                    participant_text
                ),
            )
        )

        final_title = (
            metadata.title
        )

        final_participant = (
            metadata.participant_text
        )

        final_summary = (
            str(
                summary
                or metadata.summary
                or ""
            )
            .strip()
            or None
        )

        # ----------------------------------------------------
        # MYSQL
        # ----------------------------------------------------

        conversation_id = (
            self.store_important_conversation(
                transcript=(
                    transcript
                ),

                importance=(
                    importance
                ),

                title=(
                    final_title
                ),

                participant_text=(
                    final_participant
                ),

                started_at=(
                    actual_started_at
                ),

                ended_at=(
                    actual_ended_at
                ),

                summary=(
                    final_summary
                ),
            )
        )

        completed = (
            get_conversation_session(
                conversation_id
            )
        )

        if not completed:

            raise RuntimeError(
                (
                    "Conversation was stored but "
                    "could not be retrieved."
                )
            )

        # ----------------------------------------------------
        # CHROMADB
        # ----------------------------------------------------

        vector_indexed = (
            self._try_index_conversation(
                conversation_id=(
                    conversation_id
                ),

                importance=(
                    importance
                ),
            )
        )

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        return (
            ConversationProcessResult(
                stored=True,

                vector_indexed=(
                    vector_indexed
                ),

                conversation_id=(
                    conversation_id
                ),

                transcript=(
                    transcript
                ),

                importance_score=(
                    importance.score
                ),

                importance_reasons=(
                    importance.reasons
                ),

                matched_categories=(
                    importance.matched_categories
                ),

                started_at=(
                    completed.get(
                        "started_at"
                    )
                    or actual_started_at
                ),

                ended_at=(
                    completed.get(
                        "ended_at"
                    )
                    or actual_ended_at
                ),

                title=(
                    completed.get(
                        "title"
                    )
                ),

                participant_text=(
                    completed.get(
                        "participant_text"
                    )
                ),

                summary=(
                    completed.get(
                        "summary"
                    )
                ),
            )
        )