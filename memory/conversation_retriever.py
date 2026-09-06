from __future__ import annotations

import re

from dataclasses import (
    dataclass,
)

from datetime import (
    date,
    timedelta,
)

from typing import (
    Any,
)

from config import (
    DEFAULT_TOP_K,
    MAX_DISTANCE,
)

from conversation.conversation_repository import (
    get_conversations_for_date,
    get_recent_conversations,
    search_conversations,
)

from conversation.metadata_extractor import (
    ConversationMetadataExtractor,
)

from database.conversation_vector_memory import (
    ConversationVectorMemory,
)


# ============================================================
# RESULT
# ============================================================

@dataclass
class ConversationMemoryResult:

    conversation_id: int

    text: str

    distance: float | None

    metadata: dict[str, Any]

    source: str


# ============================================================
# RETRIEVER
# ============================================================

class ConversationMemoryRetriever:
    """
    Hybrid Conversation Memory retriever.

    MySQL is authoritative for:

        - exact conversation records
        - today / yesterday
        - recent conversations
        - original transcript
        - stored participant value

    ChromaDB is used for:

        - semantic search
        - flexible wording
        - topic-based retrieval

    Registered People are used for:

        - safe correction of speech-derived names
        - canonical display names
        - participant alias matching

    Example:

        Whisper stored:
            Saksi

        Registered person:
            Sakshi

        User asks:
            What did Sakshi tell me?

        MemorAi resolves:

            Saksi
              ↓
            Sakshi
              ↓
        same participant

    Safety:

        Unknown names must never cause an unrestricted
        semantic lookup which returns somebody else's
        conversation.
    """

    # ========================================================
    # ACTION EVIDENCE
    # ========================================================

    ACTION_EVIDENCE_PATTERNS = {

        "buy": (
            r"\bbuy\b",
            r"\bpurchase\b",
            r"\bshopping\b",
            r"\bgrocer(?:y|ies)\b",
        ),

        "bring": (
            r"\bbring\b",
            r"\bcarry\b",
            r"\btake\s+(?:my|the|your)\b",
            r"\bneed\s+to\s+bring\b",
        ),

        "pay": (
            r"\bpay\b",
            r"\bpayment\b",
            r"\bbill\b",
        ),

        "call": (
            r"\bcall\b",
            r"\bphone\b",
        ),

        "collect": (
            r"\bcollect\b",
            r"\bpick\s+up\b",
        ),

        "send": (
            r"\bsend\b",
            r"\bemail\b",
        ),

        "submit": (
            r"\bsubmit\b",
            r"\bsubmission\b",
        ),

        "visit": (
            r"\bvisit\b",
            r"\bvisiting\b",
            r"\bcome\s+over\b",
            r"\bcoming\s+over\b",
            r"\bcoming\s+to\b",
        ),

        "appointment": (
            r"\bappointment\b",
            r"\bdoctor\b",
            r"\bdentist\b",
            r"\bclinic\b",
            r"\bhospital\b",
            r"\bcheck[- ]?up\b",
        ),

        "medicine": (
            r"\bmedicine\b",
            r"\bmedication\b",
            r"\btablet\b",
            r"\bpill\b",
            r"\bdose\b",
            r"\bcapsule\b",
            r"\bprescription\b",
        ),
    }


    # ========================================================
    # INITIALIZE
    # ========================================================

    def __init__(
        self,
        vector_memory: (
            ConversationVectorMemory
            | None
        ) = None,
    ) -> None:

        self.vector_memory = (
            vector_memory
            or ConversationVectorMemory()
        )

        # ----------------------------------------------------
        # Reuse the existing safe registered-person resolver.
        # ----------------------------------------------------

        self.metadata_extractor = (
            ConversationMetadataExtractor()
        )

    # ========================================================
    # NORMALIZE TEXT
    # ========================================================

    @staticmethod
    def _normalize_text(
        value: str | None,
    ) -> str:

        return (
            str(
                value
                or ""
            )
            .strip()
            .lower()
        )

    # ========================================================
    # CANONICAL PARTICIPANT
    # ========================================================

    def canonicalize_participant(
        self,
        participant_text: str | None,
    ) -> str | None:
        """
        Convert a stored / speech-derived participant spelling
        into the registered person's canonical display name
        when there is a safe unambiguous match.

        Examples:

            Saksi  -> Sakshi
            Ria    -> Riya

        only when the registered-person resolver considers
        the match safe.

        Unknown or ambiguous names remain unchanged.
        """

        cleaned = (
            str(
                participant_text
                or ""
            )
            .strip()
        )

        if not cleaned:

            return None

        try:

            displayed = (
                self.metadata_extractor
                .display_participant_name(
                    cleaned
                )
            )

        except Exception:

            displayed = None

        return (
            str(
                displayed
                or cleaned
            )
            .strip()
            or None
        )

    # ========================================================
    # CONVERSATION INTENT
    # ========================================================

    @staticmethod
    def is_conversation_question(
        question: str,
    ) -> bool:

        lowered = (
            str(
                question
                or ""
            )
            .strip()
            .lower()
        )

        if not lowered:

            return False

        # ====================================================
        # EXPLICIT CONVERSATION LANGUAGE
        # ====================================================

        phrases = (
            "what did",
            "what was said",
            "what did we talk",
            "what did we discuss",
            "what were we talking",
            "what were we discussing",
            "what did they say",
            "what did he say",
            "what did she say",
            "what did you hear",
            "what did someone say",
            "what did someone tell",
            "did anyone say",
            "did anyone tell",
            "did anyone mention",
            "did someone mention",
            "who told me",
            "who said",
            "tell me about our conversation",
            "conversation with",
            "conversation about",
            "our conversation",
            "recent conversation",
            "recent conversations",
            "latest conversation",
            "last conversation",
            "talked about",
            "spoke about",
            "discussed",
            "mentioned",
            "reminded me",
            "told me",
            "said that",
        )

        if any(
            phrase in lowered
            for phrase in phrases
        ):

            return True

        # ====================================================
        # EXPLICIT PERSON + SPEECH VERB
        # ====================================================

        if re.search(
            (
                r"\bwhat\s+did\s+.+?\s+"
                r"(say|tell|mention|discuss|remind)\b"
            ),
            lowered,
        ):

            return True

        # ====================================================
        # NATURAL TASK QUESTIONS
        # ====================================================

        natural_task_patterns = (

            r"\bwhat\s+should\s+i\s+bring\b",

            r"\bwhat\s+do\s+i\s+need\s+to\s+bring\b",

            r"\bwhat\s+am\s+i\s+supposed\s+to\s+bring\b",

            (
                r"\bwhat\s+(?:do\s+i\s+need\s+to|"
                r"should\s+i|am\s+i\s+supposed\s+to)\s+"
                r"(?:buy|do|call|collect|send|submit|"
                r"pay|bring|remember)\b"
            ),

            r"\bwhat\s+was\s+i\s+reminded\s+to\b",

            r"\bwhat\s+was\s+i\s+supposed\s+to\b",

            r"\bwhat\s+do\s+i\s+need\s+to\s+remember\b",
        )

        if any(
            re.search(
                pattern,
                lowered,
            )
            is not None

            for pattern
            in natural_task_patterns
        ):

            return True

        # ====================================================
        # APPOINTMENTS
        # ====================================================

        appointment_words = (
            "appointment",
            "meeting",
            "check-up",
            "checkup",
        )

        has_appointment_context = any(
            word in lowered
            for word in appointment_words
        )

        if has_appointment_context:

            appointment_patterns = (

                r"\bwhen\s+is\b",

                r"\bwhen\s+was\b",

                r"\bwhat\s+time\b",

                r"\bwhat\s+date\b",

                r"\bwhat\s+day\b",

                r"\bwhat\s+should\s+i\s+bring\b",

                r"\bwhat\s+do\s+i\s+need\s+to\s+bring\b",

                r"\bwhat\s+do\s+i\s+need\s+for\b",
            )

            if any(
                re.search(
                    pattern,
                    lowered,
                )
                is not None

                for pattern
                in appointment_patterns
            ):

                return True

        # ====================================================
        # VISITS / PLANS
        # ====================================================

        visit_patterns = (

            r"\bwho\s+is\s+visiting\b",

            r"\bwho\s+will\s+visit\b",

            r"\bwho\s+is\s+coming\s+over\b",

            r"\bwho\s+will\s+come\s+over\b",

            r"\bwho\s+is\s+coming\s+(?:today|tomorrow)\b",

            r"\bwho\s+will\s+come\s+(?:today|tomorrow)\b",

            r"\bwhen\s+is\s+.+\s+visiting\b",

            r"\bwhen\s+will\s+.+\s+visit\b",
        )

        if any(
            re.search(
                pattern,
                lowered,
            )
            is not None

            for pattern
            in visit_patterns
        ):

            return True

        # ====================================================
        # IMPORTANT DATES
        # ====================================================

        important_date_words = (
            "birthday",
            "anniversary",
            "deadline",
            "due date",
            "renewal date",
            "expiry date",
            "expiration date",
        )

        if any(
            word in lowered
            for word in important_date_words
        ):

            if any(
                phrase in lowered
                for phrase in (
                    "when is",
                    "when was",
                    "what date",
                    "what day",
                )
            ):

                return True

        return False

    # ========================================================
    # EXTRACT REFERENCED PARTICIPANT
    # ========================================================

    @staticmethod
    def extract_referenced_participant(
        question: str,
    ) -> str | None:
        """
        Extract a person explicitly referenced by a
        conversation-style question.

        Examples:

            What did Sakshi tell me?
                -> Sakshi

            What did Saksi tell me?
                -> Saksi

            What was my latest conversation with Rahul?
                -> Rahul

        This only extracts the phrase.

        It does NOT decide who that person actually is.
        """

        cleaned = (
            str(
                question
                or ""
            )
            .strip()
        )

        if not cleaned:

            return None

        patterns = (

            (
                r"\bwhat\s+did\s+"
                r"([A-Za-z][A-Za-z' -]{0,60}?)\s+"
                r"(?:tell|say|mention|discuss|remind)\b"
            ),

            (
                r"\bconversation\s+with\s+"
                r"([A-Za-z][A-Za-z' -]{0,60}?)"
                r"(?:\?|$)"
            ),
        )

        for pattern in patterns:

            match = (
                re.search(
                    pattern,
                    cleaned,
                    flags=re.IGNORECASE,
                )
            )

            if not match:

                continue

            participant = (
                str(
                    match.group(
                        1
                    )
                )
                .strip()
            )

            if not participant:

                continue

            if participant.lower() in {
                "he",
                "she",
                "they",
                "someone",
                "anyone",
                "somebody",
                "anybody",
                "we",
                "you",
            }:

                return None

            return participant

        return None

    # ========================================================
    # GET STORED PARTICIPANT VARIANTS
    # ========================================================

    def _stored_participants(
        self,
    ) -> list[
        tuple[
            str,
            str,
        ]
    ]:
        """
        Return:

            [
                (
                    raw stored participant,
                    canonical display participant,
                )
            ]

        Example:

            [
                ("Saksi", "Sakshi"),
                ("Rahul", "Rahul"),
            ]
        """

        conversations = (
            get_recent_conversations(
                limit=500,
                completed_only=True,
            )
        )

        results: list[
            tuple[
                str,
                str,
            ]
        ] = []

        seen: set[
            tuple[
                str,
                str,
            ]
        ] = set()

        for conversation in conversations:

            raw = (
                str(
                    conversation.get(
                        "participant_text"
                    )
                    or ""
                )
                .strip()
            )

            if not raw:

                continue

            canonical = (
                self.canonicalize_participant(
                    raw
                )
                or raw
            )

            key = (
                raw.lower(),
                canonical.lower(),
            )

            if key in seen:

                continue

            seen.add(
                key
            )

            results.append(
                (
                    raw,
                    canonical,
                )
            )

        results.sort(
            key=lambda item: max(
                len(
                    item[0]
                ),
                len(
                    item[1]
                ),
            ),
            reverse=True,
        )

        return results

    # ========================================================
    # PARTICIPANT RESOLUTION
    # ========================================================

    def resolve_participant(
        self,
        question: str,
    ) -> str | None:
        """
        Resolve a question participant against both:

            raw stored spelling
            canonical registered-person spelling

        Important:

        This method returns the RAW stored spelling because
        ChromaDB metadata may still contain that old spelling.

        Example:

            DB participant:
                Saksi

            canonical:
                Sakshi

            question:
                What did Sakshi tell me?

            returns:
                Saksi

        This allows filtering old records without rewriting
        historical transcripts or database rows.
        """

        lowered = (
            self._normalize_text(
                question
            )
        )

        if not lowered:

            return None

        participants = (
            self._stored_participants()
        )

        # ====================================================
        # DIRECT RAW OR CANONICAL MATCH
        # ====================================================

        for (
            raw,
            canonical,
        ) in participants:

            for candidate in (
                raw,
                canonical,
            ):

                escaped = (
                    re.escape(
                        candidate.lower()
                    )
                )

                if re.search(
                    rf"\b{escaped}\b",
                    lowered,
                ):

                    return raw

        # ====================================================
        # EXPLICIT NAME FUZZY/CANONICAL RESOLUTION
        # ====================================================

        referenced = (
            self.extract_referenced_participant(
                question
            )
        )

        if not referenced:

            return None

        # ----------------------------------------------------
        # First resolve the spoken/query spelling against
        # registered people.
        # ----------------------------------------------------

        try:

            reference_resolution = (
                self.metadata_extractor
                .resolve_registered_participant(
                    referenced
                )
            )

        except Exception:

            reference_resolution = None

        referenced_canonical = (
            (
                reference_resolution
                .resolved_name
            )
            if (
                reference_resolution is not None
                and reference_resolution.matched
            )
            else referenced
        )

        # ----------------------------------------------------
        # Compare that canonical person against each stored
        # raw/canonical participant pair.
        # ----------------------------------------------------

        for (
            raw,
            canonical,
        ) in participants:

            if (
                canonical.lower()
                == referenced_canonical.lower()
            ):

                return raw

        return None

    # ========================================================
    # CANONICAL DISPLAY FOR MYSQL RESULT
    # ========================================================

    def _mysql_to_result(
        self,
        conversation: dict[str, Any],
    ) -> ConversationMemoryResult:

        conversation_id = int(
            conversation.get(
                "id"
            )  # type: ignore
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

        summary = (
            str(
                conversation.get(
                    "summary"
                )
                or ""
            )
            .strip()
        )

        raw_participant = (
            str(
                conversation.get(
                    "participant_text"
                )
                or ""
            )
            .strip()
            or None
        )

        canonical_participant = (
            self.canonicalize_participant(
                raw_participant
            )
        )

        text = (
            summary
            if summary
            else transcript
        )

        return (
            ConversationMemoryResult(
                conversation_id=(
                    conversation_id
                ),

                text=(
                    text
                ),

                distance=None,

                metadata={
                    "conversation_id": (
                        conversation_id
                    ),

                    "title": (
                        conversation.get(
                            "title"
                        )
                    ),

                    # -----------------------------------------
                    # Canonical value used in answers/UI.
                    # -----------------------------------------

                    "participant_text": (
                        canonical_participant
                    ),

                    # -----------------------------------------
                    # Preserve historical raw DB spelling.
                    # -----------------------------------------

                    "participant_storage_text": (
                        raw_participant
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

                    "full_transcript": (
                        transcript
                    ),

                    "summary": (
                        summary
                    ),

                    "status": (
                        conversation.get(
                            "status"
                        )
                    ),

                    "memory_type": (
                        "conversation"
                    ),
                },

                source="mysql",
            )
        )

    # ========================================================
    # CHROMA -> RESULT
    # ========================================================

    @staticmethod
    def _semantic_results(
        raw: dict[str, Any],
    ) -> list[
        ConversationMemoryResult
    ]:

        ids = (
            raw.get(
                "ids",
                [[]],
            )[0]
        )

        documents = (
            raw.get(
                "documents",
                [[]],
            )[0]
        )

        metadatas = (
            raw.get(
                "metadatas",
                [[]],
            )[0]
        )

        distances = (
            raw.get(
                "distances",
                [[]],
            )[0]
        )

        results: list[
            ConversationMemoryResult
        ] = []

        for index, memory_id in enumerate(
            ids
        ):

            distance = (
                float(
                    distances[
                        index
                    ]
                )
                if (
                    index
                    < len(
                        distances
                    )
                    and distances[
                        index
                    ]
                    is not None
                )
                else None
            )

            if (
                distance is not None
                and distance
                > MAX_DISTANCE
            ):

                continue

            metadata = (
                metadatas[
                    index
                ]
                if (
                    index
                    < len(
                        metadatas
                    )
                    and metadatas[
                        index
                    ]
                )
                else {}
            )

            document = (
                str(
                    documents[
                        index
                    ]
                )
                if (
                    index
                    < len(
                        documents
                    )
                )
                else ""
            )

            conversation_id = (
                metadata.get(
                    "conversation_id"
                )
            )

            if conversation_id is None:

                try:

                    conversation_id = int(
                        str(
                            memory_id
                        )
                        .replace(
                            "conversation_",
                            "",
                            1,
                        )
                    )

                except ValueError:

                    continue

            results.append(
                ConversationMemoryResult(
                    conversation_id=int(
                        conversation_id
                    ),

                    text=(
                        document
                    ),

                    distance=(
                        distance
                    ),

                    metadata=dict(
                        metadata
                    ),

                    source="chroma",
                )
            )

        return results

    # ========================================================
    # QUERY ACTION
    # ========================================================

    @classmethod
    def _requested_action(
        cls,
        question: str,
    ) -> str | None:
        """
        Detect strong action/topic requirements from a query.

        This is used only as a relevance guard after semantic
        retrieval.

        Example:

            What do I need to buy tomorrow?
                -> buy

        A candidate appointment which only contains
        "tomorrow" must then be rejected.
        """

        lowered = (
            str(
                question
                or ""
            )
            .strip()
            .lower()
        )

        if not lowered:

            return None

        # ----------------------------------------------------
        # Strong actions first.
        # ----------------------------------------------------

        action_queries = {

            "buy": (
                r"\bbuy\b",
                r"\bpurchase\b",
                r"\bshopping\b",
                r"\bgrocer(?:y|ies)\b",
            ),

            "bring": (
                r"\bbring\b",
                r"\bcarry\b",
            ),

            "pay": (
                r"\bpay\b",
                r"\bbill\b",
            ),

            "call": (
                r"\bcall\b",
                r"\bphone\b",
            ),

            "collect": (
                r"\bcollect\b",
                r"\bpick\s+up\b",
            ),

            "send": (
                r"\bsend\b",
            ),

            "submit": (
                r"\bsubmit\b",
            ),

            "visit": (
                r"\bvisit\b",
                r"\bvisiting\b",
                r"\bcoming\s+over\b",
                r"\bcome\s+over\b",
            ),
        }

        for (
            action,
            patterns,
        ) in action_queries.items():

            if any(
                re.search(
                    pattern,
                    lowered,
                )
                is not None

                for pattern
                in patterns
            ):

                return action

        # ----------------------------------------------------
        # Appointment only becomes required evidence when the
        # question itself is asking about the appointment.
        # ----------------------------------------------------

        if (
            "appointment"
            in lowered
            or "check-up"
            in lowered
            or "checkup"
            in lowered
        ):

            return "appointment"

        return None

    # ========================================================
    # RESULT EVIDENCE TEXT
    # ========================================================

    @staticmethod
    def _result_evidence_text(
        result: ConversationMemoryResult,
    ) -> str:

        metadata = (
            result.metadata
        )

        return (
            " ".join(
                [
                    str(
                        result.text
                        or ""
                    ),

                    str(
                        metadata.get(
                            "title"
                        )
                        or ""
                    ),

                    str(
                        metadata.get(
                            "summary"
                        )
                        or ""
                    ),

                    str(
                        metadata.get(
                            "full_transcript"
                        )
                        or ""
                    ),
                ]
            )
            .strip()
            .lower()
        )

    # ========================================================
    # VALIDATE SEMANTIC EVIDENCE
    # ========================================================

    @classmethod
    def _result_matches_requested_action(
        cls,
        result: ConversationMemoryResult,
        action: str | None,
    ) -> bool:
        """
        If the user asked a strongly action-specific question,
        require evidence for that action in the retrieved
        conversation.

        This prevents:

            Question:
                What do I need to buy tomorrow?

            Wrong match:
                Doctor appointment tomorrow at 10 AM.
        """

        if not action:

            return True

        patterns = (
            cls.ACTION_EVIDENCE_PATTERNS.get(
                action
            )
        )

        if not patterns:

            return True

        evidence = (
            cls._result_evidence_text(
                result
            )
        )

        return any(
            re.search(
                pattern,
                evidence,
                flags=re.IGNORECASE,
            )
            is not None

            for pattern
            in patterns
        )

    # ========================================================
    # FILTER SEMANTIC RESULTS
    # ========================================================

    @classmethod
    def _filter_by_requested_action(
        cls,
        results: list[
            ConversationMemoryResult
        ],
        question: str,
    ) -> list[
        ConversationMemoryResult
    ]:

        action = (
            cls._requested_action(
                question
            )
        )

        if not action:

            return results

        return [
            result

            for result in results

            if cls._result_matches_requested_action(
                result,
                action,
            )
        ]

    # ========================================================
    # DATE RETRIEVAL
    # ========================================================

    def _retrieve_for_date(
        self,
        target_date: date,
        participant: str | None = None,
    ) -> list[
        ConversationMemoryResult
    ]:

        conversations = (
            get_conversations_for_date(
                target_date=(
                    target_date
                ),
                completed_only=True,
            )
        )

        if participant:

            participant_lower = (
                participant.lower()
            )

            conversations = [
                conversation

                for conversation
                in conversations

                if (
                    str(
                        conversation.get(
                            "participant_text"
                        )
                        or ""
                    )
                    .strip()
                    .lower()
                    == participant_lower
                )
            ]

        return [
            self._mysql_to_result(
                conversation
            )

            for conversation
            in conversations

            if str(
                conversation.get(
                    "full_transcript"
                )
                or ""
            ).strip()
        ]

    # ========================================================
    # RECENT
    # ========================================================

    def _retrieve_recent(
        self,
        participant: str | None,
        limit: int,
    ) -> list[
        ConversationMemoryResult
    ]:

        conversations = (
            get_recent_conversations(
                limit=max(
                    limit,
                    50,
                ),
                completed_only=True,
            )
        )

        if participant:

            participant_lower = (
                participant.lower()
            )

            conversations = [
                conversation

                for conversation
                in conversations

                if (
                    str(
                        conversation.get(
                            "participant_text"
                        )
                        or ""
                    )
                    .strip()
                    .lower()
                    == participant_lower
                )
            ]

        return [
            self._mysql_to_result(
                conversation
            )

            for conversation
            in conversations[
                :limit
            ]

            if str(
                conversation.get(
                    "full_transcript"
                )
                or ""
            ).strip()
        ]

    # ========================================================
    # MYSQL TEXT SEARCH
    # ========================================================

    def _retrieve_mysql_search(
        self,
        question: str,
        participant: str | None,
        limit: int,
    ) -> list[
        ConversationMemoryResult
    ]:

        conversations = (
            search_conversations(
                search_text=(
                    question
                ),
                limit=max(
                    limit,
                    20,
                ),
            )
        )

        if participant:

            participant_lower = (
                participant.lower()
            )

            conversations = [
                conversation

                for conversation
                in conversations

                if (
                    str(
                        conversation.get(
                            "participant_text"
                        )
                        or ""
                    )
                    .strip()
                    .lower()
                    == participant_lower
                )
            ]

        results = [
            self._mysql_to_result(
                conversation
            )

            for conversation
            in conversations[
                :limit
            ]

            if str(
                conversation.get(
                    "full_transcript"
                )
                or ""
            ).strip()
        ]

        return (
            self._filter_by_requested_action(
                results,
                question,
            )
        )

    # ========================================================
    # SEMANTIC
    # ========================================================

    def _retrieve_semantic(
        self,
        question: str,
        participant: str | None,
        top_k: int,
    ) -> list[
        ConversationMemoryResult
    ]:

        raw = (
            self.vector_memory.search(
                query=(
                    question
                ),
                n_results=(
                    top_k
                ),
                participant_text=(
                    participant
                ),
            )
        )

        return (
            self._semantic_results(
                raw
            )
        )

    # ========================================================
    # DEDUPLICATE
    # ========================================================

    @staticmethod
    def _deduplicate(
        results: list[
            ConversationMemoryResult
        ],
    ) -> list[
        ConversationMemoryResult
    ]:

        deduplicated: list[
            ConversationMemoryResult
        ] = []

        seen_ids: set[int] = set()

        for result in results:

            if (
                result.conversation_id
                in seen_ids
            ):

                continue

            seen_ids.add(
                result.conversation_id
            )

            deduplicated.append(
                result
            )

        return deduplicated

    # ========================================================
    # HYDRATE SEMANTIC RESULTS
    # ========================================================

    def _hydrate(
        self,
        results: list[
            ConversationMemoryResult
        ],
    ) -> list[
        ConversationMemoryResult
    ]:

        from conversation.conversation_repository import (
            get_conversation_session,
        )

        hydrated: list[
            ConversationMemoryResult
        ] = []

        for result in results:

            conversation = (
                get_conversation_session(
                    result.conversation_id
                )
            )

            if not conversation:

                continue

            mysql_result = (
                self._mysql_to_result(
                    conversation
                )
            )

            mysql_result.distance = (
                result.distance
            )

            mysql_result.source = (
                "chroma+mysql"
            )

            hydrated.append(
                mysql_result
            )

        return hydrated

    # ========================================================
    # RETRIEVE
    # ========================================================

    def retrieve(
        self,
        question: str,
        top_k: int = DEFAULT_TOP_K,
    ) -> list[
        ConversationMemoryResult
    ]:

        cleaned = (
            str(
                question
                or ""
            )
            .strip()
        )

        if not cleaned:

            return []

        lowered = (
            cleaned.lower()
        )

        # ====================================================
        # PARTICIPANT RESOLUTION
        # ====================================================

        referenced_participant = (
            self.extract_referenced_participant(
                cleaned
            )
        )

        participant = (
            self.resolve_participant(
                cleaned
            )
        )

        # ====================================================
        # UNKNOWN NAMED PARTICIPANT SAFETY GATE
        # ====================================================

        if (
            referenced_participant
            and participant is None
        ):

            return []

        # ====================================================
        # TODAY
        # ====================================================

        if "today" in lowered:

            results = (
                self._retrieve_for_date(
                    target_date=(
                        date.today()
                    ),
                    participant=(
                        participant
                    ),
                )
            )

            results = (
                self._filter_by_requested_action(
                    results,
                    cleaned,
                )
            )

            return (
                results[
                    :top_k
                ]
            )

        # ====================================================
        # YESTERDAY
        # ====================================================

        if "yesterday" in lowered:

            results = (
                self._retrieve_for_date(
                    target_date=(
                        date.today()
                        - timedelta(
                            days=1
                        )
                    ),
                    participant=(
                        participant
                    ),
                )
            )

            results = (
                self._filter_by_requested_action(
                    results,
                    cleaned,
                )
            )

            return (
                results[
                    :top_k
                ]
            )

        # ====================================================
        # RECENT / LAST
        # ====================================================

        recent_phrases = (
            "recent conversation",
            "recent conversations",
            "latest conversation",
            "last conversation",
            "what did we talk about recently",
        )

        if any(
            phrase in lowered
            for phrase in recent_phrases
        ):

            results = (
                self._retrieve_recent(
                    participant=(
                        participant
                    ),
                    limit=(
                        top_k
                    ),
                )
            )

            return (
                self._filter_by_requested_action(
                    results,
                    cleaned,
                )
            )

        # ====================================================
        # SEMANTIC
        # ====================================================

        semantic = (
            self._retrieve_semantic(
                question=(
                    cleaned
                ),
                participant=(
                    participant
                ),
                top_k=max(
                    top_k,
                    10,
                ),
            )
        )

        semantic = (
            self._hydrate(
                semantic
            )
        )

        semantic = (
            self._filter_by_requested_action(
                semantic,
                cleaned,
            )
        )

        if semantic:

            return (
                self._deduplicate(
                    semantic
                )[
                    :top_k
                ]
            )

        # ====================================================
        # MYSQL FALLBACK
        # ====================================================

        mysql_results = (
            self._retrieve_mysql_search(
                question=(
                    cleaned
                ),
                participant=(
                    participant
                ),
                limit=(
                    top_k
                ),
            )
        )

        return (
            self._deduplicate(
                mysql_results
            )[
                :top_k
            ]
        )