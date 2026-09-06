from __future__ import annotations

import re

from dataclasses import (
    dataclass,
)

from difflib import (
    SequenceMatcher,
)

from database.person_repository import (
    get_all_people,
)


# ============================================================
# METADATA RESULT
# ============================================================

@dataclass(frozen=True)
class ConversationMetadataResult:

    title: str | None

    participant_text: str | None

    summary: str


# ============================================================
# PARTICIPANT RESOLUTION RESULT
# ============================================================

@dataclass(frozen=True)
class ParticipantResolutionResult:
    """
    Result of resolving a transcription-derived name against
    MemorAi's registered people database.
    """

    original_text: str

    resolved_name: str

    person_id: int | None

    similarity: float

    matched: bool


# ============================================================
# METADATA EXTRACTOR
# ============================================================

class ConversationMetadataExtractor:
    """
    Extract useful metadata from important conversations.

    Privacy:

        This runs only after a conversation has already
        passed the importance filter.

    Participant handling:

        Whisper transcript
                ↓
        Extract possible spoken name
                ↓
        Compare against registered people
                ↓
        Safe unambiguous match?
          │                 │
         Yes                No
          │                 │
    Registered name      Keep transcript name

    Example:

        Whisper:
            Saksi

        Registered:
            Sakshi

        Result:
            Sakshi

    The original Whisper transcript itself is NOT modified.
    """

    # ========================================================
    # PARTICIPANT MATCHING
    # ========================================================

    PARTICIPANT_MATCH_THRESHOLD = 0.78

    PARTICIPANT_AMBIGUITY_MARGIN = 0.08


    # ========================================================
    # PARTICIPANT PATTERNS
    # ========================================================

    PARTICIPANT_PATTERNS = (

        r"^\s*([A-Z][A-Za-z]{1,40})\s+told\s+me\b",

        r"^\s*([A-Z][A-Za-z]{1,40})\s+said\b",

        r"^\s*([A-Z][A-Za-z]{1,40})\s+reminded\s+me\b",

        r"^\s*([A-Z][A-Za-z]{1,40})\s+mentioned\b",

        r"^\s*([A-Z][A-Za-z]{1,40})\s+asked\s+me\b",

        r"\b(?:with|from)\s+([A-Z][A-Za-z]{1,40})\b",
    )


    # ========================================================
    # TITLE RULES
    # ========================================================

    TITLE_RULES = (

        # ----------------------------------------------------
        # More specific rules should come first.
        # ----------------------------------------------------

        (
            (
                "dentist",
                "dental",
            ),
            "Dental Appointment",
        ),

        (
            (
                "physiotherapy",
                "physio",
            ),
            "Physiotherapy Appointment",
        ),

        (
            (
                "medicine",
                "medication",
                "tablet",
                "pill",
                "dose",
                "prescription",
            ),
            "Medicine Reminder",
        ),

        (
            (
                "doctor",
                "hospital",
                "clinic",
                "appointment",
                "check-up",
                "checkup",
            ),
            "Medical Appointment",
        ),

        (
            (
                "passport",
                "document",
                "documents",
            ),
            "Important Document Reminder",
        ),

        (
            (
                "buy",
                "shopping",
                "groceries",
                "milk",
                "bread",
            ),
            "Shopping Reminder",
        ),

        (
            (
                "meeting",
                "office",
                "work",
            ),
            "Meeting Reminder",
        ),

        (
            (
                "visit",
                "coming over",
                "come over",
            ),
            "Visit Plan",
        ),

        (
            (
                "call",
                "phone",
            ),
            "Call Reminder",
        ),
    )


    # ========================================================
    # CLEAN TEXT
    # ========================================================

    @staticmethod
    def _clean_text(
        value: str | None,
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


    # ========================================================
    # NORMALIZE WHITESPACE
    # ========================================================

    @staticmethod
    def _normalize_whitespace(
        text: str,
    ) -> str:

        return (
            re.sub(
                r"\s+",
                " ",
                str(
                    text
                ),
            )
            .strip()
        )


    # ========================================================
    # NORMALIZE PERSON NAME
    # ========================================================

    @staticmethod
    def _normalize_person_name(
        value: str,
    ) -> str:
        """
        Normalize a name only for comparison.

        The actual stored/display spelling is not changed
        by this function.
        """

        cleaned = (
            str(
                value
                or ""
            )
            .strip()
            .lower()
        )

        cleaned = re.sub(
            r"[^a-z0-9]+",
            "",
            cleaned,
        )

        return (
            cleaned
        )


    # ========================================================
    # NAME SIMILARITY
    # ========================================================

    @classmethod
    def _name_similarity(
        cls,
        first: str,
        second: str,
    ) -> float:

        first_normalized = (
            cls._normalize_person_name(
                first
            )
        )

        second_normalized = (
            cls._normalize_person_name(
                second
            )
        )

        if (
            not first_normalized
            or not second_normalized
        ):

            return 0.0

        # ----------------------------------------------------
        # EXACT MATCH
        # ----------------------------------------------------

        if (
            first_normalized
            == second_normalized
        ):

            return 1.0

        # ----------------------------------------------------
        # SPELLING SIMILARITY
        # ----------------------------------------------------

        similarity = (
            SequenceMatcher(
                None,
                first_normalized,
                second_normalized,
            )
            .ratio()
        )

        return float(
            similarity
        )


    # ========================================================
    # GET REGISTERED PEOPLE
    # ========================================================

    @staticmethod
    def _get_registered_people(
    ) -> list[dict]:
        """
        Include inactive people too because an old historical
        conversation may refer to somebody who was later
        deactivated.
        """

        try:

            return (
                get_all_people(
                    active_only=False
                )
            )

        except Exception:

            # ------------------------------------------------
            # Conversation metadata should not completely fail
            # merely because person lookup temporarily fails.
            # ------------------------------------------------

            return []


    # ========================================================
    # CANONICAL PERSON NAME
    # ========================================================

    @staticmethod
    def _canonical_person_name(
        person: dict,
    ) -> str:
        """
        Prefer display_name.

        Fall back to internal name if display_name is empty.
        """

        display_name = (
            str(
                person.get(
                    "display_name"
                )
                or ""
            )
            .strip()
        )

        if display_name:

            return (
                display_name
            )

        name = (
            str(
                person.get(
                    "name"
                )
                or ""
            )
            .strip()
        )

        return (
            name
        )


    # ========================================================
    # RESOLVE REGISTERED PARTICIPANT
    # ========================================================

    def resolve_registered_participant(
        self,
        participant_text: str | None,
    ) -> ParticipantResolutionResult | None:
        """
        Compare a possible participant against registered
        people.

        Exact matches resolve immediately.

        Fuzzy matches resolve only when:

            similarity >= threshold

        AND

            the best result is clearly better than
            the second best result.

        This prevents MemorAi from changing an uncertain
        name into the wrong registered person.
        """

        cleaned = (
            self._clean_text(
                participant_text
            )
        )

        if not cleaned:

            return None

        people = (
            self._get_registered_people()
        )

        # ====================================================
        # NO REGISTERED PEOPLE AVAILABLE
        # ====================================================

        if not people:

            return (
                ParticipantResolutionResult(
                    original_text=(
                        cleaned
                    ),

                    resolved_name=(
                        cleaned
                    ),

                    person_id=None,

                    similarity=0.0,

                    matched=False,
                )
            )

        # ====================================================
        # BUILD CANDIDATES
        # ====================================================

        candidates: list[
            tuple[
                float,
                dict,
                str,
            ]
        ] = []

        for person in people:

            canonical_name = (
                self._canonical_person_name(
                    person
                )
            )

            if not canonical_name:

                continue

            # ------------------------------------------------
            # Compare against both display_name and database
            # name where available.
            # ------------------------------------------------

            possible_names = [
                canonical_name
            ]

            internal_name = (
                str(
                    person.get(
                        "name"
                    )
                    or ""
                )
                .strip()
            )

            if (
                internal_name
                and internal_name
                not in possible_names
            ):

                possible_names.append(
                    internal_name
                )

            best_person_similarity = 0.0

            for possible_name in (
                possible_names
            ):

                similarity = (
                    self._name_similarity(
                        cleaned,
                        possible_name,
                    )
                )

                best_person_similarity = max(
                    best_person_similarity,
                    similarity,
                )

            candidates.append(
                (
                    best_person_similarity,
                    person,
                    canonical_name,
                )
            )

        # ====================================================
        # NOTHING TO COMPARE
        # ====================================================

        if not candidates:

            return (
                ParticipantResolutionResult(
                    original_text=(
                        cleaned
                    ),

                    resolved_name=(
                        cleaned
                    ),

                    person_id=None,

                    similarity=0.0,

                    matched=False,
                )
            )

        # ====================================================
        # BEST MATCH FIRST
        # ====================================================

        candidates.sort(
            key=lambda item: (
                item[0]
            ),
            reverse=True,
        )

        (
            best_similarity,
            best_person,
            best_name,
        ) = candidates[0]

        second_similarity = (
            candidates[1][0]
            if len(
                candidates
            ) > 1
            else 0.0
        )

        person_id = (
            int(
                best_person[
                    "id"
                ]
            )
            if best_person.get(
                "id"
            )
            is not None
            else None
        )

        # ====================================================
        # EXACT MATCH
        # ====================================================

        if (
            best_similarity
            >= 0.999
        ):

            return (
                ParticipantResolutionResult(
                    original_text=(
                        cleaned
                    ),

                    resolved_name=(
                        best_name
                    ),

                    person_id=(
                        person_id
                    ),

                    similarity=(
                        best_similarity
                    ),

                    matched=True,
                )
            )

        # ====================================================
        # TOO DIFFERENT
        # ====================================================

        if (
            best_similarity
            < self.PARTICIPANT_MATCH_THRESHOLD
        ):

            return (
                ParticipantResolutionResult(
                    original_text=(
                        cleaned
                    ),

                    resolved_name=(
                        cleaned
                    ),

                    person_id=None,

                    similarity=(
                        best_similarity
                    ),

                    matched=False,
                )
            )

        # ====================================================
        # AMBIGUOUS
        # ====================================================

        difference = (
            best_similarity
            - second_similarity
        )

        if (
            len(
                candidates
            ) > 1
            and difference
            < self.PARTICIPANT_AMBIGUITY_MARGIN
        ):

            return (
                ParticipantResolutionResult(
                    original_text=(
                        cleaned
                    ),

                    resolved_name=(
                        cleaned
                    ),

                    person_id=None,

                    similarity=(
                        best_similarity
                    ),

                    matched=False,
                )
            )

        # ====================================================
        # SAFE FUZZY MATCH
        # ====================================================

        return (
            ParticipantResolutionResult(
                original_text=(
                    cleaned
                ),

                resolved_name=(
                    best_name
                ),

                person_id=(
                    person_id
                ),

                similarity=(
                    best_similarity
                ),

                matched=True,
            )
        )


    # ========================================================
    # DISPLAY PARTICIPANT NAME
    # ========================================================

    def display_participant_name(
        self,
        participant_text: str | None,
    ) -> str | None:
        """
        Resolve old stored participant spellings for display.

        Example:

            Database:
                Saksi

            Registered people:
                Sakshi

            UI:
                Sakshi

        The existing database value and original transcript
        are not modified by this method.
        """

        cleaned = (
            self._clean_text(
                participant_text
            )
        )

        if not cleaned:

            return None

        resolution = (
            self.resolve_registered_participant(
                cleaned
            )
        )

        if (
            resolution is not None
            and resolution.matched
        ):

            return (
                resolution.resolved_name
            )

        return (
            cleaned
        )


    # ========================================================
    # EXTRACT PARTICIPANT
    # ========================================================

    def extract_participant(
        self,
        transcript: str,
    ) -> str | None:
        """
        Find a possible name in the transcript and resolve it
        against registered people where safe.
        """

        cleaned = (
            self._normalize_whitespace(
                transcript
            )
        )

        if not cleaned:

            return None

        for pattern in (
            self.PARTICIPANT_PATTERNS
        ):

            match = (
                re.search(
                    pattern,
                    cleaned,
                )
            )

            if not match:

                continue

            extracted_name = (
                match.group(
                    1
                )
                .strip()
            )

            if not extracted_name:

                continue

            # ------------------------------------------------
            # PROTECT COMMON NON-NAMES
            # ------------------------------------------------

            if extracted_name.lower() in {
                "i",
                "he",
                "she",
                "they",
                "we",
                "doctor",
                "someone",
                "anyone",
            }:

                continue

            # ------------------------------------------------
            # RESOLVE AGAINST REGISTERED PEOPLE
            # ------------------------------------------------

            resolution = (
                self.resolve_registered_participant(
                    extracted_name
                )
            )

            if (
                resolution is not None
                and resolution.matched
            ):

                return (
                    resolution.resolved_name
                )

            # ------------------------------------------------
            # NO SAFE REGISTERED MATCH
            #
            # Preserve what Whisper actually heard.
            # ------------------------------------------------

            return (
                extracted_name
            )

        return None


    # ========================================================
    # GENERATE TITLE
    # ========================================================

    def generate_title(
        self,
        transcript: str,
        matched_categories: (
            tuple[str, ...]
            | list[str]
            | None
        ) = None,
    ) -> str:

        lowered = (
            self._normalize_whitespace(
                transcript
            )
            .lower()
        )

        # ====================================================
        # SPECIALIZED TITLE RULES
        # ====================================================

        for (
            keywords,
            title,
        ) in self.TITLE_RULES:

            if any(
                keyword in lowered
                for keyword
                in keywords
            ):

                return (
                    title
                )

        # ====================================================
        # CATEGORY FALLBACK
        # ====================================================

        categories = {
            str(
                category
            )
            .strip()
            .lower()

            for category
            in (
                matched_categories
                or ()
            )

            if str(
                category
            ).strip()
        }

        if "appointment" in categories:

            return (
                "Important Appointment"
            )

        if "medicine" in categories:

            return (
                "Medicine Reminder"
            )

        if "location_memory" in categories:

            return (
                "Important Location Reminder"
            )

        if "task" in categories:

            return (
                "Important Task"
            )

        if "commitment" in categories:

            return (
                "Important Plan"
            )

        if "reminder" in categories:

            return (
                "Important Reminder"
            )

        return (
            "Important Conversation"
        )


    # ========================================================
    # SPLIT SENTENCES
    # ========================================================

    @staticmethod
    def _split_sentences(
        transcript: str,
    ) -> list[str]:

        cleaned = (
            ConversationMetadataExtractor
            ._normalize_whitespace(
                transcript
            )
        )

        if not cleaned:

            return []

        parts = (
            re.split(
                r"(?<=[.!?])\s+",
                cleaned,
            )
        )

        return [
            part.strip()

            for part
            in parts

            if part.strip()
        ]


    # ========================================================
    # CLEAN SUMMARY SENTENCE
    # ========================================================

    @staticmethod
    def _clean_summary_sentence(
        sentence: str,
    ) -> str:

        cleaned = (
            str(
                sentence
            )
            .strip()
        )

        # ----------------------------------------------------
        # Example:
        #
        # Saksi told me that my doctor appointment...
        #
        # becomes:
        #
        # My doctor appointment...
        # ----------------------------------------------------

        cleaned = (
            re.sub(
                (
                    r"^[A-Z][A-Za-z]{1,40}\s+"
                    r"(?:told me that|said that|"
                    r"reminded me that|mentioned that)\s+"
                ),
                "",
                cleaned,
            )
        )

        if cleaned:

            cleaned = (
                cleaned[0].upper()
                + cleaned[1:]
            )

        return (
            cleaned
        )


    # ========================================================
    # GENERATE SUMMARY
    # ========================================================

    def generate_summary(
        self,
        transcript: str,
        max_sentences: int = 3,
    ) -> str:

        sentences = (
            self._split_sentences(
                transcript
            )
        )

        if not sentences:

            return ""

        selected: list[str] = []

        important_words = (
            "appointment",
            "doctor",
            "hospital",
            "medicine",
            "medication",
            "remember",
            "remind",
            "need to",
            "must",
            "bring",
            "buy",
            "call",
            "visit",
            "meeting",
            "tomorrow",
            "today",
            "monday",
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
            "saturday",
            "sunday",
            "kept",
            "put",
            "left",
            "arrive",
            "pick me up",
            "pick up",
        )

        # ====================================================
        # PRIORITIZE USEFUL SENTENCES
        # ====================================================

        for sentence in sentences:

            lowered = (
                sentence.lower()
            )

            if any(
                word in lowered
                for word
                in important_words
            ):

                cleaned_sentence = (
                    self._clean_summary_sentence(
                        sentence
                    )
                )

                if (
                    cleaned_sentence
                    and cleaned_sentence
                    not in selected
                ):

                    selected.append(
                        cleaned_sentence
                    )

            if len(
                selected
            ) >= max_sentences:

                break

        # ====================================================
        # FALLBACK
        # ====================================================

        if not selected:

            for sentence in sentences[
                :max_sentences
            ]:

                cleaned_sentence = (
                    self._clean_summary_sentence(
                        sentence
                    )
                )

                if cleaned_sentence:

                    selected.append(
                        cleaned_sentence
                    )

        return (
            " ".join(
                selected
            )
            .strip()
        )


    # ========================================================
    # EXTRACT ALL
    # ========================================================

    def extract(
        self,
        transcript: str,
        provided_title: str | None = None,
        provided_participant: str | None = None,
        matched_categories: (
            tuple[str, ...]
            | list[str]
            | None
        ) = None,
    ) -> ConversationMetadataResult:

        cleaned_transcript = (
            self._normalize_whitespace(
                transcript
            )
        )

        if not cleaned_transcript:

            raise ValueError(
                (
                    "Cannot extract conversation "
                    "metadata from an empty transcript."
                )
            )

        explicit_title = (
            self._clean_text(
                provided_title
            )
        )

        explicit_participant = (
            self._clean_text(
                provided_participant
            )
        )

        # ====================================================
        # TITLE
        # ====================================================

        final_title = (
            explicit_title
            or self.generate_title(
                transcript=(
                    cleaned_transcript
                ),

                matched_categories=(
                    matched_categories
                ),
            )
        )

        # ====================================================
        # PARTICIPANT
        # ====================================================

        if explicit_participant:

            # ------------------------------------------------
            # The user entered a participant explicitly.
            #
            # Prefer their value, but normalize it to the
            # registered display name when it clearly matches
            # an existing registered person.
            # ------------------------------------------------

            resolution = (
                self.resolve_registered_participant(
                    explicit_participant
                )
            )

            if (
                resolution is not None
                and resolution.matched
            ):

                final_participant = (
                    resolution.resolved_name
                )

            else:

                final_participant = (
                    explicit_participant
                )

        else:

            final_participant = (
                self.extract_participant(
                    cleaned_transcript
                )
            )

        # ====================================================
        # SUMMARY
        # ====================================================

        summary = (
            self.generate_summary(
                cleaned_transcript
            )
        )

        # ====================================================
        # RESULT
        # ====================================================

        return (
            ConversationMetadataResult(
                title=(
                    final_title
                ),

                participant_text=(
                    final_participant
                ),

                summary=(
                    summary
                ),
            )
        )