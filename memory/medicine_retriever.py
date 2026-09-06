from __future__ import annotations

import re

from datetime import (
    date,
    datetime,
    timedelta,
)

from typing import Any

from database.medicine_repository import (
    STATUS_CAMERA_CONFIRMED,
    get_all_medicines,
    get_latest_medicine_event,
    get_medicine_events_for_date,
    get_recent_medicine_events,
)

from rag.medicine_memory_search import (
    MedicineMemorySearch,
    MedicineMemorySearchResult,
)


# ============================================================
# MEDICINE MEMORY RETRIEVER
# ============================================================

class MedicineMemoryRetriever:
    """
    Hybrid retriever for MemorAi medicine memories.

    MySQL is authoritative for:

        - Did I take Omez today?
        - Was Omez confirmed today?
        - Was Omez confirmed yesterday?
        - When was Omez last confirmed?
        - Which medicines were confirmed today?
        - What medicine was confirmed recently?

    ChromaDB is used as a semantic fallback for less
    structured natural-language medicine-memory questions.

    IMPORTANT SAFETY RULE:

    "camera_confirmed" means MemorAi observed sufficient
    camera evidence of a medication-taking event.

    It does NOT prove that medicine was swallowed.
    """

    # ========================================================
    # INITIALIZE
    # ========================================================

    def __init__(
        self,
    ) -> None:

        self.semantic_search = (
            MedicineMemorySearch()
        )

    # ========================================================
    # NORMALIZE
    # ========================================================

    @staticmethod
    def _normalize_text(
        value: str,
    ) -> str:

        return " ".join(
            str(
                value
                or ""
            )
            .strip()
            .lower()
            .split()
        )

    # ========================================================
    # PHRASE MATCH
    # ========================================================

    @staticmethod
    def _contains_phrase(
        text: str,
        phrase: str,
    ) -> bool:

        normalized_phrase = (
            MedicineMemoryRetriever
            ._normalize_text(
                phrase
            )
        )

        if not normalized_phrase:

            return False

        pattern = (
            r"(?<!\w)"
            + re.escape(
                normalized_phrase
            )
            + r"(?!\w)"
        )

        return bool(
            re.search(
                pattern,
                text,
            )
        )

    # ========================================================
    # REGISTERED MEDICINES
    # ========================================================

    @staticmethod
    def _registered_medicines(
    ) -> list[dict[str, Any]]:

        return (
            get_all_medicines(
                active_only=False
            )
        )

    # ========================================================
    # RESOLVE MEDICINE FROM QUESTION
    # ========================================================

    def resolve_medicine(
        self,
        question: str,
    ) -> dict[str, Any] | None:

        lowered = (
            self._normalize_text(
                question
            )
        )

        if not lowered:

            return None

        medicines = (
            self._registered_medicines()
        )

        candidates: list[
            tuple[
                int,
                dict[str, Any],
            ]
        ] = []

        for medicine in medicines:

            medicine_name = (
                str(
                    medicine.get(
                        "medicine_name"
                    )
                    or ""
                )
                .strip()
            )

            dosage = (
                str(
                    medicine.get(
                        "dosage"
                    )
                    or ""
                )
                .strip()
            )

            ocr_keywords = (
                str(
                    medicine.get(
                        "ocr_keywords"
                    )
                    or ""
                )
                .strip()
            )

            aliases: list[str] = []

            if medicine_name:

                aliases.append(
                    medicine_name
                )

            if (
                medicine_name
                and dosage
            ):

                aliases.append(
                    f"{medicine_name} {dosage}"
                )

            if ocr_keywords:

                aliases.extend(
                    [
                        item.strip()
                        for item in (
                            ocr_keywords
                            .replace(
                                "\n",
                                ",",
                            )
                            .split(
                                ","
                            )
                        )
                        if item.strip()
                    ]
                )

            matched_aliases: list[str] = []

            for alias in aliases:

                if self._contains_phrase(
                    lowered,
                    alias,
                ):

                    matched_aliases.append(
                        alias
                    )

            if not matched_aliases:

                continue

            best_alias_length = max(
                len(alias)
                for alias in matched_aliases
            )

            candidates.append(
                (
                    best_alias_length,
                    medicine,
                )
            )

        if not candidates:

            return None

        candidates.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        return candidates[0][1]

    # ========================================================
    # UNKNOWN SPECIFIC MEDICINE DETECTION
    # ========================================================

    def has_unresolved_specific_medicine_reference(
        self,
        question: str,
    ) -> bool:
        """
        Detect when the user appears to name a specific
        medicine but that name cannot be resolved.

        Example:

            Did I take FakeDrug today?

        FakeDrug is specific, but is not registered.

        This MUST NOT become:

            Show every medicine event from today.

        Generic questions remain allowed:

            Did I take my medicine today?
            Which medicines were confirmed today?
            What medication events happened recently?
        """

        cleaned = (
            self._normalize_text(
                question
            )
        )

        if not cleaned:

            return False

        # ----------------------------------------------------
        # Already registered
        # ----------------------------------------------------

        if (
            self.resolve_medicine(
                question
            )
            is not None
        ):

            return False

        # ----------------------------------------------------
        # Generic medicine references
        # ----------------------------------------------------

        generic_targets = {
            "medicine",
            "medicines",
            "medication",
            "medications",
            "pill",
            "pills",
            "tablet",
            "tablets",
            "capsule",
            "capsules",
            "dose",
            "dosage",

            "my medicine",
            "my medicines",
            "my medication",
            "my medications",
            "my pill",
            "my pills",
            "my tablet",
            "my tablets",
            "my capsule",
            "my capsules",
            "my dose",

            "any medicine",
            "any medication",
            "any pill",
            "any tablet",
            "any capsule",
        }

        # ----------------------------------------------------
        # Patterns which normally contain a specific target
        # ----------------------------------------------------

        patterns = (
            r"\bdid i take\s+(.+?)(?:\s+today|\s+yesterday|\?|$)",
            r"\bdid i have\s+(.+?)(?:\s+today|\s+yesterday|\?|$)",
            r"\bhave i taken\s+(.+?)(?:\s+today|\s+yesterday|\?|$)",
            r"\bhave i had\s+(.+?)(?:\s+today|\s+yesterday|\?|$)",
            r"\bwhen was\s+(.+?)\s+last confirmed\b",
            r"\bwhen did i (?:take|have)\s+(.+?)(?:\?|$)",
            r"\bwas\s+(.+?)\s+(?:camera\s+)?confirmed(?:\s+today|\s+yesterday|\?|$)",
            r"\bis\s+(.+?)\s+(?:camera\s+)?confirmed(?:\s+today|\s+yesterday|\?|$)",
            r"\bdid memorai confirm\s+(.+?)(?:\s+today|\s+yesterday|\?|$)",
        )

        for pattern in patterns:

            match = re.search(
                pattern,
                cleaned,
            )

            if not match:

                continue

            candidate = (
                self._normalize_text(
                    match.group(1)
                )
            )

            # ------------------------------------------------
            # Remove harmless articles
            # ------------------------------------------------

            candidate = re.sub(
                r"^(?:the|a|an)\s+",
                "",
                candidate,
            )

            # ------------------------------------------------
            # Example:
            #
            # FakeDrug tablet
            #
            # becomes:
            #
            # FakeDrug
            # ------------------------------------------------

            candidate = re.sub(
                (
                    r"\s+(?:medicine|medication|"
                    r"tablet|pill|capsule)$"
                ),
                "",
                candidate,
            ).strip()

            if not candidate:

                return False

            if candidate in generic_targets:

                return False

            if candidate in {
                "my",
                "any",
                "a",
                "the",
            }:

                return False

            # ------------------------------------------------
            # We found something that looks like a medicine
            # name but resolve_medicine() could not resolve it.
            # ------------------------------------------------

            return True

        return False

    # ========================================================
    # MEDICINE QUESTION DETECTION
    # ========================================================

    def is_medicine_question(
        self,
        question: str,
    ) -> bool:

        cleaned = (
            self._normalize_text(
                question
            )
        )

        if not cleaned:

            return False

        # ----------------------------------------------------
        # REGISTERED MEDICINE
        # ----------------------------------------------------

        if (
            self.resolve_medicine(
                question
            )
            is not None
        ):

            return True

        # ----------------------------------------------------
        # UNKNOWN BUT SPECIFIC MEDICINE
        #
        # Important:
        # Route this to Medicine Memory so that chat.py can
        # safely reject it instead of accidentally routing it
        # into object/conversation memory.
        # ----------------------------------------------------

        if (
            self.has_unresolved_specific_medicine_reference(
                question
            )
        ):

            return True

        # ----------------------------------------------------
        # EXPLICIT MEDICATION LANGUAGE
        # ----------------------------------------------------

        phrases = (
            "medicine",
            "medicines",
            "medication",
            "medications",
            "tablet",
            "tablets",
            "pill",
            "pills",
            "capsule",
            "capsules",
            "dose",
            "dosage",

            "take my medicine",
            "took my medicine",
            "taken my medicine",
            "take medicine",
            "took medicine",
            "taken medicine",

            "medicine confirmed",
            "medication confirmed",
            "camera confirmed medicine",
            "camera confirmed medication",

            "morning medicine",
            "afternoon medicine",
            "evening medicine",
            "night medicine",

            "my morning dose",
            "my afternoon dose",
            "my evening dose",
            "my night dose",

            "medicine event",
            "medication event",

            "which medicine",
            "what medicine",
            "which medication",
            "what medication",
        )

        return any(
            phrase in cleaned
            for phrase in phrases
        )

    # ========================================================
    # EVENT DATETIME
    # ========================================================

    @staticmethod
    def _event_datetime(
        event: dict[str, Any],
    ) -> datetime:

        value = (
            event.get(
                "observed_at"
            )
        )

        if isinstance(
            value,
            datetime,
        ):

            return value

        if value:

            try:

                parsed = (
                    datetime.fromisoformat(
                        str(value)
                        .replace(
                            "Z",
                            "+00:00",
                        )
                    )
                )

                if parsed.tzinfo is not None:

                    parsed = (
                        parsed.replace(
                            tzinfo=None
                        )
                    )

                return parsed

            except (
                TypeError,
                ValueError,
            ):

                pass

        return datetime.min

    # ========================================================
    # SORT NEWEST
    # ========================================================

    @classmethod
    def _prefer_newer_events(
        cls,
        events: list[
            dict[str, Any]
        ],
    ) -> list[
        dict[str, Any]
    ]:

        return sorted(
            events,
            key=cls._event_datetime,
            reverse=True,
        )

    # ========================================================
    # SEMANTIC -> DICT
    # ========================================================

    @staticmethod
    def _semantic_to_dict(
        memory: MedicineMemorySearchResult,
    ) -> dict[str, Any]:

        result = dict(
            memory.metadata
        )

        result["memory_id"] = (
            memory.memory_id
        )

        result["memory_text"] = (
            memory.text
        )

        result["distance"] = (
            memory.distance
        )

        if (
            "id" not in result
            and result.get(
                "medicine_event_id"
            )
            is not None
        ):

            result["id"] = (
                result[
                    "medicine_event_id"
                ]
            )

        return result

    # ========================================================
    # SEMANTIC RETRIEVAL
    # ========================================================

    def _semantic_retrieve(
        self,
        question: str,
        top_k: int,
        medicine: dict[str, Any] | None,
    ) -> list[
        dict[str, Any]
    ]:

        medicine_id = (
            int(
                medicine["id"]
            )
            if medicine
            else None
        )

        memories = (
            self.semantic_search
            .search(
                query=question,
                top_k=max(
                    top_k,
                    5,
                ),
                medicine_id=medicine_id,
            )
        )

        return [
            self._semantic_to_dict(
                memory
            )
            for memory in memories[
                :top_k
            ]
        ]

    # ========================================================
    # TODAY
    # ========================================================

    def _retrieve_today(
        self,
        medicine: dict[str, Any] | None,
        top_k: int,
    ) -> list[
        dict[str, Any]
    ]:

        medicine_id = (
            int(
                medicine["id"]
            )
            if medicine
            else None
        )

        events = (
            get_medicine_events_for_date(
                target_date=date.today(),
                medicine_id=medicine_id,
                confirmed_only=False,
            )
        )

        return (
            self._prefer_newer_events(
                events
            )[:top_k]
        )

    # ========================================================
    # YESTERDAY
    # ========================================================

    def _retrieve_yesterday(
        self,
        medicine: dict[str, Any] | None,
        top_k: int,
    ) -> list[
        dict[str, Any]
    ]:

        medicine_id = (
            int(
                medicine["id"]
            )
            if medicine
            else None
        )

        target_date = (
            date.today()
            - timedelta(
                days=1
            )
        )

        events = (
            get_medicine_events_for_date(
                target_date=target_date,
                medicine_id=medicine_id,
                confirmed_only=False,
            )
        )

        return (
            self._prefer_newer_events(
                events
            )[:top_k]
        )

    # ========================================================
    # LATEST SPECIFIC MEDICINE
    # ========================================================

    @staticmethod
    def _retrieve_latest_specific(
        medicine: dict[str, Any],
        confirmed_only: bool,
    ) -> list[
        dict[str, Any]
    ]:

        event = (
            get_latest_medicine_event(
                medicine_id=int(
                    medicine["id"]
                ),
                confirmed_only=confirmed_only,
            )
        )

        return (
            [event]
            if event
            else []
        )

    # ========================================================
    # RECENT
    # ========================================================

    @staticmethod
    def _retrieve_recent(
        top_k: int,
        confirmed_only: bool = False,
    ) -> list[
        dict[str, Any]
    ]:

        return (
            get_recent_medicine_events(
                limit=max(
                    top_k,
                    10,
                ),
                confirmed_only=confirmed_only,
            )[:top_k]
        )

    # ========================================================
    # MAIN RETRIEVE
    # ========================================================

    def retrieve(
        self,
        question: str,
        top_k: int = 5,
    ) -> list[
        dict[str, Any]
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
            self._normalize_text(
                cleaned
            )
        )

        safe_top_k = max(
            1,
            int(top_k),
        )

        medicine = (
            self.resolve_medicine(
                cleaned
            )
        )

        # ====================================================
        # UNKNOWN SPECIFIC MEDICINE SAFETY GATE
        #
        # CRITICAL FIX:
        #
        # Previously:
        #
        # unknown medicine
        # -> medicine=None
        # -> today query
        # -> retrieve every medicine today
        #
        # Now:
        #
        # unknown specific medicine
        # -> return no memories
        #
        # ====================================================

        if (
            medicine is None
            and self.has_unresolved_specific_medicine_reference(
                cleaned
            )
        ):

            return []

        # ====================================================
        # TODAY
        # ====================================================

        if "today" in lowered:

            return (
                self._retrieve_today(
                    medicine=medicine,
                    top_k=safe_top_k,
                )
            )

        # ====================================================
        # YESTERDAY
        # ====================================================

        if "yesterday" in lowered:

            return (
                self._retrieve_yesterday(
                    medicine=medicine,
                    top_k=safe_top_k,
                )
            )

        # ====================================================
        # LAST / LATEST / WHEN
        # ====================================================

        latest_phrases = (
            "when was",
            "when did",
            "what time",
            "last confirmed",
            "last medicine",
            "last medication",
            "latest medicine",
            "latest medication",
            "most recent medicine",
            "most recent medication",
        )

        if (
            medicine is not None
            and any(
                phrase in lowered
                for phrase in latest_phrases
            )
        ):

            return (
                self._retrieve_latest_specific(
                    medicine=medicine,
                    confirmed_only=True,
                )
            )

        # ====================================================
        # DIRECT CONFIRMATION QUESTION
        # ====================================================

        confirmation_phrases = (
            "was confirmed",
            "is confirmed",
            "camera confirmed",
            "did i take",
            "did i have",
            "have i taken",
            "have i had",
            "did memorai confirm",
        )

        if (
            medicine is not None
            and any(
                phrase in lowered
                for phrase in confirmation_phrases
            )
        ):

            return (
                self._retrieve_latest_specific(
                    medicine=medicine,
                    confirmed_only=True,
                )
            )

        # ====================================================
        # RECENT
        # ====================================================

        recent_phrases = (
            "recent",
            "recently",
            "latest medicine",
            "latest medication",
            "recent medicine",
            "recent medication",
            "medicine event",
            "medication event",
        )

        if any(
            phrase in lowered
            for phrase in recent_phrases
        ):

            events = (
                self._retrieve_recent(
                    top_k=safe_top_k,
                    confirmed_only=False,
                )
            )

            if medicine is None:

                return events

            medicine_id = int(
                medicine["id"]
            )

            return [
                event
                for event in events
                if int(
                    event.get(
                        "medicine_id",
                        -1,
                    )
                )
                == medicine_id
            ][:safe_top_k]

        # ====================================================
        # SPECIFIC MEDICINE WITHOUT TIME WORD
        # ====================================================

        if medicine is not None:

            return (
                self._retrieve_latest_specific(
                    medicine=medicine,
                    confirmed_only=False,
                )
            )

        # ====================================================
        # SEMANTIC FALLBACK
        # ====================================================

        if (
            self.is_medicine_question(
                cleaned
            )
        ):

            return (
                self._semantic_retrieve(
                    question=cleaned,
                    top_k=safe_top_k,
                    medicine=medicine,
                )
            )

        return []