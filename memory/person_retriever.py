from __future__ import annotations

import re

from datetime import (
    date,
    datetime,
    timedelta,
)
from typing import Any

from database.person_repository import (
    get_all_people,
    get_latest_person_event,
    get_person_events_for_date,
    get_recent_person_events,
)

from rag.person_memory_search import (
    PersonMemorySearch,
    PersonMemorySearchResult,
)


# ============================================================
# PERSON MEMORY RETRIEVER
# ============================================================

class PersonMemoryRetriever:
    """
    Hybrid retriever for MemorAi person memories.

    MySQL is authoritative for structured questions such as:

    - When did I last see Rahul?
    - Did I see Rahul today?
    - Did I see Rahul yesterday?
    - Who did I see today?
    - Who did I see yesterday?
    - Who was the last person I saw?
    - Have I seen Rahul recently?
    - Did I see my son today?

    Person ChromaDB is used only as a semantic fallback for
    natural-language person-memory questions which do not
    match the deterministic MySQL rules.

    Important:

    - Registered identity resolution remains MySQL-based.
    - Exact date/time questions remain MySQL-based.
    - Unknown proper names do not fall through into semantic
      search, preventing an unrelated person memory from being
      returned.
    """

    # ========================================================
    # INITIALIZE
    # ========================================================

    def __init__(
        self,
    ) -> None:

        self.semantic_search = (
            PersonMemorySearch()
        )

    # ========================================================
    # NORMALIZE
    # ========================================================

    @staticmethod
    def _normalize_text(
        value: str,
    ) -> str:

        return " ".join(
            value.strip().lower().split()
        )

    # ========================================================
    # WORD / PHRASE MATCH
    # ========================================================

    @staticmethod
    def _contains_phrase(
        text: str,
        phrase: str,
    ) -> bool:

        normalized_phrase = (
            PersonMemoryRetriever
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
    # REGISTERED PEOPLE
    # ========================================================

    @staticmethod
    def _registered_people(
    ) -> list[dict[str, Any]]:

        # Historical memories remain queryable even if a
        # person has later been deactivated.

        return get_all_people(
            active_only=False
        )

    # ========================================================
    # FIND PEOPLE BY NAME
    # ========================================================

    def _find_people_by_name(
        self,
        question: str,
    ) -> list[dict[str, Any]]:

        lowered = (
            self._normalize_text(
                question
            )
        )

        people = (
            self._registered_people()
        )

        matches: list[
            dict[str, Any]
        ] = []

        # Prefer longer names first.

        people = sorted(
            people,
            key=lambda person: len(
                str(
                    person.get(
                        "display_name"
                    )
                    or person.get(
                        "name"
                    )
                    or ""
                )
            ),
            reverse=True,
        )

        for person in people:

            possible_names: list[
                str
            ] = []

            display_name = (
                person.get(
                    "display_name"
                )
            )

            canonical_name = (
                person.get(
                    "name"
                )
            )

            if display_name:

                possible_names.append(
                    str(
                        display_name
                    )
                )

            if canonical_name:

                possible_names.append(
                    str(
                        canonical_name
                    )
                )

            matched = False

            for candidate in possible_names:

                if self._contains_phrase(
                    lowered,
                    candidate,
                ):

                    matched = True

                    break

            if matched:

                matches.append(
                    person
                )

        return matches

    # ========================================================
    # FIND PEOPLE BY RELATION
    # ========================================================

    def _find_people_by_relation(
        self,
        question: str,
    ) -> list[dict[str, Any]]:

        lowered = (
            self._normalize_text(
                question
            )
        )

        people = (
            self._registered_people()
        )

        matches: list[
            dict[str, Any]
        ] = []

        for person in people:

            relation = (
                person.get(
                    "relation"
                )
            )

            if not relation:

                continue

            normalized_relation = (
                self._normalize_text(
                    str(
                        relation
                    )
                )
            )

            if not normalized_relation:

                continue

            phrases = (
                normalized_relation,
                (
                    f"my "
                    f"{normalized_relation}"
                ),
                (
                    f"the "
                    f"{normalized_relation}"
                ),
            )

            if any(
                self._contains_phrase(
                    lowered,
                    phrase,
                )
                for phrase in phrases
            ):

                matches.append(
                    person
                )

        return matches

    # ========================================================
    # RESOLVE PEOPLE
    # ========================================================

    def resolve_people(
        self,
        question: str,
    ) -> list[dict[str, Any]]:
        """
        Resolve registered identities referenced by:

        - name
        - display name
        - relation

        Name matches take priority.
        """

        name_matches = (
            self._find_people_by_name(
                question
            )
        )

        if name_matches:

            return name_matches

        return (
            self._find_people_by_relation(
                question
            )
        )

    # ========================================================
    # UNRESOLVED NAMED PERSON HEURISTIC
    # ========================================================

    @staticmethod
    def _looks_like_unresolved_named_person_question(
        question: str,
    ) -> bool:
        """
        Detect questions which appear to refer to a person's
        proper name even when that identity does not exist in
        registered_people.

        Example:

            When did I last see Rahul?

        If Rahul is unregistered, this should still be routed
        to the person-memory subsystem and return the normal
        reliable-memory fallback.

        We deliberately avoid expressions such as:

            When did I last see my phone?

        because "my phone" clearly looks like an object
        reference rather than a person's proper name.
        """

        cleaned = " ".join(
            question.strip().split()
        )

        if not cleaned:

            return False

        patterns = (
            (
                r"\bwhen\s+did\s+i\s+"
                r"(?:last\s+)?see\s+"
                r"(.+?)(?:\?|$)"
            ),
            (
                r"\bdid\s+i\s+see\s+"
                r"(.+?)(?:\s+today|"
                r"\s+yesterday|"
                r"\s+recently|\?|$)"
            ),
            (
                r"\bhave\s+i\s+seen\s+"
                r"(.+?)(?:\s+today|"
                r"\s+yesterday|"
                r"\s+recently|\?|$)"
            ),
            (
                r"\bwhen\s+was\s+"
                r"(.+?)\s+"
                r"(?:here|seen)"
                r"(?:\?|$)"
            ),
        )

        target: (
            str
            | None
        ) = None

        for pattern in patterns:

            match = re.search(
                pattern,
                cleaned,
                flags=re.IGNORECASE,
            )

            if match:

                target = (
                    match.group(
                        1
                    )
                    .strip(
                        " .,!?:;"
                    )
                )

                break

        if not target:

            return False

        lowered_target = (
            target.lower()
        )

        # ----------------------------------------------------
        # CLEAR NON-PERSON EXPRESSIONS
        # ----------------------------------------------------

        non_person_prefixes = (
            "my ",
            "the ",
            "a ",
            "an ",
            "this ",
            "that ",
            "these ",
            "those ",
            "any ",
            "some ",
        )

        if lowered_target.startswith(
            non_person_prefixes
        ):

            return False

        non_person_targets = {
            "anyone",
            "someone",
            "somebody",
            "anybody",
            "everyone",
            "everybody",
            "anything",
            "something",
        }

        if (
            lowered_target
            in non_person_targets
        ):

            return False

        # ----------------------------------------------------
        # PROPER-NAME SIGNAL
        # ----------------------------------------------------

        words = re.findall(
            r"[A-Za-z]"
            r"[A-Za-z0-9_'’-]*",
            target,
        )

        if not words:

            return False

        if any(
            word[0].isupper()
            for word in words
            if word
        ):

            return True

        return False

    # ========================================================
    # PERSON QUESTION DETECTION
    # ========================================================

    def is_person_question(
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
        # REGISTERED NAME OR RELATION
        # ----------------------------------------------------

        if self.resolve_people(
            question
        ):

            return True

        # ----------------------------------------------------
        # GENERIC PERSON QUESTIONS
        # ----------------------------------------------------

        person_phrases = (
            "who did i see",
            "who have i seen",
            "who was here",
            "who visited",
            "who came",
            "who was the last person",
            "last person i saw",
            "latest person",
            "recent person",
            "recent people",
            "people i saw",
            "people seen",
            "person i saw",

            # Natural semantic person-memory language.
            "remember seeing",
            "remember meeting",
            "remember anyone",
            "remember anybody",
            "remember someone",
            "remember somebody",
            "people around",
            "person around",
            "visitor",
            "visitors",
            "visited me",
            "came to see me",
            "came by",
            "stopped by",
        )

        if any(
            phrase in cleaned
            for phrase in person_phrases
        ):

            return True

        # ----------------------------------------------------
        # UNREGISTERED PROPER NAME
        # ----------------------------------------------------

        if (
            self
            ._looks_like_unresolved_named_person_question(
                question
            )
        ):

            return True

        return False

    # ========================================================
    # FILTER EVENTS FOR PEOPLE
    # ========================================================

    @staticmethod
    def _filter_events_for_people(
        events: list[dict[str, Any]],
        people: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:

        if not people:

            return events

        person_ids = {
            int(
                person[
                    "id"
                ]
            )
            for person in people
        }

        return [
            event
            for event in events
            if int(
                event.get(
                    "person_id",
                    -1,
                )
            )
            in person_ids
        ]

    # ========================================================
    # LATEST EVENTS FOR PEOPLE
    # ========================================================

    @staticmethod
    def _latest_events_for_people(
        people: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:

        events: list[
            dict[str, Any]
        ] = []

        for person in people:

            event = (
                get_latest_person_event(
                    int(
                        person[
                            "id"
                        ]
                    )
                )
            )

            if event is not None:

                events.append(
                    event
                )

        events.sort(
            key=lambda event: (
                event.get(
                    "detected_at"
                )
                or datetime.min
            ),
            reverse=True,
        )

        return events

    # ========================================================
    # SEMANTIC RESULT -> DICT
    # ========================================================

    @staticmethod
    def _semantic_to_dict(
        memory: PersonMemorySearchResult,
    ) -> dict[str, Any]:
        """
        Convert PersonMemorySearchResult into the same general
        dictionary structure expected by the person RAG layer.
        """

        result = dict(
            memory.metadata
        )

        result[
            "memory_id"
        ] = memory.memory_id

        result[
            "memory_text"
        ] = memory.text

        result[
            "distance"
        ] = memory.distance

        # PersonVectorMemory stores the MySQL ID as
        # person_event_id. The rest of MemorAi generally
        # expects an encounter row to expose it as id.

        if (
            "id"
            not in result
            and result.get(
                "person_event_id"
            )
            is not None
        ):

            result[
                "id"
            ] = result[
                "person_event_id"
            ]

        return result

    # ========================================================
    # SEMANTIC SEARCH
    # ========================================================

    def _semantic_retrieve(
        self,
        question: str,
        top_k: int,
        people: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Search person ChromaDB.

        When exactly one registered person has been resolved,
        restrict semantic search to that person's ID.

        If multiple people match a relation such as "friend",
        search the person-memory collection normally and then
        restrict returned results to the resolved identities.
        """

        person_id: (
            int
            | None
        ) = None

        if len(
            people
        ) == 1:

            person_id = int(
                people[0][
                    "id"
                ]
            )

        memories = (
            self.semantic_search.search(
                query=question,
                top_k=max(
                    top_k,
                    5,
                ),
                person_id=(
                    person_id
                ),
            )
        )

        results = [
            self._semantic_to_dict(
                memory
            )
            for memory in memories
        ]

        # ----------------------------------------------------
        # MULTIPLE RESOLVED PEOPLE
        # ----------------------------------------------------

        if len(
            people
        ) > 1:

            allowed_ids = {
                int(
                    person[
                        "id"
                    ]
                )
                for person in people
            }

            results = [
                result
                for result in results
                if int(
                    result.get(
                        "person_id",
                        -1,
                    )
                )
                in allowed_ids
            ]

        return results[
            :top_k
        ]

    # ========================================================
    # MAIN RETRIEVAL
    # ========================================================

    def retrieve(
        self,
        question: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:

        cleaned = (
            question.strip()
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
            int(
                top_k
            ),
        )

        people = (
            self.resolve_people(
                cleaned
            )
        )

        # ====================================================
        # IMPORTANT:
        # apparent named person but no registered identity
        # ====================================================

        if (
            not people
            and self
            ._looks_like_unresolved_named_person_question(
                cleaned
            )
        ):

            # Never semantic-search an unknown proper name.
            #
            # Otherwise a question such as:
            #
            # "When did I last see Robert?"
            #
            # could incorrectly return Rahul because their
            # encounter sentences are semantically similar.

            return []

        # ====================================================
        # TODAY
        # ====================================================

        if "today" in lowered:

            events = (
                get_person_events_for_date(
                    date.today()
                )
            )

            events = (
                self._filter_events_for_people(
                    events,
                    people,
                )
            )

            return events[
                :safe_top_k
            ]

        # ====================================================
        # YESTERDAY
        # ====================================================

        if "yesterday" in lowered:

            target_date = (
                date.today()
                - timedelta(
                    days=1
                )
            )

            events = (
                get_person_events_for_date(
                    target_date
                )
            )

            events = (
                self._filter_events_for_people(
                    events,
                    people,
                )
            )

            return events[
                :safe_top_k
            ]

        # ====================================================
        # LAST / LATEST GENERIC PERSON
        # ====================================================

        generic_latest_phrases = (
            "who was the last person",
            "last person i saw",
            "latest person",
            "most recent person",
        )

        if (
            not people
            and any(
                phrase in lowered
                for phrase
                in generic_latest_phrases
            )
        ):

            return (
                get_recent_person_events(
                    limit=1
                )
            )

        # ====================================================
        # SPECIFIC PERSON / RELATION LATEST
        # ====================================================

        latest_phrases = (
            "when did",
            "when was",
            "what time",
            "last see",
            "last saw",
            "last seen",
            "latest",
            "most recent",
            "when have i",
        )

        if (
            people
            and any(
                phrase in lowered
                for phrase
                in latest_phrases
            )
        ):

            return (
                self._latest_events_for_people(
                    people
                )[
                    :safe_top_k
                ]
            )

        # ====================================================
        # RECENT PERSON MEMORIES
        # ====================================================

        recent_phrases = (
            "recent",
            "recently",
            "have i seen",
            "who have i seen",
            "people i saw",
            "people seen",
        )

        if any(
            phrase in lowered
            for phrase in recent_phrases
        ):

            events = (
                get_recent_person_events(
                    limit=max(
                        safe_top_k,
                        10,
                    )
                )
            )

            events = (
                self._filter_events_for_people(
                    events,
                    people,
                )
            )

            return events[
                :safe_top_k
            ]

        # ====================================================
        # GENERIC WHO QUESTIONS
        # ====================================================

        who_phrases = (
            "who did i see",
            "who was here",
            "who visited",
            "who came",
        )

        if any(
            phrase in lowered
            for phrase in who_phrases
        ):

            return (
                get_recent_person_events(
                    limit=safe_top_k
                )
            )

        # ====================================================
        # KNOWN PERSON WITHOUT EXPLICIT TIME WORD
        # ====================================================

        if people:

            # Keep existing deterministic behaviour.
            #
            # A direct question mentioning a known registered
            # person should prefer their latest authoritative
            # MySQL encounter.

            return (
                self._latest_events_for_people(
                    people
                )[
                    :safe_top_k
                ]
            )

        # ====================================================
        # SEMANTIC FALLBACK
        # ====================================================

        if self.is_person_question(
            cleaned
        ):

            return (
                self._semantic_retrieve(
                    question=cleaned,
                    top_k=safe_top_k,
                    people=people,
                )
            )

        return []