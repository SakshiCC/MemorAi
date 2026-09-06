from __future__ import annotations

from datetime import (
    date,
    datetime,
    timedelta,
)

import requests

from ollama import (
    Client,
    ResponseError,
)

from config import (
    DEFAULT_TOP_K,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
)

from database.medicine_repository import (
    STATUS_CAMERA_CONFIRMED,
    STATUS_LOW_CONFIDENCE,
    STATUS_MISSED,
    STATUS_UNCONFIRMED,
)

from memory.conversation_retriever import (
    ConversationMemoryResult,
    ConversationMemoryRetriever,
)

from memory.medicine_retriever import (
    MedicineMemoryRetriever,
)

from memory.person_retriever import (
    PersonMemoryRetriever,
)

from memory.retriever import (
    MemoryRetriever,
)


# ============================================================
# FALLBACK
# ============================================================

FALLBACK_ANSWER = (
    "I could not find a reliable memory for that."
)


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are MemorAi, a local AI memory assistant.

Use only the supplied memory records.

Rules:

1. Never invent an object, person, conversation,
   medicine, location, date, time, instruction or event.

2. Treat object and person records as past observations.

3. When answering object-location questions, say:
   "last observed"
   or
   "the latest record shows".

4. Never claim that a previously observed object is
   definitely still at that location.

5. Conversation records contain remembered statements
   from previous conversations.

6. Medicine records describe camera-assisted observations.

7. A medicine event with status "camera_confirmed" means
   MemorAi observed sufficient visual evidence of a
   medication-taking event.

8. Never claim that a medicine was definitely swallowed
   merely because a camera-confirmed event exists.

9. Prefer wording such as:
   "MemorAi recorded a camera-confirmed medication event."

10. Small transcription differences in names may occur.
    Do not reject otherwise strong conversation evidence
    solely because a spoken name was transcribed with a
    minor spelling variation.

11. If the supplied records truly do not answer the
    question, reply exactly:

    "I could not find a reliable memory for that."

12. Keep answers brief, clear and easy to understand.
""".strip()


# ============================================================
# MEMORAI CHAT
# ============================================================

class MemorAiChat:

    # ========================================================
    # INITIALIZE
    # ========================================================

    def __init__(
        self,
    ) -> None:

        # ----------------------------------------------------
        # OBJECT MEMORY
        # ----------------------------------------------------

        self.retriever = (
            MemoryRetriever()
        )

        # ----------------------------------------------------
        # MEDICINE MEMORY
        # ----------------------------------------------------

        self.medicine_retriever = (
            MedicineMemoryRetriever()
        )

        # ----------------------------------------------------
        # PERSON MEMORY
        # ----------------------------------------------------

        self.person_retriever = (
            PersonMemoryRetriever()
        )

        # ----------------------------------------------------
        # CONVERSATION MEMORY
        # ----------------------------------------------------

        self.conversation_retriever = (
            ConversationMemoryRetriever()
        )

        # ----------------------------------------------------
        # OLLAMA
        # ----------------------------------------------------

        self.client = Client(
            host=(
                OLLAMA_BASE_URL
            )
        )

    # ========================================================
    # OLLAMA STATUS
    # ========================================================

    def is_ollama_running(
        self,
    ) -> bool:

        try:

            response = requests.get(
                (
                    f"{OLLAMA_BASE_URL.rstrip('/')}"
                    "/api/tags"
                ),
                timeout=5,
            )

            return (
                response.status_code
                == 200
            )

        except requests.RequestException:

            return False

    # ========================================================
    # FORMAT OBJECT MEMORIES
    # ========================================================

    @staticmethod
    def _format_memories(
        memories: list[dict],
    ) -> str:

        if not memories:

            return (
                "NO_RELIABLE_MEMORY_RECORDS"
            )

        formatted_records = []

        for index, memory in enumerate(
            memories,
            start=1,
        ):

            memory_text = (
                memory.get(
                    "memory_text"
                )
            )

            if not memory_text:

                object_name = (
                    memory.get(
                        "object_name",
                        "unknown object",
                    )
                )

                location = (
                    memory.get(
                        "location",
                        "unknown location",
                    )
                )

                detected_at = (
                    memory.get(
                        "detected_at",
                        "unknown time",
                    )
                )

                memory_text = (
                    f"MemorAi observed "
                    f"{object_name} "
                    f"at {location} "
                    f"at {detected_at}."
                )

            formatted_record = f"""
Record {index}:
- Observation: {memory_text}
- Object: {memory.get("object_name", "unknown")}
- Location: {memory.get("location", "unknown")}
- Time: {memory.get("detected_at", "unknown")}
""".strip()

            formatted_records.append(
                formatted_record
            )

        return "\n\n".join(
            formatted_records
        )

    # ========================================================
    # FORMAT CONVERSATION MEMORIES
    # ========================================================

    @staticmethod
    def _format_conversation_memories(
        memories: list[
            ConversationMemoryResult
        ],
    ) -> str:

        if not memories:

            return (
                "NO_RELIABLE_MEMORY_RECORDS"
            )

        records: list[
            str
        ] = []

        for index, memory in enumerate(
            memories,
            start=1,
        ):

            metadata = (
                memory.metadata
            )

            participant = (
                metadata.get(
                    "participant_text"
                )
                or "unknown participant"
            )

            title = (
                metadata.get(
                    "title"
                )
                or "untitled conversation"
            )

            started_at = (
                metadata.get(
                    "started_at"
                )
                or "unknown time"
            )

            transcript = (
                metadata.get(
                    "full_transcript"
                )
                or memory.text
                or ""
            )

            summary = (
                metadata.get(
                    "summary"
                )
                or ""
            )

            record = f"""
Conversation {index}:
- Participant: {participant}
- Topic: {title}
- Time: {started_at}
- Summary: {summary or "No summary stored."}
- Original transcript: {transcript}
""".strip()

            records.append(
                record
            )

        return "\n\n".join(
            records
        )

    # ========================================================
    # FORMAT MEDICINE MEMORIES
    # ========================================================

    @staticmethod
    def _format_medicine_memories(
        memories: list[
            dict
        ],
    ) -> str:

        if not memories:

            return (
                "NO_RELIABLE_MEMORY_RECORDS"
            )

        records: list[
            str
        ] = []

        for index, memory in enumerate(
            memories,
            start=1,
        ):

            record = f"""
Medicine Record {index}:
- Medicine: {memory.get("medicine_name", "unknown medicine")}
- Dosage: {memory.get("dosage", "unknown")}
- Status: {memory.get("status", "unknown")}
- Observed at: {memory.get("observed_at", "unknown")}
- Scheduled for: {memory.get("scheduled_for", "unknown")}
- Person: {memory.get("person_name", "unknown")}
- Memory: {memory.get("memory_text", "")}
""".strip()

            records.append(
                record
            )

        return "\n\n".join(
            records
        )

    # ========================================================
    # ASK OLLAMA
    # ========================================================

    def _answer_with_ollama(
        self,
        question: str,
        formatted_memories: str,
        memory_kind: str,
    ) -> str:

        user_prompt = f"""
USER QUESTION:
{question}

MEMORY TYPE:
{memory_kind}

MEMORY RECORDS:
{formatted_memories}

Answer the user's question using only these memory records.

Important:

- Do not invent information.
- A camera-confirmed medicine event does not prove swallowing.
- If discussing medicine confirmation, describe it as a
  camera-confirmed medication event.
- Small transcription spelling differences in names may occur.
- If records are insufficient, answer exactly:
  "I could not find a reliable memory for that."
- Return only the final answer.
""".strip()

        try:

            response = (
                self.client.chat(
                    model=(
                        OLLAMA_MODEL
                    ),
                    messages=[
                        {
                            "role": (
                                "system"
                            ),
                            "content": (
                                SYSTEM_PROMPT
                            ),
                        },
                        {
                            "role": (
                                "user"
                            ),
                            "content": (
                                user_prompt
                            ),
                        },
                    ],
                    stream=False,
                    options={
                        "temperature": 0.1,
                        "top_p": 0.8,
                        "num_predict": 180,
                    },
                )
            )

            answer = (
                response[
                    "message"
                ][
                    "content"
                ]
                .strip()
            )

            if not answer:

                return (
                    FALLBACK_ANSWER
                )

            return answer

        except ResponseError as error:

            print(
                (
                    "Ollama response error: "
                    f"{error}"
                )
            )

            return (
                FALLBACK_ANSWER
            )

        except Exception as error:

            print(
                (
                    "Unexpected LLM error: "
                    f"{error}"
                )
            )

            return (
                FALLBACK_ANSWER
            )

    # ========================================================
    # NORMALIZE FALLBACK CHECK
    # ========================================================

    @staticmethod
    def _is_fallback_answer(
        answer: str,
    ) -> bool:

        cleaned = (
            str(
                answer
                or ""
            )
            .strip()
            .lower()
            .rstrip(
                "."
            )
        )

        expected = (
            FALLBACK_ANSWER
            .strip()
            .lower()
            .rstrip(
                "."
            )
        )

        return (
            cleaned
            == expected
        )

    # ========================================================
    # FORMAT GENERIC DATETIME
    # ========================================================

    @staticmethod
    def _format_datetime(
        value,
        include_date: bool = True,
    ) -> str:

        if isinstance(
            value,
            datetime,
        ):

            if include_date:

                return (
                    value.strftime(
                        "%d %B %Y at %I:%M %p"
                    )
                )

            return (
                value.strftime(
                    "%I:%M %p"
                )
            )

        if value:

            return (
                str(
                    value
                )
            )

        return (
            "an unknown time"
        )

    # ========================================================
    # FORMAT CONVERSATION TIME
    # ========================================================

    @staticmethod
    def _format_conversation_time(
        value,
    ) -> str | None:

        if isinstance(
            value,
            datetime,
        ):

            return (
                value.strftime(
                    "%d %B %Y at %I:%M %p"
                )
            )

        if value:

            return (
                str(
                    value
                )
            )

        return None

    # ========================================================
    # SAFE GROUNDED CONVERSATION FALLBACK
    # ========================================================

    @classmethod
    def _grounded_conversation_fallback(
        cls,
        memories: list[
            ConversationMemoryResult
        ],
    ) -> str:

        if not memories:

            return (
                FALLBACK_ANSWER
            )

        memory = (
            memories[
                0
            ]
        )

        metadata = (
            memory.metadata
        )

        summary = (
            str(
                metadata.get(
                    "summary"
                )
                or ""
            )
            .strip()
        )

        transcript = (
            str(
                metadata.get(
                    "full_transcript"
                )
                or memory.text
                or ""
            )
            .strip()
        )

        memory_text = (
            summary
            if summary
            else transcript
        )

        if not memory_text:

            return (
                FALLBACK_ANSWER
            )

        participant = (
            str(
                metadata.get(
                    "participant_text"
                )
                or ""
            )
            .strip()
        )

        started_at = (
            cls._format_conversation_time(
                metadata.get(
                    "started_at"
                )
            )
        )

        if participant:

            if started_at:

                return (
                    f"In the conversation with "
                    f"{participant} on {started_at}, "
                    f"the stored memory says: "
                    f"{memory_text}"
                )

            return (
                f"The relevant conversation with "
                f"{participant} says: "
                f"{memory_text}"
            )

        if started_at:

            return (
                f"The relevant conversation from "
                f"{started_at} says: "
                f"{memory_text}"
            )

        return (
            "The relevant stored conversation says: "
            f"{memory_text}"
        )

    # ========================================================
    # CONVERSATION ANSWER
    # ========================================================

    def _answer_conversation_question(
        self,
        question: str,
        top_k: int,
    ) -> str:

        memories = (
            self.conversation_retriever
            .retrieve(
                question,
                top_k=(
                    top_k
                ),
            )
        )

        if not memories:

            return (
                FALLBACK_ANSWER
            )

        if self.is_ollama_running():

            formatted = (
                self._format_conversation_memories(
                    memories
                )
            )

            llm_answer = (
                self._answer_with_ollama(
                    question=(
                        question
                    ),
                    formatted_memories=(
                        formatted
                    ),
                    memory_kind=(
                        "conversation memory"
                    ),
                )
            )

            if not (
                self._is_fallback_answer(
                    llm_answer
                )
            ):

                return (
                    llm_answer
                )

            return (
                self._grounded_conversation_fallback(
                    memories
                )
            )

        return (
            self._grounded_conversation_fallback(
                memories
            )
        )

    # ========================================================
    # MEDICINE HELPERS
    # ========================================================

    @staticmethod
    def _medicine_description(
        memory: dict,
    ) -> str:

        name = (
            str(
                memory.get(
                    "medicine_name"
                )
                or "medicine"
            )
            .strip()
        )

        dosage = (
            str(
                memory.get(
                    "dosage"
                )
                or ""
            )
            .strip()
        )

        if dosage:

            return (
                f"{name} {dosage}"
            )

        return (
            name
        )

    # ========================================================
    # CONFIRMED MEDICINE EVENTS
    # ========================================================

    @staticmethod
    def _confirmed_medicine_events(
        memories: list[
            dict
        ],
    ) -> list[
        dict
    ]:

        return [
            memory

            for memory
            in memories

            if (
                str(
                    memory.get(
                        "status"
                    )
                    or ""
                )
                .strip()
                .lower()
                == STATUS_CAMERA_CONFIRMED
            )
        ]

    # ========================================================
    # JOIN MEDICINE DESCRIPTIONS
    # ========================================================

    @classmethod
    def _join_unique_medicines(
        cls,
        memories: list[
            dict
        ],
    ) -> str:

        descriptions: list[
            str
        ] = []

        seen: set[
            str
        ] = set()

        for memory in memories:

            description = (
                cls._medicine_description(
                    memory
                )
            )

            key = (
                description.lower()
            )

            if key in seen:

                continue

            seen.add(
                key
            )

            descriptions.append(
                description
            )

        if not descriptions:

            return ""

        if len(
            descriptions
        ) == 1:

            return (
                descriptions[
                    0
                ]
            )

        if len(
            descriptions
        ) == 2:

            return (
                f"{descriptions[0]} "
                f"and {descriptions[1]}"
            )

        return (
            ", ".join(
                descriptions[
                    :-1
                ]
            )
            + f", and {descriptions[-1]}"
        )

    # ========================================================
    # MEDICINE EVENT ANSWER
    # ========================================================

    @classmethod
    def _answer_single_medicine_event(
        cls,
        memory: dict,
        day_word: str | None = None,
    ) -> str:

        description = (
            cls._medicine_description(
                memory
            )
        )

        observed_at = (
            memory.get(
                "observed_at"
            )
        )

        status = (
            str(
                memory.get(
                    "status"
                )
                or ""
            )
            .strip()
            .lower()
        )

        if (
            status
            == STATUS_CAMERA_CONFIRMED
        ):

            if day_word:

                time_text = (
                    cls._format_datetime(
                        observed_at,
                        include_date=False,
                    )
                )

                return (
                    "MemorAi recorded a "
                    "camera-confirmed medication "
                    f"event for {description} "
                    f"{day_word} at {time_text}."
                )

            time_text = (
                cls._format_datetime(
                    observed_at,
                    include_date=True,
                )
            )

            return (
                "MemorAi recorded a "
                "camera-confirmed medication "
                f"event for {description} "
                f"on {time_text}."
            )

        if (
            status
            == STATUS_LOW_CONFIDENCE
        ):

            return (
                "MemorAi observed possible "
                f"medication activity involving "
                f"{description}, but the camera "
                "evidence was not strong enough "
                "for confirmation."
            )

        if (
            status
            == STATUS_MISSED
        ):

            return (
                "MemorAi has a missed-schedule "
                f"record for {description}. "
                "This means no camera confirmation "
                "was recorded for that schedule; "
                "it does not prove the medicine "
                "was not taken."
            )

        if (
            status
            == STATUS_UNCONFIRMED
        ):

            return (
                "MemorAi has an unconfirmed "
                f"medication record for "
                f"{description}."
            )

        memory_text = (
            str(
                memory.get(
                    "memory_text"
                )
                or ""
            )
            .strip()
        )

        if memory_text:

            return (
                memory_text
            )

        return (
            FALLBACK_ANSWER
        )

    # ========================================================
    # MEDICINE QUESTION ANSWER
    # ========================================================

    def _answer_medicine_question(
        self,
        question: str,
        top_k: int,
    ) -> str:

        lowered = (
            question.lower()
        )

        medicine = (
            self.medicine_retriever
            .resolve_medicine(
                question
            )
        )

        # ====================================================
        # UNKNOWN SPECIFIC MEDICINE SAFETY GATE
        # ====================================================

        if (
            medicine is None
            and self.medicine_retriever
            .has_unresolved_specific_medicine_reference(
                question
            )
        ):

            return (
                FALLBACK_ANSWER
            )

        memories = (
            self.medicine_retriever
            .retrieve(
                question,
                top_k=max(
                    top_k,
                    20,
                ),
            )
        )

        # ====================================================
        # TODAY
        # ====================================================

        if (
            "today"
            in lowered
        ):

            confirmed = (
                self._confirmed_medicine_events(
                    memories
                )
            )

            if medicine is not None:

                description = (
                    str(
                        medicine.get(
                            "medicine_name"
                        )
                        or "medicine"
                    )
                )

                dosage = (
                    medicine.get(
                        "dosage"
                    )
                )

                if dosage:

                    description += (
                        f" {dosage}"
                    )

                if not confirmed:

                    return (
                        "MemorAi has no "
                        "camera-confirmed record "
                        f"for {description} today."
                    )

                return (
                    self._answer_single_medicine_event(
                        confirmed[
                            0
                        ],
                        day_word="today",
                    )
                )

            if not confirmed:

                return (
                    "MemorAi has no "
                    "camera-confirmed medication "
                    "events recorded today."
                )

            medicine_text = (
                self._join_unique_medicines(
                    confirmed
                )
            )

            return (
                "Today, MemorAi recorded "
                "camera-confirmed medication "
                f"events for {medicine_text}."
            )

        # ====================================================
        # YESTERDAY
        # ====================================================

        if (
            "yesterday"
            in lowered
        ):

            confirmed = (
                self._confirmed_medicine_events(
                    memories
                )
            )

            if medicine is not None:

                description = (
                    str(
                        medicine.get(
                            "medicine_name"
                        )
                        or "medicine"
                    )
                )

                dosage = (
                    medicine.get(
                        "dosage"
                    )
                )

                if dosage:

                    description += (
                        f" {dosage}"
                    )

                if not confirmed:

                    return (
                        "MemorAi has no "
                        "camera-confirmed record "
                        f"for {description} yesterday."
                    )

                return (
                    self._answer_single_medicine_event(
                        confirmed[
                            0
                        ],
                        day_word="yesterday",
                    )
                )

            if not confirmed:

                return (
                    "MemorAi has no "
                    "camera-confirmed medication "
                    "events recorded yesterday."
                )

            medicine_text = (
                self._join_unique_medicines(
                    confirmed
                )
            )

            return (
                "Yesterday, MemorAi recorded "
                "camera-confirmed medication "
                f"events for {medicine_text}."
            )

        # ====================================================
        # LAST / WHEN CONFIRMED
        # ====================================================

        last_phrases = (
            "last confirmed",
            "when was",
            "when did",
            "what time",
            "latest medicine",
            "latest medication",
            "most recent",
        )

        if any(
            phrase in lowered

            for phrase
            in last_phrases
        ):

            confirmed = (
                self._confirmed_medicine_events(
                    memories
                )
            )

            if not confirmed:

                if medicine is not None:

                    description = (
                        str(
                            medicine.get(
                                "medicine_name"
                            )
                            or "medicine"
                        )
                    )

                    dosage = (
                        medicine.get(
                            "dosage"
                        )
                    )

                    if dosage:

                        description += (
                            f" {dosage}"
                        )

                    return (
                        "MemorAi has no "
                        "camera-confirmed medication "
                        f"event for {description}."
                    )

                return (
                    FALLBACK_ANSWER
                )

            return (
                self._answer_single_medicine_event(
                    confirmed[
                        0
                    ]
                )
            )

        # ====================================================
        # RECENT MEDICINE EVENTS
        # ====================================================

        if (
            "recent"
            in lowered
            or "recently"
            in lowered
        ):

            confirmed = (
                self._confirmed_medicine_events(
                    memories
                )
            )

            if confirmed:

                medicine_text = (
                    self._join_unique_medicines(
                        confirmed
                    )
                )

                return (
                    "Recently, MemorAi recorded "
                    "camera-confirmed medication "
                    f"events for {medicine_text}."
                )

            if memories:

                return (
                    self._answer_single_medicine_event(
                        memories[
                            0
                        ]
                    )
                )

            return (
                FALLBACK_ANSWER
            )

        # ====================================================
        # DIRECT SPECIFIC MEDICINE QUESTION
        # ====================================================

        if medicine is not None:

            if not memories:

                description = (
                    str(
                        medicine.get(
                            "medicine_name"
                        )
                        or "medicine"
                    )
                )

                dosage = (
                    medicine.get(
                        "dosage"
                    )
                )

                if dosage:

                    description += (
                        f" {dosage}"
                    )

                return (
                    "MemorAi has no reliable "
                    "medication event recorded for "
                    f"{description}."
                )

            return (
                self._answer_single_medicine_event(
                    memories[
                        0
                    ]
                )
            )

        # ====================================================
        # GENERIC SEMANTIC MEDICINE QUESTION
        # ====================================================

        if not memories:

            return (
                FALLBACK_ANSWER
            )

        strongest = (
            self._answer_single_medicine_event(
                memories[
                    0
                ]
            )
        )

        if (
            strongest
            != FALLBACK_ANSWER
        ):

            return (
                strongest
            )

        if (
            self.is_ollama_running()
        ):

            formatted = (
                self._format_medicine_memories(
                    memories
                )
            )

            return (
                self._answer_with_ollama(
                    question=(
                        question
                    ),
                    formatted_memories=(
                        formatted
                    ),
                    memory_kind=(
                        "medicine memory"
                    ),
                )
            )

        return (
            FALLBACK_ANSWER
        )

    # ========================================================
    # OBJECT QUESTION TYPES
    # ========================================================

    @staticmethod
    def _is_location_question(
        question: str,
    ) -> bool:

        lowered = (
            question.lower()
        )

        phrases = (
            "where is",
            "where are",
            "where was",
            "where were",
            "where did i keep",
            "where did i put",
            "where did i leave",
            "where have i kept",
        )

        return any(
            phrase in lowered

            for phrase
            in phrases
        )

    @staticmethod
    def _is_time_question(
        question: str,
    ) -> bool:

        lowered = (
            question.lower()
        )

        phrases = (
            "when did",
            "what time",
            "when was",
            "when were",
            "when did i last",
            "last seen",
            "last observed",
        )

        return any(
            phrase in lowered

            for phrase
            in phrases
        )

    @staticmethod
    def _is_today_question(
        question: str,
    ) -> bool:

        return (
            "today"
            in question.lower()
        )

    @staticmethod
    def _is_yesterday_question(
        question: str,
    ) -> bool:

        return (
            "yesterday"
            in question.lower()
        )

    @staticmethod
    def _is_recent_question(
        question: str,
    ) -> bool:

        lowered = (
            question.lower()
        )

        phrases = (
            "recent",
            "recently",
            "latest objects",
            "recent objects",
            "what objects were detected",
            "what was detected",
            "what happened recently",
        )

        return any(
            phrase in lowered

            for phrase
            in phrases
        )

    # ========================================================
    # OBJECT LOCATION ANSWER
    # ========================================================

    @staticmethod
    def _answer_location_question(
        memory: dict,
    ) -> str:

        object_name = (
            memory.get(
                "object_name",
                "object",
            )
        )

        location = (
            memory.get(
                "location",
                "an unknown location",
            )
        )

        detected_at = (
            memory.get(
                "detected_at"
            )
        )

        if detected_at:

            try:

                time_text = (
                    detected_at.strftime(
                        "%d %B %Y at %I:%M %p"
                    )
                )

            except AttributeError:

                time_text = (
                    str(
                        detected_at
                    )
                )

            return (
                f"Your {object_name} "
                f"was last observed "
                f"in the {location} "
                f"on {time_text}."
            )

        return (
            f"Your {object_name} "
            f"was last observed "
            f"in the {location}."
        )

    # ========================================================
    # OBJECT TIME ANSWER
    # ========================================================

    @staticmethod
    def _answer_time_question(
        memory: dict,
    ) -> str:

        object_name = (
            memory.get(
                "object_name",
                "object",
            )
        )

        location = (
            memory.get(
                "location"
            )
        )

        detected_at = (
            memory.get(
                "detected_at"
            )
        )

        if not detected_at:

            return (
                FALLBACK_ANSWER
            )

        try:

            time_text = (
                detected_at.strftime(
                    "%d %B %Y at %I:%M %p"
                )
            )

        except AttributeError:

            time_text = (
                str(
                    detected_at
                )
            )

        if location:

            return (
                f"Your {object_name} "
                f"was last observed "
                f"on {time_text} "
                f"in the {location}."
            )

        return (
            f"Your {object_name} "
            f"was last observed "
            f"on {time_text}."
        )

    # ========================================================
    # OBJECT TODAY ANSWER
    # ========================================================

    @staticmethod
    def _answer_today_question(
        memories: list[
            dict
        ],
    ) -> str:

        if not memories:

            return (
                FALLBACK_ANSWER
            )

        latest = (
            memories[
                0
            ]
        )

        object_name = (
            latest.get(
                "object_name",
                "object",
            )
        )

        location = (
            latest.get(
                "location"
            )
        )

        detected_at = (
            latest.get(
                "detected_at"
            )
        )

        if detected_at:

            try:

                time_text = (
                    detected_at.strftime(
                        "%I:%M %p"
                    )
                )

            except AttributeError:

                time_text = (
                    str(
                        detected_at
                    )
                )

        else:

            time_text = (
                "an unknown time"
            )

        if location:

            return (
                f"Yes. MemorAi observed "
                f"your {object_name} today "
                f"at {time_text} "
                f"in the {location}."
            )

        return (
            f"Yes. MemorAi observed "
            f"your {object_name} today "
            f"at {time_text}."
        )

    # ========================================================
    # OBJECT YESTERDAY ANSWER
    # ========================================================

    @staticmethod
    def _answer_yesterday_question(
        memories: list[
            dict
        ],
    ) -> str:

        if not memories:

            return (
                FALLBACK_ANSWER
            )

        latest = (
            memories[
                0
            ]
        )

        object_name = (
            latest.get(
                "object_name",
                "object",
            )
        )

        location = (
            latest.get(
                "location"
            )
        )

        detected_at = (
            latest.get(
                "detected_at"
            )
        )

        if detected_at:

            try:

                time_text = (
                    detected_at.strftime(
                        "%I:%M %p"
                    )
                )

            except AttributeError:

                time_text = (
                    str(
                        detected_at
                    )
                )

        else:

            time_text = (
                "an unknown time"
            )

        if location:

            return (
                f"Yes. MemorAi observed "
                f"your {object_name} yesterday "
                f"at {time_text} "
                f"in the {location}."
            )

        return (
            f"Yes. MemorAi observed "
            f"your {object_name} yesterday "
            f"at {time_text}."
        )

    # ========================================================
    # RECENT OBJECTS ANSWER
    # ========================================================

    @staticmethod
    def _answer_recent_question(
        memories: list[
            dict
        ],
    ) -> str:

        if not memories:

            return (
                FALLBACK_ANSWER
            )

        unique_objects = []

        seen = set()

        for memory in memories:

            object_name = (
                memory.get(
                    "object_name"
                )
            )

            if not object_name:

                continue

            if object_name in seen:

                continue

            seen.add(
                object_name
            )

            unique_objects.append(
                object_name
            )

        if not unique_objects:

            return (
                FALLBACK_ANSWER
            )

        if len(
            unique_objects
        ) == 1:

            return (
                "Recently, MemorAi observed "
                f"a {unique_objects[0]}."
            )

        if len(
            unique_objects
        ) == 2:

            return (
                "Recently, MemorAi observed "
                f"a {unique_objects[0]} "
                f"and a {unique_objects[1]}."
            )

        object_text = (
            ", ".join(
                f"a {name}"

                for name
                in unique_objects[
                    :-1
                ]
            )
        )

        final_object = (
            unique_objects[
                -1
            ]
        )

        return (
            "Recently, MemorAi observed "
            f"{object_text}, and "
            f"a {final_object}."
        )

    # ========================================================
    # DATE FILTER HELPER
    # ========================================================

    @staticmethod
    def _filter_memories_by_date(
        memories: list[
            dict
        ],
        target_date: date,
    ) -> list[
        dict
    ]:

        filtered: list[
            dict
        ] = []

        for memory in memories:

            detected_at = (
                memory.get(
                    "detected_at"
                )
            )

            if isinstance(
                detected_at,
                datetime,
            ):

                if (
                    detected_at.date()
                    == target_date
                ):

                    filtered.append(
                        memory
                    )

                continue

            if detected_at:

                try:

                    parsed = (
                        datetime.fromisoformat(
                            str(
                                detected_at
                            )
                            .replace(
                                "Z",
                                "+00:00",
                            )
                        )
                    )

                    if (
                        parsed.date()
                        == target_date
                    ):

                        filtered.append(
                            memory
                        )

                except (
                    TypeError,
                    ValueError,
                ):

                    continue

        return (
            filtered
        )

    # ========================================================
    # PERSON HELPERS
    # ========================================================

    @staticmethod
    def _person_name(
        record: dict,
    ) -> str:

        return (
            str(
                record.get(
                    "person_name"
                )
                or record.get(
                    "display_name"
                )
                or record.get(
                    "name"
                )
                or "the person"
            )
        )

    @staticmethod
    def _person_relation(
        record: dict,
    ) -> str | None:

        relation = (
            record.get(
                "relation"
            )
        )

        if not relation:

            return None

        cleaned = (
            str(
                relation
            )
            .strip()
        )

        return (
            cleaned
            if cleaned
            else None
        )

    @classmethod
    def _person_label(
        cls,
        record: dict,
    ) -> str:

        name = (
            cls._person_name(
                record
            )
        )

        relation = (
            cls._person_relation(
                record
            )
        )

        if relation:

            return (
                f"{name}, your {relation}"
            )

        return name

    @staticmethod
    def _format_person_time(
        detected_at,
    ) -> str:

        if isinstance(
            detected_at,
            datetime,
        ):

            return (
                detected_at.strftime(
                    "%I:%M %p"
                )
                .lstrip(
                    "0"
                )
            )

        return (
            str(
                detected_at
                or "an unknown time"
            )
        )

    @staticmethod
    def _join_person_items(
        items: list[
            str
        ],
    ) -> str:

        cleaned = [
            item

            for item
            in items

            if item
        ]

        if not cleaned:

            return ""

        if len(
            cleaned
        ) == 1:

            return (
                cleaned[
                    0
                ]
            )

        if len(
            cleaned
        ) == 2:

            return (
                f"{cleaned[0]} "
                f"and {cleaned[1]}"
            )

        return (
            ", ".join(
                cleaned[
                    :-1
                ]
            )
            + f", and {cleaned[-1]}"
        )

    # ========================================================
    # LATEST SPECIFIC PERSON
    # ========================================================

    @classmethod
    def _answer_latest_person(
        cls,
        memory: dict,
    ) -> str:

        label = (
            cls._person_label(
                memory
            )
        )

        detected_at = (
            memory.get(
                "detected_at"
            )
        )

        if not isinstance(
            detected_at,
            datetime,
        ):

            return (
                f"I remember seeing "
                f"{label}, but I do not "
                "have a reliable time "
                f"for that encounter."
            )

        time_text = (
            cls._format_person_time(
                detected_at
            )
        )

        today = (
            date.today()
        )

        if (
            detected_at.date()
            == today
        ):

            return (
                f"I last saw {label} "
                f"today at {time_text}."
            )

        yesterday = (
            today
            - timedelta(
                days=1
            )
        )

        if (
            detected_at.date()
            == yesterday
        ):

            return (
                f"I last saw {label} "
                f"yesterday at {time_text}."
            )

        date_text = (
            detected_at.strftime(
                "%d %B %Y"
            )
        )

        return (
            f"I last saw {label} "
            f"on {date_text} "
            f"at {time_text}."
        )

    # ========================================================
    # LAST PERSON GENERIC
    # ========================================================

    @classmethod
    def _answer_last_person(
        cls,
        memory: dict,
    ) -> str:

        label = (
            cls._person_label(
                memory
            )
        )

        detected_at = (
            memory.get(
                "detected_at"
            )
        )

        if not isinstance(
            detected_at,
            datetime,
        ):

            return (
                "The last person I remember "
                f"seeing was {label}."
            )

        time_text = (
            cls._format_person_time(
                detected_at
            )
        )

        if (
            detected_at.date()
            == date.today()
        ):

            return (
                "The last person I remember "
                f"seeing was {label}, "
                f"today at {time_text}."
            )

        yesterday = (
            date.today()
            - timedelta(
                days=1
            )
        )

        if (
            detected_at.date()
            == yesterday
        ):

            return (
                "The last person I remember "
                f"seeing was {label}, "
                f"yesterday at {time_text}."
            )

        date_text = (
            detected_at.strftime(
                "%d %B %Y"
            )
        )

        return (
            "The last person I remember "
            f"seeing was {label}, "
            f"on {date_text} "
            f"at {time_text}."
        )

    # ========================================================
    # SPECIFIC PEOPLE ON DAY
    # ========================================================

    @classmethod
    def _answer_specific_people_on_day(
        cls,
        people: list[
            dict
        ],
        memories: list[
            dict
        ],
        day_word: str,
    ) -> str:

        if not memories:

            labels = [
                cls._person_label(
                    person
                )

                for person
                in people
            ]

            if not labels:

                return (
                    FALLBACK_ANSWER
                )

            return (
                "I do not have a reliable "
                "memory of seeing "
                f"{cls._join_person_items(labels)} "
                f"{day_word}."
            )

        latest_by_person: dict[
            int,
            dict,
        ] = {}

        for memory in memories:

            person_id = int(
                memory.get(
                    "person_id",
                    -1,
                )
            )

            existing = (
                latest_by_person.get(
                    person_id
                )
            )

            if existing is None:

                latest_by_person[
                    person_id
                ] = memory

                continue

            current_time = (
                memory.get(
                    "detected_at"
                )
            )

            existing_time = (
                existing.get(
                    "detected_at"
                )
            )

            if (
                isinstance(
                    current_time,
                    datetime,
                )
                and isinstance(
                    existing_time,
                    datetime,
                )
                and current_time
                > existing_time
            ):

                latest_by_person[
                    person_id
                ] = memory

        records = list(
            latest_by_person.values()
        )

        records.sort(
            key=lambda record: (
                record.get(
                    "detected_at"
                )
                or datetime.min
            ),
            reverse=True,
        )

        descriptions = []

        for record in records:

            label = (
                cls._person_label(
                    record
                )
            )

            time_text = (
                cls._format_person_time(
                    record.get(
                        "detected_at"
                    )
                )
            )

            descriptions.append(
                f"{label} at {time_text}"
            )

        return (
            "Yes. I saw "
            f"{cls._join_person_items(descriptions)} "
            f"{day_word}."
        )

    # ========================================================
    # GENERIC PEOPLE ON DAY
    # ========================================================

    @classmethod
    def _answer_people_on_day(
        cls,
        memories: list[
            dict
        ],
        day_word: str,
    ) -> str:

        if not memories:

            return (
                "I do not have a reliable "
                "memory of seeing anyone "
                f"{day_word}."
            )

        latest_by_person: dict[
            int,
            dict,
        ] = {}

        for memory in memories:

            person_id = int(
                memory.get(
                    "person_id",
                    -1,
                )
            )

            existing = (
                latest_by_person.get(
                    person_id
                )
            )

            if existing is None:

                latest_by_person[
                    person_id
                ] = memory

                continue

            current_time = (
                memory.get(
                    "detected_at"
                )
            )

            existing_time = (
                existing.get(
                    "detected_at"
                )
            )

            if (
                isinstance(
                    current_time,
                    datetime,
                )
                and isinstance(
                    existing_time,
                    datetime,
                )
                and current_time
                > existing_time
            ):

                latest_by_person[
                    person_id
                ] = memory

        records = list(
            latest_by_person.values()
        )

        records.sort(
            key=lambda record: (
                record.get(
                    "detected_at"
                )
                or datetime.min
            ),
            reverse=True,
        )

        descriptions = []

        for record in records:

            label = (
                cls._person_label(
                    record
                )
            )

            time_text = (
                cls._format_person_time(
                    record.get(
                        "detected_at"
                    )
                )
            )

            descriptions.append(
                f"{label} at {time_text}"
            )

        return (
            "I saw "
            f"{cls._join_person_items(descriptions)} "
            f"{day_word}."
        )

    # ========================================================
    # RECENT PEOPLE
    # ========================================================

    @classmethod
    def _answer_recent_people(
        cls,
        memories: list[
            dict
        ],
    ) -> str:

        if not memories:

            return (
                FALLBACK_ANSWER
            )

        labels = []

        seen_person_ids = set()

        for memory in memories:

            person_id = (
                memory.get(
                    "person_id"
                )
            )

            if (
                person_id
                in seen_person_ids
            ):

                continue

            seen_person_ids.add(
                person_id
            )

            labels.append(
                cls._person_label(
                    memory
                )
            )

        if not labels:

            return (
                FALLBACK_ANSWER
            )

        return (
            "The people I remember seeing "
            "recently are "
            f"{cls._join_person_items(labels)}."
        )

    # ========================================================
    # PERSON QUESTION
    # ========================================================

    def _answer_person_question(
        self,
        question: str,
        top_k: int,
    ) -> str:

        lowered = (
            question.lower()
        )

        people = (
            self.person_retriever
            .resolve_people(
                question
            )
        )

        memories = (
            self.person_retriever
            .retrieve(
                question,
                top_k=max(
                    top_k,
                    50,
                ),
            )
        )

        if (
            "today"
            in lowered
        ):

            if people:

                return (
                    self._answer_specific_people_on_day(
                        people=(
                            people
                        ),
                        memories=(
                            memories
                        ),
                        day_word=(
                            "today"
                        ),
                    )
                )

            return (
                self._answer_people_on_day(
                    memories=(
                        memories
                    ),
                    day_word=(
                        "today"
                    ),
                )
            )

        if (
            "yesterday"
            in lowered
        ):

            if people:

                return (
                    self._answer_specific_people_on_day(
                        people=(
                            people
                        ),
                        memories=(
                            memories
                        ),
                        day_word=(
                            "yesterday"
                        ),
                    )
                )

            return (
                self._answer_people_on_day(
                    memories=(
                        memories
                    ),
                    day_word=(
                        "yesterday"
                    ),
                )
            )

        generic_last_phrases = (
            "who was the last person",
            "last person i saw",
            "latest person",
            "most recent person",
        )

        if (
            not people
            and any(
                phrase
                in lowered

                for phrase
                in generic_last_phrases
            )
        ):

            if not memories:

                return (
                    FALLBACK_ANSWER
                )

            return (
                self._answer_last_person(
                    memories[
                        0
                    ]
                )
            )

        if people:

            if not memories:

                return (
                    FALLBACK_ANSWER
                )

            return (
                self._answer_latest_person(
                    memories[
                        0
                    ]
                )
            )

        if not memories:

            return (
                FALLBACK_ANSWER
            )

        return (
            self._answer_recent_people(
                memories
            )
        )

    # ========================================================
    # MAIN ANSWER
    # ========================================================

    def answer(
        self,
        question: str,
        top_k: int = DEFAULT_TOP_K,
    ) -> str:

        cleaned_question = (
            question.strip()
        )

        if not cleaned_question:

            return (
                "Please ask a "
                "memory-related question."
            )

        # ====================================================
        # 1. CONVERSATION MEMORY
        #
        # Must come before Person Memory.
        #
        # Example:
        #
        # What did Rahul tell me?
        #
        # This is Conversation Memory even though Rahul may
        # also be a registered person.
        # ====================================================

        if (
            self.conversation_retriever
            .is_conversation_question(
                cleaned_question
            )
        ):

            return (
                self._answer_conversation_question(
                    question=(
                        cleaned_question
                    ),
                    top_k=(
                        top_k
                    ),
                )
            )

        # ====================================================
        # 2. MEDICINE MEMORY
        #
        # Must come before Person/Object routing.
        # ====================================================

        if (
            self.medicine_retriever
            .is_medicine_question(
                cleaned_question
            )
        ):

            return (
                self._answer_medicine_question(
                    question=(
                        cleaned_question
                    ),
                    top_k=(
                        top_k
                    ),
                )
            )

        # ====================================================
        # 3. PERSON MEMORY
        # ====================================================

        if (
            self.person_retriever
            .is_person_question(
                cleaned_question
            )
        ):

            return (
                self._answer_person_question(
                    cleaned_question,
                    top_k=(
                        top_k
                    ),
                )
            )

        # ====================================================
        # 4. OBJECT MEMORY
        # ====================================================

        memories = (
            self.retriever.retrieve(
                cleaned_question,
                top_k=(
                    top_k
                ),
            )
        )

        # ====================================================
        # OBJECT NOTHING FOUND
        #
        # IMPORTANT:
        #
        # Do NOT fall through to an unrestricted Conversation
        # semantic search here.
        #
        # Conversation Memory already had its routing
        # opportunity above.
        #
        # Example of the old unsafe behaviour:
        #
        #     Where are my car keys?
        #
        # Object memory:
        #     no result
        #
        # Old cross-module fallback:
        #     semantic Conversation search
        #
        # Result:
        #     unrelated doctor appointment
        #
        # Correct:
        #
        #     I could not find a reliable memory for that.
        # ====================================================

        if not memories:

            return (
                FALLBACK_ANSWER
            )

        latest_memory = (
            memories[
                0
            ]
        )

        # ====================================================
        # TODAY
        # ====================================================

        if (
            self._is_today_question(
                cleaned_question
            )
        ):

            today_memories = (
                self._filter_memories_by_date(
                    memories=(
                        memories
                    ),
                    target_date=(
                        date.today()
                    ),
                )
            )

            if not today_memories:

                return (
                    FALLBACK_ANSWER
                )

            return (
                self._answer_today_question(
                    today_memories
                )
            )

        # ====================================================
        # YESTERDAY
        # ====================================================

        if (
            self._is_yesterday_question(
                cleaned_question
            )
        ):

            yesterday = (
                date.today()
                - timedelta(
                    days=1
                )
            )

            yesterday_memories = (
                self._filter_memories_by_date(
                    memories=(
                        memories
                    ),
                    target_date=(
                        yesterday
                    ),
                )
            )

            if not yesterday_memories:

                return (
                    FALLBACK_ANSWER
                )

            return (
                self._answer_yesterday_question(
                    yesterday_memories
                )
            )

        # ====================================================
        # RECENT
        # ====================================================

        if (
            self._is_recent_question(
                cleaned_question
            )
        ):

            return (
                self._answer_recent_question(
                    memories
                )
            )

        # ====================================================
        # LOCATION
        # ====================================================

        if (
            self._is_location_question(
                cleaned_question
            )
        ):

            return (
                self._answer_location_question(
                    latest_memory
                )
            )

        # ====================================================
        # TIME
        # ====================================================

        if (
            self._is_time_question(
                cleaned_question
            )
        ):

            return (
                self._answer_time_question(
                    latest_memory
                )
            )

        # ====================================================
        # OBJECT NON-LLM FALLBACK
        # ====================================================

        if not (
            self.is_ollama_running()
        ):

            memory_text = (
                latest_memory.get(
                    "memory_text"
                )
            )

            if memory_text:

                return (
                    "The most relevant "
                    "stored memory says: "
                    f"{memory_text}"
                )

            return (
                "The most relevant stored "
                f"memory is: "
                f"{latest_memory}"
            )

        # ====================================================
        # OBJECT RAG
        # ====================================================

        formatted_memories = (
            self._format_memories(
                memories
            )
        )

        return (
            self._answer_with_ollama(
                question=(
                    cleaned_question
                ),
                formatted_memories=(
                    formatted_memories
                ),
                memory_kind=(
                    "object memory"
                ),
            )
        )


# ============================================================
# TERMINAL CHAT
# ============================================================

def main() -> None:

    chat = (
        MemorAiChat()
    )

    print(
        "MemorAi memory chat"
    )

    print(
        "Type 'exit' to close."
    )

    while True:

        try:

            question = input(
                "\nYou: "
            ).strip()

        except (
            KeyboardInterrupt,
            EOFError,
        ):

            print(
                "\nClosing MemorAi."
            )

            break

        if (
            question.lower()
            in {
                "exit",
                "quit",
                "close",
            }
        ):

            print(
                "MemorAi: Goodbye."
            )

            break

        response = (
            chat.answer(
                question
            )
        )

        print(
            f"MemorAi: {response}"
        )


if __name__ == "__main__":

    main()