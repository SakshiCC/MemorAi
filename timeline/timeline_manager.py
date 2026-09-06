from __future__ import annotations

from collections import defaultdict

from datetime import (
    date,
    datetime,
    time,
    timedelta,
)

from typing import Any

from conversation.conversation_repository import (
    get_conversations_between,
    get_recent_conversations,
)

from conversation.metadata_extractor import (
    ConversationMetadataExtractor,
)

from database.event_repository import (
    count_events,
    get_events_for_date,
    get_recent_events,
    search_events,
)

from database.person_repository import (
    get_person_events_between,
    get_recent_person_events,
)

from database.medicine_repository import (
    get_medicine_events_between,
    get_recent_medicine_events,
)


# ============================================================
# FILTER CONSTANTS
# ============================================================

FILTER_TODAY = "today"

FILTER_YESTERDAY = "yesterday"

FILTER_DATE = "date"

# Kept for backward compatibility.
# The NiceGUI Timeline page no longer exposes this option.
FILTER_ALL = "all"


# ============================================================
# MEMORY TYPE CONSTANTS
# ============================================================

MEMORY_TYPE_ALL = "all"

MEMORY_TYPE_OBJECT = "object"

MEMORY_TYPE_PERSON = "person"

MEMORY_TYPE_CONVERSATION = "conversation"

MEMORY_TYPE_MEDICINE = "medicine"


# ============================================================
# PARTICIPANT RESOLVER
# ============================================================

_participant_resolver = (
    ConversationMetadataExtractor()
)


# ============================================================
# HELPERS
# ============================================================

def _safe_limit(
    limit: int,
    maximum: int = 10000,
) -> int:

    try:

        value = int(
            limit
        )

    except (
        TypeError,
        ValueError,
    ):

        value = 50

    return max(
        1,
        min(
            value,
            maximum,
        ),
    )


def _clean_text(
    value,
) -> str:

    return (
        str(
            value
            or ""
        )
        .strip()
    )


def _normalize_search_text(
    value,
) -> str:

    return (
        " ".join(
            _clean_text(
                value
            )
            .lower()
            .replace(
                "_",
                " ",
            )
            .split()
        )
    )


# ============================================================
# CANONICAL PARTICIPANT
# ============================================================

def _canonical_participant(
    value,
) -> str:
    """
    Resolve a stored conversation participant against
    registered people.

    Example:

        Stored participant:
            Saksi

        Registered person:
            Sakshi

        Timeline participant:
            Sakshi

    The original transcript and database value are not
    modified here.
    """

    cleaned = (
        _clean_text(
            value
        )
    )

    if not cleaned:

        return ""

    try:

        resolved = (
            _participant_resolver
            .display_participant_name(
                cleaned
            )
        )

        if resolved:

            return (
                str(
                    resolved
                )
                .strip()
            )

    except Exception:

        pass

    return (
        cleaned
    )


# ============================================================
# MEMORY TYPE NORMALIZATION
# ============================================================

def normalize_memory_type(
    memory_type: str | None,
) -> str:

    cleaned = (
        _clean_text(
            memory_type
        )
        .lower()
    )

    if not cleaned:

        return (
            MEMORY_TYPE_ALL
        )

    aliases = {

        "all":
            MEMORY_TYPE_ALL,

        "object":
            MEMORY_TYPE_OBJECT,

        "objects":
            MEMORY_TYPE_OBJECT,

        "object_detected":
            MEMORY_TYPE_OBJECT,

        "person":
            MEMORY_TYPE_PERSON,

        "people":
            MEMORY_TYPE_PERSON,

        "person_detected":
            MEMORY_TYPE_PERSON,

        "person_encounter":
            MEMORY_TYPE_PERSON,

        "conversation":
            MEMORY_TYPE_CONVERSATION,

        "conversations":
            MEMORY_TYPE_CONVERSATION,

        "important_conversation":
            MEMORY_TYPE_CONVERSATION,

        "medicine":
            MEMORY_TYPE_MEDICINE,

        "medicines":
            MEMORY_TYPE_MEDICINE,

        "medicine_event":
            MEMORY_TYPE_MEDICINE,

        "medicine_events":
            MEMORY_TYPE_MEDICINE,

        "camera_confirmation":
            MEMORY_TYPE_MEDICINE,
    }

    return (
        aliases.get(
            cleaned,
            cleaned,
        )
    )


# ============================================================
# ORIGINAL OBJECT TIMELINE
# ============================================================

def build_timeline(
    target_date: date | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """
    Original object-only timeline.

    Preserved for backward compatibility.
    """

    events = (
        get_events_for_date(
            target_date
        )
        if target_date
        is not None
        else get_recent_events(
            limit=limit
        )
    )

    timeline: list[
        dict[str, Any]
    ] = []

    for event in events:

        detected_at = (
            event.get(
                "detected_at"
            )
        )

        timeline.append(
            {
                "id":
                    event.get(
                        "id"
                    ),

                "event_type":
                    event.get(
                        "event_type"
                    ),

                "time":
                    detected_at,

                "object_name":
                    event.get(
                        "object_name"
                    ),

                "location":
                    event.get(
                        "location"
                    ),

                "confidence":
                    event.get(
                        "confidence"
                    ),

                "image_path":
                    event.get(
                        "image_path"
                    ),

                "memory_text":
                    event.get(
                        "memory_text"
                    ),
            }
        )

    return (
        timeline
    )


# ============================================================
# GROUP BY DATE
# ============================================================

def group_timeline_by_date(
    events: list[
        dict[str, Any]
    ],
) -> dict[
    str,
    list[
        dict[str, Any]
    ],
]:

    grouped: dict[
        str,
        list[
            dict[str, Any]
        ],
    ] = defaultdict(
        list
    )

    for event in events:

        raw_time = (
            event.get(
                "time"
            )
            or event.get(
                "detected_at"
            )
        )

        if isinstance(
            raw_time,
            datetime,
        ):

            key = (
                raw_time
                .date()
                .isoformat()
            )

        elif raw_time:

            key = (
                str(
                    raw_time
                )[:10]
            )

        else:

            key = (
                "unknown"
            )

        grouped[
            key
        ].append(
            event
        )

    return (
        dict(
            grouped
        )
    )


# ============================================================
# PERIOD RANGE
# ============================================================

def resolve_period_range(
    period: str,
    selected_date: date | None = None,
) -> tuple[
    datetime | None,
    datetime | None,
]:

    normalized_period = (
        str(
            period
            or FILTER_TODAY
        )
        .strip()
        .lower()
    )

    # ========================================================
    # TODAY
    # ========================================================

    if (
        normalized_period
        == FILTER_TODAY
    ):

        target = (
            date.today()
        )

        return (
            datetime.combine(
                target,
                time.min,
            ),

            datetime.combine(
                target,
                time.max,
            ),
        )

    # ========================================================
    # YESTERDAY
    # ========================================================

    if (
        normalized_period
        == FILTER_YESTERDAY
    ):

        target = (
            date.today()
            - timedelta(
                days=1
            )
        )

        return (
            datetime.combine(
                target,
                time.min,
            ),

            datetime.combine(
                target,
                time.max,
            ),
        )

    # ========================================================
    # SPECIFIC DATE
    # ========================================================

    if (
        normalized_period
        == FILTER_DATE
    ):

        if selected_date is None:

            raise ValueError(
                (
                    "Please select a date "
                    "before showing memories."
                )
            )

        return (
            datetime.combine(
                selected_date,
                time.min,
            ),

            datetime.combine(
                selected_date,
                time.max,
            ),
        )

    # ========================================================
    # ALL
    # ========================================================

    if (
        normalized_period
        == FILTER_ALL
    ):

        return (
            None,
            None,
        )

    raise ValueError(
        (
            "Unknown timeline period: "
            f"{period}"
        )
    )


# ============================================================
# OBJECT MEMORY
# ============================================================

def _object_memory(
    event: dict[str, Any],
) -> dict[str, Any]:

    object_name = (
        _clean_text(
            event.get(
                "object_name"
            )
        )
        or "Object"
    )

    location = (
        _clean_text(
            event.get(
                "location"
            )
        )
        or "Unknown location"
    )

    memory_text = (
        _clean_text(
            event.get(
                "memory_text"
            )
        )
    )

    if not memory_text:

        memory_text = (
            f"{object_name} was observed "
            f"in {location}."
        )

    return {

        "memory_key":
            f"object_{event.get('id')}",

        "memory_type":
            MEMORY_TYPE_OBJECT,

        "id":
            event.get(
                "id"
            ),

        "time":
            event.get(
                "detected_at"
            ),

        "title":
            object_name,

        "description":
            memory_text,

        "image_path":
            event.get(
                "image_path"
            ),

        "event_type":
            (
                event.get(
                    "event_type"
                )
                or "object_detected"
            ),

        "object_name":
            event.get(
                "object_name"
            ),

        "location":
            event.get(
                "location"
            ),

        "confidence":
            event.get(
                "confidence"
            ),

        "memory_text":
            memory_text,

        "person_id":
            None,

        "person_name":
            None,

        "relation":
            None,

        "similarity":
            None,

        "conversation_id":
            None,

        "conversation_title":
            None,

        "participant_text":
            None,

        "participant_raw":
            None,

        "summary":
            None,

        "full_transcript":
            None,

        "medicine_id":
            None,

        "medicine_name":
            None,

        "dosage":
            None,

        "medicine_status":
            None,

        "medicine_similarity":
            None,

        "person_similarity":
            None,

        "action_confidence":
            None,

        "scheduled_for":
            None,

        "confirmation_method":
            None,

        "metadata":
            dict(
                event
            ),
    }


# ============================================================
# PERSON MEMORY
# ============================================================

def _person_memory(
    event: dict[str, Any],
) -> dict[str, Any]:

    person_name = (
        _clean_text(
            event.get(
                "person_name"
            )
        )
        or "Unknown person"
    )

    relation = (
        _clean_text(
            event.get(
                "relation"
            )
        )
    )

    memory_text = (
        _clean_text(
            event.get(
                "memory_text"
            )
        )
    )

    if not memory_text:

        if relation:

            memory_text = (
                f"MemorAi encountered "
                f"{person_name}, your "
                f"{relation}."
            )

        else:

            memory_text = (
                f"MemorAi encountered "
                f"{person_name}."
            )

    return {

        "memory_key":
            f"person_{event.get('id')}",

        "memory_type":
            MEMORY_TYPE_PERSON,

        "id":
            event.get(
                "id"
            ),

        "time":
            event.get(
                "detected_at"
            ),

        "title":
            person_name,

        "description":
            memory_text,

        "image_path":
            event.get(
                "image_path"
            ),

        "event_type":
            "person_encounter",

        "person_id":
            event.get(
                "person_id"
            ),

        "person_name":
            person_name,

        "relation":
            event.get(
                "relation"
            ),

        "similarity":
            event.get(
                "similarity"
            ),

        "memory_text":
            memory_text,

        "object_name":
            None,

        "location":
            None,

        "confidence":
            None,

        "conversation_id":
            None,

        "conversation_title":
            None,

        "participant_text":
            None,

        "participant_raw":
            None,

        "summary":
            None,

        "full_transcript":
            None,

        "medicine_id":
            None,

        "medicine_name":
            None,

        "dosage":
            None,

        "medicine_status":
            None,

        "medicine_similarity":
            None,

        "person_similarity":
            None,

        "action_confidence":
            None,

        "scheduled_for":
            None,

        "confirmation_method":
            None,

        "metadata":
            dict(
                event
            ),
    }


# ============================================================
# CONVERSATION MEMORY
# ============================================================

def _conversation_memory(
    conversation: dict[str, Any],
) -> dict[str, Any]:

    conversation_id = (
        conversation.get(
            "id"
        )
    )

    title = (
        _clean_text(
            conversation.get(
                "title"
            )
        )
        or "Important Conversation"
    )

    raw_participant = (
        _clean_text(
            conversation.get(
                "participant_text"
            )
        )
    )

    canonical_participant = (
        _canonical_participant(
            raw_participant
        )
    )

    summary = (
        _clean_text(
            conversation.get(
                "summary"
            )
        )
    )

    transcript = (
        _clean_text(
            conversation.get(
                "full_transcript"
            )
        )
    )

    description = (
        summary
        or transcript
        or "Important conversation remembered."
    )

    return {

        "memory_key":
            f"conversation_{conversation_id}",

        "memory_type":
            MEMORY_TYPE_CONVERSATION,

        "id":
            conversation_id,

        "time":
            conversation.get(
                "started_at"
            ),

        "title":
            title,

        "description":
            description,

        "image_path":
            None,

        "event_type":
            "important_conversation",

        "memory_text":
            description,

        "conversation_id":
            conversation_id,

        "conversation_title":
            title,

        "participant_text":
            (
                canonical_participant
                or None
            ),

        "participant_raw":
            (
                raw_participant
                or None
            ),

        "summary":
            (
                summary
                or None
            ),

        "full_transcript":
            (
                transcript
                or None
            ),

        "ended_at":
            conversation.get(
                "ended_at"
            ),

        "object_name":
            None,

        "location":
            None,

        "confidence":
            None,

        "person_id":
            None,

        "person_name":
            None,

        "relation":
            None,

        "similarity":
            None,

        "medicine_id":
            None,

        "medicine_name":
            None,

        "dosage":
            None,

        "medicine_status":
            None,

        "medicine_similarity":
            None,

        "person_similarity":
            None,

        "action_confidence":
            None,

        "scheduled_for":
            None,

        "confirmation_method":
            None,

        "metadata":
            dict(
                conversation
            ),
    }


# ============================================================
# MEDICINE MEMORY
# ============================================================

def _medicine_memory(
    event: dict[str, Any],
) -> dict[str, Any]:

    medicine_name = (
        _clean_text(
            event.get(
                "medicine_name"
            )
        )
        or "Medicine"
    )

    dosage = (
        _clean_text(
            event.get(
                "dosage"
            )
        )
    )

    person_name = (
        _clean_text(
            event.get(
                "person_name"
            )
        )
    )

    person_relation = (
        _clean_text(
            event.get(
                "person_relation"
            )
        )
    )

    memory_text = (
        _clean_text(
            event.get(
                "memory_text"
            )
        )
    )

    if not memory_text:

        medicine_description = (
            medicine_name
        )

        if dosage:

            medicine_description += (
                f" {dosage}"
            )

        memory_text = (
            "MemorAi observed a "
            "camera-confirmed medication "
            f"event for {medicine_description}."
        )

    return {

        "memory_key":
            f"medicine_{event.get('id')}",

        "memory_type":
            MEMORY_TYPE_MEDICINE,

        "id":
            event.get(
                "id"
            ),

        "time":
            event.get(
                "observed_at"
            ),

        "title":
            medicine_name,

        "description":
            memory_text,

        "image_path":
            event.get(
                "evidence_image_path"
            ),

        "event_type":
            (
                event.get(
                    "event_type"
                )
                or "camera_confirmation"
            ),

        "memory_text":
            memory_text,

        "medicine_id":
            event.get(
                "medicine_id"
            ),

        "medicine_name":
            medicine_name,

        "dosage":
            (
                dosage
                or None
            ),

        "medicine_status":
            event.get(
                "status"
            ),

        "medicine_similarity":
            event.get(
                "medicine_similarity"
            ),

        "person_similarity":
            event.get(
                "person_similarity"
            ),

        "action_confidence":
            event.get(
                "action_confidence"
            ),

        "scheduled_for":
            event.get(
                "scheduled_for"
            ),

        "confirmation_method":
            event.get(
                "confirmation_method"
            ),

        "confidence":
            event.get(
                "confidence"
            ),

        "person_id":
            event.get(
                "person_id"
            ),

        "person_name":
            (
                person_name
                or None
            ),

        "relation":
            (
                person_relation
                or None
            ),

        "object_name":
            None,

        "location":
            None,

        "similarity":
            None,

        "conversation_id":
            None,

        "conversation_title":
            None,

        "participant_text":
            None,

        "participant_raw":
            None,

        "summary":
            None,

        "full_transcript":
            None,

        "instructions":
            event.get(
                "instructions"
            ),

        "metadata":
            dict(
                event
            ),
    }


# ============================================================
# SEARCH MATCH
# ============================================================

def _matches_search(
    memory: dict[str, Any],
    search_text: str | None,
) -> bool:

    cleaned_search = (
        _normalize_search_text(
            search_text
        )
    )

    if not cleaned_search:

        return (
            True
        )

    searchable_values = (

        memory.get(
            "title"
        ),

        memory.get(
            "description"
        ),

        memory.get(
            "object_name"
        ),

        memory.get(
            "location"
        ),

        memory.get(
            "person_name"
        ),

        memory.get(
            "relation"
        ),

        memory.get(
            "conversation_title"
        ),

        memory.get(
            "participant_text"
        ),

        memory.get(
            "participant_raw"
        ),

        memory.get(
            "summary"
        ),

        memory.get(
            "full_transcript"
        ),

        memory.get(
            "medicine_name"
        ),

        memory.get(
            "dosage"
        ),

        memory.get(
            "medicine_status"
        ),

        memory.get(
            "instructions"
        ),

        memory.get(
            "memory_text"
        ),
    )

    searchable_text = (
        " ".join(
            _normalize_search_text(
                value
            )

            for value
            in searchable_values

            if value
        )
    )

    return (
        cleaned_search
        in searchable_text
    )


# ============================================================
# TIMELINE SORT KEY
# ============================================================

def _timeline_sort_key(
    memory: dict[str, Any],
) -> tuple[
    datetime,
    int,
]:

    value = (
        memory.get(
            "time"
        )
    )

    if isinstance(
        value,
        datetime,
    ):

        parsed_time = (
            value
        )

    elif value:

        try:

            parsed_time = (
                datetime.fromisoformat(
                    str(
                        value
                    )
                    .replace(
                        "Z",
                        "+00:00",
                    )
                )
                .replace(
                    tzinfo=None
                )
            )

        except (
            TypeError,
            ValueError,
        ):

            parsed_time = (
                datetime.min
            )

    else:

        parsed_time = (
            datetime.min
        )

    memory_type = (
        str(
            memory.get(
                "memory_type"
            )
            or ""
        )
        .strip()
        .lower()
    )

    type_priority = {

        MEMORY_TYPE_OBJECT:
            4,

        MEMORY_TYPE_PERSON:
            3,

        MEMORY_TYPE_CONVERSATION:
            2,

        MEMORY_TYPE_MEDICINE:
            1,
    }

    priority = (
        type_priority.get(
            memory_type,
            0,
        )
    )

    return (
        parsed_time,
        priority,
    )


# ============================================================
# LOAD OBJECT MEMORIES
# ============================================================

def _load_object_memories(
    start: datetime | None,
    end: datetime | None,
    search_text: str | None,
    limit: int,
) -> list[dict[str, Any]]:

    events = (
        search_events(
            start=start,

            end=end,

            search_text=(
                search_text
            ),

            event_type=None,

            limit=(
                limit
            ),
        )
    )

    return [
        _object_memory(
            event
        )

        for event
        in events
    ]


# ============================================================
# LOAD PERSON MEMORIES
# ============================================================

def _load_person_memories(
    start: datetime | None,
    end: datetime | None,
    search_text: str | None,
    limit: int,
) -> list[dict[str, Any]]:

    if (
        start is not None
        and end is not None
    ):

        events = (
            get_person_events_between(
                start_time=(
                    start
                ),

                end_time=(
                    end
                ),

                person_id=None,
            )
        )

    else:

        events = (
            get_recent_person_events(
                limit=max(
                    limit,
                    1000,
                )
            )
        )

    memories = [
        _person_memory(
            event
        )

        for event
        in events
    ]

    memories = [
        memory

        for memory
        in memories

        if _matches_search(
            memory,
            search_text,
        )
    ]

    return (
        memories[
            :limit
        ]
    )


# ============================================================
# LOAD CONVERSATION MEMORIES
# ============================================================

def _load_conversation_memories(
    start: datetime | None,
    end: datetime | None,
    search_text: str | None,
    limit: int,
) -> list[dict[str, Any]]:

    if (
        start is not None
        and end is not None
    ):

        conversations = (
            get_conversations_between(
                start=(
                    start
                ),

                end=(
                    end
                ),

                completed_only=True,
            )
        )

    else:

        conversations = (
            get_recent_conversations(
                limit=max(
                    limit,
                    1000,
                ),

                completed_only=True,
            )
        )

    memories = [
        _conversation_memory(
            conversation
        )

        for conversation
        in conversations

        if (
            str(
                conversation.get(
                    "status"
                )
                or ""
            )
            .strip()
            .lower()
            == "completed"
        )
    ]

    memories = [
        memory

        for memory
        in memories

        if _matches_search(
            memory,
            search_text,
        )
    ]

    return (
        memories[
            :limit
        ]
    )


# ============================================================
# LOAD MEDICINE MEMORIES
# ============================================================

def _load_medicine_memories(
    start: datetime | None,
    end: datetime | None,
    search_text: str | None,
    limit: int,
) -> list[dict[str, Any]]:

    if (
        start is not None
        and end is not None
    ):

        events = (
            get_medicine_events_between(
                start_time=(
                    start
                ),

                end_time=(
                    end
                ),

                medicine_id=None,

                confirmed_only=True,
            )
        )

    else:

        events = (
            get_recent_medicine_events(
                limit=max(
                    limit,
                    1000,
                ),

                confirmed_only=True,
            )
        )

    memories = [
        _medicine_memory(
            event
        )

        for event
        in events
    ]

    memories = [
        memory

        for memory
        in memories

        if _matches_search(
            memory,
            search_text,
        )
    ]

    return (
        memories[
            :limit
        ]
    )


# ============================================================
# SEARCH UNIFIED TIMELINE
# ============================================================

def search_timeline(
    period: str = FILTER_TODAY,
    selected_date: date | None = None,
    search_text: str | None = None,
    limit: int = 50,
    event_type: str | None = None,
) -> list[dict[str, Any]]:
    """
    Search the complete MemorAi timeline.

    Memory sources:

        Object detections
        Person encounters
        Important conversations
        Camera-confirmed medicine events

    Ordering:

        newest first
    """

    start, end = (
        resolve_period_range(
            period=(
                period
            ),

            selected_date=(
                selected_date
            ),
        )
    )

    safe_limit = (
        _safe_limit(
            limit
        )
    )

    requested_type = (
        normalize_memory_type(
            event_type
        )
    )

    source_limit = max(
        safe_limit * 4,
        100,
    )

    combined: list[
        dict[str, Any]
    ] = []

    # ========================================================
    # OBJECTS
    # ========================================================

    if requested_type in {
        MEMORY_TYPE_ALL,
        MEMORY_TYPE_OBJECT,
    }:

        combined.extend(
            _load_object_memories(
                start=(
                    start
                ),

                end=(
                    end
                ),

                search_text=(
                    search_text
                ),

                limit=(
                    source_limit
                ),
            )
        )

    # ========================================================
    # PEOPLE
    # ========================================================

    if requested_type in {
        MEMORY_TYPE_ALL,
        MEMORY_TYPE_PERSON,
    }:

        combined.extend(
            _load_person_memories(
                start=(
                    start
                ),

                end=(
                    end
                ),

                search_text=(
                    search_text
                ),

                limit=(
                    source_limit
                ),
            )
        )

    # ========================================================
    # CONVERSATIONS
    # ========================================================

    if requested_type in {
        MEMORY_TYPE_ALL,
        MEMORY_TYPE_CONVERSATION,
    }:

        combined.extend(
            _load_conversation_memories(
                start=(
                    start
                ),

                end=(
                    end
                ),

                search_text=(
                    search_text
                ),

                limit=(
                    source_limit
                ),
            )
        )

    # ========================================================
    # MEDICINES
    # ========================================================

    if requested_type in {
        MEMORY_TYPE_ALL,
        MEMORY_TYPE_MEDICINE,
    }:

        combined.extend(
            _load_medicine_memories(
                start=(
                    start
                ),

                end=(
                    end
                ),

                search_text=(
                    search_text
                ),

                limit=(
                    source_limit
                ),
            )
        )

    combined.sort(
        key=(
            _timeline_sort_key
        ),
        reverse=True,
    )

    return (
        combined[
            :safe_limit
        ]
    )


# ============================================================
# COUNT PERSON MEMORIES
# ============================================================

def _count_person_memories(
    start: datetime | None,
    end: datetime | None,
    search_text: str | None,
) -> int:

    if (
        start is not None
        and end is not None
    ):

        events = (
            get_person_events_between(
                start_time=(
                    start
                ),

                end_time=(
                    end
                ),

                person_id=None,
            )
        )

    else:

        events = (
            get_recent_person_events(
                limit=10000
            )
        )

    memories = [
        _person_memory(
            event
        )

        for event
        in events
    ]

    return sum(
        1

        for memory
        in memories

        if _matches_search(
            memory,
            search_text,
        )
    )


# ============================================================
# COUNT CONVERSATION MEMORIES
# ============================================================

def _count_conversation_memories(
    start: datetime | None,
    end: datetime | None,
    search_text: str | None,
) -> int:

    if (
        start is not None
        and end is not None
    ):

        conversations = (
            get_conversations_between(
                start=(
                    start
                ),

                end=(
                    end
                ),

                completed_only=True,
            )
        )

    else:

        conversations = (
            get_recent_conversations(
                limit=10000,

                completed_only=True,
            )
        )

    memories = [
        _conversation_memory(
            conversation
        )

        for conversation
        in conversations

        if (
            str(
                conversation.get(
                    "status"
                )
                or ""
            )
            .strip()
            .lower()
            == "completed"
        )
    ]

    return sum(
        1

        for memory
        in memories

        if _matches_search(
            memory,
            search_text,
        )
    )


# ============================================================
# COUNT MEDICINE MEMORIES
# ============================================================

def _count_medicine_memories(
    start: datetime | None,
    end: datetime | None,
    search_text: str | None,
) -> int:

    if (
        start is not None
        and end is not None
    ):

        events = (
            get_medicine_events_between(
                start_time=(
                    start
                ),

                end_time=(
                    end
                ),

                medicine_id=None,

                confirmed_only=True,
            )
        )

    else:

        events = (
            get_recent_medicine_events(
                limit=10000,

                confirmed_only=True,
            )
        )

    memories = [
        _medicine_memory(
            event
        )

        for event
        in events
    ]

    return sum(
        1

        for memory
        in memories

        if _matches_search(
            memory,
            search_text,
        )
    )


# ============================================================
# COUNT UNIFIED TIMELINE
# ============================================================

def count_timeline(
    period: str = FILTER_TODAY,
    selected_date: date | None = None,
    search_text: str | None = None,
    event_type: str | None = None,
) -> int:

    start, end = (
        resolve_period_range(
            period=(
                period
            ),

            selected_date=(
                selected_date
            ),
        )
    )

    requested_type = (
        normalize_memory_type(
            event_type
        )
    )

    total = 0

    # ========================================================
    # OBJECTS
    # ========================================================

    if requested_type in {
        MEMORY_TYPE_ALL,
        MEMORY_TYPE_OBJECT,
    }:

        total += (
            count_events(
                start=(
                    start
                ),

                end=(
                    end
                ),

                search_text=(
                    search_text
                ),

                event_type=None,
            )
        )

    # ========================================================
    # PEOPLE
    # ========================================================

    if requested_type in {
        MEMORY_TYPE_ALL,
        MEMORY_TYPE_PERSON,
    }:

        total += (
            _count_person_memories(
                start=(
                    start
                ),

                end=(
                    end
                ),

                search_text=(
                    search_text
                ),
            )
        )

    # ========================================================
    # CONVERSATIONS
    # ========================================================

    if requested_type in {
        MEMORY_TYPE_ALL,
        MEMORY_TYPE_CONVERSATION,
    }:

        total += (
            _count_conversation_memories(
                start=(
                    start
                ),

                end=(
                    end
                ),

                search_text=(
                    search_text
                ),
            )
        )

    # ========================================================
    # MEDICINES
    # ========================================================

    if requested_type in {
        MEMORY_TYPE_ALL,
        MEMORY_TYPE_MEDICINE,
    }:

        total += (
            _count_medicine_memories(
                start=(
                    start
                ),

                end=(
                    end
                ),

                search_text=(
                    search_text
                ),
            )
        )

    return (
        total
    )