from __future__ import annotations

from datetime import (
    date,
    datetime,
)

from pathlib import Path

from nicegui import ui

from app.nicegui_ui.layout import (
    prepare_page,
    sidebar,
)

from conversation.metadata_extractor import (
    ConversationMetadataExtractor,
)

from timeline.timeline_manager import (
    FILTER_DATE,
    FILTER_TODAY,
    FILTER_YESTERDAY,
    MEMORY_TYPE_ALL,
    MEMORY_TYPE_CONVERSATION,
    MEMORY_TYPE_MEDICINE,
    MEMORY_TYPE_OBJECT,
    MEMORY_TYPE_PERSON,
    count_timeline,
    search_timeline,
)


# ============================================================
# PARTICIPANT DISPLAY RESOLVER
# ============================================================

participant_name_resolver = (
    ConversationMetadataExtractor()
)


# ============================================================
# CLEAN TEXT
# ============================================================

def _clean_text(
    value,
    fallback: str = "",
) -> str:

    cleaned = (
        str(
            value
            or ""
        )
        .strip()
    )

    return (
        cleaned
        if cleaned
        else fallback
    )


# ============================================================
# DISPLAY PARTICIPANT
# ============================================================

def _display_participant(
    value,
) -> str:

    cleaned = (
        _clean_text(
            value
        )
    )

    if not cleaned:

        return (
            "Participant not identified"
        )

    try:

        resolved = (
            participant_name_resolver
            .display_participant_name(
                cleaned
            )
        )

        return (
            resolved
            or cleaned
        )

    except Exception:

        return (
            cleaned
        )


# ============================================================
# FORMAT DATE
# ============================================================

def _format_date(
    value,
) -> str:

    if isinstance(
        value,
        datetime,
    ):

        return (
            value.strftime(
                "%d %b %Y"
            )
        )

    if value:

        try:

            parsed = (
                datetime.fromisoformat(
                    str(
                        value
                    )
                    .replace(
                        "Z",
                        "+00:00",
                    )
                )
            )

            return (
                parsed.strftime(
                    "%d %b %Y"
                )
            )

        except (
            TypeError,
            ValueError,
        ):

            pass

    return (
        "Unknown date"
    )


# ============================================================
# FORMAT TIME
# ============================================================

def _format_time(
    value,
) -> str:

    if isinstance(
        value,
        datetime,
    ):

        return (
            value.strftime(
                "%I:%M %p"
            )
            .lstrip(
                "0"
            )
        )

    if value:

        try:

            parsed = (
                datetime.fromisoformat(
                    str(
                        value
                    )
                    .replace(
                        "Z",
                        "+00:00",
                    )
                )
            )

            return (
                parsed.strftime(
                    "%I:%M %p"
                )
                .lstrip(
                    "0"
                )
            )

        except (
            TypeError,
            ValueError,
        ):

            pass

    return (
        "Unknown time"
    )


# ============================================================
# FORMAT CONFIDENCE
# ============================================================

def _format_confidence(
    value,
) -> str:

    if value is None:

        return (
            "Unknown"
        )

    try:

        confidence = float(
            value
        )

    except (
        TypeError,
        ValueError,
    ):

        return (
            str(
                value
            )
        )

    if confidence <= 1:

        confidence *= 100

    return (
        f"{confidence:.1f}%"
    )


# ============================================================
# FORMAT SIMILARITY
# ============================================================

def _format_similarity(
    value,
) -> str:

    if value is None:

        return (
            "Unknown"
        )

    try:

        similarity = float(
            value
        )

    except (
        TypeError,
        ValueError,
    ):

        return (
            str(
                value
            )
        )

    if similarity <= 1:

        similarity *= 100

    return (
        f"{similarity:.1f}%"
    )


# ============================================================
# FORMAT STATUS
# ============================================================

def _format_medicine_status(
    value,
) -> str:

    cleaned = (
        _clean_text(
            value
        )
        .replace(
            "_",
            " ",
        )
        .strip()
    )

    if not cleaned:

        return (
            "Unknown"
        )

    return (
        cleaned.title()
    )


# ============================================================
# RESOLVE EVIDENCE PATH
# ============================================================

def _resolve_evidence_path(
    image_path,
    evidence_dir: str | Path | None = None,
) -> Path | None:

    cleaned_path = (
        _clean_text(
            image_path
        )
    )

    if not cleaned_path:

        return None

    path = (
        Path(
            cleaned_path
        )
    )

    if path.exists():

        return (
            path
        )

    if evidence_dir is not None:

        base_dir = (
            Path(
                evidence_dir
            )
        )

        candidate = (
            base_dir
            / path.name
        )

        if candidate.exists():

            return (
                candidate
            )

    return None


# ============================================================
# SHOW EVIDENCE IMAGE
# ============================================================

def _show_evidence_image(
    image_path,
    evidence_dir: str | Path | None = None,
) -> None:

    path = (
        _resolve_evidence_path(
            image_path=(
                image_path
            ),

            evidence_dir=(
                evidence_dir
            ),
        )
    )

    if path is None:

        ui.notify(
            (
                "The evidence image "
                "could not be found."
            ),
            type="warning",
        )

        return

    with ui.dialog() as dialog:

        with ui.card().classes(
            "w-full max-w-3xl"
        ):

            ui.label(
                "Evidence Image"
            ).classes(
                "text-2xl font-bold"
            )

            ui.separator()

            ui.image(
                str(
                    path
                )
            ).classes(
                "w-full rounded-lg"
            )

            with ui.row().classes(
                "w-full justify-end"
            ):

                ui.button(
                    "Close",

                    on_click=(
                        dialog.close
                    ),
                ).props(
                    "outline no-caps"
                )

    dialog.open()


# ============================================================
# SHOW CONVERSATION
# ============================================================

def _show_conversation(
    memory: dict,
) -> None:

    title = (
        _clean_text(
            memory.get(
                "conversation_title"
            ),
            "Important Conversation",
        )
    )

    participant = (
        _display_participant(
            memory.get(
                "participant_text"
            )
        )
    )

    summary = (
        _clean_text(
            memory.get(
                "summary"
            ),
            "No summary available.",
        )
    )

    transcript = (
        _clean_text(
            memory.get(
                "full_transcript"
            ),
            "No transcript available.",
        )
    )

    started_at = (
        memory.get(
            "time"
        )
    )

    ended_at = (
        memory.get(
            "ended_at"
        )
    )

    with ui.dialog() as dialog:

        with ui.card().classes(
            "w-full max-w-3xl gap-4"
        ):

            ui.label(
                "🗣 Conversation Memory"
            ).classes(
                "text-2xl font-bold"
            )

            ui.label(
                title
            ).classes(
                "text-xl font-semibold"
            )

            ui.label(
                (
                    "With: "
                    f"{participant}"
                )
            ).classes(
                "text-base opacity-80"
            )

            ui.label(
                (
                    f"{_format_date(started_at)}"
                    " • "
                    f"{_format_time(started_at)}"
                )
            ).classes(
                "text-sm opacity-60"
            )

            if ended_at:

                ui.label(
                    (
                        "Ended: "
                        f"{_format_time(ended_at)}"
                    )
                ).classes(
                    "text-sm opacity-60"
                )

            ui.separator()

            ui.label(
                "Summary"
            ).classes(
                "text-lg font-semibold"
            )

            ui.label(
                summary
            ).classes(
                "text-base"
            )

            ui.separator()

            ui.label(
                "Original Transcript"
            ).classes(
                "text-lg font-semibold"
            )

            ui.label(
                transcript
            ).classes(
                (
                    "text-base "
                    "whitespace-pre-wrap"
                )
            )

            with ui.row().classes(
                "w-full justify-end"
            ):

                ui.button(
                    "Close",

                    on_click=(
                        dialog.close
                    ),
                ).props(
                    "outline no-caps"
                )

    dialog.open()


# ============================================================
# OBJECT CARD
# ============================================================

def _render_object_card(
    memory: dict,
    evidence_dir: str | Path | None = None,
) -> None:

    object_name = (
        _clean_text(
            memory.get(
                "object_name"
            ),
            "Object",
        )
    )

    location = (
        _clean_text(
            memory.get(
                "location"
            ),
            "Unknown location",
        )
    )

    memory_text = (
        _clean_text(
            memory.get(
                "memory_text"
            ),
            (
                f"{object_name} was observed "
                f"in {location}."
            ),
        )
    )

    timestamp = (
        memory.get(
            "time"
        )
    )

    with ui.card().classes(
        "memorai-card gap-3"
    ):

        with ui.row().classes(
            (
                "w-full "
                "items-center "
                "justify-between"
            )
        ):

            with ui.row().classes(
                "items-center gap-3"
            ):

                ui.label(
                    "📦"
                ).classes(
                    "text-3xl"
                )

                with ui.column().classes(
                    "gap-0"
                ):

                    ui.label(
                        "Object Memory"
                    ).classes(
                        (
                            "text-sm "
                            "font-semibold "
                            "opacity-60"
                        )
                    )

                    ui.label(
                        object_name
                    ).classes(
                        "text-xl font-bold"
                    )

            ui.badge(
                "Object"
            ).props(
                "outline"
            )

        ui.label(
            (
                f"📍 {location}"
            )
        ).classes(
            "text-base"
        )

        ui.label(
            (
                "Confidence: "
                f"{_format_confidence(memory.get('confidence'))}"
            )
        ).classes(
            "text-sm opacity-70"
        )

        ui.label(
            (
                f"{_format_date(timestamp)}"
                " • "
                f"{_format_time(timestamp)}"
            )
        ).classes(
            "text-sm opacity-60"
        )

        ui.separator()

        ui.label(
            memory_text
        ).classes(
            "text-base"
        )

        image_path = (
            memory.get(
                "image_path"
            )
        )

        if image_path:

            ui.button(
                "🖼 View Evidence",

                on_click=(
                    lambda p=image_path: (
                        _show_evidence_image(
                            image_path=(
                                p
                            ),

                            evidence_dir=(
                                evidence_dir
                            ),
                        )
                    )
                ),
            ).props(
                "outline no-caps"
            ).classes(
                "memorai-action-button"
            )


# ============================================================
# PERSON CARD
# ============================================================

def _render_person_card(
    memory: dict,
    evidence_dir: str | Path | None = None,
) -> None:

    person_name = (
        _clean_text(
            memory.get(
                "person_name"
            ),
            "Unknown person",
        )
    )

    relation = (
        _clean_text(
            memory.get(
                "relation"
            )
        )
    )

    timestamp = (
        memory.get(
            "time"
        )
    )

    memory_text = (
        _clean_text(
            memory.get(
                "memory_text"
            )
        )
    )

    with ui.card().classes(
        "memorai-card gap-3"
    ):

        with ui.row().classes(
            (
                "w-full "
                "items-center "
                "justify-between"
            )
        ):

            with ui.row().classes(
                "items-center gap-3"
            ):

                ui.label(
                    "👤"
                ).classes(
                    "text-3xl"
                )

                with ui.column().classes(
                    "gap-0"
                ):

                    ui.label(
                        "Person Encounter"
                    ).classes(
                        (
                            "text-sm "
                            "font-semibold "
                            "opacity-60"
                        )
                    )

                    ui.label(
                        person_name
                    ).classes(
                        "text-xl font-bold"
                    )

            ui.badge(
                "Person"
            ).props(
                "outline"
            )

        if relation:

            ui.label(
                (
                    "Relation: "
                    f"{relation}"
                )
            ).classes(
                "text-base"
            )

        ui.label(
            (
                "Recognition confidence: "
                f"{_format_similarity(memory.get('similarity'))}"
            )
        ).classes(
            "text-sm opacity-70"
        )

        ui.label(
            (
                f"{_format_date(timestamp)}"
                " • "
                f"{_format_time(timestamp)}"
            )
        ).classes(
            "text-sm opacity-60"
        )

        if memory_text:

            ui.separator()

            ui.label(
                memory_text
            ).classes(
                "text-base"
            )

        image_path = (
            memory.get(
                "image_path"
            )
        )

        if image_path:

            ui.button(
                "🖼 View Evidence",

                on_click=(
                    lambda p=image_path: (
                        _show_evidence_image(
                            image_path=(
                                p
                            ),

                            evidence_dir=(
                                evidence_dir
                            ),
                        )
                    )
                ),
            ).props(
                "outline no-caps"
            ).classes(
                "memorai-action-button"
            )


# ============================================================
# CONVERSATION CARD
# ============================================================

def _render_conversation_card(
    memory: dict,
) -> None:

    title = (
        _clean_text(
            memory.get(
                "conversation_title"
            ),
            "Important Conversation",
        )
    )

    participant = (
        _display_participant(
            memory.get(
                "participant_text"
            )
        )
    )

    summary = (
        _clean_text(
            (
                memory.get(
                    "summary"
                )
                or memory.get(
                    "description"
                )
            ),
            (
                "Important conversation "
                "remembered."
            ),
        )
    )

    timestamp = (
        memory.get(
            "time"
        )
    )

    with ui.card().classes(
        "memorai-card gap-3"
    ):

        with ui.row().classes(
            (
                "w-full "
                "items-center "
                "justify-between"
            )
        ):

            with ui.row().classes(
                "items-center gap-3"
            ):

                ui.label(
                    "🗣"
                ).classes(
                    "text-3xl"
                )

                with ui.column().classes(
                    "gap-0"
                ):

                    ui.label(
                        "Conversation Memory"
                    ).classes(
                        (
                            "text-sm "
                            "font-semibold "
                            "opacity-60"
                        )
                    )

                    ui.label(
                        title
                    ).classes(
                        "text-xl font-bold"
                    )

            ui.badge(
                "Conversation"
            ).props(
                "outline"
            )

        if (
            participant
            != "Participant not identified"
        ):

            ui.label(
                (
                    "With: "
                    f"{participant}"
                )
            ).classes(
                "text-base"
            )

        ui.label(
            (
                f"{_format_date(timestamp)}"
                " • "
                f"{_format_time(timestamp)}"
            )
        ).classes(
            "text-sm opacity-60"
        )

        ui.separator()

        ui.label(
            summary
        ).classes(
            "text-base"
        )

        ui.button(
            "💬 View Conversation",

            on_click=(
                lambda m=memory: (
                    _show_conversation(
                        m # type: ignore
                    )
                )
            ),
        ).props(
            "outline no-caps"
        ).classes(
            "memorai-action-button"
        )


# ============================================================
# MEDICINE CARD
# ============================================================

def _render_medicine_card(
    memory: dict,
    evidence_dir: str | Path | None = None,
) -> None:

    medicine_name = (
        _clean_text(
            memory.get(
                "medicine_name"
            ),
            "Medicine",
        )
    )

    dosage = (
        _clean_text(
            memory.get(
                "dosage"
            )
        )
    )

    person_name = (
        _clean_text(
            memory.get(
                "person_name"
            )
        )
    )

    relation = (
        _clean_text(
            memory.get(
                "relation"
            )
        )
    )

    timestamp = (
        memory.get(
            "time"
        )
    )

    scheduled_for = (
        memory.get(
            "scheduled_for"
        )
    )

    memory_text = (
        _clean_text(
            memory.get(
                "memory_text"
            ),
            (
                "MemorAi observed a "
                "camera-confirmed "
                "medication event."
            ),
        )
    )

    with ui.card().classes(
        "memorai-card gap-3"
    ):

        with ui.row().classes(
            (
                "w-full "
                "items-center "
                "justify-between"
            )
        ):

            with ui.row().classes(
                "items-center gap-3"
            ):

                ui.label(
                    "💊"
                ).classes(
                    "text-3xl"
                )

                with ui.column().classes(
                    "gap-0"
                ):

                    ui.label(
                        "Medicine Memory"
                    ).classes(
                        (
                            "text-sm "
                            "font-semibold "
                            "opacity-60"
                        )
                    )

                    ui.label(
                        medicine_name
                    ).classes(
                        "text-xl font-bold"
                    )

            ui.badge(
                "Medicine"
            ).props(
                "outline"
            )

        if dosage:

            ui.label(
                (
                    "Dosage: "
                    f"{dosage}"
                )
            ).classes(
                "text-base"
            )

        ui.label(
            (
                "Status: "
                f"{_format_medicine_status(memory.get('medicine_status'))}"
            )
        ).classes(
            "text-base font-semibold"
        )

        if person_name:

            person_line = (
                f"Observed with: "
                f"{person_name}"
            )

            if relation:

                person_line += (
                    f" ({relation})"
                )

            ui.label(
                person_line
            ).classes(
                "text-base"
            )

        medicine_similarity = (
            memory.get(
                "medicine_similarity"
            )
        )

        if medicine_similarity is not None:

            ui.label(
                (
                    "Medicine recognition: "
                    f"{_format_similarity(medicine_similarity)}"
                )
            ).classes(
                "text-sm opacity-70"
            )

        action_confidence = (
            memory.get(
                "action_confidence"
            )
        )

        if action_confidence is not None:

            ui.label(
                (
                    "Action confidence: "
                    f"{_format_confidence(action_confidence)}"
                )
            ).classes(
                "text-sm opacity-70"
            )

        person_similarity = (
            memory.get(
                "person_similarity"
            )
        )

        if person_similarity is not None:

            ui.label(
                (
                    "Person recognition: "
                    f"{_format_similarity(person_similarity)}"
                )
            ).classes(
                "text-sm opacity-70"
            )

        ui.label(
            (
                "Observed: "
                f"{_format_date(timestamp)}"
                " • "
                f"{_format_time(timestamp)}"
            )
        ).classes(
            "text-sm opacity-60"
        )

        if scheduled_for:

            ui.label(
                (
                    "Scheduled: "
                    f"{_format_date(scheduled_for)}"
                    " • "
                    f"{_format_time(scheduled_for)}"
                )
            ).classes(
                "text-sm opacity-60"
            )

        ui.separator()

        ui.label(
            memory_text
        ).classes(
            "text-base"
        )

        with ui.card().classes(
            "w-full bg-transparent border border-gray-300"
        ):

            ui.label(
                "Camera-assisted memory"
            ).classes(
                "font-semibold"
            )

            ui.label(
                (
                    "This record means MemorAi "
                    "observed camera evidence related "
                    "to the medicine. It does not prove "
                    "that the medicine was swallowed."
                )
            ).classes(
                "text-sm opacity-70"
            )

        image_path = (
            memory.get(
                "image_path"
            )
        )

        if image_path:

            ui.button(
                "🖼 View Evidence",

                on_click=(
                    lambda p=image_path: (
                        _show_evidence_image(
                            image_path=(
                                p
                            ),

                            evidence_dir=(
                                evidence_dir
                            ),
                        )
                    )
                ),
            ).props(
                "outline no-caps"
            ).classes(
                "memorai-action-button"
            )


# ============================================================
# RENDER MEMORY
# ============================================================

def _render_memory(
    memory: dict,
    evidence_dir: str | Path | None = None,
) -> None:

    memory_type = (
        _clean_text(
            memory.get(
                "memory_type"
            )
        )
        .lower()
    )

    if (
        memory_type
        == MEMORY_TYPE_OBJECT
    ):

        _render_object_card(
            memory=(
                memory
            ),

            evidence_dir=(
                evidence_dir
            ),
        )

        return

    if (
        memory_type
        == MEMORY_TYPE_PERSON
    ):

        _render_person_card(
            memory=(
                memory
            ),

            evidence_dir=(
                evidence_dir
            ),
        )

        return

    if (
        memory_type
        == MEMORY_TYPE_CONVERSATION
    ):

        _render_conversation_card(
            memory
        )

        return

    if (
        memory_type
        == MEMORY_TYPE_MEDICINE
    ):

        _render_medicine_card(
            memory=(
                memory
            ),

            evidence_dir=(
                evidence_dir
            ),
        )

        return

    with ui.card().classes(
        "memorai-card"
    ):

        ui.label(
            "Unknown Memory"
        ).classes(
            "text-xl font-bold"
        )

        ui.label(
            _clean_text(
                memory.get(
                    "description"
                ),
                "No description available.",
            )
        )


# ============================================================
# BUILD TIMELINE PAGE
# ============================================================

def build_timeline_page(
    evidence_dir: str | Path | None = None,
) -> None:

    prepare_page()

    state = {
        "period": (
            FILTER_TODAY
        ),
    }

    with ui.row().classes(
        "memorai-page items-start no-wrap"
    ):

        sidebar(
            active_path="/timeline"
        )

        with ui.column().classes(
            "memorai-main gap-6"
        ):

            ui.label(
                "🧠 Memory Timeline"
            ).classes(
                "text-4xl font-bold"
            )

            ui.label(
                (
                    "Review remembered objects, "
                    "people, important conversations "
                    "and camera-confirmed medicine events."
                )
            ).classes(
                "text-lg opacity-70"
            )

            with ui.card().classes(
                "memorai-card w-full gap-5"
            ):

                ui.label(
                    "Timeline Filters"
                ).classes(
                    "memorai-section-title"
                )

                ui.label(
                    (
                        "Choose when and what kind "
                        "of memories you want to review."
                    )
                ).classes(
                    "opacity-70"
                )

                ui.separator()

                result_container = None

                result_label = None

                selected_date = None

                memory_type_select = None

                search_input = None

                limit_select = None

                today_button = None

                yesterday_button = None

                date_button = None

                date_area = None

                # =================================================
                # PERIOD SELECTION
                # =================================================

                def set_period(
                    period: str,
                ) -> None:

                    state[
                        "period"
                    ] = (
                        period
                    )

                    if date_area is not None:

                        date_area.set_visibility(
                            period
                            == FILTER_DATE
                        )

                    if (
                        period
                        == FILTER_TODAY
                    ):

                        if today_button:

                            today_button.props(
                                remove="outline"
                            )

                        if yesterday_button:

                            yesterday_button.props(
                                "outline"
                            )

                        if date_button:

                            date_button.props(
                                "outline"
                            )

                    elif (
                        period
                        == FILTER_YESTERDAY
                    ):

                        if today_button:

                            today_button.props(
                                "outline"
                            )

                        if yesterday_button:

                            yesterday_button.props(
                                remove="outline"
                            )

                        if date_button:

                            date_button.props(
                                "outline"
                            )

                    else:

                        if today_button:

                            today_button.props(
                                "outline"
                            )

                        if yesterday_button:

                            yesterday_button.props(
                                "outline"
                            )

                        if date_button:

                            date_button.props(
                                remove="outline"
                            )

                # =================================================
                # LOAD MEMORIES
                # =================================================

                def load_memories() -> None:

                    if (
                        memory_type_select is None
                        or search_input is None
                        or limit_select is None
                        or result_container is None
                        or result_label is None
                    ):

                        return

                    period = (
                        str(
                            state.get(
                                "period"
                            )
                            or FILTER_TODAY
                        )
                    )

                    memory_type = (
                        str(
                            memory_type_select.value
                            or MEMORY_TYPE_ALL
                        )
                    )

                    search_text = (
                        _clean_text(
                            search_input.value
                        )
                        or None
                    )

                    try:

                        limit = int(
                            limit_select.value
                            or 50
                        )

                    except (
                        TypeError,
                        ValueError,
                    ):

                        limit = 50

                    target_date = None

                    if (
                        period
                        == FILTER_DATE
                    ):

                        if selected_date is None:

                            return

                        raw_date = (
                            str(
                                selected_date.value
                                or ""
                            )
                            .strip()
                        )

                        if not raw_date:

                            ui.notify(
                                (
                                    "Please select "
                                    "a date."
                                ),
                                type="warning",
                            )

                            return

                        try:

                            target_date = (
                                date.fromisoformat(
                                    raw_date
                                )
                            )

                        except ValueError:

                            ui.notify(
                                (
                                    "The selected date "
                                    "is invalid."
                                ),
                                type="negative",
                            )

                            return

                    try:

                        memories = (
                            search_timeline(
                                period=(
                                    period
                                ),

                                selected_date=(
                                    target_date
                                ),

                                search_text=(
                                    search_text
                                ),

                                limit=(
                                    limit
                                ),

                                event_type=(
                                    memory_type
                                ),
                            )
                        )

                        total = (
                            count_timeline(
                                period=(
                                    period
                                ),

                                selected_date=(
                                    target_date
                                ),

                                search_text=(
                                    search_text
                                ),

                                event_type=(
                                    memory_type
                                ),
                            )
                        )

                    except Exception as exc:

                        result_container.clear()

                        result_label.set_text(
                            (
                                "Could not load "
                                "memories."
                            )
                        )

                        ui.notify(
                            (
                                "Timeline error: "
                                f"{exc}"
                            ),
                            type="negative",
                        )

                        return

                    result_container.clear()

                    if not memories:

                        result_label.set_text(
                            (
                                "No matching "
                                "memories found."
                            )
                        )

                        with result_container:

                            with ui.card().classes(
                                "memorai-card"
                            ):

                                ui.label(
                                    "No memories found"
                                ).classes(
                                    (
                                        "text-xl "
                                        "font-semibold"
                                    )
                                )

                                ui.label(
                                    (
                                        "Try another date, "
                                        "memory type or "
                                        "search term."
                                    )
                                ).classes(
                                    "opacity-70"
                                )

                        return

                    if total > len(
                        memories
                    ):

                        result_label.set_text(
                            (
                                f"Showing "
                                f"{len(memories)} "
                                f"of {total} memories."
                            )
                        )

                    else:

                        result_label.set_text(
                            (
                                f"{len(memories)} "
                                "memories found."
                            )
                        )

                    with result_container:

                        current_date = None

                        for memory in memories:

                            memory_time = (
                                memory.get(
                                    "time"
                                )
                            )

                            memory_date = (
                                _format_date(
                                    memory_time
                                )
                            )

                            if (
                                memory_date
                                != current_date
                            ):

                                current_date = (
                                    memory_date
                                )

                                ui.label(
                                    current_date
                                ).classes(
                                    (
                                        "text-2xl "
                                        "font-bold mt-3"
                                    )
                                )

                                ui.separator()

                            _render_memory(
                                memory=(
                                    memory
                                ),

                                evidence_dir=(
                                    evidence_dir
                                ),
                            )

                # =================================================
                # CLEAR FILTERS
                # =================================================

                def clear_filters() -> None:

                    set_period(
                        FILTER_TODAY
                    )

                    if memory_type_select:

                        memory_type_select.value = (
                            MEMORY_TYPE_ALL
                        )

                    if selected_date:

                        selected_date.value = (
                            date.today()
                            .isoformat()
                        )

                    if search_input:

                        search_input.value = ""

                    if limit_select:

                        limit_select.value = 50

                    if result_container:

                        result_container.clear()

                    if result_label:

                        result_label.set_text(
                            (
                                "Press Show Memories "
                                "to load the timeline."
                            )
                        )

                # =================================================
                # PERIOD BUTTONS
                # =================================================

                ui.label(
                    "When?"
                ).classes(
                    "text-lg font-semibold"
                )

                with ui.row().classes(
                    (
                        "w-full gap-3 "
                        "items-center flex-wrap"
                    )
                ):

                    today_button = (
                        ui.button(
                            "Today",
                            icon="today",

                            on_click=lambda: (
                                set_period(
                                    FILTER_TODAY
                                )
                            ),
                        )
                        .props(
                            "no-caps"
                        )
                        .classes(
                            "memorai-action-button"
                        )
                    )

                    yesterday_button = (
                        ui.button(
                            "Yesterday",
                            icon="history",

                            on_click=lambda: (
                                set_period(
                                    FILTER_YESTERDAY
                                )
                            ),
                        )
                        .props(
                            "outline no-caps"
                        )
                        .classes(
                            "memorai-action-button"
                        )
                    )

                    date_button = (
                        ui.button(
                            "Specific Date",
                            icon="calendar_month",

                            on_click=lambda: (
                                set_period(
                                    FILTER_DATE
                                )
                            ),
                        )
                        .props(
                            "outline no-caps"
                        )
                        .classes(
                            "memorai-action-button"
                        )
                    )

                    ui.separator().props(
                        "vertical"
                    ).classes(
                        "h-10 mx-1"
                    )

                    ui.button(
                        "Show Memories",
                        icon="search",
                        on_click=(
                            load_memories
                        ),
                    ).classes(
                        "memorai-action-button"
                    )

                    ui.button(
                        "Clear Filters",
                        icon="restart_alt",
                        on_click=(
                            clear_filters
                        ),
                    ).props(
                        "outline no-caps"
                    ).classes(
                        "memorai-action-button"
                    )

                # =================================================
                # DATE AREA
                # =================================================

                date_area = (
                    ui.column()
                    .classes(
                        "gap-2"
                    )
                )

                with date_area:

                    ui.label(
                        "Choose a specific date"
                    ).classes(
                        "font-semibold"
                    )

                    selected_date = (
                        ui.input(
                            value=(
                                date.today()
                                .isoformat()
                            ),

                            label="Date",
                        )
                        .props(
                            "outlined type=date"
                        )
                        .classes(
                            "w-64"
                        )
                    )

                date_area.set_visibility(
                    False
                )

                ui.separator()

                # =================================================
                # MEMORY FILTERS
                # =================================================

                ui.label(
                    "What do you want to find?"
                ).classes(
                    "text-lg font-semibold"
                )

                with ui.row().classes(
                    (
                        "w-full gap-4 "
                        "items-end flex-wrap"
                    )
                ):

                    memory_type_select = (
                        ui.select(
                            options={

                                MEMORY_TYPE_ALL:
                                    "All Memories",

                                MEMORY_TYPE_OBJECT:
                                    "📦 Objects",

                                MEMORY_TYPE_PERSON:
                                    "👤 People Encountered",

                                MEMORY_TYPE_CONVERSATION:
                                    "🗣 Conversations",

                                MEMORY_TYPE_MEDICINE:
                                    "💊 Medicine Events",
                            },

                            value=(
                                MEMORY_TYPE_ALL
                            ),

                            label=(
                                "Memory Type"
                            ),
                        )
                        .props(
                            "outlined"
                        )
                        .classes(
                            "w-60"
                        )
                    )

                    search_input = (
                        ui.input(
                            label=(
                                "Search Memories"
                            ),

                            placeholder=(
                                (
                                    "phone, Sakshi, "
                                    "appointment, Omez..."
                                )
                            ),
                        )
                        .props(
                            "outlined clearable"
                        )
                        .classes(
                            (
                                "w-full "
                                "max-w-md"
                            )
                        )
                    )

                    limit_select = (
                        ui.select(
                            options=[
                                20,
                                50,
                                100,
                                200,
                            ],

                            value=50,

                            label=(
                                "Maximum Results"
                            ),
                        )
                        .props(
                            "outlined"
                        )
                        .classes(
                            "w-48"
                        )
                    )

                result_label = (
                    ui.label(
                        (
                            "Press Show Memories "
                            "to load the timeline."
                        )
                    )
                    .classes(
                        "text-base opacity-70 mt-2"
                    )
                )

            result_container = (
                ui.column()
                .classes(
                    "w-full gap-4"
                )
            )

            set_period(
                FILTER_TODAY
            )