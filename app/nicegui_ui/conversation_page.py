from __future__ import annotations

from datetime import (
    date,
    datetime,
    timedelta,
)

from nicegui import ui

from app.nicegui_ui.layout import (
    sidebar,
)

from conversation.conversation_repository import (
    get_conversations_for_date,
    get_recent_conversations,
)

from conversation.conversation_settings import (
    get_conversation_settings,
)

from conversation.metadata_extractor import (
    ConversationMetadataExtractor,
)


# ============================================================
# PARTICIPANT NAME RESOLVER
# ============================================================

participant_name_resolver = (
    ConversationMetadataExtractor()
)


# ============================================================
# HELPERS
# ============================================================

def _format_datetime(
    value,
) -> str:

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

    return (
        "Unknown time"
    )


def _format_short_time(
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

        return (
            str(
                value
            )
        )

    return (
        "Unknown time"
    )


def _clean_text(
    value,
    fallback: str,
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
# DISPLAY CANONICAL PARTICIPANT
# ============================================================

def _display_participant(
    value,
) -> str:
    """
    Resolve old Whisper-derived participant spellings against
    registered people before displaying them.

    Example:

        stored:
            Saksi

        registered:
            Sakshi

        displayed:
            Sakshi

    The stored transcript remains unchanged.
    """

    cleaned = (
        str(
            value
            or ""
        )
        .strip()
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

    except Exception:

        resolved = None

    return (
        resolved
        or cleaned
    )


def _threshold_name(
    threshold: float,
) -> str:

    if threshold >= 0.70:

        return (
            "High Selectivity"
        )

    if threshold <= 0.50:

        return (
            "Remember More"
        )

    return (
        "Balanced"
    )


# ============================================================
# CONVERSATION PAGE
# ============================================================

def build_conversation_page() -> None:

    state = {
        "recording": False,
        "started_at": None,
    }

    # ========================================================
    # PAGE
    # ========================================================

    with ui.row().classes(
        "memorai-page items-start no-wrap"
    ):

        # ====================================================
        # SIDEBAR
        # ====================================================

        sidebar(
            active_path="/conversations"
        )

        # ====================================================
        # MAIN CONTENT
        # ====================================================

        with ui.column().classes(
            "memorai-main gap-6"
        ):

            # =================================================
            # HEADER
            # =================================================

            ui.label(
                "🗣 Conversations"
            ).classes(
                "text-4xl font-bold"
            )

            ui.label(
                (
                    "Allow MemorAi to remember only "
                    "important information from conversations."
                )
            ).classes(
                "text-lg opacity-70"
            )

            # =================================================
            # PRIVACY STATUS
            # =================================================

            with ui.card().classes(
                "memorai-card w-full"
            ):

                ui.label(
                    "🔒 Privacy Status"
                ).classes(
                    "memorai-section-title"
                )

                privacy_status = (
                    ui.label()
                    .classes(
                        "text-lg"
                    )
                )

                selectivity_status = (
                    ui.label()
                    .classes(
                        "opacity-70"
                    )
                )

                def refresh_privacy_status() -> bool:

                    settings = (
                        get_conversation_settings()
                    )

                    enabled = bool(
                        settings.get(
                            "memory_enabled",
                            False,
                        )
                    )

                    threshold = float(
                        settings.get(
                            "importance_threshold",
                            0.60,
                        )
                        or 0.60
                    )

                    if enabled:

                        privacy_status.set_text(
                            (
                                "✅ Conversation Memory "
                                "is enabled."
                            )
                        )

                    else:

                        privacy_status.set_text(
                            (
                                "🔒 Conversation Memory "
                                "is disabled."
                            )
                        )

                    selectivity_status.set_text(
                        (
                            "Current memory selectivity: "
                            f"{_threshold_name(threshold)}"
                        )
                    )

                    return (
                        enabled
                    )

                refresh_privacy_status()

                ui.button(
                    "Open Settings",
                    icon="settings",

                    on_click=lambda: (
                        ui.navigate.to(
                            "/settings"
                        )
                    ),
                ).props(
                    "outline"
                ).classes(
                    "memorai-action-button mt-2"
                )

            # =================================================
            # RECORD CONVERSATION
            # =================================================

            with ui.card().classes(
                "memorai-card w-full"
            ):

                ui.label(
                    "🎙 Record Conversation"
                ).classes(
                    "memorai-section-title"
                )

                ui.label(
                    (
                        "Recording starts only after you "
                        "press Start Conversation."
                    )
                ).classes(
                    "opacity-70"
                )

                # =============================================
                # PARTICIPANT
                # =============================================

                participant_input = (
                    ui.input(
                        label=(
                            "Who are you talking with? "
                            "(optional)"
                        ),

                        placeholder=(
                            "Example: Sakshi"
                        ),
                    )
                    .props(
                        "outlined"
                    )
                    .classes(
                        "w-full max-w-xl mt-3"
                    )
                )

                ui.label(
                    (
                        "Entering the person's name helps "
                        "MemorAi avoid speech-recognition "
                        "spelling mistakes."
                    )
                ).classes(
                    "text-sm opacity-60"
                )

                # =============================================
                # TITLE
                # =============================================

                title_input = (
                    ui.input(
                        label=(
                            "Conversation title "
                            "(optional)"
                        ),

                        placeholder=(
                            "Example: Doctor appointment"
                        ),
                    )
                    .props(
                        "outlined"
                    )
                    .classes(
                        "w-full max-w-xl"
                    )
                )

                ui.label(
                    (
                        "If left blank, MemorAi creates "
                        "a title automatically for an "
                        "important conversation."
                    )
                ).classes(
                    "text-sm opacity-60"
                )

                # =============================================
                # RECORDING STATUS
                # =============================================

                recording_status = (
                    ui.label(
                        "⚪ Ready"
                    )
                    .classes(
                        (
                            "text-xl "
                            "font-semibold "
                            "mt-3"
                        )
                    )
                )

                # ------------------------------------------------
                # References populated after result card creation.
                # ------------------------------------------------

                result_status = None

                transcript_label = None

                importance_label = None

                memory_id_label = None

                result_title_label = None

                result_participant_label = None

                result_summary_label = None


                # =============================================
                # START
                # =============================================

                async def start_conversation() -> None:

                    if state[
                        "recording"
                    ]:

                        ui.notify(
                            (
                                "A conversation is already "
                                "being recorded."
                            ),
                            type="warning",
                        )

                        return

                    if not refresh_privacy_status():

                        ui.notify(
                            (
                                "Conversation Memory is disabled. "
                                "Enable it in Settings first."
                            ),
                            type="warning",
                        )

                        return

                    try:

                        result = (
                            await ui.run_javascript(
                                """
                                return await (async () => {

                                    if (
                                        !navigator.mediaDevices
                                        ||
                                        !navigator.mediaDevices
                                            .getUserMedia
                                    ) {

                                        return {
                                            ok: false,
                                            error:
                                                'Microphone access is not supported by this browser.'
                                        };
                                    }

                                    if (
                                        window
                                            .memoraiConversationRecorder
                                        &&
                                        window
                                            .memoraiConversationRecorder
                                            .state === 'recording'
                                    ) {

                                        return {
                                            ok: false,
                                            error:
                                                'A conversation recording is already running.'
                                        };
                                    }

                                    try {

                                        const stream =
                                            await navigator
                                                .mediaDevices
                                                .getUserMedia({
                                                    audio: true
                                                });

                                        let options = {};

                                        if (
                                            MediaRecorder
                                                .isTypeSupported(
                                                    'audio/webm;codecs=opus'
                                                )
                                        ) {

                                            options = {
                                                mimeType:
                                                    'audio/webm;codecs=opus'
                                            };

                                        }

                                        else if (
                                            MediaRecorder
                                                .isTypeSupported(
                                                    'audio/webm'
                                                )
                                        ) {

                                            options = {
                                                mimeType:
                                                    'audio/webm'
                                            };

                                        }

                                        const recorder =
                                            new MediaRecorder(
                                                stream,
                                                options
                                            );

                                        window
                                            .memoraiConversationStream =
                                            stream;

                                        window
                                            .memoraiConversationChunks =
                                            [];

                                        window
                                            .memoraiConversationStartedAt =
                                            new Date()
                                                .toISOString();

                                        recorder
                                            .ondataavailable =
                                            event => {

                                                if (
                                                    event.data
                                                    &&
                                                    event.data.size > 0
                                                ) {

                                                    window
                                                        .memoraiConversationChunks
                                                        .push(
                                                            event.data
                                                        );
                                                }
                                            };

                                        window
                                            .memoraiConversationRecorder =
                                            recorder;

                                        recorder.start();

                                        return {
                                            ok: true,
                                            started_at:
                                                window
                                                    .memoraiConversationStartedAt
                                        };

                                    }

                                    catch (error) {

                                        return {
                                            ok: false,
                                            error:
                                                error.message
                                                ||
                                                String(error)
                                        };
                                    }

                                })();
                                """,

                                timeout=10.0,
                            )
                        )

                        if (
                            not result
                            or not result.get(
                                "ok",
                                False,
                            )
                        ):

                            message = (
                                result.get(
                                    "error",
                                    (
                                        "Could not start "
                                        "conversation recording."
                                    ),
                                )
                                if result
                                else (
                                    "Could not start "
                                    "conversation recording."
                                )
                            )

                            ui.notify(
                                message,
                                type="negative",
                            )

                            return

                        state[
                            "recording"
                        ] = True

                        state[
                            "started_at"
                        ] = result.get(
                            "started_at"
                        )

                        recording_status.set_text(
                            (
                                "🔴 Recording conversation..."
                            )
                        )

                        if result_status:

                            result_status.set_text(
                                (
                                    "Conversation is currently "
                                    "being recorded."
                                )
                            )

                        ui.notify(
                            (
                                "Conversation recording started."
                            ),
                            type="positive",
                        )

                    except Exception as exc:

                        state[
                            "recording"
                        ] = False

                        recording_status.set_text(
                            "🔴 Recording error"
                        )

                        ui.notify(
                            str(
                                exc
                            ),
                            type="negative",
                        )


                # =============================================
                # STOP
                # =============================================

                async def stop_conversation() -> None:

                    if not state[
                        "recording"
                    ]:

                        ui.notify(
                            (
                                "Start a conversation "
                                "recording first."
                            ),
                            type="warning",
                        )

                        return

                    state[
                        "recording"
                    ] = False

                    recording_status.set_text(
                        (
                            "🔵 Processing conversation..."
                        )
                    )

                    if result_status:

                        result_status.set_text(
                            (
                                "Transcribing and deciding "
                                "whether this conversation "
                                "should be remembered..."
                            )
                        )

                    participant = (
                        str(
                            participant_input.value
                            or ""
                        )
                        .strip()
                    )

                    title = (
                        str(
                            title_input.value
                            or ""
                        )
                        .strip()
                    )

                    try:

                        result = (
                            await ui.run_javascript(
                                f"""
                                return await (async () => {{

                                    const recorder =
                                        window
                                            .memoraiConversationRecorder;

                                    if (!recorder) {{

                                        return {{
                                            ok: false,
                                            error:
                                                'No active conversation recording was found.'
                                        }};
                                    }}

                                    try {{

                                        const stopped =
                                            new Promise(
                                                resolve => {{
                                                    recorder.onstop =
                                                        resolve;
                                                }}
                                            );

                                        if (
                                            recorder.state
                                            !== 'inactive'
                                        ) {{

                                            recorder.stop();
                                        }}

                                        await stopped;

                                        const chunks =
                                            window
                                                .memoraiConversationChunks
                                            || [];

                                        if (
                                            chunks.length === 0
                                        ) {{

                                            return {{
                                                ok: false,
                                                error:
                                                    'No conversation audio was recorded.'
                                            }};
                                        }}

                                        const mimeType =
                                            recorder.mimeType
                                            ||
                                            'audio/webm';

                                        const blob =
                                            new Blob(
                                                chunks,
                                                {{
                                                    type:
                                                        mimeType
                                                }}
                                            );

                                        if (
                                            window
                                                .memoraiConversationStream
                                        ) {{

                                            window
                                                .memoraiConversationStream
                                                .getTracks()
                                                .forEach(
                                                    track =>
                                                        track.stop()
                                                );
                                        }}

                                        const startedAt =
                                            window
                                                .memoraiConversationStartedAt
                                            || '';

                                        window
                                            .memoraiConversationRecorder =
                                            null;

                                        window
                                            .memoraiConversationStream =
                                            null;

                                        window
                                            .memoraiConversationChunks =
                                            [];

                                        window
                                            .memoraiConversationStartedAt =
                                            null;

                                        const response =
                                            await fetch(
                                                '/api/conversation/process',
                                                {{
                                                    method:
                                                        'POST',

                                                    headers: {{
                                                        'Content-Type':
                                                            mimeType,

                                                        'X-Conversation-Started-At':
                                                            startedAt,

                                                        'X-Conversation-Participant':
                                                            {participant!r},

                                                        'X-Conversation-Title':
                                                            {title!r}
                                                    }},

                                                    body:
                                                        blob
                                                }}
                                            );

                                        const data =
                                            await response
                                                .json();

                                        return data;

                                    }}

                                    catch (error) {{

                                        return {{
                                            ok: false,
                                            error:
                                                error.message
                                                ||
                                                String(error)
                                        }};
                                    }}

                                }})();
                                """,

                                timeout=180.0,
                            )
                        )

                        if not result:

                            raise RuntimeError(
                                (
                                    "Conversation processing "
                                    "returned no result."
                                )
                            )

                        if not result.get(
                            "ok",
                            False,
                        ):

                            raise RuntimeError(
                                result.get(
                                    "error",
                                    (
                                        "Conversation "
                                        "processing failed."
                                    ),
                                )
                            )

                        transcript = (
                            str(
                                result.get(
                                    "transcript",
                                    "",
                                )
                                or ""
                            )
                            .strip()
                        )

                        stored = bool(
                            result.get(
                                "stored",
                                False,
                            )
                        )

                        score = float(
                            result.get(
                                "importance_score",
                                0.0,
                            )
                            or 0.0
                        )

                        if transcript_label:

                            transcript_label.set_text(
                                (
                                    transcript
                                    if transcript
                                    else (
                                        "No transcript available."
                                    )
                                )
                            )

                        if importance_label:

                            importance_label.set_text(
                                (
                                    "Importance score: "
                                    f"{score:.2f}"
                                )
                            )

                        # =====================================
                        # IMPORTANT
                        # =====================================

                        if stored:

                            conversation_id = (
                                result.get(
                                    "conversation_id"
                                )
                            )

                            result_title = (
                                _clean_text(
                                    result.get(
                                        "title"
                                    ),
                                    (
                                        "Important "
                                        "Conversation"
                                    ),
                                )
                            )

                            result_participant = (
                                _display_participant(
                                    result.get(
                                        "participant_text"
                                    )
                                )
                            )

                            result_summary = (
                                _clean_text(
                                    result.get(
                                        "summary"
                                    ),
                                    (
                                        "No summary "
                                        "available."
                                    ),
                                )
                            )

                            recording_status.set_text(
                                (
                                    "🟢 Important conversation "
                                    "remembered"
                                )
                            )

                            if result_status:

                                result_status.set_text(
                                    (
                                        "MemorAi considered this "
                                        "conversation important "
                                        "and stored it as "
                                        "long-term memory."
                                    )
                                )

                            if memory_id_label:

                                memory_id_label.set_text(
                                    (
                                        "Conversation memory ID: "
                                        f"{conversation_id}"
                                    )
                                )

                            if result_title_label:

                                result_title_label.set_text(
                                    (
                                        "Title: "
                                        f"{result_title}"
                                    )
                                )

                            if result_participant_label:

                                result_participant_label.set_text(
                                    (
                                        "Participant: "
                                        f"{result_participant}"
                                    )
                                )

                            if result_summary_label:

                                result_summary_label.set_text(
                                    (
                                        "Summary: "
                                        f"{result_summary}"
                                    )
                                )

                            ui.notify(
                                (
                                    "Important conversation "
                                    "remembered."
                                ),
                                type="positive",
                            )

                            load_recent_conversations()

                        # =====================================
                        # NOT STORED
                        # =====================================

                        else:

                            recording_status.set_text(
                                (
                                    "⚪ Conversation not stored"
                                )
                            )

                            if result_status:

                                result_status.set_text(
                                    (
                                        "This conversation did "
                                        "not meet the selected "
                                        "importance threshold, "
                                        "so MemorAi discarded it."
                                    )
                                )

                            if memory_id_label:

                                memory_id_label.set_text(
                                    (
                                        "No long-term memory "
                                        "was created."
                                    )
                                )

                            if result_title_label:

                                result_title_label.set_text(
                                    "Title: —"
                                )

                            if result_participant_label:

                                result_participant_label.set_text(
                                    "Participant: —"
                                )

                            if result_summary_label:

                                result_summary_label.set_text(
                                    (
                                        "Summary: This "
                                        "conversation was "
                                        "not stored."
                                    )
                                )

                            ui.notify(
                                (
                                    "Conversation was not "
                                    "important enough to store."
                                ),
                                type="info",
                            )

                    except Exception as exc:

                        recording_status.set_text(
                            (
                                "🔴 Conversation "
                                "processing failed"
                            )
                        )

                        if result_status:

                            result_status.set_text(
                                (
                                    "MemorAi could not process "
                                    "this conversation."
                                )
                            )

                        ui.notify(
                            str(
                                exc
                            ),
                            type="negative",
                        )


                # =============================================
                # CANCEL
                # =============================================

                async def cancel_conversation() -> None:

                    try:

                        await ui.run_javascript(
                            """
                            return await (async () => {

                                const recorder =
                                    window
                                        .memoraiConversationRecorder;

                                if (
                                    recorder
                                    &&
                                    recorder.state
                                    !== 'inactive'
                                ) {

                                    recorder.stop();
                                }

                                if (
                                    window
                                        .memoraiConversationStream
                                ) {

                                    window
                                        .memoraiConversationStream
                                        .getTracks()
                                        .forEach(
                                            track =>
                                                track.stop()
                                        );
                                }

                                window
                                    .memoraiConversationRecorder =
                                    null;

                                window
                                    .memoraiConversationStream =
                                    null;

                                window
                                    .memoraiConversationChunks =
                                    [];

                                window
                                    .memoraiConversationStartedAt =
                                    null;

                                return true;

                            })();
                            """,

                            timeout=5.0,
                        )

                    except Exception:

                        pass

                    state[
                        "recording"
                    ] = False

                    state[
                        "started_at"
                    ] = None

                    recording_status.set_text(
                        "⚪ Ready"
                    )

                    if result_status:

                        result_status.set_text(
                            (
                                "No conversation is currently "
                                "being recorded."
                            )
                        )

                    ui.notify(
                        (
                            "Conversation recording "
                            "cancelled."
                        )
                    )


                # =============================================
                # BUTTONS
                # =============================================

                with ui.row().classes(
                    "gap-3 flex-wrap mt-3"
                ):

                    ui.button(
                        "Start Conversation",
                        icon="mic",
                        on_click=(
                            start_conversation
                        ),
                    ).classes(
                        "memorai-action-button"
                    )

                    ui.button(
                        "Stop & Analyze",
                        icon="stop_circle",
                        on_click=(
                            stop_conversation
                        ),
                    ).classes(
                        "memorai-action-button"
                    )

                    ui.button(
                        "Cancel",
                        icon="close",
                        on_click=(
                            cancel_conversation
                        ),
                    ).props(
                        "outline"
                    ).classes(
                        "memorai-action-button"
                    )

            # =================================================
            # RESULT
            # =================================================

            with ui.card().classes(
                "memorai-card w-full"
            ):

                ui.label(
                    "🧠 Conversation Result"
                ).classes(
                    "memorai-section-title"
                )

                result_status = (
                    ui.label(
                        (
                            "No conversation is currently "
                            "being recorded."
                        )
                    )
                    .classes(
                        "text-lg"
                    )
                )

                ui.separator().classes(
                    "my-3"
                )

                ui.label(
                    "Transcript"
                ).classes(
                    "text-lg font-semibold"
                )

                transcript_label = (
                    ui.label(
                        (
                            "The transcript will appear "
                            "here after processing."
                        )
                    )
                    .classes(
                        (
                            "leading-relaxed "
                            "opacity-80"
                        )
                    )
                )

                importance_label = (
                    ui.label(
                        "Importance score: —"
                    )
                    .classes(
                        "mt-3"
                    )
                )

                memory_id_label = (
                    ui.label(
                        (
                            "No long-term memory "
                            "created yet."
                        )
                    )
                    .classes(
                        "opacity-70"
                    )
                )

                ui.separator().classes(
                    "my-3"
                )

                ui.label(
                    "Generated Memory Details"
                ).classes(
                    "text-lg font-semibold"
                )

                result_title_label = (
                    ui.label(
                        "Title: —"
                    )
                )

                result_participant_label = (
                    ui.label(
                        "Participant: —"
                    )
                )

                result_summary_label = (
                    ui.label(
                        "Summary: —"
                    )
                    .classes(
                        (
                            "leading-relaxed "
                            "opacity-80"
                        )
                    )
                )

            # =================================================
            # HISTORY
            # =================================================

            with ui.card().classes(
                "memorai-card w-full"
            ):

                ui.label(
                    "📚 Important Conversation History"
                ).classes(
                    "memorai-section-title"
                )

                ui.label(
                    (
                        "Only conversations MemorAi "
                        "considered important are shown here."
                    )
                ).classes(
                    "opacity-70"
                )

                ui.separator().classes(
                    "my-3"
                )

                # =============================================
                # DATE
                # =============================================

                with ui.row().classes(
                    (
                        "items-end gap-3 "
                        "flex-wrap"
                    )
                ):

                    date_input = (
                        ui.input(
                            label="Choose Date",

                            value=(
                                date.today()
                                .isoformat()
                            ),
                        )
                        .props(
                            "outlined type=date"
                        )
                        .classes(
                            "w-56"
                        )
                    )

                    history_status = (
                        ui.label(
                            ""
                        )
                        .classes(
                            "opacity-70"
                        )
                    )

                history_container = (
                    ui.column()
                    .classes(
                        "w-full gap-3 mt-4"
                    )
                )

                # =============================================
                # FULL CONVERSATION
                # =============================================

                def open_conversation_dialog(
                    conversation: dict,
                ) -> None:

                    title = (
                        _clean_text(
                            conversation.get(
                                "title"
                            ),
                            (
                                "Important "
                                "Conversation"
                            ),
                        )
                    )

                    participant = (
                        _display_participant(
                            conversation.get(
                                "participant_text"
                            )
                        )
                    )

                    summary = (
                        _clean_text(
                            conversation.get(
                                "summary"
                            ),
                            (
                                "No summary "
                                "available."
                            ),
                        )
                    )

                    transcript = (
                        _clean_text(
                            conversation.get(
                                "full_transcript"
                            ),
                            (
                                "No transcript "
                                "available."
                            ),
                        )
                    )

                    started_at = (
                        _format_datetime(
                            conversation.get(
                                "started_at"
                            )
                        )
                    )

                    with ui.dialog() as dialog:

                        with ui.card().classes(
                            (
                                "w-full "
                                "max-w-3xl "
                                "p-6"
                            )
                        ):

                            ui.label(
                                (
                                    f"🗣 {title}"
                                )
                            ).classes(
                                (
                                    "text-2xl "
                                    "font-bold"
                                )
                            )

                            ui.label(
                                (
                                    "Participant: "
                                    f"{participant}"
                                )
                            ).classes(
                                "text-lg"
                            )

                            ui.label(
                                started_at
                            ).classes(
                                "opacity-70"
                            )

                            ui.separator().classes(
                                "my-3"
                            )

                            ui.label(
                                "Summary"
                            ).classes(
                                (
                                    "text-lg "
                                    "font-semibold"
                                )
                            )

                            ui.label(
                                summary
                            ).classes(
                                "leading-relaxed"
                            )

                            ui.separator().classes(
                                "my-3"
                            )

                            ui.label(
                                (
                                    "Original Conversation "
                                    "Transcript"
                                )
                            ).classes(
                                (
                                    "text-lg "
                                    "font-semibold"
                                )
                            )

                            ui.label(
                                transcript
                            ).classes(
                                (
                                    "leading-relaxed "
                                    "whitespace-pre-wrap"
                                )
                            )

                            ui.separator().classes(
                                "my-3"
                            )

                            ui.label(
                                (
                                    "Conversation Memory ID: "
                                    f"{conversation.get('id')}"
                                )
                            ).classes(
                                "text-sm opacity-60"
                            )

                            ui.button(
                                "Close",
                                icon="close",
                                on_click=(
                                    dialog.close
                                ),
                            ).classes(
                                "memorai-action-button"
                            )

                    dialog.open()


                # =============================================
                # RENDER HISTORY
                # =============================================

                def render_history(
                    conversations: list[dict],
                    status_text: str,
                ) -> None:

                    history_container.clear()

                    history_status.set_text(
                        status_text
                    )

                    if not conversations:

                        with history_container:

                            ui.label(
                                (
                                    "No important "
                                    "conversations were "
                                    "remembered for this period."
                                )
                            ).classes(
                                (
                                    "text-lg "
                                    "opacity-70 "
                                    "py-4"
                                )
                            )

                        return

                    with history_container:

                        for conversation in (
                            conversations
                        ):

                            title = (
                                _clean_text(
                                    conversation.get(
                                        "title"
                                    ),
                                    (
                                        "Important "
                                        "Conversation"
                                    ),
                                )
                            )

                            # =============================
                            # CANONICAL PARTICIPANT
                            # =============================

                            participant = (
                                _display_participant(
                                    conversation.get(
                                        "participant_text"
                                    )
                                )
                            )

                            summary = (
                                _clean_text(
                                    conversation.get(
                                        "summary"
                                    ),
                                    (
                                        "No summary "
                                        "is available."
                                    ),
                                )
                            )

                            started_at = (
                                conversation.get(
                                    "started_at"
                                )
                            )

                            time_text = (
                                _format_short_time(
                                    started_at
                                )
                            )

                            date_text = (
                                started_at.strftime(
                                    "%d %B %Y"
                                )
                                if isinstance(
                                    started_at,
                                    datetime,
                                )
                                else ""
                            )

                            # =============================
                            # HISTORY CARD
                            # =============================

                            with ui.card().classes(
                                (
                                    "memorai-memory-card "
                                    "w-full"
                                )
                            ):

                                with ui.row().classes(
                                    (
                                        "w-full "
                                        "items-start "
                                        "justify-between "
                                        "gap-4"
                                    )
                                ):

                                    with ui.column().classes(
                                        "gap-1"
                                    ):

                                        ui.label(
                                            (
                                                f"🗣 {title}"
                                            )
                                        ).classes(
                                            (
                                                "text-xl "
                                                "font-semibold"
                                            )
                                        )

                                        ui.label(
                                            (
                                                "With: "
                                                f"{participant}"
                                            )
                                        ).classes(
                                            "opacity-80"
                                        )

                                        ui.label(
                                            (
                                                f"{date_text} "
                                                f"• {time_text}"
                                            )
                                        ).classes(
                                            (
                                                "text-sm "
                                                "opacity-60"
                                            )
                                        )

                                    ui.button(
                                        (
                                            "View Full "
                                            "Conversation"
                                        ),

                                        icon="visibility",

                                        on_click=(
                                            lambda
                                            current=conversation:
                                            open_conversation_dialog(
                                                current
                                            )
                                        ),
                                    ).props(
                                        "outline"
                                    ).classes(
                                        "memorai-action-button"
                                    )

                                ui.separator().classes(
                                    "my-2"
                                )

                                ui.label(
                                    "Summary"
                                ).classes(
                                    "font-semibold"
                                )

                                ui.label(
                                    summary
                                ).classes(
                                    (
                                        "leading-relaxed "
                                        "opacity-80"
                                    )
                                )


                # =============================================
                # RECENT
                # =============================================

                def load_recent_conversations() -> None:

                    try:

                        conversations = (
                            get_recent_conversations(
                                limit=10,
                                completed_only=True,
                            )
                        )

                        render_history(
                            conversations,
                            (
                                f"Showing "
                                f"{len(conversations)} "
                                "recent important "
                                "conversation(s)."
                            ),
                        )

                    except Exception as exc:

                        history_container.clear()

                        history_status.set_text(
                            (
                                "Could not load "
                                "conversation history."
                            )
                        )

                        ui.notify(
                            (
                                "Could not load "
                                "conversation history: "
                                f"{exc}"
                            ),
                            type="negative",
                        )


                # =============================================
                # TODAY
                # =============================================

                def load_today() -> None:

                    selected = (
                        date.today()
                    )

                    date_input.value = (
                        selected.isoformat()
                    )

                    try:

                        conversations = (
                            get_conversations_for_date(
                                target_date=(
                                    selected
                                ),
                                completed_only=True,
                            )
                        )

                        render_history(
                            conversations,
                            (
                                f"Showing "
                                f"{len(conversations)} "
                                "conversation(s) from today."
                            ),
                        )

                    except Exception as exc:

                        ui.notify(
                            (
                                "Could not load today's "
                                "conversations: "
                                f"{exc}"
                            ),
                            type="negative",
                        )


                # =============================================
                # YESTERDAY
                # =============================================

                def load_yesterday() -> None:

                    selected = (
                        date.today()
                        - timedelta(
                            days=1
                        )
                    )

                    date_input.value = (
                        selected.isoformat()
                    )

                    try:

                        conversations = (
                            get_conversations_for_date(
                                target_date=(
                                    selected
                                ),
                                completed_only=True,
                            )
                        )

                        render_history(
                            conversations,
                            (
                                f"Showing "
                                f"{len(conversations)} "
                                "conversation(s) "
                                "from yesterday."
                            ),
                        )

                    except Exception as exc:

                        ui.notify(
                            (
                                "Could not load "
                                "yesterday's conversations: "
                                f"{exc}"
                            ),
                            type="negative",
                        )


                # =============================================
                # LOAD SELECTED DATE
                # =============================================

                def load_selected_date() -> None:

                    raw_date = (
                        str(
                            date_input.value
                            or ""
                        )
                        .strip()
                    )

                    if not raw_date:

                        ui.notify(
                            (
                                "Please choose a date."
                            ),
                            type="warning",
                        )

                        return

                    try:

                        selected = (
                            date.fromisoformat(
                                raw_date
                            )
                        )

                    except ValueError:

                        ui.notify(
                            (
                                "Please choose a valid date."
                            ),
                            type="negative",
                        )

                        return

                    try:

                        conversations = (
                            get_conversations_for_date(
                                target_date=(
                                    selected
                                ),
                                completed_only=True,
                            )
                        )

                        render_history(
                            conversations,
                            (
                                f"Showing "
                                f"{len(conversations)} "
                                "conversation(s) from "
                                f"{selected.strftime('%d %B %Y')}."
                            ),
                        )

                    except Exception as exc:

                        ui.notify(
                            (
                                "Could not load conversation "
                                f"history: {exc}"
                            ),
                            type="negative",
                        )


                # =============================================
                # FILTER BUTTONS
                # =============================================

                with ui.row().classes(
                    "gap-3 flex-wrap mt-3"
                ):

                    ui.button(
                        "Today",
                        icon="today",
                        on_click=(
                            load_today
                        ),
                    ).props(
                        "outline"
                    ).classes(
                        "memorai-action-button"
                    )

                    ui.button(
                        "Yesterday",
                        icon="history",
                        on_click=(
                            load_yesterday
                        ),
                    ).props(
                        "outline"
                    ).classes(
                        "memorai-action-button"
                    )

                    ui.button(
                        "Load Date",
                        icon="calendar_month",
                        on_click=(
                            load_selected_date
                        ),
                    ).classes(
                        "memorai-action-button"
                    )

                    ui.button(
                        "Recent",
                        icon="schedule",
                        on_click=(
                            load_recent_conversations
                        ),
                    ).props(
                        "outline"
                    ).classes(
                        "memorai-action-button"
                    )

                    ui.button(
                        "Refresh",
                        icon="refresh",
                        on_click=(
                            load_recent_conversations
                        ),
                    ).props(
                        "outline"
                    ).classes(
                        "memorai-action-button"
                    )

                # =============================================
                # INITIAL HISTORY
                # =============================================

                load_recent_conversations()

            # =================================================
            # PRIVACY INFO
            # =================================================

            with ui.card().classes(
                "memorai-card w-full"
            ):

                ui.label(
                    "🛡 How Conversation Privacy Works"
                ).classes(
                    "memorai-section-title"
                )

                with ui.column().classes(
                    "gap-2"
                ):

                    ui.label(
                        (
                            "• Recording begins only when "
                            "you press Start Conversation."
                        )
                    )

                    ui.label(
                        (
                            "• Recording stops when you "
                            "press Stop & Analyze or Cancel."
                        )
                    )

                    ui.label(
                        (
                            "• Raw audio is temporary and "
                            "deleted after processing."
                        )
                    )

                    ui.label(
                        (
                            "• Casual conversations are "
                            "discarded."
                        )
                    )

                    ui.label(
                        (
                            "• Metadata is generated only "
                            "after a conversation passes "
                            "the importance check."
                        )
                    )

                    ui.label(
                        (
                            "• Registered-person names may "
                            "be used to correct clear "
                            "speech-recognition spelling "
                            "differences for display."
                        )
                    )

                    ui.label(
                        (
                            "• The original Whisper transcript "
                            "is preserved unchanged."
                        )
                    )