from __future__ import annotations

from nicegui import ui

from app.nicegui_ui.browser_tts import (
    DEFAULT_SPEECH_RATE,
    DEFAULT_SPEECH_VOLUME,
    get_saved_speech_rate,
    get_saved_speech_volume,
    reset_speech_settings,
    save_speech_rate,
    save_speech_volume,
    speak_text,
    stop_speaking,
)

from app.nicegui_ui.layout import (
    apply_global_styles,
    sidebar,
)

from app.nicegui_ui.theme import (
    THEME_DARK,
    THEME_LIGHT,
    THEME_SYSTEM,
    apply_theme_to_controller,
    get_saved_theme,
    get_theme_description,
    save_theme,
)

from conversation.conversation_settings import (
    get_conversation_settings,
    set_conversation_memory_enabled,
    set_importance_threshold,
)


# ============================================================
# CONVERSATION IMPORTANCE LEVELS
# ============================================================

IMPORTANCE_HIGH = (
    "high"
)

IMPORTANCE_BALANCED = (
    "balanced"
)

IMPORTANCE_MORE = (
    "more"
)


IMPORTANCE_LEVEL_OPTIONS = {

    IMPORTANCE_HIGH:
        "High Selectivity",

    IMPORTANCE_BALANCED:
        "Balanced",

    IMPORTANCE_MORE:
        "Remember More",
}


IMPORTANCE_LEVEL_THRESHOLDS = {

    IMPORTANCE_HIGH:
        0.75,

    IMPORTANCE_BALANCED:
        0.60,

    IMPORTANCE_MORE:
        0.45,
}


# ============================================================
# SPEECH RATE OPTIONS
# ============================================================

SPEECH_RATE_OPTIONS = {

    "slow":
        "Slow",

    "normal":
        "Normal",

    "fast":
        "Faster",
}


SPEECH_RATE_VALUES = {

    "slow":
        0.80,

    "normal":
        0.95,

    "fast":
        1.10,
}


# ============================================================
# SPEECH VOLUME OPTIONS
# ============================================================

SPEECH_VOLUME_OPTIONS = {

    "quiet":
        "Quiet",

    "medium":
        "Medium",

    "full":
        "Full",
}


SPEECH_VOLUME_VALUES = {

    "quiet":
        0.60,

    "medium":
        0.80,

    "full":
        1.00,
}


# ============================================================
# IMPORTANCE HELPERS
# ============================================================

def _threshold_to_level(
    threshold: float,
) -> str:

    value = float(
        threshold
    )

    differences = {

        level: abs(
            value
            - threshold_value
        )

        for (
            level,
            threshold_value,
        ) in (
            IMPORTANCE_LEVEL_THRESHOLDS
            .items()
        )
    }

    return min(
        differences,
        key=differences.get,  # type: ignore
    )  # type: ignore


def _importance_description(
    level: str,
) -> str:

    if (
        level
        == IMPORTANCE_HIGH
    ):

        return (
            "MemorAi will remember only conversations "
            "with strong important-memory signals such "
            "as appointments, medicine instructions, "
            "important plans, reminders or commitments."
        )

    if (
        level
        == IMPORTANCE_MORE
    ):

        return (
            "MemorAi will remember more potentially useful "
            "conversations. Some lower-priority conversations "
            "may also qualify as long-term memories."
        )

    return (
        "Recommended. MemorAi ignores casual conversation "
        "while remembering useful appointments, reminders, "
        "plans, medicines and other important information."
    )


# ============================================================
# SPEECH HELPERS
# ============================================================

def _speech_rate_to_level(
    rate: float,
) -> str:

    differences = {

        level: abs(
            float(
                rate
            )
            - value
        )

        for (
            level,
            value,
        ) in (
            SPEECH_RATE_VALUES
            .items()
        )
    }

    return min(
        differences,
        key=differences.get,  # type: ignore
    )  # type: ignore


def _speech_volume_to_level(
    volume: float,
) -> str:

    differences = {

        level: abs(
            float(
                volume
            )
            - value
        )

        for (
            level,
            value,
        ) in (
            SPEECH_VOLUME_VALUES
            .items()
        )
    }

    return min(
        differences,
        key=differences.get,  # type: ignore
    )  # type: ignore


def _speech_rate_description(
    level: str,
) -> str:

    if level == "slow":

        return (
            "Speaks more slowly for easier listening."
        )

    if level == "fast":

        return (
            "Speaks a little faster while remaining clear."
        )

    return (
        "Recommended speaking speed."
    )


def _speech_volume_description(
    level: str,
) -> str:

    if level == "quiet":

        return (
            "Uses a softer browser speech volume."
        )

    if level == "medium":

        return (
            "Uses a moderate browser speech volume."
        )

    return (
        "Uses the browser speech voice at full volume."
    )


# ============================================================
# SETTINGS PAGE
# ============================================================

def build_settings_page() -> None:

    # ========================================================
    # THEME CONTROLLER
    # ========================================================

    dark_mode = (
        ui.dark_mode()
    )

    current_theme = (
        get_saved_theme()
    )

    apply_theme_to_controller(
        dark_mode,
        current_theme,
    )

    apply_global_styles()

    # ========================================================
    # CONVERSATION SETTINGS
    # ========================================================

    conversation_settings = (
        get_conversation_settings()
    )

    conversation_enabled = bool(
        conversation_settings.get(
            "memory_enabled",
            False,
        )
    )

    importance_threshold = float(
        conversation_settings.get(
            "importance_threshold",
            0.60,
        )
        or 0.60
    )

    current_importance_level = (
        _threshold_to_level(
            importance_threshold
        )
    )

    # ========================================================
    # SPEECH SETTINGS
    # ========================================================

    saved_speech_rate = (
        get_saved_speech_rate()
    )

    saved_speech_volume = (
        get_saved_speech_volume()
    )

    current_speech_rate_level = (
        _speech_rate_to_level(
            saved_speech_rate
        )
    )

    current_speech_volume_level = (
        _speech_volume_to_level(
            saved_speech_volume
        )
    )

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
            active_path="/settings"
        )

        # ====================================================
        # MAIN
        # ====================================================

        with ui.column().classes(
            "memorai-main gap-6"
        ):

            # =================================================
            # HEADER
            # =================================================

            ui.label(
                "⚙ Settings"
            ).classes(
                "text-4xl font-bold"
            )

            ui.label(
                (
                    "Choose how MemorAi looks, speaks, "
                    "and remembers important conversations."
                )
            ).classes(
                "text-lg opacity-70"
            )

            # =================================================
            # APPEARANCE
            # =================================================

            with ui.card().classes(
                "memorai-card w-full"
            ):

                ui.label(
                    "🎨 Appearance"
                ).classes(
                    "memorai-section-title"
                )

                ui.label(
                    (
                        "Choose the appearance that is "
                        "most comfortable for you."
                    )
                ).classes(
                    "opacity-70"
                )

                ui.separator().classes(
                    "my-3"
                )

                theme_select = (
                    ui.select(
                        options={

                            THEME_SYSTEM:
                                "🖥 System",

                            THEME_LIGHT:
                                "☀️ Light",

                            THEME_DARK:
                                "🌙 Dark",
                        },

                        value=(
                            current_theme
                        ),

                        label="Theme",
                    )
                    .props(
                        "outlined"
                    )
                    .classes(
                        "w-80"
                    )
                )

                description_label = (
                    ui.label(
                        get_theme_description(
                            current_theme
                        )
                    )
                    .classes(
                        "text-sm opacity-70"
                    )
                )

                def change_theme(
                    event,
                ) -> None:

                    selected_theme = (
                        str(
                            event.value
                        )
                    )

                    selected_theme = (
                        save_theme(
                            selected_theme
                        )
                    )

                    apply_theme_to_controller(
                        dark_mode,
                        selected_theme,
                    )

                    description_label.set_text(
                        get_theme_description(
                            selected_theme
                        )
                    )

                    if (
                        selected_theme
                        == THEME_LIGHT
                    ):

                        message = (
                            "Light theme enabled."
                        )

                    elif (
                        selected_theme
                        == THEME_DARK
                    ):

                        message = (
                            "Dark theme enabled."
                        )

                    else:

                        message = (
                            "System theme enabled."
                        )

                    ui.notify(
                        message,
                        type="positive",
                    )

                theme_select.on_value_change(
                    change_theme
                )

            # =================================================
            # SPOKEN ANSWERS
            # =================================================

            with ui.card().classes(
                "memorai-card w-full"
            ):

                ui.label(
                    "🔊 Spoken Answers"
                ).classes(
                    "memorai-section-title"
                )

                ui.label(
                    (
                        "Choose how MemorAi sounds when you "
                        "press Speak Answer."
                    )
                ).classes(
                    "text-lg"
                )

                ui.label(
                    (
                        "MemorAi never speaks automatically. "
                        "Speech begins only after you press "
                        "a Speak button."
                    )
                ).classes(
                    "opacity-70"
                )

                ui.separator().classes(
                    "my-3"
                )

                # =============================================
                # SPEECH RATE
                # =============================================

                ui.label(
                    "Speaking Speed"
                ).classes(
                    "text-lg font-semibold"
                )

                speech_rate_select = (
                    ui.select(
                        options=(
                            SPEECH_RATE_OPTIONS
                        ),

                        value=(
                            current_speech_rate_level
                        ),

                        label="Speech Speed",
                    )
                    .props(
                        "outlined"
                    )
                    .classes(
                        "w-80 mt-2"
                    )
                )

                speech_rate_description = (
                    ui.label(
                        _speech_rate_description(
                            current_speech_rate_level
                        )
                    )
                    .classes(
                        "text-sm opacity-70"
                    )
                )

                ui.separator().classes(
                    "my-3"
                )

                # =============================================
                # SPEECH VOLUME
                # =============================================

                ui.label(
                    "Speech Volume"
                ).classes(
                    "text-lg font-semibold"
                )

                speech_volume_select = (
                    ui.select(
                        options=(
                            SPEECH_VOLUME_OPTIONS
                        ),

                        value=(
                            current_speech_volume_level
                        ),

                        label="Speech Volume",
                    )
                    .props(
                        "outlined"
                    )
                    .classes(
                        "w-80 mt-2"
                    )
                )

                speech_volume_description = (
                    ui.label(
                        _speech_volume_description(
                            current_speech_volume_level
                        )
                    )
                    .classes(
                        "text-sm opacity-70"
                    )
                )

                # =============================================
                # SPEECH RATE CHANGE
                # =============================================

                def change_speech_rate(
                    event,
                ) -> None:

                    selected_level = (
                        str(
                            event.value
                        )
                    )

                    rate = (
                        SPEECH_RATE_VALUES
                        .get(
                            selected_level
                        )
                    )

                    if rate is None:

                        ui.notify(
                            (
                                "Invalid speech "
                                "speed setting."
                            ),
                            type="negative",
                        )

                        return

                    try:

                        save_speech_rate(
                            rate
                        )

                        speech_rate_description.set_text(
                            _speech_rate_description(
                                selected_level
                            )
                        )

                        ui.notify(
                            (
                                "Speech speed saved."
                            ),
                            type="positive",
                        )

                    except Exception as exc:

                        ui.notify(
                            (
                                "Could not save speech "
                                f"speed: {exc}"
                            ),
                            type="negative",
                        )

                speech_rate_select.on_value_change(
                    change_speech_rate
                )

                # =============================================
                # SPEECH VOLUME CHANGE
                # =============================================

                def change_speech_volume(
                    event,
                ) -> None:

                    selected_level = (
                        str(
                            event.value
                        )
                    )

                    volume = (
                        SPEECH_VOLUME_VALUES
                        .get(
                            selected_level
                        )
                    )

                    if volume is None:

                        ui.notify(
                            (
                                "Invalid speech "
                                "volume setting."
                            ),
                            type="negative",
                        )

                        return

                    try:

                        save_speech_volume(
                            volume
                        )

                        speech_volume_description.set_text(
                            _speech_volume_description(
                                selected_level
                            )
                        )

                        ui.notify(
                            (
                                "Speech volume saved."
                            ),
                            type="positive",
                        )

                    except Exception as exc:

                        ui.notify(
                            (
                                "Could not save speech "
                                f"volume: {exc}"
                            ),
                            type="negative",
                        )

                speech_volume_select.on_value_change(
                    change_speech_volume
                )

                # =============================================
                # TEST VOICE
                # =============================================

                def test_voice() -> None:

                    stop_speaking()

                    speak_text(
                        (
                            "Hello. This is MemorAi. "
                            "This is how spoken answers "
                            "will sound."
                        )
                    )

                # =============================================
                # RESET SPEECH
                # =============================================

                def reset_voice_preferences() -> None:

                    stop_speaking()

                    try:

                        reset_speech_settings()

                        speech_rate_select.value = (
                            _speech_rate_to_level(
                                DEFAULT_SPEECH_RATE
                            )
                        )

                        speech_volume_select.value = (
                            _speech_volume_to_level(
                                DEFAULT_SPEECH_VOLUME
                            )
                        )

                        speech_rate_description.set_text(
                            _speech_rate_description(
                                "normal"
                            )
                        )

                        speech_volume_description.set_text(
                            _speech_volume_description(
                                "full"
                            )
                        )

                        ui.notify(
                            (
                                "Speech settings restored "
                                "to defaults."
                            ),
                            type="positive",
                        )

                    except Exception as exc:

                        ui.notify(
                            (
                                "Could not reset speech "
                                f"settings: {exc}"
                            ),
                            type="negative",
                        )

                with ui.row().classes(
                    "gap-3 flex-wrap mt-4"
                ):

                    ui.button(
                        "Test Voice",
                        icon="volume_up",
                        on_click=(
                            test_voice
                        ),
                    ).classes(
                        "memorai-action-button"
                    )

                    ui.button(
                        "Stop Speaking",
                        icon="stop",
                        on_click=(
                            stop_speaking
                        ),
                    ).props(
                        "outline"
                    ).classes(
                        "memorai-action-button"
                    )

                    ui.button(
                        "Reset Speech",
                        icon="restart_alt",
                        on_click=(
                            reset_voice_preferences
                        ),
                    ).props(
                        "outline"
                    ).classes(
                        "memorai-action-button"
                    )

            # =================================================
            # CONVERSATION MEMORY
            # =================================================

            with ui.card().classes(
                "memorai-card w-full"
            ):

                ui.label(
                    "🔒 Conversation Memory"
                ).classes(
                    "memorai-section-title"
                )

                ui.label(
                    (
                        "Control whether MemorAi is allowed "
                        "to process and remember important "
                        "conversations."
                    )
                ).classes(
                    "text-lg"
                )

                ui.separator().classes(
                    "my-3"
                )

                ui.label(
                    (
                        "Conversation Memory is "
                        "privacy-controlled."
                    )
                ).classes(
                    "text-lg font-semibold"
                )

                ui.label(
                    (
                        "Enabling this setting does not "
                        "automatically start your microphone."
                    )
                ).classes(
                    "opacity-70"
                )

                ui.label(
                    (
                        "You must explicitly press "
                        "Start Conversation each time."
                    )
                ).classes(
                    "opacity-70"
                )

                ui.separator().classes(
                    "my-3"
                )

                # =============================================
                # PERMISSION
                # =============================================

                with ui.row().classes(
                    (
                        "w-full items-center "
                        "justify-between gap-4"
                    )
                ):

                    with ui.column().classes(
                        "gap-1"
                    ):

                        ui.label(
                            "Allow Conversation Memory"
                        ).classes(
                            "text-lg font-semibold"
                        )

                        permission_description = (
                            ui.label()
                            .classes(
                                "text-sm opacity-70"
                            )
                        )

                    memory_switch = (
                        ui.switch(
                            value=(
                                conversation_enabled
                            )
                        )
                    )

                ui.separator().classes(
                    "my-3"
                )

                # =============================================
                # SELECTIVITY
                # =============================================

                ui.label(
                    "Memory Selectivity"
                ).classes(
                    "text-lg font-semibold"
                )

                ui.label(
                    (
                        "Choose how selective MemorAi should "
                        "be when deciding whether a "
                        "conversation deserves long-term "
                        "memory."
                    )
                ).classes(
                    "opacity-70"
                )

                importance_select = (
                    ui.select(
                        options=(
                            IMPORTANCE_LEVEL_OPTIONS
                        ),

                        value=(
                            current_importance_level
                        ),

                        label=(
                            "Conversation Importance"
                        ),
                    )
                    .props(
                        "outlined"
                    )
                    .classes(
                        "w-80 mt-2"
                    )
                )

                importance_description = (
                    ui.label(
                        _importance_description(
                            current_importance_level
                        )
                    )
                    .classes(
                        (
                            "text-sm opacity-70 "
                            "max-w-2xl"
                        )
                    )
                )

                # =============================================
                # PRIVACY STATUS
                # =============================================

                with ui.card().classes(
                    (
                        "w-full mt-4 "
                        "border border-gray-300"
                    )
                ):

                    ui.label(
                        "Privacy Behavior"
                    ).classes(
                        "text-lg font-semibold"
                    )

                    privacy_status = (
                        ui.label()
                        .classes(
                            "leading-relaxed"
                        )
                    )

                # =============================================
                # REFRESH
                # =============================================

                def refresh_conversation_ui(
                    enabled: bool,
                ) -> None:

                    if enabled:

                        permission_description.set_text(
                            (
                                "Enabled. Conversation recording "
                                "is available, but still starts "
                                "only when you explicitly request it."
                            )
                        )

                        privacy_status.set_text(
                            (
                                "✅ Conversation Memory is enabled. "
                                "You must manually start each "
                                "recording. Casual conversations "
                                "are discarded and only important "
                                "ones may become long-term memories."
                            )
                        )

                    else:

                        permission_description.set_text(
                            (
                                "Disabled. Conversation audio "
                                "processing and conversation memory "
                                "creation are blocked."
                            )
                        )

                        privacy_status.set_text(
                            (
                                "🔒 Conversation Memory is disabled. "
                                "The microphone cannot be used by "
                                "the Conversation Memory feature."
                            )
                        )

                refresh_conversation_ui(
                    conversation_enabled
                )

                # =============================================
                # PERMISSION CHANGE
                # =============================================

                def change_conversation_permission(
                    event,
                ) -> None:

                    enabled = bool(
                        event.value
                    )

                    try:

                        set_conversation_memory_enabled(
                            enabled
                        )

                        refresh_conversation_ui(
                            enabled
                        )

                        if enabled:

                            ui.notify(
                                (
                                    "Conversation Memory enabled. "
                                    "Recording still requires "
                                    "your explicit action."
                                ),
                                type="positive",
                            )

                        else:

                            ui.notify(
                                (
                                    "Conversation Memory disabled."
                                ),
                                type="positive",
                            )

                    except Exception as exc:

                        ui.notify(
                            (
                                "Could not update Conversation "
                                f"Memory: {exc}"
                            ),
                            type="negative",
                        )

                        latest = (
                            get_conversation_settings()
                        )

                        actual_enabled = bool(
                            latest.get(
                                "memory_enabled",
                                False,
                            )
                        )

                        memory_switch.value = (
                            actual_enabled
                        )

                        refresh_conversation_ui(
                            actual_enabled
                        )

                memory_switch.on_value_change(
                    change_conversation_permission
                )

                # =============================================
                # IMPORTANCE CHANGE
                # =============================================

                def change_importance_level(
                    event,
                ) -> None:

                    selected_level = (
                        str(
                            event.value
                        )
                    )

                    threshold = (
                        IMPORTANCE_LEVEL_THRESHOLDS
                        .get(
                            selected_level
                        )
                    )

                    if threshold is None:

                        ui.notify(
                            (
                                "Invalid conversation "
                                "importance level."
                            ),
                            type="negative",
                        )

                        return

                    try:

                        set_importance_threshold(
                            threshold
                        )

                        importance_description.set_text(
                            _importance_description(
                                selected_level
                            )
                        )

                        if (
                            selected_level
                            == IMPORTANCE_HIGH
                        ):

                            message = (
                                "High Selectivity saved."
                            )

                        elif (
                            selected_level
                            == IMPORTANCE_MORE
                        ):

                            message = (
                                "Remember More saved."
                            )

                        else:

                            message = (
                                "Balanced selectivity saved."
                            )

                        ui.notify(
                            message,
                            type="positive",
                        )

                    except Exception as exc:

                        ui.notify(
                            (
                                "Could not update conversation "
                                f"selectivity: {exc}"
                            ),
                            type="negative",
                        )

                importance_select.on_value_change(
                    change_importance_level
                )

            # =================================================
            # PRIVACY INFORMATION
            # =================================================

            with ui.card().classes(
                "memorai-card w-full"
            ):

                ui.label(
                    "🛡 Privacy & Control"
                ).classes(
                    "memorai-section-title"
                )

                ui.label(
                    (
                        "MemorAi keeps microphone and "
                        "conversation recording under "
                        "your control."
                    )
                ).classes(
                    "text-lg"
                )

                with ui.column().classes(
                    "gap-2 mt-3"
                ):

                    ui.label(
                        (
                            "• Conversation recording starts "
                            "only after you press Start Conversation."
                        )
                    )

                    ui.label(
                        (
                            "• Voice Assistant recording starts "
                            "only after you press Start Recording."
                        )
                    )

                    ui.label(
                        (
                            "• Raw conversation audio is used "
                            "temporarily for transcription."
                        )
                    )

                    ui.label(
                        (
                            "• Casual conversations are discarded."
                        )
                    )

                    ui.label(
                        (
                            "• Only important conversations can "
                            "become long-term Conversation Memories."
                        )
                    )

                    ui.label(
                        (
                            "• MemorAi never reads an answer aloud "
                            "until you explicitly press Speak Answer."
                        )
                    )