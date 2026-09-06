# app/nicegui_ui/browser_tts.py

from __future__ import annotations

import json

from nicegui import (
    app,
    ui,
)


# ============================================================
# SPEECH SETTINGS
# ============================================================

DEFAULT_SPEECH_RATE = 0.95

DEFAULT_SPEECH_VOLUME = 1.0

DEFAULT_SPEECH_PITCH = 1.0

DEFAULT_SPEECH_LANGUAGE = "en-US"


MIN_SPEECH_RATE = 0.70
MAX_SPEECH_RATE = 1.30

MIN_SPEECH_VOLUME = 0.0
MAX_SPEECH_VOLUME = 1.0


# ============================================================
# CLAMP
# ============================================================

def _clamp(
    value: float,
    minimum: float,
    maximum: float,
) -> float:

    return max(
        minimum,
        min(
            maximum,
            float(
                value
            ),
        ),
    )


# ============================================================
# GET SAVED SPEECH RATE
# ============================================================

def get_saved_speech_rate() -> float:
    """
    Return the user's saved browser speech rate.

    Falls back safely if storage is unavailable.
    """

    try:

        value = (
            app.storage.user.get(
                "speech_rate",
                DEFAULT_SPEECH_RATE,
            )
        )

        return (
            _clamp(
                float(
                    value
                ),
                MIN_SPEECH_RATE,
                MAX_SPEECH_RATE,
            )
        )

    except Exception:

        return (
            DEFAULT_SPEECH_RATE
        )


# ============================================================
# SAVE SPEECH RATE
# ============================================================

def save_speech_rate(
    rate: float,
) -> float:
    """
    Save the user's preferred browser speech rate.
    """

    value = (
        _clamp(
            float(
                rate
            ),
            MIN_SPEECH_RATE,
            MAX_SPEECH_RATE,
        )
    )

    app.storage.user[
        "speech_rate"
    ] = value

    return (
        value
    )


# ============================================================
# GET SAVED SPEECH VOLUME
# ============================================================

def get_saved_speech_volume() -> float:
    """
    Return the user's saved browser speech volume.
    """

    try:

        value = (
            app.storage.user.get(
                "speech_volume",
                DEFAULT_SPEECH_VOLUME,
            )
        )

        return (
            _clamp(
                float(
                    value
                ),
                MIN_SPEECH_VOLUME,
                MAX_SPEECH_VOLUME,
            )
        )

    except Exception:

        return (
            DEFAULT_SPEECH_VOLUME
        )


# ============================================================
# SAVE SPEECH VOLUME
# ============================================================

def save_speech_volume(
    volume: float,
) -> float:
    """
    Save the user's preferred browser speech volume.
    """

    value = (
        _clamp(
            float(
                volume
            ),
            MIN_SPEECH_VOLUME,
            MAX_SPEECH_VOLUME,
        )
    )

    app.storage.user[
        "speech_volume"
    ] = value

    return (
        value
    )


# ============================================================
# RESET SPEECH SETTINGS
# ============================================================

def reset_speech_settings() -> None:
    """
    Restore the MemorAi speech defaults.
    """

    app.storage.user[
        "speech_rate"
    ] = (
        DEFAULT_SPEECH_RATE
    )

    app.storage.user[
        "speech_volume"
    ] = (
        DEFAULT_SPEECH_VOLUME
    )


# ============================================================
# SPEAK
# ============================================================

def speak_text(
    text: str,
    rate: float | None = None,
    pitch: float = DEFAULT_SPEECH_PITCH,
    volume: float | None = None,
    language: str = DEFAULT_SPEECH_LANGUAGE,
) -> None:
    """
    Speak text using the web browser's Speech Synthesis API.

    Speech is produced on the user's browser/device.

    Important:

        This function NEVER runs automatically.

        MemorAi pages call it only after the user
        explicitly presses a Speak button.

    If rate or volume are not supplied explicitly,
    the saved user preferences are used.
    """

    if not text:

        return

    cleaned_text = (
        str(
            text
        )
        .strip()
    )

    if not cleaned_text:

        return

    # --------------------------------------------------------
    # SAVED SETTINGS
    # --------------------------------------------------------

    final_rate = (
        get_saved_speech_rate()
        if rate is None
        else _clamp(
            float(
                rate
            ),
            MIN_SPEECH_RATE,
            MAX_SPEECH_RATE,
        )
    )

    final_volume = (
        get_saved_speech_volume()
        if volume is None
        else _clamp(
            float(
                volume
            ),
            MIN_SPEECH_VOLUME,
            MAX_SPEECH_VOLUME,
        )
    )

    final_pitch = (
        _clamp(
            float(
                pitch
            ),
            0.5,
            2.0,
        )
    )

    # --------------------------------------------------------
    # SAFE JAVASCRIPT VALUES
    # --------------------------------------------------------

    safe_text = (
        json.dumps(
            cleaned_text
        )
    )

    safe_language = (
        json.dumps(
            str(
                language
                or DEFAULT_SPEECH_LANGUAGE
            )
        )
    )

    # --------------------------------------------------------
    # SPEECH SYNTHESIS
    # --------------------------------------------------------

    javascript = f"""
    (() => {{

        if (!('speechSynthesis' in window)) {{

            console.error(
                'Speech synthesis is not supported by this browser.'
            );

            return;
        }}

        window
            .speechSynthesis
            .cancel();

        const utterance =
            new SpeechSynthesisUtterance(
                {safe_text}
            );

        utterance.lang =
            {safe_language};

        utterance.rate =
            {float(final_rate)};

        utterance.pitch =
            {float(final_pitch)};

        utterance.volume =
            {float(final_volume)};

        window
            .speechSynthesis
            .speak(
                utterance
            );

    }})();
    """

    ui.run_javascript(
        javascript
    )


# ============================================================
# STOP
# ============================================================

def stop_speaking() -> None:
    """
    Immediately stop browser speech.
    """

    ui.run_javascript(
        """
        if ('speechSynthesis' in window) {

            window
                .speechSynthesis
                .cancel();
        }
        """
    )


# ============================================================
# PAUSE
# ============================================================

def pause_speaking() -> None:
    """
    Pause current browser speech.
    """

    ui.run_javascript(
        """
        if ('speechSynthesis' in window) {

            window
                .speechSynthesis
                .pause();
        }
        """
    )


# ============================================================
# RESUME
# ============================================================

def resume_speaking() -> None:
    """
    Resume paused browser speech.
    """

    ui.run_javascript(
        """
        if ('speechSynthesis' in window) {

            window
                .speechSynthesis
                .resume();
        }
        """
    )