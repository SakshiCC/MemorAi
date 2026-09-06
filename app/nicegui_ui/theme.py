from __future__ import annotations

from typing import Any

from nicegui import app


# ============================================================
# THEME CONSTANTS
# ============================================================

THEME_SYSTEM = "system"
THEME_LIGHT = "light"
THEME_DARK = "dark"

DEFAULT_THEME = THEME_SYSTEM


# ============================================================
# VALIDATION
# ============================================================

def normalize_theme(
    theme: str | None,
) -> str:

    if theme in {
        THEME_SYSTEM,
        THEME_LIGHT,
        THEME_DARK,
    }:
        return theme

    return DEFAULT_THEME


# ============================================================
# GET SAVED THEME
# ============================================================

def get_saved_theme() -> str:

    try:

        theme = app.storage.user.get(
            "theme",
            DEFAULT_THEME,
        )

    except Exception:

        return DEFAULT_THEME

    return normalize_theme(
        theme
    )


# ============================================================
# SAVE THEME
# ============================================================

def save_theme(
    theme: str,
) -> str:

    selected = normalize_theme(
        theme
    )

    app.storage.user[
        "theme"
    ] = selected

    return selected


# ============================================================
# APPLY THEME TO CONTROLLER
# ============================================================

def apply_theme_to_controller(
    dark_mode: Any,
    theme: str,
) -> None:
    """
    Apply MemorAi theme to an existing NiceGUI
    dark-mode controller.

    dark   -> enable
    light  -> disable
    system -> auto
    """

    selected = normalize_theme(
        theme
    )

    if selected == THEME_DARK:

        dark_mode.enable()

    elif selected == THEME_LIGHT:

        dark_mode.disable()

    else:

        dark_mode.auto()


# ============================================================
# DESCRIPTION
# ============================================================

def get_theme_description(
    theme: str,
) -> str:

    descriptions = {

        THEME_SYSTEM: (
            "Uses your computer or browser "
            "appearance preference."
        ),

        THEME_LIGHT: (
            "MemorAi will always use the light theme."
        ),

        THEME_DARK: (
            "MemorAi will always use the dark theme."
        ),
    }

    return descriptions[
        normalize_theme(theme)
    ]