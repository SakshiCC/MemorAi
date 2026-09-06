from __future__ import annotations

from nicegui import ui

from app.nicegui_ui.theme import (
    apply_theme_to_controller,
    get_saved_theme,
)


# ============================================================
# NAVIGATION
# ============================================================

NAVIGATION_ITEMS = [
    ("🏠 Home", "/"),
    ("💬 Ask MemorAi", "/ask"),
    ("🎤 Voice Assistant", "/voice"),
    ("🗣 Conversations", "/conversations"),
    ("🧠 Memory Timeline", "/timeline"),
    ("📷 Camera / Monitoring", "/camera"),
    ("👤 People", "/people"),
    ("💊 Medicines", "/medicines"),
    ("⚙ Settings", "/settings"),
]


# ============================================================
# GLOBAL STYLE
# ============================================================

def apply_global_styles() -> None:
    """
    Apply global MemorAi styles.

    Important:
    We deliberately do NOT force a dark background here.

    NiceGUI's dark-mode controller is responsible for:

        System
        Light
        Dark
    """

    ui.add_css(
        """
        :root {
            --memorai-radius: 16px;
        }


        /* ===================================================
           GENERAL
        =================================================== */

        body {
            margin: 0;

            font-family:
                Inter,
                -apple-system,
                BlinkMacSystemFont,
                "Segoe UI",
                sans-serif;
        }


        /* ===================================================
           FULL PAGE
        =================================================== */

        .memorai-page {
            width: 100%;
            min-height: 100vh;
        }


        /* ===================================================
           MAIN CONTENT
        =================================================== */

        .memorai-main {
            flex: 1;

            width: 100%;

            padding: 32px;

            max-width: 1500px;
        }


        /* ===================================================
           SIDEBAR
        =================================================== */

        .memorai-sidebar {
            width: 270px;

            min-width: 270px;

            min-height: 100vh;

            padding: 22px;

            border-right:
                1px solid rgba(127, 127, 127, 0.20);
        }


        /* ===================================================
           BRAND
        =================================================== */

        .memorai-title {
            font-size: 32px;

            font-weight: 750;
        }


        .memorai-subtitle {
            font-size: 15px;

            opacity: 0.65;
        }


        /* ===================================================
           SECTION TITLE
        =================================================== */

        .memorai-section-title {
            font-size: 24px;

            font-weight: 650;
        }


        /* ===================================================
           CARDS
        =================================================== */

        .memorai-card {
            width: 100%;

            border:
                1px solid rgba(127, 127, 127, 0.22);

            border-radius:
                var(--memorai-radius);

            padding: 20px;

            box-shadow:
                0 2px 12px rgba(0, 0, 0, 0.06);
        }


        .memorai-memory-card {
            width: 100%;

            border:
                1px solid rgba(127, 127, 127, 0.22);

            border-radius: 13px;

            padding: 16px;
        }


        /* ===================================================
           VIDEO
        =================================================== */

        .memorai-video {
            width: 100%;

            max-width: 900px;

            border-radius:
                var(--memorai-radius);

            border:
                1px solid rgba(127, 127, 127, 0.25);

            background: #000;
        }


        /* ===================================================
           NAVIGATION BUTTONS
        =================================================== */

        .memorai-nav-button {
            width: 100%;

            justify-content: flex-start;

            text-align: left;

            min-height: 48px;

            font-size: 16px;

            border-radius: 10px;
        }


        /* ===================================================
           ELDER-FRIENDLY ACTION BUTTONS
        =================================================== */

        .memorai-action-button {
            min-height: 50px;

            font-size: 16px;

            padding-left: 20px;

            padding-right: 20px;
        }


        /* ===================================================
           STATUS
        =================================================== */

        .memorai-status-ready {
            color: #22c55e;

            font-weight: 650;
        }


        /* ===================================================
           RESPONSIVE
        =================================================== */

        @media (max-width: 850px) {

            .memorai-sidebar {
                width: 220px;

                min-width: 220px;

                padding: 14px;
            }


            .memorai-main {
                padding: 18px;
            }


            .memorai-title {
                font-size: 27px;
            }
        }
        """
    )


# ============================================================
# APPLY THE SAVED THEME
# ============================================================

def prepare_page():
    """
    Apply the stored MemorAi theme to the current page.

    Saved options:

        system
        light
        dark

    Returns
    -------
    NiceGUI dark-mode controller.

    The Settings page can use this controller to update
    appearance immediately.
    """

    dark_mode = ui.dark_mode()

    selected_theme = (
        get_saved_theme()
    )

    apply_theme_to_controller(
        dark_mode,
        selected_theme,
    )

    apply_global_styles()

    return dark_mode


# ============================================================
# SIDEBAR
# ============================================================

def sidebar(
    active_path: str,
) -> None:
    """
    Render the common MemorAi sidebar.
    """

    with ui.column().classes(
        "memorai-sidebar gap-2"
    ):

        # ----------------------------------------------------
        # BRAND
        # ----------------------------------------------------

        ui.label(
            "🧠 MemorAi"
        ).classes(
            "memorai-title"
        )

        ui.label(
            "Your Personal Memory Assistant"
        ).classes(
            "memorai-subtitle"
        )

        ui.separator().classes(
            "my-4"
        )

        # ----------------------------------------------------
        # NAVIGATION ITEMS
        # ----------------------------------------------------

        for label, path in (
            NAVIGATION_ITEMS
        ):

            button = (
                ui.button(
                    label,
                    on_click=lambda p=path: (
                        ui.navigate.to(p)  # type: ignore
                    ),
                )
                .props(
                    "flat no-caps align=left"
                )
                .classes(
                    "memorai-nav-button"
                )
            )

            if path == active_path:

                button.props(
                    "color=primary"
                )

        # ----------------------------------------------------
        # SYSTEM STATUS
        # ----------------------------------------------------

        ui.separator().classes(
            "my-4"
        )

        ui.label(
            "MemorAi"
        ).classes(
            "text-sm opacity-60"
        )

        ui.label(
            "● System Ready"
        ).classes(
            "memorai-status-ready"
        )


# ============================================================
# PAGE SHELL
# ============================================================

def page_shell(
    active_path: str,
):
    """
    Create a standard MemorAi page container.

    This helper is optional; pages may also create their own
    ui.row layout and call sidebar() directly.
    """

    prepare_page()

    return (
        ui.row()
        .classes(
            "memorai-page items-start no-wrap"
        )
    )