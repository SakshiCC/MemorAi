from nicegui import ui

from app.nicegui_ui.browser_tts import (
    speak_text,
    stop_speaking,
)


TEST_TEXT = (
    "Hello. I am MemorAi, "
    "your personal memory assistant."
)


@ui.page("/")
def test_page():

    ui.label(
        "MemorAi Browser TTS Test"
    ).classes(
        "text-2xl font-bold"
    )

    ui.label(
        "Use the buttons below to test "
        "browser-based speech."
    )

    with ui.row():

        ui.button(
            "Speak",
            icon="volume_up",
            on_click=lambda: speak_text(
                TEST_TEXT
            ),
        )

        ui.button(
            "Stop",
            icon="stop",
            on_click=stop_speaking,
        )


ui.run(
    title="MemorAi TTS Test",
)