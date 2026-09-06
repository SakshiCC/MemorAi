from __future__ import annotations

import asyncio

from nicegui import ui

from llm.chat import (
    MemorAiChat,
)

from app.nicegui_ui.browser_tts import (
    speak_text,
    stop_speaking,
)

from app.nicegui_ui.layout import (
    sidebar,
)


# ============================================================
# ASK MEMORAI PAGE
# ============================================================


def build_ask_page(
    chat: MemorAiChat,
) -> None:
    """
    Build the main MemorAi text-chat page.

    Important behaviour:

    - User asks a question.
    - MemorAi answers in text.
    - MemorAi NEVER speaks automatically.
    - Speech only starts when the user explicitly
      presses the Speak Answer button.
    """

    # --------------------------------------------------------
    # PAGE STATE
    # --------------------------------------------------------

    state = {
        "latest_answer": "",
    }

    # ========================================================
    # PAGE LAYOUT
    # ========================================================

    with ui.row().classes(
        "memorai-page items-start no-wrap"
    ):

        # ====================================================
        # SIDEBAR
        # ====================================================

        sidebar(
            active_path="/ask"
        )

        # ====================================================
        # MAIN CONTENT
        # ====================================================

        with ui.column().classes(
            "memorai-main gap-6"
        ):

            # ------------------------------------------------
            # HEADER
            # ------------------------------------------------

            ui.label(
                "💬 Ask MemorAi"
            ).classes(
                "text-4xl font-bold"
            )

            ui.label(
                (
                    "Ask about objects, locations, "
                    "recent memories, or events "
                    "MemorAi has observed."
                )
            ).classes(
                "text-lg opacity-70"
            )

            # =================================================
            # CHAT CARD
            # =================================================

            with ui.card().classes(
                "memorai-card w-full"
            ):

                ui.label(
                    "Ask a Question"
                ).classes(
                    "memorai-section-title"
                )

                ui.label(
                    (
                        "Type your question below. "
                        "MemorAi will search its memories "
                        "and answer you."
                    )
                ).classes(
                    "opacity-70"
                )

                # =============================================
                # QUESTION INPUT
                # =============================================

                question_input = (
                    ui.input(
                        label="Your question",
                        placeholder=(
                            "For example: "
                            "Where is my cell phone?"
                        ),
                    )
                    .props(
                        "outlined clearable"
                    )
                    .classes(
                        "w-full text-lg"
                    )
                )

                # =============================================
                # ANSWER SECTION
                # =============================================

                with ui.column().classes(
                    "w-full gap-2 mt-3"
                ):

                    ui.label(
                        "MemorAi"
                    ).classes(
                        "text-lg font-semibold"
                    )

                    answer_label = (
                        ui.label(
                            (
                                "Ask me something about "
                                "your memories."
                            )
                        )
                        .classes(
                            (
                                "text-lg "
                                "leading-relaxed "
                                "w-full"
                            )
                        )
                    )

                # =============================================
                # ASK QUESTION
                # =============================================

                async def ask_question() -> None:
                    """
                    Send the user's question to MemorAi.

                    This function only displays the response.
                    It never triggers speech automatically.
                    """

                    question = (
                        question_input.value
                        or ""
                    ).strip()

                    if not question:

                        ui.notify(
                            "Please enter a question.",
                            type="warning",
                        )

                        return

                    # -----------------------------------------
                    # Stop any speech left from an older answer.
                    # -----------------------------------------

                    stop_speaking()

                    # -----------------------------------------
                    # Clear previous answer state.
                    # -----------------------------------------

                    state[
                        "latest_answer"
                    ] = ""

                    answer_label.set_text(
                        "Searching your memories..."
                    )

                    try:

                        # -------------------------------------
                        # RAG / MemorAi answer
                        # -------------------------------------

                        answer = (
                            await asyncio.to_thread(
                                chat.answer,
                                question,
                            )
                        )

                        answer = str(
                            answer
                        ).strip()

                        if not answer:

                            answer = (
                                "I could not find a reliable "
                                "answer in my memories."
                            )

                        # -------------------------------------
                        # Store latest answer
                        # -------------------------------------

                        state[
                            "latest_answer"
                        ] = answer

                        # -------------------------------------
                        # Display answer
                        # -------------------------------------

                        answer_label.set_text(
                            answer
                        )

                        # IMPORTANT:
                        #
                        # Do NOT call speak_text(answer) here.
                        #
                        # MemorAi must only speak after the
                        # user explicitly presses Speak Answer.

                    except Exception as exc:

                        state[
                            "latest_answer"
                        ] = ""

                        answer_label.set_text(
                            (
                                "MemorAi could not answer "
                                "the question."
                            )
                        )

                        ui.notify(
                            (
                                "Could not retrieve the "
                                f"answer: {exc}"
                            ),
                            type="negative",
                        )

                # =============================================
                # SPEAK ANSWER
                # =============================================

                def speak_latest_answer() -> None:
                    """
                    Speak only the latest successful answer.

                    Speech starts only because the user
                    explicitly pressed the button.
                    """

                    answer = (
                        state.get(
                            "latest_answer",
                            ""
                        )
                        or ""
                    ).strip()

                    if not answer:

                        ui.notify(
                            (
                                "Ask MemorAi a question "
                                "before using Speak Answer."
                            ),
                            type="warning",
                        )

                        return

                    speak_text(
                        answer
                    )

                # =============================================
                # STOP SPEECH
                # =============================================

                def stop_answer_speech() -> None:

                    stop_speaking()

                # =============================================
                # ENTER KEY
                # =============================================

                question_input.on(
                    "keydown.enter",
                    ask_question,
                )

                # =============================================
                # BUTTONS
                # =============================================

                with ui.row().classes(
                    "w-full gap-3 mt-3"
                ):

                    ui.button(
                        "Ask MemorAi",
                        icon="send",
                        on_click=ask_question,
                    ).classes(
                        "memorai-action-button"
                    )

                    ui.button(
                        "Speak Answer",
                        icon="volume_up",
                        on_click=speak_latest_answer,
                    ).props(
                        "outline"
                    ).classes(
                        "memorai-action-button"
                    )

                    ui.button(
                        "Stop Speaking",
                        icon="stop",
                        on_click=stop_answer_speech,
                    ).props(
                        "outline"
                    ).classes(
                        "memorai-action-button"
                    )

            # =================================================
            # EXAMPLE QUESTIONS
            # =================================================

            with ui.card().classes(
                "memorai-card w-full"
            ):

                ui.label(
                    "💡 Example Questions"
                ).classes(
                    "memorai-section-title"
                )

                ui.label(
                    (
                        "You can ask questions like these."
                    )
                ).classes(
                    "opacity-70"
                )

                # =============================================
                # SET EXAMPLE QUESTION
                # =============================================

                def set_question(
                    question: str,
                ) -> None:

                    question_input.value = (
                        question
                    )

                with ui.row().classes(
                    "w-full gap-3 flex-wrap"
                ):

                    ui.button(
                        "Where is my cell phone?",
                        on_click=lambda: (
                            set_question(
                                (
                                    "Where is my "
                                    "cell phone?"
                                )
                            )
                        ),
                    ).props(
                        "outline no-caps"
                    )

                    ui.button(
                        "What did you see today?",
                        on_click=lambda: (
                            set_question(
                                (
                                    "What objects were "
                                    "detected today?"
                                )
                            )
                        ),
                    ).props(
                        "outline no-caps"
                    )

                    ui.button(
                        "What was detected recently?",
                        on_click=lambda: (
                            set_question(
                                (
                                    "What objects were "
                                    "detected recently?"
                                )
                            )
                        ),
                    ).props(
                        "outline no-caps"
                    )

                    ui.button(
                        "When did I last see my phone?",
                        on_click=lambda: (
                            set_question(
                                (
                                    "When did I last see "
                                    "my cell phone?"
                                )
                            )
                        ),
                    ).props(
                        "outline no-caps"
                    )

            # =================================================
            # SPEECH INFORMATION
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
                        "MemorAi does not speak automatically. "
                        "After receiving an answer, press "
                        "'Speak Answer' whenever you want "
                        "MemorAi to read the response aloud."
                    )
                ).classes(
                    "text-lg"
                )