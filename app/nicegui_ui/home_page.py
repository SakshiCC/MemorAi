from __future__ import annotations

import asyncio
import time

from datetime import (
    date,
    datetime,
)

from nicegui import ui

from app.nicegui_ui.browser_tts import (
    speak_text,
    stop_speaking,
)

from app.nicegui_ui.layout import (
    sidebar,
)

from app.services.camera_service import (
    CameraService,
    CameraState,
)

from database.event_repository import (
    get_recent_events,
)

from database.medicine_repository import (
    get_medicine_by_name,
)

from database.person_repository import (
    get_person_events_for_date,
)

from llm.chat import (
    MemorAiChat,
)


# ============================================================
# HOME PAGE
# ============================================================

def build_home_page(
    camera_service: CameraService,
    chat: MemorAiChat,
) -> None:
    """
    MemorAi Home Page.

    Medicine panel contains only four compact stages:

    1. Medicine yet to be detected
    2. Medicine checking
    3. Medicine confirmed
    4. Medicine event confirmed

    IMPORTANT:

    A confirmed medicine event is latched in the UI for
    several seconds.

    The medicine backend resets very quickly after storing
    an event so it can begin watching for the next event.
    Without this UI latch, the event_saved state can disappear
    before the browser's next refresh.

    No medicine-recognition or action-recognition logic is
    changed here.
    """

    # ========================================================
    # UI CONFIG
    # ========================================================

    MEDICINE_EVENT_SUCCESS_DISPLAY_SECONDS = 5.0

    # ========================================================
    # LOCAL STATE
    # ========================================================

    chat_state = {
        "latest_answer": "",
    }

    quick_state = {
        "latest_answer": "",
    }

    person_state = {
        "version":
            camera_service
            .get_person_version(),
    }

    # --------------------------------------------------------
    # Medicine UI state
    # --------------------------------------------------------

    medicine_state = {
        # Medicine metadata cache
        "name": None,
        "dosage": None,

        # Last camera event version already observed by UI
        "version":
            camera_service
            .get_medicine_version(),

        # Time until which event-confirmed UI should remain
        # visible.
        "event_confirmed_until": 0.0,

        # Medicine shown during the latched success state.
        "confirmed_name": None,
        "confirmed_dosage": None,
    }

    # ========================================================
    # MEDICINE DOSAGE LOOKUP
    # ========================================================

    def get_medicine_dosage(
        medicine_name: str | None,
    ) -> str | None:

        if not medicine_name:

            return None

        normalized_name = str(
            medicine_name
        ).strip()

        if not normalized_name:

            return None

        # ----------------------------------------------------
        # Use cache if this is the same medicine.
        # ----------------------------------------------------

        if (
            medicine_state[
                "name"
            ]
            == normalized_name
        ):

            return medicine_state[
                "dosage"
            ]

        try:

            medicine = (
                get_medicine_by_name(
                    normalized_name
                )
            )

        except Exception:

            medicine = None

        dosage = None

        if medicine:

            raw_dosage = (
                medicine.get(
                    "dosage"
                )
            )

            if raw_dosage:

                dosage = str(
                    raw_dosage
                ).strip()

        medicine_state[
            "name"
        ] = normalized_name

        medicine_state[
            "dosage"
        ] = dosage

        return dosage

    # ========================================================
    # FORMAT MEDICINE NAME
    # ========================================================

    def format_medicine_name(
        medicine_name: str | None,
        dosage: str | None,
    ) -> str:

        if not medicine_name:

            return (
                "Medicine"
            )

        text = str(
            medicine_name
        ).strip()

        if dosage:

            dosage_text = str(
                dosage
            ).strip()

            if dosage_text:

                text += (
                    f" {dosage_text}"
                )

        return text

    # ========================================================
    # PAGE
    # ========================================================

    with ui.row().classes(
        "w-full items-start no-wrap"
    ):

        sidebar(
            active_path="/"
        )

        with ui.column().classes(
            "flex-1 p-8 gap-6"
        ):

            # =================================================
            # HEADER
            # =================================================

            with ui.row().classes(
                "w-full justify-between items-center"
            ):

                with ui.column().classes(
                    "gap-0"
                ):

                    ui.label(
                        "🧠 MemorAi"
                    ).classes(
                        "text-4xl font-bold"
                    )

                    ui.label(
                        "Your Personal Memory Assistant"
                    ).classes(
                        "text-gray-400"
                    )

                ui.label(
                    "🟢 MemorAi Ready"
                ).classes(
                    "text-green-400"
                )

            # =================================================
            # GREETING
            # =================================================

            hour = (
                datetime.now().hour
            )

            if hour < 12:

                greeting = (
                    "Good morning"
                )

            elif hour < 17:

                greeting = (
                    "Good afternoon"
                )

            else:

                greeting = (
                    "Good evening"
                )

            ui.label(
                f"{greeting} 👋"
            ).classes(
                "text-3xl font-semibold"
            )

            ui.label(
                "How can I help you today?"
            ).classes(
                "text-lg text-gray-300"
            )

            # =================================================
            # ASK MEMORAI
            # =================================================

            with ui.card().classes(
                "memorai-card w-full"
            ):

                ui.label(
                    "💬 Ask MemorAi"
                ).classes(
                    "memorai-section-title"
                )

                ui.label(
                    (
                        "Ask MemorAi about your recent "
                        "memories and detected objects."
                    )
                ).classes(
                    "opacity-70"
                )

                question_input = (
                    ui.input(
                        label=(
                            "Your question"
                        ),

                        placeholder=(
                            "For example: "
                            "Where is my cell phone?"
                        ),
                    )
                    .props(
                        "outlined clearable"
                    )
                    .classes(
                        "w-full"
                    )
                )

                answer_label = (
                    ui.label(
                        (
                            "Ask me something "
                            "about your memories."
                        )
                    )
                    .classes(
                        "text-lg text-blue-300"
                    )
                )

                # =============================================
                # ASK QUESTION
                # =============================================

                async def ask_question() -> None:

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

                    stop_speaking()

                    chat_state[
                        "latest_answer"
                    ] = ""

                    answer_label.set_text(
                        "Searching your memories..."
                    )

                    try:

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

                        chat_state[
                            "latest_answer"
                        ] = answer

                        answer_label.set_text(
                            answer
                        )

                    except Exception as exc:

                        chat_state[
                            "latest_answer"
                        ] = ""

                        answer_label.set_text(
                            (
                                "MemorAi could not "
                                "answer the question."
                            )
                        )

                        ui.notify(
                            str(
                                exc
                            ),
                            type="negative",
                        )

                # =============================================
                # SPEAK ANSWER
                # =============================================

                def speak_latest_answer() -> None:

                    answer = (
                        chat_state.get(
                            "latest_answer"
                        )
                        or ""
                    ).strip()

                    if not answer:

                        ui.notify(
                            (
                                "Ask MemorAi a question first, "
                                "then press Speak Answer."
                            ),
                            type="warning",
                        )

                        return

                    speak_text(
                        answer
                    )

                question_input.on(
                    "keydown.enter",
                    ask_question,
                )

                with ui.row().classes(
                    "w-full gap-3 flex-wrap"
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
                        on_click=(
                            speak_latest_answer
                        ),
                    ).props(
                        "outline"
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

            # =================================================
            # LIVE CAMERA
            # =================================================

            with ui.card().classes(
                "memorai-card w-full"
            ):

                ui.label(
                    "📷 Live Memory Camera"
                ).classes(
                    "memorai-section-title"
                )

                ui.label(
                    (
                        "MemorAi observes important objects, "
                        "registered people and registered "
                        "medicine packages."
                    )
                ).classes(
                    "text-gray-400"
                )

                camera_state_label = (
                    ui.label(
                        "Starting camera..."
                    )
                    .classes(
                        "text-lg"
                    )
                )

                # =============================================
                # CAMERA CONTROLS
                # =============================================

                with ui.row().classes(
                    "w-full gap-3 flex-wrap"
                ):

                    ui.button(
                        "▶ Start",
                        on_click=(
                            camera_service.start
                        ),
                    )

                    ui.button(
                        "⏸ Pause",
                        on_click=(
                            camera_service.pause
                        ),
                    )

                    ui.button(
                        "▶ Resume",
                        on_click=(
                            camera_service.resume
                        ),
                    )

                    ui.button(
                        "⏹ Stop",
                        on_click=(
                            camera_service.stop
                        ),
                    )

                # =============================================
                # CAMERA + COMPACT MEDICINE STATUS
                # =============================================

                with ui.row().classes(
                    (
                        "w-full gap-5 items-start "
                        "flex-wrap xl:flex-nowrap"
                    )
                ):

                    # =========================================
                    # CAMERA
                    # =========================================

                    with ui.column().classes(
                        (
                            "w-full xl:flex-[2] "
                            "min-w-0 gap-2"
                        )
                    ):

                        ui.label(
                            "Live Camera"
                        ).classes(
                            "text-lg font-semibold"
                        )

                        ui.html(
                            """
                            <div style="
                                width:100%;
                                overflow:hidden;
                                border-radius:16px;
                            ">
                                <img
                                    src="/video_feed"
                                    class="memorai-video"
                                    alt="MemorAi Live Camera"
                                    style="
                                        width:100%;
                                        height:auto;
                                        display:block;
                                    "
                                >
                            </div>
                            """
                        )

                    # =========================================
                    # MEDICINE STATUS
                    # =========================================

                    with ui.column().classes(
                        (
                            "w-full "
                            "xl:w-[330px] "
                            "xl:min-w-[330px] "
                            "gap-2"
                        )
                    ):

                        with ui.card().classes(
                            (
                                "w-full gap-2 "
                                "rounded-2xl "
                                "border "
                                "border-gray-500/30 "
                                "bg-gray-500/5 "
                                "px-4 py-3"
                            )
                        ):

                            ui.label(
                                "💊 Medicine Status"
                            ).classes(
                                "text-lg font-bold mb-1"
                            )

                            # =================================
                            # STATUS ROW BUILDER
                            # =================================

                            def build_status_row(
                                icon: str,
                                text: str,
                                classes: str,
                            ):

                                row = (
                                    ui.row()
                                    .classes(
                                        (
                                            "w-full "
                                            "items-center "
                                            "gap-3 "
                                            "rounded-xl "
                                            "px-3 py-2 "
                                            "border "
                                            f"{classes}"
                                        )
                                    )
                                )

                                with row:

                                    icon_label = (
                                        ui.label(
                                            icon
                                        )
                                        .classes(
                                            "text-lg"
                                        )
                                    )

                                    title_label = (
                                        ui.label(
                                            text
                                        )
                                        .classes(
                                            (
                                                "font-semibold "
                                                "leading-tight"
                                            )
                                        )
                                    )

                                return (
                                    row,
                                    icon_label,
                                    title_label,
                                )

                            # =================================
                            # STEP 1
                            # =================================

                            (
                                step1_row,
                                step1_icon,
                                step1_title,
                            ) = build_status_row(
                                "🔴",
                                (
                                    "Medicine yet "
                                    "to be detected"
                                ),
                                (
                                    "border-red-500/40 "
                                    "bg-red-500/10"
                                ),
                            )

                            step1_title.classes(
                                add="text-red-400"
                            )

                            # =================================
                            # STEP 2
                            # =================================

                            (
                                step2_row,
                                step2_icon,
                                step2_title,
                            ) = build_status_row(
                                "○",
                                "Medicine checking",
                                (
                                    "border-gray-500/20 "
                                    "bg-gray-500/5 "
                                    "opacity-50"
                                ),
                            )

                            # =================================
                            # STEP 3
                            # =================================

                            (
                                step3_row,
                                step3_icon,
                                step3_title,
                            ) = build_status_row(
                                "○",
                                "Medicine confirmed",
                                (
                                    "border-gray-500/20 "
                                    "bg-gray-500/5 "
                                    "opacity-50"
                                ),
                            )

                            # =================================
                            # STEP 4
                            # =================================

                            (
                                step4_row,
                                step4_icon,
                                step4_title,
                            ) = build_status_row(
                                "○",
                                (
                                    "Medicine event "
                                    "confirmed"
                                ),
                                (
                                    "border-gray-500/20 "
                                    "bg-gray-500/5 "
                                    "opacity-50"
                                ),
                            )

                # =================================================
                # LATEST DETECTION
                # =================================================

                ui.label(
                    "Latest Detection"
                ).classes(
                    "text-xl font-semibold mt-4"
                )

                detection_label = (
                    ui.label(
                        (
                            "No important object "
                            "has been detected yet."
                        )
                    )
                    .classes(
                        "text-gray-300"
                    )
                )

                camera_stats_label = (
                    ui.label(
                        ""
                    )
                    .classes(
                        "text-sm text-gray-500"
                    )
                )

                # =================================================
                # STYLE CONSTANTS
                # =================================================

                mutable_row_classes = (
                    "border-red-500/40 "
                    "border-yellow-500/50 "
                    "border-green-500/45 "
                    "border-gray-500/20 "
                    "bg-red-500/10 "
                    "bg-yellow-500/10 "
                    "bg-green-500/10 "
                    "bg-gray-500/5 "
                    "opacity-50 "
                    "opacity-100"
                )

                mutable_text_classes = (
                    "text-red-400 "
                    "text-yellow-400 "
                    "text-green-400 "
                    "opacity-60"
                )

                # =================================================
                # STYLE ONE STEP
                # =================================================

                def style_step(
                    row,
                    icon_label,
                    title_label,
                    icon: str,
                    color: str,
                    active: bool,
                ) -> None:

                    row.classes(
                        remove=(
                            mutable_row_classes
                        )
                    )

                    title_label.classes(
                        remove=(
                            mutable_text_classes
                        )
                    )

                    icon_label.set_text(
                        icon
                    )

                    opacity = (
                        "opacity-100"
                        if active
                        else "opacity-50"
                    )

                    if color == "red":

                        row.classes(
                            add=(
                                "border-red-500/40 "
                                "bg-red-500/10 "
                                f"{opacity}"
                            )
                        )

                        title_label.classes(
                            add="text-red-400"
                        )

                    elif color == "yellow":

                        row.classes(
                            add=(
                                "border-yellow-500/50 "
                                "bg-yellow-500/10 "
                                f"{opacity}"
                            )
                        )

                        title_label.classes(
                            add="text-yellow-400"
                        )

                    elif color == "green":

                        row.classes(
                            add=(
                                "border-green-500/45 "
                                "bg-green-500/10 "
                                f"{opacity}"
                            )
                        )

                        title_label.classes(
                            add="text-green-400"
                        )

                    else:

                        row.classes(
                            add=(
                                "border-gray-500/20 "
                                "bg-gray-500/5 "
                                f"{opacity}"
                            )
                        )

                        if not active:

                            title_label.classes(
                                add="opacity-60"
                            )

                # =================================================
                # RESET ALL STEPS
                # =================================================

                def reset_steps() -> None:

                    style_step(
                        step1_row,
                        step1_icon,
                        step1_title,
                        "○",
                        "neutral",
                        False,
                    )

                    style_step(
                        step2_row,
                        step2_icon,
                        step2_title,
                        "○",
                        "neutral",
                        False,
                    )

                    style_step(
                        step3_row,
                        step3_icon,
                        step3_title,
                        "○",
                        "neutral",
                        False,
                    )

                    style_step(
                        step4_row,
                        step4_icon,
                        step4_title,
                        "○",
                        "neutral",
                        False,
                    )

                    step3_title.set_text(
                        "Medicine confirmed"
                    )

                # =================================================
                # SHOW EVENT CONFIRMED STATE
                # =================================================

                def show_event_confirmed_state() -> None:

                    reset_steps()

                    medicine_name = (
                        medicine_state.get(
                            "confirmed_name"
                        )
                    )

                    dosage = (
                        medicine_state.get(
                            "confirmed_dosage"
                        )
                    )

                    medicine_display = (
                        format_medicine_name(
                            medicine_name,
                            dosage,
                        )
                    )

                    # -----------------------------------------
                    # Previous stages complete.
                    # -----------------------------------------

                    style_step(
                        step1_row,
                        step1_icon,
                        step1_title,
                        "✓",
                        "green",
                        False,
                    )

                    style_step(
                        step2_row,
                        step2_icon,
                        step2_title,
                        "✓",
                        "green",
                        False,
                    )

                    step3_title.set_text(
                        (
                            "Medicine confirmed"
                            f" • {medicine_display}"
                        )
                    )

                    style_step(
                        step3_row,
                        step3_icon,
                        step3_title,
                        "✓",
                        "green",
                        False,
                    )

                    # -----------------------------------------
                    # Current important state.
                    # -----------------------------------------

                    style_step(
                        step4_row,
                        step4_icon,
                        step4_title,
                        "✅",
                        "green",
                        True,
                    )

                # =================================================
                # DETECT NEW CONFIRMED EVENT
                # =================================================

                def detect_new_medicine_event() -> bool:
                    """
                    Returns True when the green event state
                    should currently remain latched.
                    """

                    current_version = (
                        camera_service
                        .get_medicine_version()
                    )

                    previous_version = int(
                        medicine_state[
                            "version"
                        ]
                    )

                    # -----------------------------------------
                    # New camera-confirmed medicine event.
                    # -----------------------------------------

                    if (
                        current_version
                        > previous_version
                    ):

                        medicine_state[
                            "version"
                        ] = current_version

                        medicine_state[
                            "event_confirmed_until"
                        ] = (
                            time.monotonic()
                            +
                            MEDICINE_EVENT_SUCCESS_DISPLAY_SECONDS
                        )

                        # -------------------------------------
                        # Store exact event details so they
                        # remain visible even after pipeline
                        # resets to scanning.
                        # -------------------------------------

                        latest_event = (
                            camera_service
                            .get_latest_medicine()
                        )

                        if latest_event is not None:

                            medicine_state[
                                "confirmed_name"
                            ] = (
                                latest_event.medicine_name
                            )

                            medicine_state[
                                "confirmed_dosage"
                            ] = (
                                latest_event.dosage
                            )

                    # -----------------------------------------
                    # Check if latch is still active.
                    # -----------------------------------------

                    return (
                        time.monotonic()
                        <
                        float(
                            medicine_state[
                                "event_confirmed_until"
                            ]
                        )
                    )

                # =================================================
                # UPDATE MEDICINE STATUS
                # =================================================

                def update_medicine_status() -> None:

                    # =========================================
                    # FIRST:
                    # CHECK FOR A NEW CONFIRMED EVENT
                    # =========================================

                    event_success_visible = (
                        detect_new_medicine_event()
                    )

                    # -----------------------------------------
                    # Event success has priority over live
                    # backend scanning state.
                    # -----------------------------------------

                    if event_success_visible:

                        show_event_confirmed_state()

                        return

                    # =========================================
                    # NORMAL LIVE STATUS
                    # =========================================

                    medicine_status = (
                        camera_service
                        .get_medicine_live_status()
                    )

                    phase = str(
                        medicine_status.phase
                        or "idle"
                    ).strip().lower()

                    medicine_name = (
                        medicine_status
                        .medicine_name
                    )

                    dosage = (
                        get_medicine_dosage(
                            medicine_name
                        )
                    )

                    reset_steps()

                    # =========================================
                    # 1. MEDICINE NOT DETECTED
                    # =========================================

                    if (
                        camera_service.state
                        in {
                            CameraState.STOPPED,
                            CameraState.STARTING,
                            CameraState.ERROR,
                            CameraState.PAUSED,
                        }

                        or

                        phase
                        in {
                            "idle",
                            "no_registered_medicine",
                        }
                    ):

                        style_step(
                            step1_row,
                            step1_icon,
                            step1_title,
                            "🔴",
                            "red",
                            True,
                        )

                        return

                    # =========================================
                    # 2. MEDICINE CHECKING
                    # =========================================

                    if (
                        phase
                        == "scanning_medicine"
                    ):

                        style_step(
                            step1_row,
                            step1_icon,
                            step1_title,
                            "✓",
                            "green",
                            False,
                        )

                        style_step(
                            step2_row,
                            step2_icon,
                            step2_title,
                            "🟡",
                            "yellow",
                            True,
                        )

                        return

                    # =========================================
                    # 3. MEDICINE CONFIRMED
                    # =========================================

                    if (
                        phase
                        in {
                            "watching_action",
                            "waiting_for_person",
                            "expired",
                        }
                        and medicine_name
                    ):

                        style_step(
                            step1_row,
                            step1_icon,
                            step1_title,
                            "✓",
                            "green",
                            False,
                        )

                        style_step(
                            step2_row,
                            step2_icon,
                            step2_title,
                            "✓",
                            "green",
                            False,
                        )

                        medicine_display = (
                            format_medicine_name(
                                medicine_name,
                                dosage,
                            )
                        )

                        step3_title.set_text(
                            (
                                "Medicine confirmed"
                                f" • {medicine_display}"
                            )
                        )

                        style_step(
                            step3_row,
                            step3_icon,
                            step3_title,
                            "🟢",
                            "green",
                            True,
                        )

                        return

                    # =========================================
                    # BACKUP EVENT SAVED CHECK
                    #
                    # Normally medicine_version catches this.
                    # Keep this for completeness.
                    # =========================================

                    if (
                        phase
                        == "event_saved"
                    ):

                        medicine_state[
                            "confirmed_name"
                        ] = (
                            medicine_name
                        )

                        medicine_state[
                            "confirmed_dosage"
                        ] = (
                            dosage
                        )

                        medicine_state[
                            "event_confirmed_until"
                        ] = (
                            time.monotonic()
                            +
                            MEDICINE_EVENT_SUCCESS_DISPLAY_SECONDS
                        )

                        show_event_confirmed_state()

                        return

                    # =========================================
                    # FALLBACK:
                    # SOMETHING IS BEING CHECKED
                    # =========================================

                    style_step(
                        step1_row,
                        step1_icon,
                        step1_title,
                        "✓",
                        "green",
                        False,
                    )

                    style_step(
                        step2_row,
                        step2_icon,
                        step2_title,
                        "🟡",
                        "yellow",
                        True,
                    )

                # =================================================
                # UPDATE CAMERA STATUS
                # =================================================

                def update_camera_status() -> None:

                    status = (
                        camera_service
                        .get_status()
                    )

                    state = (
                        camera_service.state
                    )

                    # =========================================
                    # CAMERA STATE
                    # =========================================

                    if (
                        state
                        == CameraState.RUNNING
                    ):

                        camera_state_label.set_text(
                            "🟢 Monitoring surroundings"
                        )

                    elif (
                        state
                        == CameraState.PAUSED
                    ):

                        camera_state_label.set_text(
                            "🟡 Monitoring paused"
                        )

                    elif (
                        state
                        == CameraState.STARTING
                    ):

                        camera_state_label.set_text(
                            "🔵 Camera starting"
                        )

                    elif (
                        state
                        == CameraState.ERROR
                    ):

                        camera_state_label.set_text(
                            "🔴 Camera error"
                        )

                    else:

                        camera_state_label.set_text(
                            "⚫ Monitoring stopped"
                        )

                    # =========================================
                    # LATEST OBJECT
                    # =========================================

                    detection = (
                        camera_service
                        .get_latest_detection()
                    )

                    if (
                        detection
                        is not None
                    ):

                        detection_label.set_text(
                            (
                                f"{detection.object_name.title()} "
                                f"• "
                                f"{detection.confidence * 100:.1f}% "
                                f"• "
                                f"{detection.location} "
                                f"• "
                                f"{detection.detected_at:%I:%M:%S %p}"
                            )
                        )

                    # =========================================
                    # CAMERA STATS
                    # =========================================

                    camera_stats_label.set_text(
                        (
                            "Captured: "
                            f"{status['captured_frames']} | "
                            "YOLO processed: "
                            f"{status['processed_frames']} | "
                            "Face processed: "
                            f"{status['face_processed_frames']} | "
                            "Medicine processed: "
                            f"{status['medicine_processed_frames']} | "
                            "Inference: "
                            f"{status['inference_seconds']:.2f}s"
                        )
                    )

                    # =========================================
                    # MEDICINE UI
                    # =========================================

                    update_medicine_status()

                # -------------------------------------------------
                # 0.5 seconds is fine now because confirmation is
                # latched through medicine_version.
                # -------------------------------------------------

                ui.timer(
                    0.5,
                    update_camera_status,
                )

            # =================================================
            # QUICK ACTIONS
            # =================================================

            with ui.card().classes(
                "memorai-card w-full"
            ):

                ui.label(
                    "⚡ Quick Actions"
                ).classes(
                    "memorai-section-title"
                )

                ui.label(
                    (
                        "Use these shortcuts to retrieve "
                        "common memories quickly."
                    )
                ).classes(
                    "opacity-70"
                )

                quick_answer = (
                    ui.label(
                        ""
                    )
                    .classes(
                        "text-blue-300 text-lg"
                    )
                )

                # =============================================
                # QUICK QUESTION
                # =============================================

                async def run_quick_question(
                    question: str,
                ) -> None:

                    stop_speaking()

                    quick_state[
                        "latest_answer"
                    ] = ""

                    quick_answer.set_text(
                        "Checking memories..."
                    )

                    try:

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

                        quick_state[
                            "latest_answer"
                        ] = answer

                        quick_answer.set_text(
                            answer
                        )

                    except Exception as exc:

                        quick_state[
                            "latest_answer"
                        ] = ""

                        quick_answer.set_text(
                            (
                                "MemorAi could not "
                                "retrieve this memory."
                            )
                        )

                        ui.notify(
                            str(
                                exc
                            ),
                            type="negative",
                        )

                # =============================================
                # SPEAK QUICK RESULT
                # =============================================

                def speak_quick_answer() -> None:

                    answer = (
                        quick_state.get(
                            "latest_answer"
                        )
                        or ""
                    ).strip()

                    if not answer:

                        ui.notify(
                            (
                                "Choose a quick action first, "
                                "then press Speak Result."
                            ),
                            type="warning",
                        )

                        return

                    speak_text(
                        answer
                    )

                # =============================================
                # QUICK BUTTONS
                # =============================================

                with ui.row().classes(
                    "w-full gap-4 flex-wrap"
                ):

                    ui.button(
                        "🔎 Find My Phone",
                        on_click=lambda: (
                            run_quick_question(
                                "Where is my cell phone?"
                            )
                        ),
                    )

                    ui.button(
                        "🧠 Today's Memories",
                        on_click=lambda: (
                            run_quick_question(
                                (
                                    "What objects "
                                    "were detected today?"
                                )
                            )
                        ),
                    )

                    ui.button(
                        "🕐 Recent Objects",
                        on_click=lambda: (
                            run_quick_question(
                                (
                                    "What objects "
                                    "were detected recently?"
                                )
                            )
                        ),
                    )

                with ui.row().classes(
                    "w-full gap-3 flex-wrap"
                ):

                    ui.button(
                        "Speak Result",
                        icon="volume_up",
                        on_click=(
                            speak_quick_answer
                        ),
                    ).props(
                        "outline"
                    )

                    ui.button(
                        "Stop Speaking",
                        icon="stop",
                        on_click=(
                            stop_speaking
                        ),
                    ).props(
                        "outline"
                    )

            # =================================================
            # PEOPLE SEEN TODAY
            # =================================================

            with ui.card().classes(
                "memorai-card w-full"
            ):

                with ui.row().classes(
                    "w-full items-center justify-between"
                ):

                    with ui.column().classes(
                        "gap-1"
                    ):

                        ui.label(
                            "👤 People Seen Today"
                        ).classes(
                            "memorai-section-title"
                        )

                        ui.label(
                            (
                                "Latest confirmed person "
                                "encounters from today."
                            )
                        ).classes(
                            "opacity-70"
                        )

                    ui.button(
                        "View People History",
                        icon="history",
                        on_click=lambda: (
                            ui.navigate.to(
                                "/people"
                            )
                        ),
                    ).props(
                        "outline"
                    )

                # =============================================
                # PEOPLE LIST
                # =============================================

                @ui.refreshable
                def people_seen_today() -> None:

                    try:

                        events = (
                            get_person_events_for_date(
                                date.today(),
                                limit=5,
                            )
                        )

                    except Exception as exc:

                        ui.label(
                            (
                                "Could not load today's "
                                f"person encounters: {exc}"
                            )
                        ).classes(
                            "text-negative"
                        )

                        return

                    if not events:

                        ui.label(
                            (
                                "No registered people have "
                                "been encountered today."
                            )
                        ).classes(
                            "opacity-70"
                        )

                        return

                    with ui.column().classes(
                        "w-full gap-2"
                    ):

                        for event in events:

                            detected_at = (
                                event.get(
                                    "detected_at"
                                )
                            )

                            if (
                                detected_at
                                and hasattr(
                                    detected_at,
                                    "strftime",
                                )
                            ):

                                time_text = (
                                    detected_at
                                    .strftime(
                                        "%I:%M %p"
                                    )
                                )

                            else:

                                time_text = str(
                                    detected_at
                                    or ""
                                )

                            relation = (
                                event.get(
                                    "relation"
                                )
                                or "Registered Person"
                            )

                            with ui.row().classes(
                                (
                                    "w-full items-center "
                                    "justify-between "
                                    "rounded-xl "
                                    "px-4 py-3 "
                                    "bg-gray-500/10"
                                )
                            ):

                                with ui.column().classes(
                                    "gap-0"
                                ):

                                    ui.label(
                                        event[
                                            "person_name"
                                        ]
                                    ).classes(
                                        (
                                            "text-lg "
                                            "font-semibold"
                                        )
                                    )

                                    ui.label(
                                        relation
                                    ).classes(
                                        "text-sm opacity-70"
                                    )

                                ui.label(
                                    time_text
                                ).classes(
                                    "opacity-70"
                                )

                people_seen_today()

                # =============================================
                # PEOPLE AUTO REFRESH
                # =============================================

                def check_person_updates() -> None:

                    current_version = (
                        camera_service
                        .get_person_version()
                    )

                    if (
                        current_version
                        == person_state[
                            "version"
                        ]
                    ):

                        return

                    person_state[
                        "version"
                    ] = current_version

                    people_seen_today.refresh()

                ui.timer(
                    0.5,
                    check_person_updates,
                )

            # =================================================
            # RECENT MEMORIES
            # =================================================

            with ui.card().classes(
                "memorai-card w-full"
            ):

                ui.label(
                    "🧠 Recent Memories"
                ).classes(
                    "memorai-section-title"
                )

                memories_container = (
                    ui.column()
                    .classes(
                        "w-full gap-3"
                    )
                )

                # =============================================
                # REFRESH MEMORIES
                # =============================================

                async def refresh_recent_memories() -> None:

                    try:

                        memories = (
                            await asyncio.to_thread(
                                get_recent_events,
                                5,
                            )
                        )

                    except Exception as exc:

                        ui.notify(
                            (
                                "Could not load memories: "
                                f"{exc}"
                            ),
                            type="negative",
                        )

                        return

                    memories_container.clear()

                    with memories_container:

                        if not memories:

                            ui.label(
                                "No memories recorded yet."
                            )

                            return

                        for memory in memories:

                            object_name = (
                                memory.get(
                                    "object_name",
                                    "Memory",
                                )
                            )

                            location = (
                                memory.get(
                                    "location",
                                    "Unknown location",
                                )
                            )

                            detected_at = (
                                memory.get(
                                    "detected_at"
                                )
                            )

                            memory_text = (
                                memory.get(
                                    "memory_text"
                                )
                            )

                            if (
                                detected_at
                                and hasattr(
                                    detected_at,
                                    "strftime",
                                )
                            ):

                                time_text = (
                                    detected_at.strftime(
                                        (
                                            "%d %b %Y "
                                            "• %I:%M %p"
                                        )
                                    )
                                )

                            else:

                                time_text = str(
                                    detected_at
                                    or ""
                                )

                            with ui.card().classes(
                                "memorai-memory-card"
                            ):

                                with ui.row().classes(
                                    (
                                        "w-full "
                                        "justify-between"
                                    )
                                ):

                                    ui.label(
                                        object_name.title()
                                    ).classes(
                                        (
                                            "text-xl "
                                            "font-semibold"
                                        )
                                    )

                                    ui.label(
                                        time_text
                                    ).classes(
                                        "text-gray-500"
                                    )

                                ui.label(
                                    memory_text
                                    or (
                                        f"{object_name.title()} "
                                        f"was observed in "
                                        f"{location}."
                                    )
                                )

                ui.timer(
                    2.0,
                    refresh_recent_memories,
                )