from __future__ import annotations

from datetime import datetime
from typing import Any

import streamlit as st

from app.ui.camera_monitor import CameraState


# ============================================================
# HOME PAGE
# ============================================================


def render_home(
    camera_monitor=None,
    rag_service=None,
    memory_repository=None,
) -> None:
    """
    Render the MemorAi Home page.

    Parameters
    ----------
    camera_monitor:
        Shared CameraMonitor object.

    rag_service:
        Existing MemorAi RAG service.

    memory_repository:
        Existing MySQL memory repository.
    """

    render_page_styles()

    render_header()

    st.divider()

    render_ask_memorai(
        rag_service=rag_service
    )

    st.divider()

    render_camera_section(
        camera_monitor=camera_monitor
    )

    st.divider()

    render_quick_actions(
        rag_service=rag_service
    )

    st.divider()

    render_recent_memories(
        memory_repository=memory_repository
    )


# ============================================================
# PAGE STYLING
# ============================================================


def render_page_styles() -> None:
    """
    Elder-friendly Home-page styling.
    """

    st.markdown(
        """
        <style>

        .block-container {
            max-width: 1350px;
            padding-top: 1.8rem;
            padding-bottom: 4rem;
        }

        .memorai-title {
            font-size: 42px;
            font-weight: 750;
            margin-bottom: 0px;
        }

        .memorai-subtitle {
            font-size: 19px;
            opacity: 0.75;
            margin-top: 2px;
            margin-bottom: 20px;
        }

        .memorai-greeting {
            font-size: 30px;
            font-weight: 650;
            margin-top: 15px;
            margin-bottom: 3px;
        }

        .memorai-question {
            font-size: 20px;
            opacity: 0.85;
            margin-bottom: 18px;
        }

        .memorai-card-title {
            font-size: 25px;
            font-weight: 650;
            margin-bottom: 8px;
        }

        .status-ready {
            padding: 10px 15px;
            border-radius: 10px;
            font-size: 17px;
            font-weight: 600;
            border: 1px solid rgba(120, 120, 120, 0.25);
            text-align: center;
        }

        div.stButton > button {
            min-height: 52px;
            font-size: 17px;
            font-weight: 600;
            border-radius: 11px;
        }

        div[data-testid="stFormSubmitButton"] > button {
            min-height: 52px;
            font-size: 17px;
            font-weight: 600;
            border-radius: 11px;
        }

        div[data-baseweb="input"] input {
            font-size: 18px;
            min-height: 52px;
        }

        div[data-testid="stMetric"] {
            border: 1px solid rgba(130, 130, 130, 0.25);
            border-radius: 12px;
            padding: 15px;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# HEADER
# ============================================================


def get_greeting() -> str:
    """
    Generate greeting according to current local time.
    """

    hour = datetime.now().hour

    if hour < 12:
        return "Good morning"

    if hour < 17:
        return "Good afternoon"

    return "Good evening"


def render_header() -> None:
    """
    MemorAi title, subtitle, greeting and status.
    """

    left, right = st.columns(
        [4, 1]
    )

    with left:

        st.markdown(
            """
            <div class="memorai-title">
                🧠 MemorAi
            </div>

            <div class="memorai-subtitle">
                Your Personal Memory Assistant
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:

        st.markdown(
            """
            <div class="status-ready">
                🟢 MemorAi Ready
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
        <div class="memorai-greeting">
            {get_greeting()} 👋
        </div>

        <div class="memorai-question">
            How can I help you today?
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# ASK MEMORAI
# ============================================================


def render_ask_memorai(
    rag_service=None,
) -> None:
    """
    Compact question box on the Home page.
    """

    st.markdown(
        """
        <div class="memorai-card-title">
            💬 Ask MemorAi
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        "Ask where something was last seen "
        "or ask about your recent memories."
    )

    with st.form(
        key="home_ask_form",
        clear_on_submit=False,
    ):

        question = st.text_input(
            "Ask MemorAi",
            placeholder="For example: Where is my cell phone?",
            label_visibility="collapsed",
        )

        submitted = st.form_submit_button(
            "Ask MemorAi",
            use_container_width=True,
        )

    if not submitted:
        return

    question = question.strip()

    if not question:

        st.warning(
            "Please enter a question."
        )

        return

    if rag_service is None:

        st.warning(
            "MemorAi question service "
            "has not been connected yet."
        )

        return

    try:

        with st.spinner(
            "Searching your memories..."
        ):

            result = (
                rag_service.answer_question(
                    question
                )
            )

        if hasattr(
            result,
            "answer",
        ):

            answer = result.answer

        else:

            answer = str(
                result
            )

        st.session_state[
            "home_last_answer"
        ] = answer

        st.markdown(
            "#### 🧠 MemorAi"
        )

        st.success(
            answer
        )

    except Exception as exc:

        st.error(
            "MemorAi could not answer the question."
        )

        with st.expander(
            "Technical details"
        ):

            st.code(
                str(exc)
            )


# ============================================================
# LIVE CAMERA FRAGMENT
# ============================================================


@st.fragment(
    run_every=0.20
)
def render_live_camera_fragment(
    camera_monitor,
) -> None:
    """
    Refresh only the live camera section.

    run_every=0.20 gives approximately five browser
    refreshes per second.

    CameraMonitor can continue processing frames
    independently in its background thread.
    """

    state = camera_monitor.state

    # --------------------------------------------------------
    # CAMERA FRAME
    # --------------------------------------------------------

    try:

        frame = (
            camera_monitor.get_latest_frame()
        )

    except Exception as exc:

        st.error(
            f"Could not read camera frame: {exc}"
        )

        return

    # --------------------------------------------------------
    # Compact camera preview
    # --------------------------------------------------------

    camera_column, information_column = (
        st.columns(
            [3, 1]
        )
    )

    with camera_column:

        if frame is not None:

            st.image(
                frame,
                channels="RGB",
                use_container_width=True,
            )

        elif state == CameraState.PAUSED:

            st.info(
                "Camera monitoring is paused."
            )

        elif state == CameraState.STOPPED:

            st.info(
                "Camera monitoring is stopped. "
                "Press Start to begin again."
            )

        elif state == CameraState.ERROR:

            st.error(
                "Camera error."
            )

        else:

            st.info(
                "Waiting for camera feed..."
            )

    # --------------------------------------------------------
    # Monitoring information
    # --------------------------------------------------------

    with information_column:

        st.markdown(
            "### Monitoring"
        )

        if state == CameraState.RUNNING:

            st.success(
                "🟢 Active"
            )

        elif state == CameraState.PAUSED:

            st.warning(
                "🟡 Paused"
            )

        elif state == CameraState.STARTING:

            st.info(
                "🔵 Starting"
            )

        elif state == CameraState.ERROR:

            st.error(
                "🔴 Error"
            )

        else:

            st.info(
                "⚫ Stopped"
            )

        try:

            status = (
                camera_monitor.get_status()
            )

            st.metric(
                "Frames",
                status.get(
                    "frames_processed",
                    0,
                ),
            )

        except Exception:

            pass

    # --------------------------------------------------------
    # Latest detection
    # --------------------------------------------------------

    render_latest_detection(
        camera_monitor
    )


# ============================================================
# CAMERA SECTION
# ============================================================


def render_camera_section(
    camera_monitor=None,
) -> None:
    """
    Home-page camera preview.

    Supports:

        Start
        Pause
        Resume
        Stop

    Live frames are refreshed by a Streamlit fragment.
    """

    st.markdown(
        """
        <div class="memorai-card-title">
            📷 Live Memory Camera
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        "MemorAi observes the surroundings and "
        "updates its memory when important objects are seen."
    )

    # --------------------------------------------------------
    # CAMERA NOT CONNECTED
    # --------------------------------------------------------

    if camera_monitor is None:

        st.warning(
            "Camera monitoring service "
            "has not been connected."
        )

        return

    # ========================================================
    # AUTOMATIC START
    # ========================================================

    if (
        "memorai_home_camera_initialized"
        not in st.session_state
    ):

        st.session_state[
            "memorai_home_camera_initialized"
        ] = True

        try:

            camera_monitor.start()

        except Exception as exc:

            st.error(
                "MemorAi could not start the camera."
            )

            st.code(
                str(exc)
            )

    # ========================================================
    # CURRENT STATE
    # ========================================================

    state = (
        camera_monitor.state
    )

    if state == CameraState.RUNNING:

        st.success(
            "🟢 Monitoring surroundings"
        )

    elif state == CameraState.PAUSED:

        st.warning(
            "🟡 Monitoring paused"
        )

    elif state == CameraState.STARTING:

        st.info(
            "🔵 Camera is starting..."
        )

    elif state == CameraState.ERROR:

        st.error(
            "🔴 Camera error"
        )

        if camera_monitor.last_error:

            st.caption(
                camera_monitor.last_error
            )

    else:

        st.info(
            "⚫ Monitoring stopped"
        )

    # ========================================================
    # CAMERA CONTROLS
    # ========================================================

    (
        start_col,
        pause_col,
        resume_col,
        stop_col,
    ) = st.columns(
        4
    )

    # --------------------------------------------------------
    # START
    # --------------------------------------------------------

    with start_col:

        start_disabled = (
            state
            in {
                CameraState.RUNNING,
                CameraState.STARTING,
            }
        )

        if st.button(
            "▶ Start",
            key="home_camera_start",
            use_container_width=True,
            disabled=start_disabled,
        ):

            try:

                camera_monitor.start()

                st.rerun()

            except Exception as exc:

                st.error(
                    f"Could not start camera: {exc}"
                )

    # --------------------------------------------------------
    # PAUSE
    # --------------------------------------------------------

    with pause_col:

        pause_disabled = (
            state
            != CameraState.RUNNING
        )

        if st.button(
            "⏸ Pause",
            key="home_camera_pause",
            use_container_width=True,
            disabled=pause_disabled,
        ):

            try:

                camera_monitor.pause()

                st.rerun()

            except Exception as exc:

                st.error(
                    f"Could not pause camera: {exc}"
                )

    # --------------------------------------------------------
    # RESUME
    # --------------------------------------------------------

    with resume_col:

        resume_disabled = (
            state
            != CameraState.PAUSED
        )

        if st.button(
            "▶ Resume",
            key="home_camera_resume",
            use_container_width=True,
            disabled=resume_disabled,
        ):

            try:

                camera_monitor.resume()

                st.rerun()

            except Exception as exc:

                st.error(
                    f"Could not resume camera: {exc}"
                )

    # --------------------------------------------------------
    # STOP
    # --------------------------------------------------------

    with stop_col:

        stop_disabled = (
            state
            == CameraState.STOPPED
        )

        if st.button(
            "⏹ Stop",
            key="home_camera_stop",
            use_container_width=True,
            disabled=stop_disabled,
        ):

            try:

                camera_monitor.stop()

                st.rerun()

            except Exception as exc:

                st.error(
                    f"Could not stop camera: {exc}"
                )

    st.write("")

    # ========================================================
    # LIVE REFRESHING CAMERA
    # ========================================================

    render_live_camera_fragment(
        camera_monitor
    )


# ============================================================
# LATEST DETECTION
# ============================================================


def render_latest_detection(
    camera_monitor,
) -> None:
    """
    Display the latest confirmed important-object detection.
    """

    st.markdown(
        "#### Latest Detection"
    )

    try:

        detection = (
            camera_monitor.get_latest_detection()
        )

    except Exception as exc:

        st.warning(
            f"Could not retrieve latest detection: {exc}"
        )

        return

    if detection is None:

        st.caption(
            "No important object has been detected yet."
        )

        return

    object_name = getattr(
        detection,
        "object_name",
        "Unknown",
    )

    location = getattr(
        detection,
        "location",
        "Unknown",
    )

    confidence = getattr(
        detection,
        "confidence",
        0.0,
    )

    detected_at = getattr(
        detection,
        "detected_at",
        None,
    )

    # --------------------------------------------------------
    # DETECTION TIME
    # --------------------------------------------------------

    if isinstance(
        detected_at,
        datetime,
    ):

        detected_time = (
            detected_at.strftime(
                "%I:%M:%S %p"
            )
        )

    elif detected_at:

        detected_time = str(
            detected_at
        )

    else:

        detected_time = "Unknown"

    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------

    try:

        confidence_text = (
            f"{float(confidence) * 100:.1f}%"
        )

    except Exception:

        confidence_text = str(
            confidence
        )

    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    (
        col1,
        col2,
        col3,
        col4,
    ) = st.columns(
        4
    )

    with col1:

        st.metric(
            "Object",
            str(
                object_name
            ).title(),
        )

    with col2:

        st.metric(
            "Location",
            str(
                location
            ).title(),
        )

    with col3:

        st.metric(
            "Confidence",
            confidence_text,
        )

    with col4:

        st.metric(
            "Detected",
            detected_time,
        )


# ============================================================
# QUICK ACTIONS
# ============================================================


def render_quick_actions(
    rag_service=None,
) -> None:
    """
    Home-page quick memory questions.
    """

    st.markdown(
        """
        <div class="memorai-card-title">
            ⚡ Quick Actions
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = (
        st.columns(
            3
        )
    )

    selected_question = None

    with col1:

        if st.button(
            "🔎 Find My Phone",
            key="quick_find_phone",
            use_container_width=True,
        ):

            selected_question = (
                "Where is my cell phone?"
            )

    with col2:

        if st.button(
            "🧠 Today's Memories",
            key="quick_today_memories",
            use_container_width=True,
        ):

            selected_question = (
                "What objects were detected today?"
            )

    with col3:

        if st.button(
            "🕐 Recent Objects",
            key="quick_recent_objects",
            use_container_width=True,
        ):

            selected_question = (
                "What objects were detected recently?"
            )

    if selected_question is None:
        return

    if rag_service is None:

        st.warning(
            "Memory question service "
            "has not been connected."
        )

        return

    try:

        with st.spinner(
            "Checking your memories..."
        ):

            result = (
                rag_service.answer_question(
                    selected_question
                )
            )

        if hasattr(
            result,
            "answer",
        ):

            answer = result.answer

        else:

            answer = str(
                result
            )

        st.info(
            f"**{selected_question}**\n\n"
            f"{answer}"
        )

    except Exception as exc:

        st.error(
            "Could not retrieve memory."
        )

        with st.expander(
            "Technical details"
        ):

            st.code(
                str(exc)
            )


# ============================================================
# RECENT MEMORIES
# ============================================================


def render_recent_memories(
    memory_repository=None,
) -> None:
    """
    Show latest five memories on Home.

    The repository backend will be connected next.
    """

    st.markdown(
        """
        <div class="memorai-card-title">
            🧠 Recent Memories
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        "Latest memories recorded by MemorAi."
    )

    if memory_repository is None:

        st.info(
            "Recent memories will appear here "
            "after the MySQL memory repository is connected."
        )

        return

    try:

        memories = (
            get_recent_memory_records(
                repository=memory_repository,
                limit=5,
            )
        )

    except Exception as exc:

        st.error(
            "Could not load recent memories."
        )

        with st.expander(
            "Technical details"
        ):

            st.code(
                str(exc)
            )

        return

    if not memories:

        st.info(
            "No memories are available yet."
        )

        return

    for memory in memories:

        render_memory_card(
            memory
        )


# ============================================================
# RECENT MEMORY REPOSITORY HELPER
# ============================================================


def get_recent_memory_records(
    repository,
    limit: int = 5,
) -> list[Any]:
    """
    Temporary compatibility layer.

    Once the exact current repository API is connected,
    this helper will be simplified.
    """

    possible_methods = [
        "get_recent_memories",
        "get_recent_events",
        "fetch_recent_memories",
        "fetch_recent_events",
        "list_recent_memories",
    ]

    for method_name in possible_methods:

        method = getattr(
            repository,
            method_name,
            None,
        )

        if not callable(
            method
        ):

            continue

        try:

            return method(
                limit=limit
            )

        except TypeError:

            return method(
                limit
            )

    raise AttributeError(
        "The connected memory repository does not "
        "provide a supported method for loading "
        "recent memories."
    )


# ============================================================
# MEMORY CARD
# ============================================================


def render_memory_card(
    memory: Any,
) -> None:
    """
    Display one recent memory.
    """

    object_name = get_memory_value(
        memory,
        [
            "object_name",
            "object_class",
            "label",
            "name",
        ],
        default="Memory",
    )

    location = get_memory_value(
        memory,
        [
            "location",
            "location_label",
            "region",
        ],
        default="Unknown location",
    )

    detected_at = get_memory_value(
        memory,
        [
            "detected_at",
            "created_at",
            "timestamp",
            "event_time",
        ],
        default=None,
    )

    memory_text = get_memory_value(
        memory,
        [
            "memory_text",
            "description",
            "text",
        ],
        default=None,
    )

    # --------------------------------------------------------
    # TIME
    # --------------------------------------------------------

    if isinstance(
        detected_at,
        datetime,
    ):

        time_text = (
            detected_at.strftime(
                "%d %b %Y • %I:%M %p"
            )
        )

    elif detected_at:

        time_text = str(
            detected_at
        )

    else:

        time_text = ""

    # --------------------------------------------------------
    # MEMORY TEXT
    # --------------------------------------------------------

    if memory_text:

        display_text = str(
            memory_text
        )

    else:

        display_text = (
            f"{str(object_name).title()} "
            f"was observed in "
            f"{str(location)}."
        )

    # --------------------------------------------------------
    # MEMORY CARD
    # --------------------------------------------------------

    with st.container(
        border=True
    ):

        left, right = (
            st.columns(
                [4, 1]
            )
        )

        with left:

            st.markdown(
                f"### {str(object_name).title()}"
            )

            st.write(
                display_text
            )

        with right:

            if time_text:

                st.caption(
                    time_text
                )


# ============================================================
# GENERIC RECORD HELPER
# ============================================================


def get_memory_value(
    record: Any,
    keys: list[str],
    default=None,
):
    """
    Read a value from dictionary/database rows
    or Python objects.
    """

    if isinstance(
        record,
        dict,
    ):

        for key in keys:

            value = record.get(
                key
            )

            if value is not None:

                return value

    for key in keys:

        if hasattr(
            record,
            key,
        ):

            value = getattr(
                record,
                key,
            )

            if value is not None:

                return value

    return default