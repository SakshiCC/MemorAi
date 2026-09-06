from __future__ import annotations

from datetime import datetime
from typing import Any

import streamlit as st


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
        Shared camera monitoring service.

    rag_service:
        Existing MemorAi RAG / question-answering service.

    memory_repository:
        Existing MySQL memory/event repository.

    Notes
    -----
    These services are passed into the Home page instead of
    creating new copies every time Streamlit reruns.
    """

    render_page_styles()

    render_header()

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    render_ask_memorai(
        rag_service=rag_service
    )

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    render_camera_section(
        camera_monitor=camera_monitor
    )

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    render_quick_actions(
        rag_service=rag_service
    )

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    render_recent_memories(
        memory_repository=memory_repository
    )


# ============================================================
# STYLING
# ============================================================


def render_page_styles() -> None:
    """
    Elder-friendly styling for the Home page.
    """

    st.markdown(
        """
        <style>

        /* ---------------------------------------------
           Main page
        --------------------------------------------- */

        .block-container {
            max-width: 1350px;
            padding-top: 1.8rem;
            padding-bottom: 4rem;
        }


        /* ---------------------------------------------
           Main MemorAi title
        --------------------------------------------- */

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


        /* ---------------------------------------------
           Greeting
        --------------------------------------------- */

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


        /* ---------------------------------------------
           Section cards
        --------------------------------------------- */

        .memorai-card-title {
            font-size: 24px;
            font-weight: 650;
            margin-bottom: 8px;
        }


        /* ---------------------------------------------
           Status
        --------------------------------------------- */

        .status-ready {
            padding: 10px 15px;
            border-radius: 10px;
            font-size: 17px;
            font-weight: 600;
            border: 1px solid rgba(120, 120, 120, 0.25);
        }


        /* ---------------------------------------------
           Buttons
        --------------------------------------------- */

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


        /* ---------------------------------------------
           Inputs
        --------------------------------------------- */

        div[data-baseweb="input"] input {
            font-size: 18px;
            min-height: 52px;
        }


        /* ---------------------------------------------
           Metrics
        --------------------------------------------- */

        div[data-testid="stMetric"] {
            border: 1px solid rgba(130, 130, 130, 0.25);
            border-radius: 12px;
            padding: 15px;
        }


        /* ---------------------------------------------
           Divider
        --------------------------------------------- */

        hr {
            margin-top: 20px;
            margin-bottom: 20px;
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
    Return a greeting according to local system time.
    """

    hour = datetime.now().hour

    if hour < 12:
        return "Good morning"

    if hour < 17:
        return "Good afternoon"

    return "Good evening"


def render_header() -> None:
    """
    MemorAi heading and simple user-friendly status.
    """

    title_column, status_column = st.columns(
        [4, 1]
    )

    with title_column:

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

    with status_column:

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
    Small Ask MemorAi component on Home.

    A complete chat interface will later exist on
    the dedicated Ask MemorAi page.
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
        key="home_ask_memorai_form",
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

    # Save the question so it stays available
    # during Streamlit reruns.
    st.session_state[
        "home_last_question"
    ] = question

    if rag_service is None:

        st.warning(
            "The RAG service has not been connected "
            "to the Home page yet."
        )

        return

    try:

        with st.spinner(
            "Searching your memories..."
        ):

            result = rag_service.answer_question(
                question
            )

        # Support both:
        # 1. result.answer
        # 2. a simple returned string.

        if hasattr(result, "answer"):

            answer = result.answer

        else:

            answer = str(result)

        st.session_state[
            "home_last_answer"
        ] = answer

        st.markdown("#### 🧠 MemorAi")

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
# CAMERA
# ============================================================


def render_camera_section(
    camera_monitor=None,
) -> None:
    """
    Compact live camera component.

    The dedicated Camera / Monitoring page will later
    display the full monitoring interface.
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
        "MemorAi observes the surrounding area "
        "and updates memories when important objects are seen."
    )

    # --------------------------------------------------------
    # Camera not connected yet
    # --------------------------------------------------------

    if camera_monitor is None:

        st.info(
            "Camera monitoring service is not connected yet."
        )

        return

    # --------------------------------------------------------
    # Automatically start monitoring
    # --------------------------------------------------------

    if (
        "home_camera_initialized"
        not in st.session_state
    ):

        st.session_state[
            "home_camera_initialized"
        ] = True

        try:

            camera_monitor.start()

        except Exception as exc:

            st.error(
                "Could not start the camera."
            )

            with st.expander(
                "Technical details"
            ):

                st.code(
                    str(exc)
                )

    # --------------------------------------------------------
    # Status
    # --------------------------------------------------------

    status_column, control_column = st.columns(
        [3, 2]
    )

    with status_column:

        if camera_monitor.running:

            st.success(
                "🟢 Monitoring surroundings"
            )

        else:

            st.warning(
                "⏸ Camera monitoring is paused"
            )

    # --------------------------------------------------------
    # Start / pause controls
    # --------------------------------------------------------

    with control_column:

        start_column, pause_column = st.columns(
            2
        )

        with start_column:

            if st.button(
                "▶ Start",
                key="home_camera_start",
                use_container_width=True,
            ):

                try:

                    camera_monitor.start()

                    st.rerun()

                except Exception as exc:

                    st.error(
                        f"Could not start camera: {exc}"
                    )

        with pause_column:

            if st.button(
                "⏸ Pause",
                key="home_camera_pause",
                use_container_width=True,
            ):

                try:

                    camera_monitor.stop()

                    st.rerun()

                except Exception as exc:

                    st.error(
                        f"Could not pause camera: {exc}"
                    )

    # --------------------------------------------------------
    # Live frame
    # --------------------------------------------------------

    frame = None

    try:

        frame = camera_monitor.get_latest_frame()

    except Exception as exc:

        st.error(
            f"Could not read camera frame: {exc}"
        )

    if frame is None:

        st.info(
            "Camera is starting. "
            "The live preview will appear here."
        )

    else:

        st.image(
            frame,
            channels="RGB",
            use_container_width=True,
        )

    # --------------------------------------------------------
    # Latest detection
    # --------------------------------------------------------

    render_latest_detection(
        camera_monitor=camera_monitor
    )


# ============================================================
# LATEST DETECTION
# ============================================================


def render_latest_detection(
    camera_monitor,
) -> None:

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

    if detected_at is not None:

        detection_time = detected_at.strftime(
            "%I:%M:%S %p"
        )

    else:

        detection_time = "Unknown"

    col1, col2, col3, col4 = st.columns(
        4
    )

    with col1:

        st.metric(
            "Object",
            str(object_name).title(),
        )

    with col2:

        st.metric(
            "Location",
            str(location).title(),
        )

    with col3:

        try:

            confidence_text = (
                f"{float(confidence) * 100:.1f}%"
            )

        except Exception:

            confidence_text = str(
                confidence
            )

        st.metric(
            "Confidence",
            confidence_text,
        )

    with col4:

        st.metric(
            "Detected",
            detection_time,
        )


# ============================================================
# QUICK ACTIONS
# ============================================================


def render_quick_actions(
    rag_service=None,
) -> None:

    st.markdown(
        """
        <div class="memorai-card-title">
            ⚡ Quick Actions
        </div>
        """,
        unsafe_allow_html=True,
    )

    first, second, third = st.columns(
        3
    )

    selected_question = None

    with first:

        if st.button(
            "🔎 Find My Phone",
            key="quick_find_phone",
            use_container_width=True,
        ):

            selected_question = (
                "Where is my cell phone?"
            )

    with second:

        if st.button(
            "🧠 Today's Memories",
            key="quick_today",
            use_container_width=True,
        ):

            selected_question = (
                "What objects were detected today?"
            )

    with third:

        if st.button(
            "🕐 Recent Objects",
            key="quick_recent",
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
            "is not connected yet."
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

            answer = str(result)

        st.info(
            f"**{selected_question}**\n\n"
            f"{answer}"
        )

    except Exception as exc:

        st.error(
            "Could not retrieve the memory."
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
    Display a small Home-page preview.

    The complete timeline will later be shown
    on the Memory Timeline page.
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
            "Recent memory preview will appear here "
            "after the MySQL repository is connected."
        )

        return

    try:

        memories = get_recent_memory_records(
            memory_repository,
            limit=5,
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
# MEMORY REPOSITORY COMPATIBILITY
# ============================================================


def get_recent_memory_records(
    repository,
    limit: int = 5,
) -> list[Any]:
    """
    Try common repository method names.

    This allows us to connect the existing MemorAi
    repository without rewriting the Home page later.

    Once we confirm the exact repository method,
    this can be simplified.
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

        if callable(method):

            try:

                return method(
                    limit=limit
                )

            except TypeError:

                return method(
                    limit
                )

    raise AttributeError(
        "The memory repository does not currently expose "
        "a supported recent-memory method."
    )


# ============================================================
# MEMORY CARD
# ============================================================


def render_memory_card(
    memory: Any,
) -> None:
    """
    Render different repository return formats:

    - dictionaries
    - objects/dataclasses
    - database rows
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

    if isinstance(
        detected_at,
        datetime,
    ):

        time_text = detected_at.strftime(
            "%d %b %Y • %I:%M %p"
        )

    elif detected_at:

        time_text = str(
            detected_at
        )

    else:

        time_text = ""

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

    with st.container(
        border=True
    ):

        left, right = st.columns(
            [4, 1]
        )

        with left:

            st.markdown(
                f"**{str(object_name).title()}**"
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
# GENERIC VALUE READER
# ============================================================


def get_memory_value(
    record: Any,
    keys: list[str],
    default=None,
):
    """
    Read values from either:
    dictionary/database row or Python object.
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