from __future__ import annotations

import streamlit as st

from app.ui.camera_monitor import (
    CameraMonitor,
)

from app.ui.home import (
    render_home,
)

from app.ui.home_memory_repository import (
    HomeMemoryRepository,
)

from app.ui.home_rag_adapter import (
    HomeRagAdapter,
)

from app.ui.vision_adapter import (
    MemorAiVisionAdapter,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================


st.set_page_config(
    page_title="MemorAi",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# GLOBAL STYLING
# ============================================================


st.markdown(
    """
    <style>

    .block-container {
        max-width: 1400px;
        padding-top: 1.5rem;
        padding-bottom: 4rem;
    }

    section[data-testid="stSidebar"] {
        min-width: 260px;
    }

    section[data-testid="stSidebar"]
    div[role="radiogroup"] label {
        padding-top: 7px;
        padding-bottom: 7px;
        font-size: 17px;
    }

    div.stButton > button {
        min-height: 50px;
        font-size: 17px;
        font-weight: 600;
        border-radius: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# VISION ADAPTER
# ============================================================


@st.cache_resource
def get_vision_adapter() -> (
    MemorAiVisionAdapter
):

    return MemorAiVisionAdapter(
        index_in_vector_memory=True
    )


# ============================================================
# CAMERA MONITOR
# ============================================================


@st.cache_resource
def get_camera_monitor() -> (
    CameraMonitor
):

    vision_adapter = (
        get_vision_adapter()
    )

    return CameraMonitor(
        frame_processor=(
            vision_adapter.process_frame
        ),

        camera_index=0,

        width=640,
        height=480,

        # Camera remains smooth.
        capture_fps=30,

        # YOLO does not need to run 30 FPS
        # for memory logging.
        inference_fps=4,
    )


# ============================================================
# RAG
# ============================================================


@st.cache_resource
def get_home_rag_service() -> (
    HomeRagAdapter
):

    return HomeRagAdapter()


# ============================================================
# RECENT MEMORIES
# ============================================================


@st.cache_resource
def get_home_memory_repository() -> (
    HomeMemoryRepository
):

    return HomeMemoryRepository()


# ============================================================
# INITIALIZE SERVICES
# ============================================================


def initialize_home_services() -> dict:

    services = {
        "camera_monitor": None,
        "rag_service": None,
        "memory_repository": None,

        "camera_error": None,
        "rag_error": None,
        "memory_error": None,
    }

    # --------------------------------------------------------
    # CAMERA
    # --------------------------------------------------------

    try:

        services[
            "camera_monitor"
        ] = (
            get_camera_monitor()
        )

    except Exception as exc:

        services[
            "camera_error"
        ] = str(
            exc
        )

    # --------------------------------------------------------
    # RAG
    # --------------------------------------------------------

    try:

        services[
            "rag_service"
        ] = (
            get_home_rag_service()
        )

    except Exception as exc:

        services[
            "rag_error"
        ] = str(
            exc
        )

    # --------------------------------------------------------
    # RECENT MEMORIES
    # --------------------------------------------------------

    try:

        services[
            "memory_repository"
        ] = (
            get_home_memory_repository()
        )

    except Exception as exc:

        services[
            "memory_error"
        ] = str(
            exc
        )

    return services


# ============================================================
# SIDEBAR
# ============================================================


def render_sidebar() -> str:

    with st.sidebar:

        st.title(
            "🧠 MemorAi"
        )

        st.caption(
            "Your Personal Memory Assistant"
        )

        st.divider()

        selected_page = (
            st.radio(
                "Navigation",
                [
                    "🏠 Home",
                    "💬 Ask MemorAi",
                    "🎤 Voice Assistant",
                    "🧠 Memory Timeline",
                    "📷 Camera / Monitoring",
                    "👤 People",
                    "⚙ Settings",
                ],
                label_visibility="collapsed",
            )
        )

        st.divider()

        st.caption(
            "MemorAi"
        )

        st.success(
            "● System Ready"
        )

    return selected_page


# ============================================================
# PLACEHOLDER
# ============================================================


def render_not_built_page(
    title: str,
    message: str,
) -> None:

    st.title(
        title
    )

    st.info(
        message
    )

    st.caption(
        "This page will be implemented after "
        "the Home page is fully tested."
    )


# ============================================================
# DEVELOPER STATUS
# ============================================================


def render_developer_status(
    services: dict,
) -> None:

    errors = {
        "Camera / Vision": (
            services[
                "camera_error"
            ]
        ),

        "Question / RAG": (
            services[
                "rag_error"
            ]
        ),

        "Recent Memories": (
            services[
                "memory_error"
            ]
        ),
    }

    active_errors = {
        name: error
        for name, error
        in errors.items()
        if error
    }

    if not active_errors:

        return

    with st.expander(
        "Developer status"
    ):

        for name, error in (
            active_errors.items()
        ):

            st.warning(
                f"{name}: {error}"
            )


# ============================================================
# MAIN
# ============================================================


def main() -> None:

    services = (
        initialize_home_services()
    )

    selected_page = (
        render_sidebar()
    )

    # ========================================================
    # HOME
    # ========================================================

    if selected_page == "🏠 Home":

        render_home(
            camera_monitor=(
                services[
                    "camera_monitor"
                ]
            ),

            rag_service=(
                services[
                    "rag_service"
                ]
            ),

            memory_repository=(
                services[
                    "memory_repository"
                ]
            ),
        )

        render_developer_status(
            services
        )

    # ========================================================
    # ASK MEMORAI
    # ========================================================

    elif (
        selected_page
        == "💬 Ask MemorAi"
    ):

        render_not_built_page(
            "💬 Ask MemorAi",
            (
                "The full MemorAi chat interface "
                "will be implemented next."
            ),
        )

    # ========================================================
    # VOICE
    # ========================================================

    elif (
        selected_page
        == "🎤 Voice Assistant"
    ):

        render_not_built_page(
            "🎤 Voice Assistant",
            (
                "The existing Whisper voice system "
                "will be connected here."
            ),
        )

    # ========================================================
    # TIMELINE
    # ========================================================

    elif (
        selected_page
        == "🧠 Memory Timeline"
    ):

        render_not_built_page(
            "🧠 Memory Timeline",
            (
                "The complete chronological memory "
                "history will be shown here."
            ),
        )

    # ========================================================
    # CAMERA PAGE
    # ========================================================

    elif (
        selected_page
        == "📷 Camera / Monitoring"
    ):

        render_not_built_page(
            "📷 Camera / Monitoring",
            (
                "The full monitoring dashboard "
                "will be implemented here."
            ),
        )

    # ========================================================
    # PEOPLE
    # ========================================================

    elif (
        selected_page
        == "👤 People"
    ):

        render_not_built_page(
            "👤 People",
            (
                "Face recognition and known-person "
                "memory will be implemented here."
            ),
        )

    # ========================================================
    # SETTINGS
    # ========================================================

    elif (
        selected_page
        == "⚙ Settings"
    ):

        render_not_built_page(
            "⚙ Settings",
            (
                "MemorAi configuration will "
                "appear here."
            ),
        )


# ============================================================
# ENTRY
# ============================================================


if __name__ == "__main__":

    main()