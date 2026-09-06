from __future__ import annotations

import asyncio
import base64
import mimetypes
from datetime import datetime
from pathlib import Path
from typing import Any

from nicegui import ui

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


# ============================================================
# CONSTANTS
# ============================================================


RECENT_DETECTION_LIMIT = 10

STATUS_REFRESH_SECONDS = 0.25

DETECTION_WATCH_SECONDS = 0.20

FALLBACK_DATABASE_REFRESH_SECONDS = 5.0


# ============================================================
# DATE / TIME
# ============================================================


def _format_detection_time(
    value: Any,
) -> str:

    if value is None:

        return "Unknown"

    if isinstance(
        value,
        datetime,
    ):

        return value.strftime(
            "%d %b %Y • %I:%M:%S %p"
        )

    return str(
        value
    )


# ============================================================
# CONFIDENCE
# ============================================================


def _format_confidence(
    value: Any,
) -> str:

    if value is None:

        return "—"

    try:

        confidence = (
            float(
                value
            )
        )

    except (
        TypeError,
        ValueError,
    ):

        return "—"

    return (
        f"{confidence * 100:.1f}%"
    )


# ============================================================
# OBJECT NAME
# ============================================================


def _format_object_name(
    value: Any,
) -> str:

    if not value:

        return (
            "Unknown object"
        )

    return (
        str(
            value
        )
        .replace(
            "_",
            " ",
        )
        .strip()
        .title()
    )


# ============================================================
# LOCATION
# ============================================================


def _format_location(
    value: Any,
) -> str:

    if not value:

        return (
            "Unknown location"
        )

    return str(
        value
    )


# ============================================================
# EVIDENCE IMAGE
# ============================================================


def _image_path_to_data_url(
    image_path: str | None,
) -> str | None:

    if not image_path:

        return (
            None
        )

    path = (
        Path(
            image_path
        )
    )

    if not path.exists():

        return (
            None
        )

    if not path.is_file():

        return (
            None
        )

    try:

        mime_type, _ = (
            mimetypes.guess_type(
                path.name
            )
        )

        if not mime_type:

            mime_type = (
                "image/jpeg"
            )

        encoded = (
            base64
            .b64encode(
                path.read_bytes()
            )
            .decode(
                "ascii"
            )
        )

        return (
            f"data:{mime_type};"
            f"base64,{encoded}"
        )

    except Exception:

        return (
            None
        )


# ============================================================
# RECENT DETECTIONS
# ============================================================


def _load_recent_detections(
    limit: int = (
        RECENT_DETECTION_LIMIT
    ),
) -> list[
    dict[
        str,
        Any,
    ]
]:

    try:

        events = (
            get_recent_events(
                limit=limit
            )
        )

    except Exception:

        return []

    return list(
        events
        or
        []
    )


# ============================================================
# BUILD TABLE ROWS
# ============================================================


def _build_table_rows(
    events: list[
        dict[
            str,
            Any,
        ]
    ],
) -> list[
    dict[
        str,
        Any,
    ]
]:

    rows: list[
        dict[
            str,
            Any,
        ]
    ] = []

    for event in events:

        rows.append(
            {
                "id": (
                    event.get(
                        "id"
                    )
                ),

                "object_name": (
                    _format_object_name(
                        event.get(
                            "object_name"
                        )
                    )
                ),

                "location": (
                    _format_location(
                        event.get(
                            "location"
                        )
                    )
                ),

                "confidence": (
                    _format_confidence(
                        event.get(
                            "confidence"
                        )
                    )
                ),

                "detected_at": (
                    _format_detection_time(
                        event.get(
                            "detected_at"
                        )
                    )
                ),
            }
        )

    return (
        rows
    )


# ============================================================
# CAMERA PAGE
# ============================================================


def build_camera_page(
    camera_service: CameraService,
) -> None:

    # ========================================================
    # PAGE
    # ========================================================

    with ui.row().classes(
        (
            "memorai-page "
            "items-start "
            "no-wrap"
        )
    ):

        # ====================================================
        # SIDEBAR
        # ====================================================

        sidebar(
            active_path="/camera"
        )

        # ====================================================
        # MAIN CONTENT
        # ====================================================

        with ui.column().classes(
            "memorai-main gap-6"
        ):

            # =================================================
            # HEADER
            # =================================================

            with ui.row().classes(
                (
                    "w-full "
                    "items-center "
                    "justify-between "
                    "gap-4 "
                    "flex-wrap"
                )
            ):

                with ui.column().classes(
                    "gap-1"
                ):

                    ui.label(
                        "📷 Camera / Monitoring"
                    ).classes(
                        "text-4xl font-bold"
                    )

                    ui.label(
                        (
                            "Monitor your surroundings "
                            "and let MemorAi remember "
                            "important objects."
                        )
                    ).classes(
                        "text-lg opacity-70"
                    )

                status_badge = (
                    ui.badge(
                        "Checking..."
                    )
                    .props(
                        "rounded"
                    )
                    .classes(
                        (
                            "text-base "
                            "px-4 py-2"
                        )
                    )
                )

            # =================================================
            # CAMERA + LATEST DETECTION
            # =================================================

            with ui.row().classes(
                (
                    "w-full "
                    "items-stretch "
                    "gap-5 "
                    "flex-wrap"
                )
            ):

                # =============================================
                # CAMERA CARD
                # =============================================

                with ui.card().classes(
                    (
                        "memorai-card "
                        "flex-1 "
                        "min-w-[520px]"
                    )
                ):

                    ui.label(
                        "Live Memory Camera"
                    ).classes(
                        "memorai-section-title"
                    )

                    ui.label(
                        (
                            "YOLO monitors the camera "
                            "for confirmed important "
                            "objects."
                        )
                    ).classes(
                        "opacity-70"
                    )

                    # =========================================
                    # VIDEO
                    # =========================================

                    with ui.element(
                        "div"
                    ).classes(
                        "w-full mt-3"
                    ):

                        ui.html(
                            """
                            <img
                                src="/video_feed"
                                class="memorai-video"
                                alt="MemorAi live camera feed"
                            >
                            """
                        )

                    # =========================================
                    # CONTROLS
                    # =========================================

                    ui.label(
                        "Monitoring Controls"
                    ).classes(
                        (
                            "text-lg "
                            "font-semibold "
                            "mt-3"
                        )
                    )

                    with ui.row().classes(
                        (
                            "w-full "
                            "gap-3 "
                            "flex-wrap"
                        )
                    ):

                        start_button = (
                            ui.button(
                                "▶ Start"
                            )
                            .classes(
                                "memorai-action-button"
                            )
                        )

                        pause_button = (
                            ui.button(
                                "⏸ Pause"
                            )
                            .classes(
                                "memorai-action-button"
                            )
                        )

                        resume_button = (
                            ui.button(
                                "▶ Resume"
                            )
                            .classes(
                                "memorai-action-button"
                            )
                        )

                        stop_button = (
                            ui.button(
                                "⏹ Stop"
                            )
                            .props(
                                "outline"
                            )
                            .classes(
                                "memorai-action-button"
                            )
                        )

                        refresh_button = (
                            ui.button(
                                "↻ Refresh"
                            )
                            .props(
                                "flat"
                            )
                            .classes(
                                "memorai-action-button"
                            )
                        )

                    camera_message = (
                        ui.label(
                            (
                                "Checking monitoring "
                                "status..."
                            )
                        )
                        .classes(
                            "text-base mt-2"
                        )
                    )

                    camera_statistics = (
                        ui.label(
                            ""
                        )
                        .classes(
                            "text-sm opacity-60"
                        )
                    )

                # =============================================
                # LATEST DETECTION
                # =============================================

                with ui.card().classes(
                    (
                        "memorai-card "
                        "w-full "
                        "max-w-[420px]"
                    )
                ):

                    ui.label(
                        "Latest Detection"
                    ).classes(
                        "memorai-section-title"
                    )

                    latest_object = (
                        ui.label(
                            "No detection yet"
                        )
                        .classes(
                            (
                                "text-2xl "
                                "font-semibold "
                                "mt-2"
                            )
                        )
                    )

                    with ui.column().classes(
                        "gap-2 mt-2"
                    ):

                        latest_location = (
                            ui.label(
                                "📍 Location: —"
                            )
                        )

                        latest_confidence = (
                            ui.label(
                                "🎯 Confidence: —"
                            )
                        )

                        latest_time = (
                            ui.label(
                                "🕐 Detected: —"
                            )
                        )

                    ui.separator().classes(
                        "my-3"
                    )

                    ui.label(
                        "Evidence"
                    ).classes(
                        "text-lg font-semibold"
                    )

                    evidence_placeholder = (
                        ui.label(
                            (
                                "No evidence image "
                                "available yet."
                            )
                        )
                        .classes(
                            "opacity-60"
                        )
                    )

                    evidence_image = (
                        ui.image()
                        .classes(
                            (
                                "w-full "
                                "rounded-xl "
                                "mt-2"
                            )
                        )
                    )

                    evidence_image.set_visibility(
                        False
                    )

            # =================================================
            # RECENT DETECTIONS
            # =================================================

            with ui.card().classes(
                "memorai-card w-full"
            ):

                with ui.row().classes(
                    (
                        "w-full "
                        "justify-between "
                        "items-center "
                        "gap-3 "
                        "flex-wrap"
                    )
                ):

                    with ui.column().classes(
                        "gap-0"
                    ):

                        ui.label(
                            "Recent Detections"
                        ).classes(
                            "memorai-section-title"
                        )

                        ui.label(
                            (
                                "Confirmed object memories "
                                "stored by MemorAi."
                            )
                        ).classes(
                            "opacity-70"
                        )

                    recent_count_label = (
                        ui.label(
                            "0 detections"
                        )
                        .classes(
                            "text-sm opacity-70"
                        )
                    )

                columns = [
                    {
                        "name": "object_name",
                        "label": "Object",
                        "field": "object_name",
                        "align": "left",
                    },
                    {
                        "name": "location",
                        "label": "Location",
                        "field": "location",
                        "align": "left",
                    },
                    {
                        "name": "confidence",
                        "label": "Confidence",
                        "field": "confidence",
                        "align": "left",
                    },
                    {
                        "name": "detected_at",
                        "label": "Detection Time",
                        "field": "detected_at",
                        "align": "left",
                    },
                ]

                recent_table = (
                    ui.table(
                        columns=columns,
                        rows=[],
                        row_key="id",
                    )
                    .props(
                        (
                            "flat "
                            "bordered "
                            "wrap-cells"
                        )
                    )
                    .classes(
                        "w-full mt-3"
                    )
                )

                no_detections_label = (
                    ui.label(
                        (
                            "No confirmed detections "
                            "are stored yet."
                        )
                    )
                    .classes(
                        "opacity-60 py-3"
                    )
                )

            # =================================================
            # MONITORED OBJECTS
            # =================================================

            with ui.card().classes(
                "memorai-card w-full"
            ):

                ui.label(
                    "Objects Being Monitored"
                ).classes(
                    "memorai-section-title"
                )

                ui.label(
                    (
                        "MemorAi currently uses the "
                        "existing important-object "
                        "configuration. Later this "
                        "will be configurable from "
                        "Settings."
                    )
                ).classes(
                    "opacity-70"
                )

                with ui.row().classes(
                    "gap-2 mt-2 flex-wrap"
                ):

                    ui.badge(
                        "📱 Phone"
                    ).props(
                        "outline"
                    )

                    ui.badge(
                        "📖 Book"
                    ).props(
                        "outline"
                    )

                    ui.badge(
                        "👓 Glasses • future"
                    ).props(
                        "outline"
                    )

                    ui.badge(
                        "💊 Medicine • future"
                    ).props(
                        "outline"
                    )

            # =================================================
            # PAGE LOCAL STATE
            # =================================================

            detection_state = {
                "version": (
                    camera_service
                    .get_detection_version()
                )
            }

            # Prevent simultaneous database refreshes
            # from manual refresh, fallback timer and
            # detection watcher.

            database_refresh_lock = (
                asyncio.Lock()
            )

            # =================================================
            # LATEST DETECTION UI
            # =================================================

            def update_latest_detection() -> None:

                detection = (
                    camera_service
                    .get_latest_detection()
                )

                if detection is None:

                    latest_object.set_text(
                        "No detection yet"
                    )

                    latest_location.set_text(
                        "📍 Location: —"
                    )

                    latest_confidence.set_text(
                        "🎯 Confidence: —"
                    )

                    latest_time.set_text(
                        "🕐 Detected: —"
                    )

                    return

                latest_object.set_text(
                    (
                        "🔎 "
                        + _format_object_name(
                            detection.object_name
                        )
                    )
                )

                latest_location.set_text(
                    (
                        "📍 Location: "
                        + _format_location(
                            detection.location
                        )
                    )
                )

                latest_confidence.set_text(
                    (
                        "🎯 Confidence: "
                        + _format_confidence(
                            detection.confidence
                        )
                    )
                )

                latest_time.set_text(
                    (
                        "🕐 Detected: "
                        + _format_detection_time(
                            detection.detected_at
                        )
                    )
                )

            # =================================================
            # REFRESH RECENT DETECTIONS
            # =================================================

            async def refresh_recent_detections() -> None:

                if database_refresh_lock.locked():

                    return

                async with database_refresh_lock:

                    events = (
                        await asyncio.to_thread(
                            _load_recent_detections
                        )
                    )

                    rows = (
                        _build_table_rows(
                            events
                        )
                    )

                    recent_table.rows = (
                        rows
                    )

                    recent_table.update()

                    count = (
                        len(
                            rows
                        )
                    )

                    if count == 1:

                        recent_count_label.set_text(
                            "1 detection"
                        )

                    else:

                        recent_count_label.set_text(
                            (
                                f"{count} "
                                "detections"
                            )
                        )

                    no_detections_label.set_visibility(
                        count == 0
                    )

                    # ========================================
                    # EVIDENCE
                    # ========================================

                    if not events:

                        evidence_image.set_visibility(
                            False
                        )

                        evidence_placeholder.set_visibility(
                            True
                        )

                        return

                    latest_event = (
                        events[0]
                    )

                    image_path = (
                        latest_event.get(
                            "image_path"
                        )
                    )

                    image_url = (
                        await asyncio.to_thread(
                            _image_path_to_data_url,
                            image_path,
                        )
                    )

                    if image_url:

                        evidence_image.set_source(
                            image_url
                        )

                        evidence_image.set_visibility(
                            True
                        )

                        evidence_placeholder.set_visibility(
                            False
                        )

                    else:

                        evidence_image.set_visibility(
                            False
                        )

                        evidence_placeholder.set_visibility(
                            True
                        )

            # =================================================
            # CAMERA STATUS
            # =================================================

            async def refresh_camera_status() -> None:

                status = (
                    camera_service
                    .get_status()
                )

                state = (
                    camera_service.state
                )

                # ============================================
                # RUNNING
                # ============================================

                if (
                    state
                    == CameraState.RUNNING
                ):

                    status_badge.set_text(
                        "🟢 Monitoring"
                    )

                    camera_message.set_text(
                        (
                            "MemorAi is actively "
                            "monitoring the camera."
                        )
                    )

                    start_button.set_enabled(
                        False
                    )

                    pause_button.set_enabled(
                        True
                    )

                    resume_button.set_enabled(
                        False
                    )

                    stop_button.set_enabled(
                        True
                    )

                # ============================================
                # PAUSED
                # ============================================

                elif (
                    state
                    == CameraState.PAUSED
                ):

                    status_badge.set_text(
                        "🟡 Paused"
                    )

                    camera_message.set_text(
                        (
                            "Live video is active, "
                            "but object detection and "
                            "memory logging are paused."
                        )
                    )

                    start_button.set_enabled(
                        False
                    )

                    pause_button.set_enabled(
                        False
                    )

                    resume_button.set_enabled(
                        True
                    )

                    stop_button.set_enabled(
                        True
                    )

                # ============================================
                # STARTING
                # ============================================

                elif (
                    state
                    == CameraState.STARTING
                ):

                    status_badge.set_text(
                        "🔵 Starting"
                    )

                    camera_message.set_text(
                        "Opening the camera..."
                    )

                    start_button.set_enabled(
                        False
                    )

                    pause_button.set_enabled(
                        False
                    )

                    resume_button.set_enabled(
                        False
                    )

                    stop_button.set_enabled(
                        True
                    )

                # ============================================
                # ERROR
                # ============================================

                elif (
                    state
                    == CameraState.ERROR
                ):

                    status_badge.set_text(
                        "🔴 Error"
                    )

                    camera_message.set_text(
                        (
                            camera_service.last_error
                            or
                            "Unknown camera error."
                        )
                    )

                    start_button.set_enabled(
                        True
                    )

                    pause_button.set_enabled(
                        False
                    )

                    resume_button.set_enabled(
                        False
                    )

                    stop_button.set_enabled(
                        True
                    )

                # ============================================
                # STOPPED
                # ============================================

                else:

                    status_badge.set_text(
                        "⚫ Stopped"
                    )

                    camera_message.set_text(
                        (
                            "Camera monitoring "
                            "is currently stopped."
                        )
                    )

                    start_button.set_enabled(
                        True
                    )

                    pause_button.set_enabled(
                        False
                    )

                    resume_button.set_enabled(
                        False
                    )

                    stop_button.set_enabled(
                        False
                    )

                # ============================================
                # STATS
                # ============================================

                captured_frames = (
                    status.get(
                        "captured_frames",
                        0,
                    )
                )

                processed_frames = (
                    status.get(
                        "processed_frames",
                        0,
                    )
                )

                inference_seconds = (
                    status.get(
                        "inference_seconds",
                        0.0,
                    )
                )

                camera_statistics.set_text(
                    (
                        "Captured frames: "
                        f"{captured_frames:,}"
                        "   •   "
                        "YOLO processed: "
                        f"{processed_frames:,}"
                        "   •   "
                        "Last inference: "
                        f"{inference_seconds:.2f}s"
                    )
                )

                update_latest_detection()

            # =================================================
            # DETECTION WATCHER
            # =================================================

            async def watch_for_new_detection() -> None:
                """
                Check an in-memory integer frequently.

                MySQL is queried only when that integer
                changes.
                """

                current_version = (
                    camera_service
                    .get_detection_version()
                )

                if (
                    current_version
                    ==
                    detection_state[
                        "version"
                    ]
                ):

                    return

                detection_state[
                    "version"
                ] = (
                    current_version
                )

                # Update the lightweight latest-detection
                # card immediately.

                update_latest_detection()

                # Then reload MySQL/evidence once.

                await (
                    refresh_recent_detections()
                )

            # =================================================
            # FULL REFRESH
            # =================================================

            async def refresh_page_data() -> None:

                await (
                    refresh_camera_status()
                )

                await (
                    refresh_recent_detections()
                )

                detection_state[
                    "version"
                ] = (
                    camera_service
                    .get_detection_version()
                )

            # =================================================
            # START
            # =================================================

            async def start_monitoring() -> None:

                camera_message.set_text(
                    "Starting camera..."
                )

                await asyncio.to_thread(
                    camera_service.start
                )

                await (
                    refresh_page_data()
                )

                if (
                    camera_service.state
                    == CameraState.RUNNING
                ):

                    ui.notify(
                        (
                            "Camera monitoring "
                            "started."
                        ),
                        type="positive",
                    )

                elif (
                    camera_service.state
                    == CameraState.ERROR
                ):

                    ui.notify(
                        (
                            camera_service.last_error
                            or
                            "Camera could not start."
                        ),
                        type="negative",
                    )

            # =================================================
            # PAUSE
            # =================================================

            async def pause_monitoring() -> None:

                await asyncio.to_thread(
                    camera_service.pause
                )

                await (
                    refresh_camera_status()
                )

                ui.notify(
                    "Monitoring paused.",
                    type="warning",
                )

            # =================================================
            # RESUME
            # =================================================

            async def resume_monitoring() -> None:

                await asyncio.to_thread(
                    camera_service.resume
                )

                await (
                    refresh_camera_status()
                )

                ui.notify(
                    "Monitoring resumed.",
                    type="positive",
                )

            # =================================================
            # STOP
            # =================================================

            async def stop_monitoring() -> None:

                await asyncio.to_thread(
                    camera_service.stop
                )

                await (
                    refresh_camera_status()
                )

                ui.notify(
                    (
                        "Camera monitoring "
                        "stopped."
                    ),
                    type="info",
                )

            # =================================================
            # CONNECT BUTTONS
            # =================================================

            start_button.on(
                "click",
                start_monitoring,
            )

            pause_button.on(
                "click",
                pause_monitoring,
            )

            resume_button.on(
                "click",
                resume_monitoring,
            )

            stop_button.on(
                "click",
                stop_monitoring,
            )

            refresh_button.on(
                "click",
                refresh_page_data,
            )

            # =================================================
            # STATUS TIMER
            # =================================================

            ui.timer(
                STATUS_REFRESH_SECONDS,
                refresh_camera_status,
            )

            # =================================================
            # NEW DETECTION WATCHER
            # =================================================

            ui.timer(
                DETECTION_WATCH_SECONDS,
                watch_for_new_detection,
            )

            # =================================================
            # FALLBACK DATABASE REFRESH
            # =================================================

            ui.timer(
                FALLBACK_DATABASE_REFRESH_SECONDS,
                refresh_recent_detections,
            )

            # =================================================
            # INITIAL LOAD
            # =================================================

            ui.timer(
                0.1,
                refresh_page_data,
                once=True,
            )