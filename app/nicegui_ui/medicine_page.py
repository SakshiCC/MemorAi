from __future__ import annotations

import asyncio
import shutil
import uuid

from datetime import (
    date,
    datetime,
    time,
    timedelta,
)

from pathlib import Path

import cv2
import numpy as np

from nicegui import ui

from app.nicegui_ui.layout import (
    sidebar,
)

from app.services.camera_service import (
    CameraService,
)

from config import (
    BASE_DIR,
    MEDICINE_REGISTRATION_SAMPLES,
    REGISTERED_MEDICINE_IMAGE_DIR,
)

from database.medicine_repository import (
    STATUS_CAMERA_CONFIRMED,
    STATUS_LOW_CONFIDENCE,
    STATUS_MISSED,
    STATUS_UNCONFIRMED,
    deactivate_medicine,
    get_all_medicines,
    get_medicine_reference_images,
    get_medicine_schedules,
    get_today_medicine_events,
    reactivate_medicine,
)

from medicine.medicine_registration import (
    MedicineRegistrationService,
)


# ============================================================
# SCHEDULE LABEL OPTIONS
# ============================================================

SCHEDULE_LABEL_OPTIONS = [
    "Morning",
    "Afternoon",
    "Evening",
    "Night",
    "Other",
]


# ============================================================
# TEMP REGISTRATION ROOT
# ============================================================

REGISTRATION_TEMP_ROOT = (
    BASE_DIR
    / "data"
    / "temp"
    / "medicine_registration"
)

REGISTRATION_TEMP_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# IMAGE URL
# ============================================================

def _medicine_image_url(
    image_path: str | None,
) -> str | None:

    if not image_path:

        return None

    try:

        path = Path(
            image_path
        ).resolve()

        relative = path.relative_to(
            REGISTERED_MEDICINE_IMAGE_DIR.resolve()
        )

        return (
            "/registered_medicines/"
            + relative.as_posix()
        )

    except Exception:

        return None


# ============================================================
# TIME FORMAT
# ============================================================

def _format_schedule_time(
    value,
) -> str:

    if value is None:

        return "—"

    if isinstance(
        value,
        time,
    ):

        return value.strftime(
            "%I:%M %p"
        )

    if isinstance(
        value,
        timedelta,
    ):

        total_seconds = int(
            value.total_seconds()
        )

        hours = (
            total_seconds
            // 3600
        ) % 24

        minutes = (
            total_seconds
            % 3600
        ) // 60

        try:

            parsed = time(
                hour=hours,
                minute=minutes,
            )

            return parsed.strftime(
                "%I:%M %p"
            )

        except Exception:

            return str(
                value
            )

    if isinstance(
        value,
        str,
    ):

        raw = (
            value.strip()
        )

        for pattern in (
            "%H:%M:%S",
            "%H:%M",
        ):

            try:

                parsed = datetime.strptime(
                    raw,
                    pattern,
                )

                return parsed.strftime(
                    "%I:%M %p"
                )

            except ValueError:

                continue

        return raw

    return str(
        value
    )


# ============================================================
# DATETIME FORMAT
# ============================================================

def _format_observed_time(
    value,
) -> str:

    if value is None:

        return "—"

    if isinstance(
        value,
        datetime,
    ):

        return (
            value.strftime(
                "%I:%M %p"
            )
        )

    try:

        parsed = (
            datetime.fromisoformat(
                str(
                    value
                )
                .replace(
                    "Z",
                    "+00:00",
                )
            )
        )

        return (
            parsed.strftime(
                "%I:%M %p"
            )
        )

    except (
        TypeError,
        ValueError,
    ):

        return str(
            value
        )


# ============================================================
# CONVERT SCHEDULE VALUE TO TIME
# ============================================================

def _schedule_value_to_time(
    value,
) -> time | None:

    if value is None:

        return None

    if isinstance(
        value,
        time,
    ):

        return value

    if isinstance(
        value,
        timedelta,
    ):

        total_seconds = int(
            value.total_seconds()
        )

        hours = (
            total_seconds
            // 3600
        ) % 24

        minutes = (
            total_seconds
            % 3600
        ) // 60

        seconds = (
            total_seconds
            % 60
        )

        try:

            return time(
                hour=hours,
                minute=minutes,
                second=seconds,
            )

        except ValueError:

            return None

    raw = (
        str(
            value
        )
        .strip()
    )

    for pattern in (
        "%H:%M:%S",
        "%H:%M",
    ):

        try:

            return (
                datetime.strptime(
                    raw,
                    pattern,
                )
                .time()
            )

        except ValueError:

            continue

    return None


# ============================================================
# VALUE TO DATETIME
# ============================================================

def _value_to_datetime(
    value,
) -> datetime | None:

    if value is None:

        return None

    if isinstance(
        value,
        datetime,
    ):

        return value

    try:

        parsed = (
            datetime.fromisoformat(
                str(
                    value
                )
                .replace(
                    "Z",
                    "+00:00",
                )
            )
        )

        if (
            parsed.tzinfo
            is not None
        ):

            parsed = (
                parsed.replace(
                    tzinfo=None
                )
            )

        return parsed

    except (
        TypeError,
        ValueError,
    ):

        return None


# ============================================================
# STATUS LABEL
# ============================================================

def _medicine_status_label(
    status: str | None,
) -> str:

    cleaned = (
        str(
            status
            or ""
        )
        .strip()
        .lower()
    )

    if (
        cleaned
        == STATUS_CAMERA_CONFIRMED
    ):

        return (
            "Camera confirmed"
        )

    if (
        cleaned
        == STATUS_LOW_CONFIDENCE
    ):

        return (
            "Low confidence"
        )

    if (
        cleaned
        == STATUS_MISSED
    ):

        return (
            "Missed"
        )

    if (
        cleaned
        == STATUS_UNCONFIRMED
    ):

        return (
            "Unconfirmed"
        )

    return (
        "No event"
    )


# ============================================================
# FIND BEST EVENT FOR SCHEDULE
# ============================================================

def _find_event_for_schedule(
    events: list[dict],
    schedule_time: time | None,
) -> dict | None:
    """
    Find the medicine event that most closely belongs to
    one scheduled dose.

    Priority:

    1. Event with scheduled_for nearest this schedule.
    2. Event observed nearest this schedule.

    This is only used for UI presentation.
    """

    if not events:

        return None

    if schedule_time is None:

        return (
            events[
                0
            ]
        )

    today = (
        date.today()
    )

    target_datetime = (
        datetime.combine(
            today,
            schedule_time,
        )
    )

    # --------------------------------------------------------
    # FIRST TRY SCHEDULED_FOR
    # --------------------------------------------------------

    scheduled_candidates: list[
        tuple[
            float,
            dict,
        ]
    ] = []

    for event in events:

        scheduled_for = (
            _value_to_datetime(
                event.get(
                    "scheduled_for"
                )
            )
        )

        if scheduled_for is None:

            continue

        difference = abs(
            (
                scheduled_for
                - target_datetime
            )
            .total_seconds()
        )

        scheduled_candidates.append(
            (
                difference,
                event,
            )
        )

    if scheduled_candidates:

        scheduled_candidates.sort(
            key=lambda item:
                item[
                    0
                ]
        )

        best_difference, best_event = (
            scheduled_candidates[
                0
            ]
        )

        # Exact/near schedule association.
        if (
            best_difference
            <= 60 * 60
        ):

            return (
                best_event
            )

    # --------------------------------------------------------
    # FALLBACK TO OBSERVED TIME
    # --------------------------------------------------------

    observed_candidates: list[
        tuple[
            float,
            dict,
        ]
    ] = []

    for event in events:

        observed_at = (
            _value_to_datetime(
                event.get(
                    "observed_at"
                )
            )
        )

        if observed_at is None:

            continue

        difference = abs(
            (
                observed_at
                - target_datetime
            )
            .total_seconds()
        )

        observed_candidates.append(
            (
                difference,
                event,
            )
        )

    if not observed_candidates:

        return None

    observed_candidates.sort(
        key=lambda item:
            item[
                0
            ]
    )

    best_difference, best_event = (
        observed_candidates[
            0
        ]
    )

    # Avoid assigning an observation from a completely
    # different part of the day to this scheduled dose.
    if (
        best_difference
        <= 3 * 60 * 60
    ):

        return (
            best_event
        )

    return None


# ============================================================
# FIND SINGLE-DOSE CONFIRMED EVENT
# ============================================================

def _find_single_dose_confirmed_event(
    events: list[dict],
) -> dict | None:
    """
    Return the most recent camera-confirmed event for a
    medicine today.

    This helper is used only when the medicine has exactly
    one active daily schedule.

    Current live camera events may be stored with
    scheduled_for=None. Therefore a valid camera-confirmed
    event can occur outside the normal three-hour matching
    window used by _find_event_for_schedule().

    For one-dose medicines, a camera-confirmed event for the
    same medicine today is sufficient to represent today's
    single dose.

    For multi-dose medicines this helper must NOT be used,
    because one event should not mark several schedules as
    confirmed.
    """

    confirmed_events: list[
        dict
    ] = []

    for event in events:

        status = (
            str(
                event.get(
                    "status"
                )
                or ""
            )
            .strip()
            .lower()
        )

        if (
            status
            != STATUS_CAMERA_CONFIRMED
        ):

            continue

        confirmed_events.append(
            event
        )

    if not confirmed_events:

        return None

    confirmed_events.sort(
        key=lambda event: (
            _value_to_datetime(
                event.get(
                    "observed_at"
                )
            )
            or datetime.min
        ),
        reverse=True,
    )

    return (
        confirmed_events[
            0
        ]
    )


# ============================================================
# DETERMINE DISPLAY STATUS
# ============================================================

def _resolve_taken_status(
    schedule_time: time | None,
    event: dict | None,
) -> tuple[
    str,
    str,
]:
    """
    Return:

        taken_text
        event_status_text

    Safety:
    "Yes" here means camera-confirmed medication evidence,
    not medically proven swallowing.
    """

    now = (
        datetime.now()
    )

    if event is not None:

        status = (
            str(
                event.get(
                    "status"
                )
                or ""
            )
            .strip()
            .lower()
        )

        if (
            status
            == STATUS_CAMERA_CONFIRMED
        ):

            return (
                "Yes • Camera confirmed",
                "Camera confirmed",
            )

        if (
            status
            == STATUS_LOW_CONFIDENCE
        ):

            return (
                "Not confirmed",
                "Low confidence",
            )

        if (
            status
            == STATUS_MISSED
        ):

            return (
                "No",
                "Missed",
            )

        if (
            status
            == STATUS_UNCONFIRMED
        ):

            return (
                "Not confirmed",
                "Unconfirmed",
            )

    # --------------------------------------------------------
    # NO EVENT
    # --------------------------------------------------------

    if schedule_time is None:

        return (
            "Not confirmed",
            "No camera event",
        )

    scheduled_datetime = (
        datetime.combine(
            date.today(),
            schedule_time,
        )
    )

    if (
        now
        < scheduled_datetime
    ):

        return (
            "Upcoming",
            "Waiting for schedule",
        )

    return (
        "Not confirmed",
        "No camera confirmation",
    )


# ============================================================
# TAKEN STATUS STYLE
# ============================================================

def _taken_status_classes(
    taken_status: str,
) -> str:

    normalized = (
        taken_status
        .strip()
        .lower()
    )

    if normalized.startswith(
        "yes"
    ):

        return (
            "w-48 font-semibold "
            "text-positive"
        )

    if normalized == "upcoming":

        return (
            "w-48 font-semibold "
            "text-info"
        )

    if normalized == "no":

        return (
            "w-48 font-semibold "
            "text-negative"
        )

    return (
        "w-48 font-semibold "
        "text-warning"
    )


# ============================================================
# REGISTRATION CROP
# ============================================================

def _extract_registration_crop(
    frame: np.ndarray,
) -> np.ndarray:
    """
    Use the same central region that the live medicine
    camera worker scans.

    This keeps registration images and live-recognition
    images visually consistent.
    """

    if frame is None:

        raise ValueError(
            "Camera frame is not available."
        )

    if not isinstance(
        frame,
        np.ndarray,
    ):

        raise TypeError(
            "Camera frame must be a NumPy array."
        )

    if frame.size == 0:

        raise ValueError(
            "Camera frame is empty."
        )

    height, width = (
        frame.shape[:2]
    )

    x1 = int(
        width
        * 0.20
    )

    x2 = int(
        width
        * 0.80
    )

    y1 = int(
        height
        * 0.20
    )

    y2 = int(
        height
        * 0.80
    )

    crop = (
        frame[
            y1:y2,
            x1:x2,
        ]
    )

    if crop.size == 0:

        return (
            frame.copy()
        )

    return (
        crop.copy()
    )


# ============================================================
# PAGE
# ============================================================

def build_medicine_page(
    camera_service: CameraService,
) -> None:

    # ========================================================
    # REGISTRATION STATE
    # ========================================================

    registration_state = {
        "sample_paths": [],
        "session_dir": None,
        "service": None,
        "saving": False,
    }

    # ========================================================
    # LIVE MEDICINE STATUS REFRESH STATE
    # ========================================================
    #
    # The camera service increments medicine_version whenever
    # a new camera-confirmed medicine event is published.
    #
    # The Medicine page watches this value so Today's Medicine
    # Status refreshes automatically after a successful
    # medication event.
    # ========================================================

    medicine_status_state = {
        "version":
            camera_service
            .get_medicine_version(),
    }

    # ========================================================
    # REGISTRATION SERVICE
    # ========================================================

    async def get_registration_service(
    ) -> MedicineRegistrationService:

        existing = (
            registration_state[
                "service"
            ]
        )

        if existing is not None:

            return existing

        ui.notify(
            (
                "Preparing medicine "
                "recognition model..."
            ),
            type="info",
        )

        service = (
            await asyncio.to_thread(
                MedicineRegistrationService
            )
        )

        registration_state[
            "service"
        ] = service

        return service

    # ========================================================
    # SESSION DIRECTORY
    # ========================================================

    def ensure_session_directory(
    ) -> Path:

        existing = (
            registration_state[
                "session_dir"
            ]
        )

        if existing is not None:

            return Path(
                existing
            )

        session_dir = (
            REGISTRATION_TEMP_ROOT
            / uuid.uuid4().hex
        )

        session_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        registration_state[
            "session_dir"
        ] = session_dir

        return session_dir

    # ========================================================
    # CLEAR TEMP FILES
    # ========================================================

    def clear_registration_files(
    ) -> None:

        registration_state[
            "sample_paths"
        ].clear()

        session_dir = (
            registration_state[
                "session_dir"
            ]
        )

        if (
            session_dir is not None
            and Path(
                session_dir
            ).exists()
        ):

            try:

                shutil.rmtree(
                    session_dir
                )

            except Exception:

                pass

        registration_state[
            "session_dir"
        ] = None

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
            active_path="/medicines"
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
                    "w-full items-center "
                    "justify-between gap-4 "
                    "flex-wrap"
                )
            ):

                with ui.column().classes(
                    "gap-1"
                ):

                    ui.label(
                        "💊 Medicine Memory"
                    ).classes(
                        "text-4xl font-bold"
                    )

                    ui.label(
                        (
                            "Register medicines, review "
                            "today's scheduled doses, and "
                            "track camera-assisted medication "
                            "memories."
                        )
                    ).classes(
                        "text-base opacity-70"
                    )

                register_button = (
                    ui.button(
                        "Register Medicine",
                        icon="medication",
                    )
                    .classes(
                        "memorai-action-button"
                    )
                )

            # =================================================
            # SAFETY INFORMATION
            # =================================================

            with ui.card().classes(
                "memorai-card w-full"
            ):

                with ui.row().classes(
                    "items-start gap-3 no-wrap"
                ):

                    ui.icon(
                        "info"
                    ).classes(
                        "text-2xl"
                    )

                    with ui.column().classes(
                        "gap-1"
                    ):

                        ui.label(
                            "Camera-assisted medicine memory"
                        ).classes(
                            "text-lg font-semibold"
                        )

                        ui.label(
                            (
                                "MemorAi recognizes registered "
                                "medicine packaging and can "
                                "record camera-confirmed "
                                "medication events. A "
                                "'Yes • Camera confirmed' "
                                "status means that MemorAi "
                                "observed sufficient camera "
                                "evidence related to the "
                                "scheduled medicine. It does "
                                "not medically prove that the "
                                "medicine was swallowed."
                            )
                        ).classes(
                            "opacity-70"
                        )

            # =================================================
            # TODAY'S MEDICINE STATUS HEADER
            # =================================================

            with ui.row().classes(
                (
                    "w-full items-center "
                    "justify-between gap-3"
                )
            ):

                with ui.column().classes(
                    "gap-0"
                ):

                    ui.label(
                        "Today's Medicine Status"
                    ).classes(
                        "text-2xl font-semibold"
                    )

                    ui.label(
                        (
                            "Today's scheduled medicines "
                            "and their camera-assisted "
                            "confirmation status."
                        )
                    ).classes(
                        "opacity-70"
                    )

                ui.button(
                    "Refresh",
                    icon="refresh",
                    on_click=lambda: (
                        today_medicine_status.refresh()
                    ),
                ).props(
                    "flat"
                )

            # =================================================
            # TODAY'S MEDICINE STATUS
            # =================================================

            @ui.refreshable
            def today_medicine_status():

                medicines = (
                    get_all_medicines(
                        active_only=True
                    )
                )

                today_events = (
                    get_today_medicine_events(
                        medicine_id=None,
                        confirmed_only=False,
                    )
                )

                if not medicines:

                    with ui.card().classes(
                        "memorai-card w-full"
                    ):

                        ui.label(
                            (
                                "No active medicines "
                                "are registered."
                            )
                        ).classes(
                            "text-lg font-medium"
                        )

                        ui.label(
                            (
                                "Register a medicine to "
                                "see today's schedule and "
                                "camera status here."
                            )
                        ).classes(
                            "opacity-70"
                        )

                    return

                # ---------------------------------------------
                # GROUP EVENTS BY MEDICINE
                # ---------------------------------------------

                events_by_medicine: dict[
                    int,
                    list[dict],
                ] = {}

                for event in today_events:

                    raw_medicine_id = (
                        event.get(
                            "medicine_id"
                        )
                    )

                    if raw_medicine_id is None:

                        continue

                    try:

                        event_medicine_id = int(
                            raw_medicine_id
                        )

                    except (
                        TypeError,
                        ValueError,
                    ):

                        continue

                    events_by_medicine.setdefault(
                        event_medicine_id,
                        [],
                    ).append(
                        event
                    )

                with ui.card().classes(
                    (
                        "memorai-card w-full "
                        "p-0 overflow-x-auto"
                    )
                ):

                    # =========================================
                    # TABLE HEADER
                    # =========================================

                    with ui.row().classes(
                        (
                            "w-full min-w-[1050px] "
                            "items-center gap-4 "
                            "px-4 py-3 font-semibold "
                            "border-b opacity-70"
                        )
                    ):

                        ui.label(
                            "Medicine"
                        ).classes(
                            "w-52"
                        )

                        ui.label(
                            "Dosage"
                        ).classes(
                            "w-28"
                        )

                        ui.label(
                            "Scheduled"
                        ).classes(
                            "w-44"
                        )

                        ui.label(
                            "Taken?"
                        ).classes(
                            "w-48"
                        )

                        ui.label(
                            "Observed"
                        ).classes(
                            "w-36"
                        )

                        ui.label(
                            "Camera Status"
                        ).classes(
                            "w-52"
                        )

                    # =========================================
                    # TABLE ROWS
                    # =========================================

                    row_count = 0

                    for medicine in medicines:

                        medicine_id = int(
                            medicine[
                                "id"
                            ]
                        )

                        medicine_name = str(
                            medicine.get(
                                "medicine_name"
                            )
                            or "Medicine"
                        )

                        dosage = (
                            medicine.get(
                                "dosage"
                            )
                            or "—"
                        )

                        schedules = (
                            get_medicine_schedules(
                                medicine_id=(
                                    medicine_id
                                ),
                                active_only=True,
                            )
                        )

                        medicine_events = (
                            events_by_medicine.get(
                                medicine_id,
                                [],
                            )
                        )

                        # -------------------------------------
                        # FALLBACK FOR OLD MEDICINES WITHOUT
                        # medicine_schedules ROW
                        # -------------------------------------

                        if not schedules:

                            fallback_schedule = (
                                medicine.get(
                                    "schedule_time"
                                )
                            )

                            schedules = [
                                {
                                    "schedule_time":
                                        fallback_schedule,

                                    "schedule_label":
                                        medicine.get(
                                            "schedule_label"
                                        ),
                                }
                            ]

                        # -------------------------------------
                        # SINGLE-DOSE TODAY FALLBACK EVENT
                        # -------------------------------------
                        #
                        # Calculate this once per medicine.
                        #
                        # If this medicine has exactly one
                        # schedule and the normal schedule
                        # matcher cannot associate a camera
                        # event, we can use this confirmed
                        # event instead.
                        # -------------------------------------

                        single_dose_confirmed_event = None

                        if (
                            len(
                                schedules
                            )
                            == 1
                        ):

                            single_dose_confirmed_event = (
                                _find_single_dose_confirmed_event(
                                    medicine_events
                                )
                            )

                        for schedule in schedules:

                            row_count += 1

                            raw_schedule_time = (
                                schedule.get(
                                    "schedule_time"
                                )
                            )

                            parsed_schedule_time = (
                                _schedule_value_to_time(
                                    raw_schedule_time
                                )
                            )

                            formatted_schedule = (
                                _format_schedule_time(
                                    raw_schedule_time
                                )
                            )

                            schedule_label = (
                                schedule.get(
                                    "schedule_label"
                                )
                                or ""
                            )

                            if schedule_label:

                                schedule_display = (
                                    f"{formatted_schedule} "
                                    f"• {schedule_label}"
                                )

                            else:

                                schedule_display = (
                                    formatted_schedule
                                )

                            # =================================
                            # NORMAL SCHEDULE MATCH
                            # =================================

                            matching_event = (
                                _find_event_for_schedule(
                                    events=(
                                        medicine_events
                                    ),

                                    schedule_time=(
                                        parsed_schedule_time
                                    ),
                                )
                            )

                            # =================================
                            # SINGLE-DOSE CAMERA CONFIRMATION
                            # FALLBACK
                            # =================================
                            #
                            # Live camera events currently may
                            # have scheduled_for=None.
                            #
                            # If the medicine has only ONE
                            # active daily schedule, a valid
                            # camera-confirmed event for the
                            # same medicine today should mark
                            # today's single schedule as
                            # confirmed even if it occurred
                            # outside the three-hour schedule
                            # matching window.
                            #
                            # This fallback intentionally does
                            # NOT apply to medicines with
                            # multiple daily schedules.
                            # =================================

                            if (
                                matching_event is None
                                and
                                len(
                                    schedules
                                )
                                == 1
                                and
                                single_dose_confirmed_event
                                is not None
                            ):

                                matching_event = (
                                    single_dose_confirmed_event
                                )

                            (
                                taken_status,
                                camera_status,
                            ) = (
                                _resolve_taken_status(
                                    schedule_time=(
                                        parsed_schedule_time
                                    ),

                                    event=(
                                        matching_event
                                    ),
                                )
                            )

                            observed_time = (
                                _format_observed_time(
                                    (
                                        matching_event.get(
                                            "observed_at"
                                        )
                                        if matching_event
                                        else None
                                    )
                                )
                            )

                            with ui.row().classes(
                                (
                                    "w-full min-w-[1050px] "
                                    "items-center gap-4 "
                                    "px-4 py-4 border-b"
                                )
                            ):

                                # -----------------------------
                                # MEDICINE
                                # -----------------------------

                                ui.label(
                                    medicine_name
                                ).classes(
                                    "w-52 font-semibold"
                                )

                                # -----------------------------
                                # DOSAGE
                                # -----------------------------

                                ui.label(
                                    str(
                                        dosage
                                    )
                                ).classes(
                                    "w-28"
                                )

                                # -----------------------------
                                # SCHEDULE
                                # -----------------------------

                                ui.label(
                                    schedule_display
                                ).classes(
                                    "w-44 opacity-80"
                                )

                                # -----------------------------
                                # TAKEN STATUS
                                # -----------------------------

                                ui.label(
                                    taken_status
                                ).classes(
                                    _taken_status_classes(
                                        taken_status
                                    )
                                )

                                # -----------------------------
                                # OBSERVED TIME
                                # -----------------------------

                                ui.label(
                                    observed_time
                                ).classes(
                                    "w-36 opacity-80"
                                )

                                # -----------------------------
                                # CAMERA STATUS
                                # -----------------------------

                                ui.label(
                                    camera_status
                                ).classes(
                                    "w-52 opacity-80"
                                )

                    if row_count == 0:

                        with ui.row().classes(
                            "w-full px-4 py-5"
                        ):

                            ui.label(
                                (
                                    "No medicine schedules "
                                    "are available for today."
                                )
                            ).classes(
                                "opacity-70"
                            )

            # Initial render
            today_medicine_status()

            # =================================================
            # AUTO-REFRESH TODAY'S STATUS AFTER CAMERA EVENT
            # =================================================
            #
            # camera_service increments medicine_version when
            # a new confirmed medicine event is published.
            #
            # This page polls that lightweight version number.
            # It does NOT rerun medicine recognition.
            #
            # Whenever the version changes, only the status
            # table is rebuilt.
            # =================================================

            def check_medicine_status_updates() -> None:

                current_version = (
                    camera_service
                    .get_medicine_version()
                )

                previous_version = (
                    medicine_status_state[
                        "version"
                    ]
                )

                if (
                    current_version
                    == previous_version
                ):

                    return

                medicine_status_state[
                    "version"
                ] = current_version

                today_medicine_status.refresh()

            ui.timer(
                0.5,
                check_medicine_status_updates,
            )

            # =================================================
            # REGISTERED MEDICINES HEADER
            # =================================================

            with ui.row().classes(
                (
                    "w-full items-center "
                    "justify-between gap-3"
                )
            ):

                ui.label(
                    "Registered Medicines"
                ).classes(
                    "text-2xl font-semibold"
                )

                ui.button(
                    "Refresh",
                    icon="refresh",
                    on_click=lambda:
                        registered_medicines.refresh(),
                ).props(
                    "flat"
                )

            # =================================================
            # REGISTERED MEDICINES
            # =================================================

            @ui.refreshable
            def registered_medicines():

                medicines = (
                    get_all_medicines(
                        active_only=False
                    )
                )

                if not medicines:

                    with ui.card().classes(
                        "memorai-card w-full"
                    ):

                        ui.label(
                            (
                                "No medicines are "
                                "registered yet."
                            )
                        ).classes(
                            "text-lg font-medium"
                        )

                        ui.label(
                            (
                                "Press Register Medicine "
                                "to add the first medicine "
                                "and capture its package "
                                "images."
                            )
                        ).classes(
                            "opacity-70"
                        )

                    return

                with ui.card().classes(
                    (
                        "memorai-card w-full "
                        "p-0 overflow-hidden"
                    )
                ):

                    # =========================================
                    # HEADER
                    # =========================================

                    with ui.row().classes(
                        (
                            "w-full items-center gap-4 "
                            "px-4 py-3 font-semibold "
                            "border-b opacity-70"
                        )
                    ):

                        ui.label(
                            "Photo"
                        ).classes(
                            "w-24"
                        )

                        ui.label(
                            "Medicine"
                        ).classes(
                            "w-52"
                        )

                        ui.label(
                            "Dosage"
                        ).classes(
                            "w-28"
                        )

                        ui.label(
                            "Schedule"
                        ).classes(
                            "w-48"
                        )

                        ui.label(
                            "Samples"
                        ).classes(
                            "w-24"
                        )

                        ui.label(
                            "Status"
                        ).classes(
                            "w-24"
                        )

                        ui.label(
                            "Action"
                        ).classes(
                            "w-32"
                        )

                    # =========================================
                    # ROWS
                    # =========================================

                    for medicine in medicines:

                        medicine_id = int(
                            medicine[
                                "id"
                            ]
                        )

                        active = bool(
                            medicine[
                                "is_active"
                            ]
                        )

                        references = (
                            get_medicine_reference_images(
                                medicine_id
                            )
                        )

                        primary_reference = next(
                            (
                                reference
                                for reference
                                in references
                                if bool(
                                    reference.get(
                                        "is_primary"
                                    )
                                )
                            ),
                            None,
                        )

                        if (
                            primary_reference
                            is None
                            and references
                        ):

                            primary_reference = (
                                references[
                                    0
                                ]
                            )

                        image_url = (
                            _medicine_image_url(
                                (
                                    primary_reference.get(
                                        "image_path"
                                    )
                                    if primary_reference
                                    else None
                                )
                            )
                        )

                        schedules = (
                            get_medicine_schedules(
                                medicine_id=(
                                    medicine_id
                                ),
                                active_only=True,
                            )
                        )

                        schedule_parts: list[
                            str
                        ] = []

                        for schedule in schedules:

                            formatted_time = (
                                _format_schedule_time(
                                    schedule.get(
                                        "schedule_time"
                                    )
                                )
                            )

                            label = (
                                schedule.get(
                                    "schedule_label"
                                )
                                or ""
                            )

                            if label:

                                schedule_parts.append(
                                    (
                                        f"{formatted_time} "
                                        f"• {label}"
                                    )
                                )

                            else:

                                schedule_parts.append(
                                    formatted_time
                                )

                        schedule_text = (
                            ", ".join(
                                schedule_parts
                            )
                            if schedule_parts
                            else "—"
                        )

                        with ui.row().classes(
                            (
                                "w-full items-center gap-4 "
                                "px-4 py-3 border-b"
                            )
                        ):

                            # ---------------------------------
                            # PHOTO
                            # ---------------------------------

                            with ui.element(
                                "div"
                            ).classes(
                                "w-24"
                            ):

                                if image_url:

                                    ui.image(
                                        image_url
                                    ).classes(
                                        (
                                            "w-20 h-16 "
                                            "rounded-lg "
                                            "object-cover"
                                        )
                                    )

                                else:

                                    with ui.element(
                                        "div"
                                    ).classes(
                                        (
                                            "w-20 h-16 "
                                            "rounded-lg "
                                            "flex items-center "
                                            "justify-center "
                                            "bg-gray-500/20"
                                        )
                                    ):

                                        ui.icon(
                                            "medication"
                                        ).classes(
                                            "text-2xl opacity-60"
                                        )

                            # ---------------------------------
                            # NAME
                            # ---------------------------------

                            with ui.column().classes(
                                "w-52 gap-0"
                            ):

                                ui.label(
                                    medicine[
                                        "medicine_name"
                                    ]
                                ).classes(
                                    "font-semibold"
                                )

                                instructions = (
                                    medicine.get(
                                        "instructions"
                                    )
                                )

                                if instructions:

                                    ui.label(
                                        instructions
                                    ).classes(
                                        (
                                            "text-xs opacity-60 "
                                            "max-w-[200px]"
                                        )
                                    )

                            # ---------------------------------
                            # DOSAGE
                            # ---------------------------------

                            ui.label(
                                (
                                    medicine.get(
                                        "dosage"
                                    )
                                    or "—"
                                )
                            ).classes(
                                "w-28"
                            )

                            # ---------------------------------
                            # SCHEDULE
                            # ---------------------------------

                            ui.label(
                                schedule_text
                            ).classes(
                                "w-48 opacity-80"
                            )

                            # ---------------------------------
                            # SAMPLES
                            # ---------------------------------

                            ui.label(
                                str(
                                    len(
                                        references
                                    )
                                )
                            ).classes(
                                "w-24"
                            )

                            # ---------------------------------
                            # STATUS
                            # ---------------------------------

                            ui.label(
                                (
                                    "Active"
                                    if active
                                    else "Inactive"
                                )
                            ).classes(
                                (
                                    "w-24 font-medium "
                                    + (
                                        "text-positive"
                                        if active
                                        else "opacity-50"
                                    )
                                )
                            )

                            # ---------------------------------
                            # ACTIVATE / DEACTIVATE
                            # ---------------------------------

                            async def toggle_medicine(
                                mid=medicine_id,
                                currently_active=active,
                            ):

                                try:

                                    if currently_active:

                                        await asyncio.to_thread(
                                            deactivate_medicine,
                                            mid,
                                        )

                                        ui.notify(
                                            (
                                                "Medicine "
                                                "deactivated."
                                            ),
                                            type="positive",
                                        )

                                    else:

                                        await asyncio.to_thread(
                                            reactivate_medicine,
                                            mid,
                                        )

                                        ui.notify(
                                            (
                                                "Medicine "
                                                "activated."
                                            ),
                                            type="positive",
                                        )

                                    registered_medicines.refresh()

                                    today_medicine_status.refresh()

                                except Exception as exc:

                                    ui.notify(
                                        str(
                                            exc
                                        ),
                                        type="negative",
                                    )

                            ui.button(
                                (
                                    "Deactivate"
                                    if active
                                    else "Activate"
                                ),
                                icon=(
                                    "visibility_off"
                                    if active
                                    else "visibility"
                                ),
                                on_click=(
                                    toggle_medicine
                                ),
                            ).props(
                                "outline dense"
                            ).classes(
                                "w-32"
                            )

            registered_medicines()

            # =================================================
            # HOW REGISTRATION WORKS
            # =================================================

            with ui.card().classes(
                "memorai-card w-full"
            ):

                ui.label(
                    "Medicine Registration Guide"
                ).classes(
                    "memorai-section-title"
                )

                ui.label(
                    (
                        "For the best recognition result, "
                        "place the medicine box, strip, or "
                        "bottle inside the centre camera "
                        "area and capture slightly different "
                        "views."
                    )
                ).classes(
                    "opacity-70"
                )

                with ui.row().classes(
                    "gap-3 mt-2 flex-wrap"
                ):

                    ui.badge(
                        "1 • Front view"
                    ).props(
                        "outline"
                    )

                    ui.badge(
                        "2 • Slight left angle"
                    ).props(
                        "outline"
                    )

                    ui.badge(
                        "3 • Slight right angle"
                    ).props(
                        "outline"
                    )

                    ui.badge(
                        "4 • Closer view"
                    ).props(
                        "outline"
                    )

                    ui.badge(
                        "5 • Slightly different distance"
                    ).props(
                        "outline"
                    )

            # =================================================
            # REGISTRATION DIALOG
            # =================================================

            with ui.dialog() as registration_dialog:

                with ui.card().classes(
                    (
                        "w-[1000px] "
                        "max-w-[96vw] "
                        "rounded-2xl p-6"
                    )
                ):

                    # =========================================
                    # TITLE
                    # =========================================

                    ui.label(
                        "Register Medicine"
                    ).classes(
                        "text-2xl font-bold"
                    )

                    ui.label(
                        (
                            "Enter the medicine details and "
                            f"capture exactly "
                            f"{MEDICINE_REGISTRATION_SAMPLES} "
                            "clear package samples."
                        )
                    ).classes(
                        "opacity-70"
                    )

                    # =========================================
                    # CONTENT
                    # =========================================

                    with ui.row().classes(
                        (
                            "w-full items-start "
                            "gap-6 flex-wrap"
                        )
                    ):

                        # =====================================
                        # FORM
                        # =====================================

                        with ui.column().classes(
                            "w-80 gap-3"
                        ):

                            medicine_name_input = (
                                ui.input(
                                    label="Medicine Name",
                                    placeholder=(
                                        "e.g. Metformin"
                                    ),
                                )
                                .classes(
                                    "w-full"
                                )
                            )

                            dosage_input = (
                                ui.input(
                                    label="Dosage",
                                    placeholder=(
                                        "e.g. 500 mg"
                                    ),
                                )
                                .classes(
                                    "w-full"
                                )
                            )

                            schedule_time_input = (
                                ui.input(
                                    label="Schedule Time",
                                    value="08:00",
                                )
                                .props(
                                    "type=time"
                                )
                                .classes(
                                    "w-full"
                                )
                            )

                            schedule_label_select = (
                                ui.select(
                                    SCHEDULE_LABEL_OPTIONS,
                                    value="Morning",
                                    label="Schedule Label",
                                )
                                .classes(
                                    "w-full"
                                )
                            )

                            custom_schedule_label = (
                                ui.input(
                                    label=(
                                        "Custom Schedule Label"
                                    ),
                                    placeholder=(
                                        "e.g. After lunch"
                                    ),
                                )
                                .classes(
                                    "w-full"
                                )
                            )

                            custom_schedule_label.set_visibility(
                                False
                            )

                            def schedule_label_changed():

                                custom_schedule_label.set_visibility(
                                    schedule_label_select.value
                                    == "Other"
                                )

                            schedule_label_select.on_value_change(
                                lambda _:
                                    schedule_label_changed()
                            )

                            instructions_input = (
                                ui.textarea(
                                    label="Instructions",
                                    placeholder=(
                                        "e.g. After breakfast"
                                    ),
                                )
                                .classes(
                                    "w-full"
                                )
                            )

                        # =====================================
                        # CAMERA
                        # =====================================

                        with ui.column().classes(
                            "flex-1 min-w-[440px] gap-3"
                        ):

                            ui.label(
                                "Live Camera"
                            ).classes(
                                "text-lg font-semibold"
                            )

                            ui.label(
                                (
                                    "Place only the medicine "
                                    "package inside the centre "
                                    "scan region before "
                                    "capturing."
                                )
                            ).classes(
                                "text-sm opacity-70"
                            )

                            ui.html(
                                """
                                <div style="
                                    width:100%;
                                    max-width:560px;
                                    aspect-ratio:4/3;
                                    overflow:hidden;
                                    border-radius:16px;
                                    background:#111;
                                    position:relative;
                                ">
                                    <img
                                        src="/video_feed"
                                        style="
                                            width:100%;
                                            height:100%;
                                            object-fit:cover;
                                        "
                                    >
                                </div>
                                """
                            )

                            sample_status = (
                                ui.label(
                                    (
                                        "0 / "
                                        f"{MEDICINE_REGISTRATION_SAMPLES} "
                                        "samples captured"
                                    )
                                )
                                .classes(
                                    "font-medium"
                                )
                            )

                            sample_dots = (
                                ui.label(
                                    " ".join(
                                        [
                                            "○"
                                            for _ in range(
                                                MEDICINE_REGISTRATION_SAMPLES
                                            )
                                        ]
                                    )
                                )
                                .classes(
                                    "text-2xl tracking-wider"
                                )
                            )

                            validation_label = (
                                ui.label(
                                    (
                                        "Hold the package clearly "
                                        "inside the centre area."
                                    )
                                )
                                .classes(
                                    "text-sm opacity-70"
                                )
                            )

                    # =========================================
                    # SAMPLE INDICATOR
                    # =========================================

                    def update_sample_indicator():

                        count = len(
                            registration_state[
                                "sample_paths"
                            ]
                        )

                        total = (
                            MEDICINE_REGISTRATION_SAMPLES
                        )

                        sample_status.set_text(
                            (
                                f"{count} / {total} "
                                "samples captured"
                            )
                        )

                        dots: list[
                            str
                        ] = []

                        for index in range(
                            total
                        ):

                            dots.append(
                                (
                                    "●"
                                    if index < count
                                    else "○"
                                )
                            )

                        sample_dots.set_text(
                            " ".join(
                                dots
                            )
                        )

                    # =========================================
                    # CAPTURE SAMPLE
                    # =========================================

                    async def capture_sample():

                        current_count = len(
                            registration_state[
                                "sample_paths"
                            ]
                        )

                        if (
                            current_count
                            >= MEDICINE_REGISTRATION_SAMPLES
                        ):

                            ui.notify(
                                (
                                    "All required medicine "
                                    "samples have already "
                                    "been captured."
                                ),
                                type="warning",
                            )

                            return

                        frame = (
                            camera_service
                            .get_latest_frame_copy()
                        )

                        if frame is None:

                            ui.notify(
                                (
                                    "Camera frame is not "
                                    "available yet."
                                ),
                                type="negative",
                            )

                            return

                        try:

                            crop = (
                                _extract_registration_crop(
                                    frame
                                )
                            )

                            session_dir = (
                                ensure_session_directory()
                            )

                            sample_number = (
                                current_count
                                + 1
                            )

                            sample_path = (
                                session_dir
                                / (
                                    f"sample_"
                                    f"{sample_number:02d}.jpg"
                                )
                            )

                            success = (
                                await asyncio.to_thread(
                                    cv2.imwrite,
                                    str(
                                        sample_path
                                    ),
                                    crop,
                                )
                            )

                            if not success:

                                raise RuntimeError(
                                    (
                                        "Could not save "
                                        "medicine sample."
                                    )
                                )

                            service = (
                                await get_registration_service()
                            )

                            await asyncio.to_thread(
                                service.extract_registration_embedding,
                                sample_path,
                            )

                            registration_state[
                                "sample_paths"
                            ].append(
                                sample_path
                            )

                            update_sample_indicator()

                            validation_label.set_text(
                                (
                                    f"Sample {sample_number} "
                                    "accepted. Change the "
                                    "package angle slightly "
                                    "before the next capture."
                                )
                            )

                            ui.notify(
                                (
                                    f"Medicine sample "
                                    f"{sample_number} captured."
                                ),
                                type="positive",
                            )

                        except Exception as exc:

                            try:

                                if (
                                    "sample_path"
                                    in locals()
                                    and sample_path.exists()
                                ):

                                    sample_path.unlink()

                            except Exception:

                                pass

                            validation_label.set_text(
                                str(
                                    exc
                                )
                            )

                            ui.notify(
                                str(
                                    exc
                                ),
                                type="negative",
                            )

                    # =========================================
                    # CLEAR SAMPLES
                    # =========================================

                    def clear_samples():

                        clear_registration_files()

                        update_sample_indicator()

                        validation_label.set_text(
                            (
                                "Samples cleared. "
                                "Capture the medicine "
                                "package again."
                            )
                        )

                        ui.notify(
                            (
                                "Medicine samples "
                                "cleared."
                            ),
                            type="info",
                        )

                    # =========================================
                    # CANCEL
                    # =========================================

                    def cancel_registration():

                        clear_registration_files()

                        update_sample_indicator()

                        registration_dialog.close()

                    # =========================================
                    # REGISTER MEDICINE
                    # =========================================

                    async def register_medicine():

                        if registration_state[
                            "saving"
                        ]:

                            return

                        medicine_name = (
                            medicine_name_input.value
                            or ""
                        ).strip()

                        if not medicine_name:

                            ui.notify(
                                (
                                    "Please enter the "
                                    "medicine name."
                                ),
                                type="warning",
                            )

                            return

                        dosage = (
                            dosage_input.value
                            or ""
                        ).strip()

                        raw_time = (
                            schedule_time_input.value
                            or ""
                        ).strip()

                        try:

                            parsed_schedule_time = (
                                datetime.strptime(
                                    raw_time,
                                    "%H:%M",
                                )
                                .time()
                            )

                        except ValueError:

                            ui.notify(
                                (
                                    "Please enter a valid "
                                    "schedule time."
                                ),
                                type="warning",
                            )

                            return

                        selected_label = (
                            schedule_label_select.value
                        )

                        if not selected_label:

                            schedule_label = None

                        elif selected_label == "Other":

                            schedule_label = (
                                custom_schedule_label.value
                                or ""
                            ).strip()

                            if not schedule_label:

                                ui.notify(
                                    (
                                        "Please enter the "
                                        "custom schedule label."
                                    ),
                                    type="warning",
                                )

                                return

                        else:

                            schedule_label = str(
                                selected_label
                            )

                        instructions = (
                            instructions_input.value
                            or ""
                        ).strip()

                        sample_paths = list(
                            registration_state[
                                "sample_paths"
                            ]
                        )

                        if (
                            len(
                                sample_paths
                            )
                            != MEDICINE_REGISTRATION_SAMPLES
                        ):

                            ui.notify(
                                (
                                    "Please capture exactly "
                                    f"{MEDICINE_REGISTRATION_SAMPLES} "
                                    "medicine samples first."
                                ),
                                type="warning",
                            )

                            return

                        registration_state[
                            "saving"
                        ] = True

                        validation_label.set_text(
                            (
                                "Generating medicine "
                                "embeddings and saving "
                                "registration..."
                            )
                        )

                        try:

                            service = (
                                await get_registration_service()
                            )

                            result = (
                                await asyncio.to_thread(
                                    service.register_medicine,
                                    medicine_name,
                                    dosage or None,
                                    parsed_schedule_time,
                                    sample_paths,
                                    schedule_label,
                                    instructions or None,
                                )
                            )

                            ui.notify(
                                (
                                    f"{result.medicine_name} "
                                    "registered successfully."
                                ),
                                type="positive",
                            )

                            clear_registration_files()

                            medicine_name_input.value = ""

                            dosage_input.value = ""

                            schedule_time_input.value = (
                                "08:00"
                            )

                            schedule_label_select.value = (
                                "Morning"
                            )

                            custom_schedule_label.value = ""

                            custom_schedule_label.set_visibility(
                                False
                            )

                            instructions_input.value = ""

                            update_sample_indicator()

                            validation_label.set_text(
                                (
                                    "Medicine registered. "
                                    "You can now use it for "
                                    "live camera recognition."
                                )
                            )

                            registered_medicines.refresh()

                            today_medicine_status.refresh()

                            registration_dialog.close()

                        except Exception as exc:

                            validation_label.set_text(
                                str(
                                    exc
                                )
                            )

                            ui.notify(
                                str(
                                    exc
                                ),
                                type="negative",
                            )

                        finally:

                            registration_state[
                                "saving"
                            ] = False

                    # =========================================
                    # BUTTONS
                    # =========================================

                    with ui.row().classes(
                        (
                            "w-full justify-between "
                            "mt-4 gap-3 flex-wrap"
                        )
                    ):

                        with ui.row().classes(
                            "gap-2 flex-wrap"
                        ):

                            ui.button(
                                "Capture Sample",
                                icon="photo_camera",
                                on_click=capture_sample,
                            ).classes(
                                "memorai-action-button"
                            )

                            ui.button(
                                "Clear Samples",
                                icon="delete_sweep",
                                on_click=clear_samples,
                            ).props(
                                "outline"
                            ).classes(
                                "memorai-action-button"
                            )

                        with ui.row().classes(
                            "gap-2 flex-wrap"
                        ):

                            ui.button(
                                "Cancel",
                                on_click=(
                                    cancel_registration
                                ),
                            ).props(
                                "flat"
                            ).classes(
                                "memorai-action-button"
                            )

                            ui.button(
                                "Register Medicine",
                                icon="medication",
                                on_click=register_medicine,
                            ).classes(
                                "memorai-action-button"
                            )

            # =================================================
            # OPEN REGISTRATION DIALOG
            # =================================================

            def open_registration_dialog():

                clear_registration_files()

                update_sample_indicator()

                validation_label.set_text(
                    (
                        "Hold the package clearly "
                        "inside the centre area."
                    )
                )

                registration_dialog.open()

            register_button.on(
                "click",
                open_registration_dialog,
            )