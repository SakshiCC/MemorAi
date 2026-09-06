from __future__ import annotations

import asyncio
import shutil
import uuid

from datetime import (
    date,
    datetime,
)
from pathlib import Path

import cv2

from nicegui import ui

from app.nicegui_ui.layout import (
    sidebar,
)
from app.services.camera_service import (
    CameraService,
)
from config import (
    BASE_DIR,
    EVIDENCE_DIR,
    FACE_REGISTRATION_SAMPLES,
    REGISTERED_FACE_IMAGE_DIR,
)
from database.person_repository import (
    deactivate_person,
    get_people_with_last_seen,
    get_person_events_for_date,
    reactivate_person,
)
from face.person_registration import (
    PersonRegistrationService,
)


# ============================================================
# RELATION OPTIONS
# ============================================================

RELATION_OPTIONS = [
    "Spouse",
    "Son",
    "Daughter",
    "Mother",
    "Father",
    "Brother",
    "Sister",
    "Grandson",
    "Granddaughter",
    "Friend",
    "Caregiver",
    "Doctor",
    "Nurse",
    "Neighbor",
    "Other",
]


# ============================================================
# TEMP REGISTRATION ROOT
# ============================================================

REGISTRATION_TEMP_ROOT = (
    BASE_DIR
    / "data"
    / "temp"
    / "people_registration"
)

REGISTRATION_TEMP_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# IMAGE URL HELPERS
# ============================================================

def _evidence_url(
    image_path: str | None,
) -> str | None:

    if not image_path:

        return None

    try:

        path = Path(
            image_path
        ).resolve()

        relative = path.relative_to(
            EVIDENCE_DIR.resolve()
        )

        return (
            "/evidence/"
            + relative.as_posix()
        )

    except Exception:

        return None


def _profile_image_url(
    image_path: str | None,
) -> str | None:

    if not image_path:

        return None

    try:

        path = Path(
            image_path
        ).resolve()

        relative = path.relative_to(
            REGISTERED_FACE_IMAGE_DIR.resolve()
        )

        return (
            "/registered_faces/"
            + relative.as_posix()
        )

    except Exception:

        return None


# ============================================================
# FORMATTING
# ============================================================

def _format_last_seen(
    value,
) -> str:

    if value is None:

        return "Never"

    if not hasattr(
        value,
        "date",
    ):

        return str(
            value
        )

    today = date.today()

    if value.date() == today:

        return (
            "Today "
            + value.strftime(
                "%I:%M %p"
            )
        )

    yesterday = (
        today.fromordinal(
            today.toordinal()
            - 1
        )
    )

    if value.date() == yesterday:

        return (
            "Yesterday "
            + value.strftime(
                "%I:%M %p"
            )
        )

    return value.strftime(
        "%d %b %Y, %I:%M %p"
    )


def _format_seen_time(
    value,
) -> str:

    if value is None:

        return "Not available"

    try:

        return value.strftime(
            "%I:%M %p"
        )

    except Exception:

        return str(
            value
        )


# ============================================================
# PAGE
# ============================================================

def build_people_page(
    camera_service: CameraService,
) -> None:

    registration_state = {
        "sample_paths": [],
        "session_dir": None,
        "service": None,
        "saving": False,
    }

    # ========================================================
    # REGISTRATION SERVICE
    # ========================================================

    async def get_registration_service():

        if registration_state[
            "service"
        ] is not None:

            return registration_state[
                "service"
            ]

        ui.notify(
            "Preparing face registration...",
            type="info",
        )

        service = await asyncio.to_thread(
            PersonRegistrationService
        )

        registration_state[
            "service"
        ] = service

        return service

    # ========================================================
    # TEMP SESSION
    # ========================================================

    def ensure_session_directory() -> Path:

        existing = registration_state[
            "session_dir"
        ]

        if existing is not None:

            return existing

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
    # CLEAN TEMP
    # ========================================================

    def clear_registration_files() -> None:

        registration_state[
            "sample_paths"
        ].clear()

        session_dir = registration_state[
            "session_dir"
        ]

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

        sidebar(
            active_path="/people"
        )

        with ui.column().classes(
            "memorai-main gap-6"
        ):

            # =================================================
            # HEADER
            # =================================================

            with ui.row().classes(
                "w-full items-center justify-between"
            ):

                with ui.column().classes(
                    "gap-1"
                ):

                    ui.label(
                        "👤 People"
                    ).classes(
                        "text-4xl font-bold"
                    )

                    ui.label(
                        (
                            "Manage registered people and "
                            "review person encounters."
                        )
                    ).classes(
                        "text-base opacity-70"
                    )

                register_button = ui.button(
                    "Register Person",
                    icon="person_add",
                )

            # =================================================
            # REGISTERED PEOPLE
            # =================================================

            with ui.row().classes(
                "w-full items-center justify-between"
            ):

                ui.label(
                    "Registered People"
                ).classes(
                    "text-2xl font-semibold"
                )

                ui.button(
                    "Refresh",
                    icon="refresh",
                    on_click=lambda:
                        registered_people.refresh(),
                ).props(
                    "flat"
                )

            @ui.refreshable
            def registered_people():

                people = (
                    get_people_with_last_seen(
                        active_only=False
                    )
                )

                if not people:

                    with ui.card().classes(
                        "memorai-card w-full"
                    ):

                        ui.label(
                            "No people registered yet."
                        ).classes(
                            "opacity-70"
                        )

                    return

                with ui.card().classes(
                    "memorai-card w-full p-0 overflow-hidden"
                ):

                    # -----------------------------------------
                    # TABLE HEADER
                    # -----------------------------------------

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
                            "w-20"
                        )

                        ui.label(
                            "Name"
                        ).classes(
                            "w-44"
                        )

                        ui.label(
                            "Relation"
                        ).classes(
                            "w-32"
                        )

                        ui.label(
                            "Status"
                        ).classes(
                            "w-24"
                        )

                        ui.label(
                            "Last Seen"
                        ).classes(
                            "flex-1"
                        )

                        ui.label(
                            "Action"
                        ).classes(
                            "w-32"
                        )

                    # -----------------------------------------
                    # TABLE ROWS
                    # -----------------------------------------

                    for person in people:

                        person_id = int(
                            person[
                                "id"
                            ]
                        )

                        active = bool(
                            person[
                                "is_active"
                            ]
                        )

                        profile_url = (
                            _profile_image_url(
                                person.get(
                                    "profile_image_path"
                                )
                            )
                        )

                        with ui.row().classes(
                            (
                                "w-full items-center gap-4 "
                                "px-4 py-3 border-b"
                            )
                        ):

                            with ui.element(
                                "div"
                            ).classes(
                                "w-20"
                            ):

                                if profile_url:

                                    ui.image(
                                        profile_url
                                    ).classes(
                                        (
                                            "w-14 h-14 rounded-full "
                                            "object-cover"
                                        )
                                    )

                                else:

                                    with ui.element(
                                        "div"
                                    ).classes(
                                        (
                                            "w-14 h-14 rounded-full "
                                            "flex items-center "
                                            "justify-center "
                                            "bg-gray-500/20"
                                        )
                                    ):

                                        ui.icon(
                                            "person"
                                        ).classes(
                                            "text-2xl opacity-60"
                                        )

                            ui.label(
                                (
                                    person[
                                        "display_name"
                                    ]
                                    or person[
                                        "name"
                                    ]
                                )
                            ).classes(
                                "w-44 font-semibold"
                            )

                            ui.label(
                                (
                                    person.get(
                                        "relation"
                                    )
                                    or "—"
                                )
                            ).classes(
                                "w-32 opacity-80"
                            )

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

                            ui.label(
                                _format_last_seen(
                                    person.get(
                                        "last_seen"
                                    )
                                )
                            ).classes(
                                "flex-1 opacity-80"
                            )

                            async def toggle_person(
                                pid=person_id,
                                currently_active=active,
                            ):

                                try:

                                    if currently_active:

                                        await asyncio.to_thread(
                                            deactivate_person,
                                            pid,
                                        )

                                        ui.notify(
                                            "Person deactivated.",
                                            type="positive",
                                        )

                                    else:

                                        await asyncio.to_thread(
                                            reactivate_person,
                                            pid,
                                        )

                                        ui.notify(
                                            "Person activated.",
                                            type="positive",
                                        )

                                    registered_people.refresh()

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
                                    "person_off"
                                    if active
                                    else "person"
                                ),
                                on_click=toggle_person,
                            ).props(
                                "outline dense"
                            ).classes(
                                "w-32"
                            )

            registered_people()

            # =================================================
            # ENCOUNTER HISTORY
            # =================================================

            ui.label(
                "Encounter History"
            ).classes(
                "text-2xl font-semibold mt-2"
            )

            ui.label(
                (
                    "Choose a date and press Load Encounters. "
                    "Changing the date alone will not query "
                    "the database."
                )
            ).classes(
                "opacity-70"
            )

            with ui.row().classes(
                "items-end gap-3"
            ):

                date_input = ui.input(
                    label="Select Date",
                    value=(
                        date.today()
                        .isoformat()
                    ),
                ).props(
                    "type=date"
                )

                load_button = ui.button(
                    "Load Encounters",
                    icon="search",
                )

            encounter_state = {
                "loaded": False,
                "date": None,
            }

            @ui.refreshable
            def encounter_history():

                if not encounter_state[
                    "loaded"
                ]:

                    with ui.card().classes(
                        "memorai-card w-full"
                    ):

                        ui.label(
                            (
                                "Select a date and press "
                                "Load Encounters to view history."
                            )
                        ).classes(
                            "opacity-70"
                        )

                    return

                selected_date = encounter_state[
                    "date"
                ]

                events = (
                    get_person_events_for_date(
                        selected_date
                    )
                )

                if not events:

                    with ui.card().classes(
                        "memorai-card w-full"
                    ):

                        ui.label(
                            (
                                "No person encounters were "
                                "recorded on "
                                f"{selected_date:%d %b %Y}."
                            )
                        ).classes(
                            "opacity-70"
                        )

                    return

                with ui.card().classes(
                    "memorai-card w-full p-0 overflow-hidden"
                ):

                    with ui.row().classes(
                        (
                            "w-full items-center gap-4 "
                            "px-4 py-3 font-semibold "
                            "border-b opacity-70"
                        )
                    ):

                        ui.label(
                            "Image"
                        ).classes(
                            "w-24"
                        )

                        ui.label(
                            "Person"
                        ).classes(
                            "w-44"
                        )

                        ui.label(
                            "Relation"
                        ).classes(
                            "w-32"
                        )

                        ui.label(
                            "Similarity"
                        ).classes(
                            "w-28"
                        )

                        ui.label(
                            "Seen At"
                        ).classes(
                            "flex-1"
                        )

                    for event in events:

                        evidence_url = (
                            _evidence_url(
                                event.get(
                                    "image_path"
                                )
                            )
                        )

                        similarity = float(
                            event.get(
                                "similarity"
                            )
                            or 0
                        )

                        with ui.row().classes(
                            (
                                "w-full items-center gap-4 "
                                "px-4 py-3 border-b"
                            )
                        ):

                            with ui.element(
                                "div"
                            ).classes(
                                "w-24"
                            ):

                                if evidence_url:

                                    ui.image(
                                        evidence_url
                                    ).classes(
                                        (
                                            "w-20 h-16 rounded-lg "
                                            "object-cover"
                                        )
                                    )

                                else:

                                    with ui.element(
                                        "div"
                                    ).classes(
                                        (
                                            "w-20 h-16 rounded-lg "
                                            "flex items-center "
                                            "justify-center "
                                            "bg-gray-500/20"
                                        )
                                    ):

                                        ui.icon(
                                            "person"
                                        )

                            ui.label(
                                event[
                                    "person_name"
                                ]
                            ).classes(
                                "w-44 font-semibold"
                            )

                            ui.label(
                                (
                                    event.get(
                                        "relation"
                                    )
                                    or "—"
                                )
                            ).classes(
                                "w-32 opacity-80"
                            )

                            ui.label(
                                (
                                    f"{similarity * 100:.1f}%"
                                )
                            ).classes(
                                "w-28"
                            )

                            ui.label(
                                _format_seen_time(
                                    event.get(
                                        "detected_at"
                                    )
                                )
                            ).classes(
                                "flex-1 opacity-80"
                            )

            encounter_history()

            async def load_encounters():

                raw_date = (
                    date_input.value
                    or ""
                ).strip()

                try:

                    selected_date = (
                        datetime.strptime(
                            raw_date,
                            "%Y-%m-%d",
                        )
                        .date()
                    )

                except ValueError:

                    ui.notify(
                        "Please select a valid date.",
                        type="warning",
                    )

                    return

                encounter_state[
                    "date"
                ] = selected_date

                encounter_state[
                    "loaded"
                ] = True

                encounter_history.refresh()

            load_button.on(
                "click",
                load_encounters,
            )

            # =================================================
            # REGISTER DIALOG
            # =================================================

            with ui.dialog() as registration_dialog:

                with ui.card().classes(
                    (
                        "w-[900px] max-w-[95vw] "
                        "rounded-2xl p-6"
                    )
                ):

                    ui.label(
                        "Register a Person"
                    ).classes(
                        "text-2xl font-bold"
                    )

                    ui.label(
                        (
                            "Capture five clear face samples. "
                            "The first accepted sample will "
                            "also become the profile photo."
                        )
                    ).classes(
                        "opacity-70"
                    )

                    with ui.row().classes(
                        "w-full items-start gap-6"
                    ):

                        with ui.column().classes(
                            "w-80 gap-3"
                        ):

                            name_input = ui.input(
                                label="Name",
                                placeholder="e.g. Rahul",
                            ).classes(
                                "w-full"
                            )

                            relation_select = ui.select(
                                RELATION_OPTIONS,
                                label="Relation",
                            ).classes(
                                "w-full"
                            )

                            custom_relation_input = ui.input(
                                label="Custom Relation",
                                placeholder="Enter relation",
                            ).classes(
                                "w-full"
                            )

                            custom_relation_input.set_visibility(
                                False
                            )

                            def relation_changed():

                                custom_relation_input.set_visibility(
                                    relation_select.value
                                    == "Other"
                                )

                            relation_select.on_value_change(
                                lambda _:
                                    relation_changed()
                            )

                            notes_input = ui.textarea(
                                label="Notes",
                                placeholder=(
                                    "Optional information "
                                    "about this person"
                                ),
                            ).classes(
                                "w-full"
                            )

                        with ui.column().classes(
                            "flex-1 gap-3"
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
                                    max-width:520px;
                                    aspect-ratio:4/3;
                                    overflow:hidden;
                                    border-radius:16px;
                                    background:#111;
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

                            sample_status = ui.label(
                                (
                                    "0 / "
                                    f"{FACE_REGISTRATION_SAMPLES} "
                                    "samples captured"
                                )
                            ).classes(
                                "font-medium"
                            )

                            sample_dots = ui.label(
                                "○ ○ ○ ○ ○"
                            ).classes(
                                "text-2xl tracking-wider"
                            )

                            validation_label = ui.label(
                                (
                                    "Face the camera and capture "
                                    "five slightly different angles."
                                )
                            ).classes(
                                "text-sm opacity-70"
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
                            FACE_REGISTRATION_SAMPLES
                        )

                        sample_status.set_text(
                            (
                                f"{count} / {total} "
                                "samples captured"
                            )
                        )

                        dots = []

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
                            >= FACE_REGISTRATION_SAMPLES
                        ):

                            ui.notify(
                                (
                                    "All five samples are "
                                    "already captured."
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

                            service = (
                                await get_registration_service()
                            )

                            faces = await asyncio.to_thread(
                                service.face_engine.detect_faces,
                                frame,
                            )

                            if not faces:

                                validation_label.set_text(
                                    "No clear face detected."
                                )

                                ui.notify(
                                    (
                                        "No clear face detected. "
                                        "Please face the camera."
                                    ),
                                    type="warning",
                                )

                                return

                            if len(
                                faces
                            ) > 1:

                                validation_label.set_text(
                                    (
                                        "More than one face detected. "
                                        "Keep only one person visible."
                                    )
                                )

                                ui.notify(
                                    (
                                        "More than one face detected. "
                                        "Only the person being registered "
                                        "should be visible."
                                    ),
                                    type="warning",
                                )

                                return

                            await asyncio.to_thread(
                                service.face_engine.generate_embedding,
                                frame,
                                faces[0],
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
                                    f"sample"
                                    f"{sample_number}.jpg"
                                )
                            )

                            success = await asyncio.to_thread(
                                cv2.imwrite,
                                str(
                                    sample_path
                                ),
                                frame,
                            )

                            if not success:

                                raise RuntimeError(
                                    (
                                        "Could not save "
                                        "captured sample."
                                    )
                                )

                            registration_state[
                                "sample_paths"
                            ].append(
                                sample_path
                            )

                            update_sample_indicator()

                            validation_label.set_text(
                                (
                                    f"Sample {sample_number} accepted. "
                                    "Change your angle slightly "
                                    "before the next sample."
                                )
                            )

                            ui.notify(
                                (
                                    f"Sample {sample_number} captured."
                                ),
                                type="positive",
                            )

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

                    # =========================================
                    # CLEAR SAMPLES
                    # =========================================

                    def clear_samples():

                        clear_registration_files()

                        update_sample_indicator()

                        validation_label.set_text(
                            (
                                "Samples cleared. "
                                "You can capture again."
                            )
                        )

                        ui.notify(
                            "Captured samples cleared.",
                            type="info",
                        )

                    # =========================================
                    # REGISTER
                    # =========================================

                    async def register_person():

                        if registration_state[
                            "saving"
                        ]:

                            return

                        name = (
                            name_input.value
                            or ""
                        ).strip()

                        if not name:

                            ui.notify(
                                (
                                    "Please enter the "
                                    "person's name."
                                ),
                                type="warning",
                            )

                            return

                        selected_relation = (
                            relation_select.value
                        )

                        if not selected_relation:

                            ui.notify(
                                "Please select a relation.",
                                type="warning",
                            )

                            return

                        if selected_relation == "Other":

                            relation = (
                                custom_relation_input.value
                                or ""
                            ).strip()

                            if not relation:

                                ui.notify(
                                    (
                                        "Please enter the "
                                        "custom relation."
                                    ),
                                    type="warning",
                                )

                                return

                        else:

                            relation = str(
                                selected_relation
                            )

                        sample_paths = list(
                            registration_state[
                                "sample_paths"
                            ]
                        )

                        if len(
                            sample_paths
                        ) != FACE_REGISTRATION_SAMPLES:

                            ui.notify(
                                (
                                    "Please capture exactly "
                                    f"{FACE_REGISTRATION_SAMPLES} "
                                    "face samples first."
                                ),
                                type="warning",
                            )

                            return

                        notes = (
                            notes_input.value
                            or ""
                        ).strip()

                        registration_state[
                            "saving"
                        ] = True

                        try:

                            service = (
                                await get_registration_service()
                            )

                            result = (
                                await asyncio.to_thread(
                                    service.register_person,
                                    name,
                                    sample_paths,
                                    name,
                                    relation,
                                    notes or None,
                                )
                            )

                            ui.notify(
                                (
                                    f"{result.display_name} "
                                    "registered successfully."
                                ),
                                type="positive",
                            )

                            clear_registration_files()

                            name_input.value = ""

                            relation_select.value = None

                            custom_relation_input.value = ""

                            custom_relation_input.set_visibility(
                                False
                            )

                            notes_input.value = ""

                            update_sample_indicator()

                            registered_people.refresh()

                            registration_dialog.close()

                        except Exception as exc:

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
                        "w-full justify-between mt-4"
                    ):

                        with ui.row().classes(
                            "gap-2"
                        ):

                            ui.button(
                                "Capture Sample",
                                icon="photo_camera",
                                on_click=capture_sample,
                            )

                            ui.button(
                                "Clear Samples",
                                icon="delete_sweep",
                                on_click=clear_samples,
                            ).props(
                                "outline"
                            )

                        with ui.row().classes(
                            "gap-2"
                        ):

                            ui.button(
                                "Cancel",
                                on_click=(
                                    registration_dialog.close
                                ),
                            ).props(
                                "flat"
                            )

                            ui.button(
                                "Register Person",
                                icon="person_add",
                                on_click=register_person,
                            )

            register_button.on(
                "click",
                lambda:
                    registration_dialog.open(),
            )