from __future__ import annotations

import asyncio
import logging
import os
import secrets
import uuid

from datetime import datetime
from pathlib import Path

from fastapi import Request

from fastapi.responses import (
    JSONResponse,
    StreamingResponse,
)

from nicegui import (
    app,
    ui,
)

from app.services.camera_service import (
    CameraService,
)

from app.nicegui_ui.ask_page import (
    build_ask_page,
)

from app.nicegui_ui.camera_page import (
    build_camera_page,
)

from app.nicegui_ui.conversation_page import (
    build_conversation_page,
)

from app.nicegui_ui.home_page import (
    build_home_page,
)

from app.nicegui_ui.layout import (
    prepare_page,
)

from app.nicegui_ui.medicine_page import (
    build_medicine_page,
)

from app.nicegui_ui.people_page import (
    build_people_page,
)

from app.nicegui_ui.settings_page import (
    build_settings_page,
)

from app.nicegui_ui.timeline_page import (
    build_timeline_page,
)

from app.nicegui_ui.voice_page import (
    build_voice_page,
)

from config import (
    REGISTERED_FACE_IMAGE_DIR,
    REGISTERED_MEDICINE_IMAGE_DIR,
)

from conversation.conversation_service import (
    ConversationService,
)

from database.db_manager import (
    initialize_database,
)

from llm.chat import (
    MemorAiChat,
)

from speech.voice_chat import (
    MemorAiVoiceChat,
)


# ============================================================
# LOGGER
# ============================================================

logger = logging.getLogger(
    __name__
)


# ============================================================
# PROJECT DIRECTORIES
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

VOICE_RECORDING_DIR = (
    PROJECT_ROOT
    / "recordings"
    / "browser"
)

CONVERSATION_RECORDING_DIR = (
    PROJECT_ROOT
    / "recordings"
    / "conversations"
)

EVIDENCE_DIR = (
    PROJECT_ROOT
    / "evidence"
)


VOICE_RECORDING_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CONVERSATION_RECORDING_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

EVIDENCE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

REGISTERED_FACE_IMAGE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

REGISTERED_MEDICINE_IMAGE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# NICEGUI STORAGE SECRET
# ============================================================

MEMORAI_STORAGE_SECRET = (
    os.getenv(
        "MEMORAI_STORAGE_SECRET",
        "",
    )
    .strip()
)

if not MEMORAI_STORAGE_SECRET:

    MEMORAI_STORAGE_SECRET = (
        secrets.token_urlsafe(
            32
        )
    )

    logger.warning(
        "MEMORAI_STORAGE_SECRET is not configured. "
        "A temporary secure storage secret was generated "
        "for this application session. Set "
        "MEMORAI_STORAGE_SECRET in the local .env file "
        "to keep NiceGUI storage persistent across restarts."
    )


# ============================================================
# STATIC FILES
# ============================================================

app.add_static_files(
    "/evidence",
    EVIDENCE_DIR,
)

app.add_static_files(
    "/registered_faces",
    REGISTERED_FACE_IMAGE_DIR,
)

app.add_static_files(
    "/registered_medicines",
    REGISTERED_MEDICINE_IMAGE_DIR,
)


# ============================================================
# SHARED SERVICES
# ============================================================

camera_service = (
    CameraService(
        capture_fps=30,
        inference_fps=4,
        stream_fps=15,
        face_inference_fps=3,
        medicine_scan_fps=1,
        medicine_action_fps=6,
    )
)

chat_service = (
    MemorAiChat()
)


# ============================================================
# LAZY VOICE SERVICE
# ============================================================

voice_service: (
    MemorAiVoiceChat
    | None
) = None

voice_service_lock = (
    asyncio.Lock()
)


async def get_voice_service(
) -> MemorAiVoiceChat:

    global voice_service

    if voice_service is not None:

        return voice_service

    async with voice_service_lock:

        if voice_service is not None:

            return voice_service

        print(
            "\nInitializing MemorAi "
            "web voice service..."
        )

        voice_service = (
            await asyncio.to_thread(
                MemorAiVoiceChat,
                chat_service,
            )
        )

        print(
            "MemorAi web voice "
            "service is ready."
        )

        return voice_service


# ============================================================
# LAZY CONVERSATION SERVICE
# ============================================================

conversation_service: (
    ConversationService
    | None
) = None

conversation_service_lock = (
    asyncio.Lock()
)


async def get_conversation_service(
) -> ConversationService:

    global conversation_service

    if conversation_service is not None:

        return conversation_service

    async with conversation_service_lock:

        if conversation_service is not None:

            return conversation_service

        print(
            "\nInitializing MemorAi "
            "Conversation Memory service..."
        )

        conversation_service = (
            await asyncio.to_thread(
                ConversationService
            )
        )

        print(
            "MemorAi Conversation Memory "
            "service is ready."
        )

        return conversation_service


# ============================================================
# VIDEO STREAM
# ============================================================

@app.get(
    "/video_feed"
)
def video_feed():

    return StreamingResponse(
        camera_service.mjpeg_stream(),
        media_type=(
            "multipart/x-mixed-replace;"
            " boundary=frame"
        ),
    )


# ============================================================
# VOICE AUDIO API
# ============================================================

@app.post(
    "/api/voice/question"
)
async def voice_question_api(
    request: Request,
):

    audio_path: Path | None = None

    try:

        audio_bytes = (
            await request.body()
        )

        if not audio_bytes:

            return JSONResponse(
                status_code=400,
                content={
                    "ok": False,
                    "error": (
                        "No audio data "
                        "was received."
                    ),
                },
            )

        if len(
            audio_bytes
        ) < 500:

            return JSONResponse(
                status_code=400,
                content={
                    "ok": False,
                    "error": (
                        "The recording was "
                        "too short or empty."
                    ),
                },
            )

        content_type = (
            request.headers.get(
                "content-type",
                "",
            )
            .lower()
        )

        if "ogg" in content_type:

            extension = ".ogg"

        elif "wav" in content_type:

            extension = ".wav"

        else:

            extension = ".webm"

        filename = (
            f"voice_{uuid.uuid4().hex}"
            f"{extension}"
        )

        audio_path = (
            VOICE_RECORDING_DIR
            / filename
        )

        await asyncio.to_thread(
            audio_path.write_bytes,
            audio_bytes,
        )

        logger.info(
            "Browser voice recording "
            "saved: %s",
            audio_path,
        )

        service = (
            await get_voice_service()
        )

        question, answer = (
            await asyncio.to_thread(
                service.ask_from_audio,
                audio_path,
            )
        )

        return JSONResponse(
            content={
                "ok": True,
                "question": question,
                "answer": answer,
            }
        )

    except ValueError as exc:

        logger.warning(
            "Invalid browser voice "
            "question: %s",
            exc,
        )

        return JSONResponse(
            status_code=400,
            content={
                "ok": False,
                "error": str(
                    exc
                ),
            },
        )

    except Exception as exc:

        logger.exception(
            "Browser voice question "
            "processing failed."
        )

        return JSONResponse(
            status_code=500,
            content={
                "ok": False,
                "error": (
                    "MemorAi could not process "
                    "the voice recording. "
                    f"{exc}"
                ),
            },
        )

    finally:

        if (
            audio_path is not None
            and audio_path.exists()
        ):

            try:

                await asyncio.to_thread(
                    audio_path.unlink
                )

            except Exception:

                logger.warning(
                    "Could not delete temporary "
                    "voice recording: %s",
                    audio_path,
                )


# ============================================================
# CONVERSATION AUDIO API
# ============================================================

@app.post(
    "/api/conversation/process"
)
async def conversation_process_api(
    request: Request,
):

    audio_path: Path | None = None

    try:

        audio_bytes = (
            await request.body()
        )

        if not audio_bytes:

            return JSONResponse(
                status_code=400,
                content={
                    "ok": False,
                    "error": (
                        "No conversation audio "
                        "was received."
                    ),
                },
            )

        if len(
            audio_bytes
        ) < 500:

            return JSONResponse(
                status_code=400,
                content={
                    "ok": False,
                    "error": (
                        "The conversation recording "
                        "was too short or empty."
                    ),
                },
            )

        # ----------------------------------------------------
        # AUDIO FORMAT
        # ----------------------------------------------------

        content_type = (
            request.headers.get(
                "content-type",
                "",
            )
            .lower()
        )

        if "ogg" in content_type:

            extension = ".ogg"

        elif "wav" in content_type:

            extension = ".wav"

        else:

            extension = ".webm"

        filename = (
            "conversation_"
            f"{uuid.uuid4().hex}"
            f"{extension}"
        )

        audio_path = (
            CONVERSATION_RECORDING_DIR
            / filename
        )

        await asyncio.to_thread(
            audio_path.write_bytes,
            audio_bytes,
        )

        logger.info(
            "Temporary conversation recording "
            "saved: %s",
            audio_path,
        )

        # ----------------------------------------------------
        # OPTIONAL DETAILS
        # ----------------------------------------------------

        participant_text = (
            request.headers.get(
                "x-conversation-participant",
                "",
            )
            .strip()
            or None
        )

        title = (
            request.headers.get(
                "x-conversation-title",
                "",
            )
            .strip()
            or None
        )

        # ----------------------------------------------------
        # START TIME
        # ----------------------------------------------------

        started_at_header = (
            request.headers.get(
                "x-conversation-started-at",
                "",
            )
            .strip()
        )

        started_at: (
            datetime
            | None
        ) = None

        if started_at_header:

            try:

                normalized = (
                    started_at_header
                    .replace(
                        "Z",
                        "+00:00",
                    )
                )

                parsed = (
                    datetime.fromisoformat(
                        normalized
                    )
                )

                if (
                    parsed.tzinfo
                    is not None
                ):

                    parsed = (
                        parsed
                        .astimezone()
                        .replace(
                            tzinfo=None
                        )
                    )

                started_at = (
                    parsed
                )

            except ValueError:

                started_at = None

        # ----------------------------------------------------
        # PROCESS
        # ----------------------------------------------------

        service = (
            await get_conversation_service()
        )

        result = (
            await asyncio.to_thread(
                service.process_complete_audio,
                audio_path,
                title,
                participant_text,
                started_at,
                datetime.now(),
            )
        )

        # ----------------------------------------------------
        # SUCCESS RESPONSE
        #
        # The UI needs not only the transcript and importance
        # result, but also the metadata generated for a stored
        # important conversation.
        #
        # ConversationProcessResult already contains:
        #
        #     title
        #     participant_text
        #     summary
        #     vector_indexed
        #
        # Returning these fields keeps the browser result card
        # synchronized with the record that was actually saved.
        # ----------------------------------------------------

        return JSONResponse(
            content={
                "ok": True,

                "stored": (
                    result.stored
                ),

                "vector_indexed": (
                    result.vector_indexed
                ),

                "conversation_id": (
                    result.conversation_id
                ),

                "transcript": (
                    result.transcript
                ),

                "importance_score": (
                    result.importance_score
                ),

                "importance_reasons": list(
                    result.importance_reasons
                ),

                "matched_categories": list(
                    result.matched_categories
                ),

                "title": (
                    result.title
                ),

                "participant_text": (
                    result.participant_text
                ),

                "summary": (
                    result.summary
                ),
            }
        )

    except PermissionError as exc:

        logger.warning(
            "Conversation processing blocked: %s",
            exc,
        )

        return JSONResponse(
            status_code=403,
            content={
                "ok": False,
                "error": str(
                    exc
                ),
            },
        )

    except ValueError as exc:

        logger.warning(
            "Invalid conversation recording: %s",
            exc,
        )

        return JSONResponse(
            status_code=400,
            content={
                "ok": False,
                "error": str(
                    exc
                ),
            },
        )

    except Exception as exc:

        logger.exception(
            "Conversation processing failed."
        )

        return JSONResponse(
            status_code=500,
            content={
                "ok": False,
                "error": (
                    "MemorAi could not process "
                    "the conversation. "
                    f"{exc}"
                ),
            },
        )

    finally:

        # ----------------------------------------------------
        # PRIVACY:
        # RAW AUDIO IS ALWAYS TEMPORARY
        # ----------------------------------------------------

        if (
            audio_path is not None
            and audio_path.exists()
        ):

            try:

                await asyncio.to_thread(
                    audio_path.unlink
                )

                logger.info(
                    "Temporary conversation "
                    "audio deleted: %s",
                    audio_path,
                )

            except Exception:

                logger.warning(
                    "Could not delete temporary "
                    "conversation audio: %s",
                    audio_path,
                )


# ============================================================
# HOME
# ============================================================

@ui.page("/")
def home():

    prepare_page()

    build_home_page(
        camera_service=(
            camera_service
        ),
        chat=(
            chat_service
        ),
    )


# ============================================================
# ASK MEMORAI
# ============================================================

@ui.page("/ask")
def ask_page():

    prepare_page()

    build_ask_page(
        chat=chat_service
    )


# ============================================================
# VOICE ASSISTANT
# ============================================================

@ui.page("/voice")
def voice_page():

    prepare_page()

    build_voice_page()


# ============================================================
# CONVERSATIONS
# ============================================================

@ui.page(
    "/conversations"
)
def conversations_page():

    prepare_page()

    build_conversation_page()


# ============================================================
# MEMORY TIMELINE
# ============================================================

@ui.page("/timeline")
def timeline_page():

    prepare_page()

    build_timeline_page(
        evidence_dir=(
            EVIDENCE_DIR
        )
    )


# ============================================================
# CAMERA / MONITORING
# ============================================================

@ui.page("/camera")
def camera_page():

    prepare_page()

    build_camera_page(
        camera_service=(
            camera_service
        )
    )


# ============================================================
# PEOPLE
# ============================================================

@ui.page("/people")
def people_page():

    prepare_page()

    build_people_page(
        camera_service=(
            camera_service
        )
    )


# ============================================================
# MEDICINES
# ============================================================

@ui.page("/medicines")
def medicines_page():

    prepare_page()

    build_medicine_page(
        camera_service=(
            camera_service
        )
    )


# ============================================================
# SETTINGS
# ============================================================

@ui.page("/settings")
def settings_page():

    build_settings_page()


# ============================================================
# STARTUP
# ============================================================

@app.on_startup
def startup():

    initialize_database()

    camera_service.start()


# ============================================================
# SHUTDOWN
# ============================================================

@app.on_shutdown
def shutdown():

    camera_service.stop()


# ============================================================
# RUN
# ============================================================

ui.run(
    title="MemorAi",

    favicon="🧠",

    host="127.0.0.1",

    port=8080,

    reload=False,

    storage_secret=(
        MEMORAI_STORAGE_SECRET
    ),
)