from __future__ import annotations

from nicegui import ui

from app.nicegui_ui.browser_tts import (
    speak_text,
    stop_speaking,
)

from app.nicegui_ui.layout import (
    sidebar,
)


# ============================================================
# VOICE ASSISTANT PAGE
# ============================================================


def build_voice_page() -> None:
    """
    Build the browser-based MemorAi Voice Assistant.

    Pipeline:

        Browser microphone
              ↓
        MediaRecorder
              ↓
        WebM audio
              ↓
        /api/voice/question
              ↓
        Faster-Whisper
              ↓
        MemorAiChat / RAG
              ↓
        Text answer

    Important:

        MemorAi NEVER speaks automatically.

        The user must explicitly press:
            Speak Answer
    """

    # ========================================================
    # PAGE STATE
    # ========================================================

    state = {
        "latest_question": "",
        "latest_answer": "",
        "recording": False,
    }

    # ========================================================
    # PAGE
    # ========================================================

    with ui.row().classes(
        "memorai-page items-start no-wrap"
    ):

        # ====================================================
        # SIDEBAR
        # ====================================================

        sidebar(
            active_path="/voice"
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

            ui.label(
                "🎤 Voice Assistant"
            ).classes(
                "text-4xl font-bold"
            )

            ui.label(
                (
                    "Speak naturally to MemorAi. "
                    "Your voice will be converted to text "
                    "and searched against your memories."
                )
            ).classes(
                "text-lg opacity-70"
            )

            # =================================================
            # ASK BY VOICE CARD
            # =================================================

            with ui.card().classes(
                "memorai-card w-full"
            ):

                ui.label(
                    "🎙 Ask by Voice"
                ).classes(
                    "memorai-section-title"
                )

                ui.label(
                    (
                        "Press Start Recording, ask your "
                        "question, and then press Stop & Ask."
                    )
                ).classes(
                    "opacity-70"
                )

                # =============================================
                # RECORDING STATUS
                # =============================================

                recording_status = (
                    ui.label(
                        "⚪ Ready to record"
                    )
                    .classes(
                        "text-xl font-semibold"
                    )
                )

                ui.label(
                    (
                        "The browser may ask for microphone "
                        "permission the first time you use "
                        "this page."
                    )
                ).classes(
                    "text-sm opacity-60"
                )

                # =============================================
                # RECOGNIZED QUESTION
                # =============================================

                ui.label(
                    "Your Question"
                ).classes(
                    "text-lg font-semibold mt-3"
                )

                question_label = (
                    ui.label(
                        (
                            "Your spoken question "
                            "will appear here."
                        )
                    )
                    .classes(
                        "text-lg"
                    )
                )

                # =============================================
                # START RECORDING
                # =============================================

                async def start_recording() -> None:
                    """
                    Request browser microphone access
                    and start MediaRecorder.
                    """

                    # Stop TTS if it is currently speaking.
                    stop_speaking()

                    # Clear previous state.
                    state[
                        "latest_question"
                    ] = ""

                    state[
                        "latest_answer"
                    ] = ""

                    question_label.set_text(
                        "Listening..."
                    )

                    answer_label.set_text(
                        (
                            "Ask a voice question "
                            "to receive an answer."
                        )
                    )

                    try:

                        result = (
                            await ui.run_javascript(
                                """
                                return await (async () => {

                                    // -------------------------
                                    // Browser support
                                    // -------------------------

                                    if (
                                        !navigator.mediaDevices
                                        ||
                                        !navigator.mediaDevices
                                            .getUserMedia
                                    ) {

                                        return {
                                            ok: false,
                                            error:
                                                'Microphone access is not supported by this browser.'
                                        };
                                    }


                                    // -------------------------
                                    // Prevent second recording
                                    // -------------------------

                                    if (
                                        window.memoraiMediaRecorder
                                        &&
                                        window.memoraiMediaRecorder
                                            .state
                                        === 'recording'
                                    ) {

                                        return {
                                            ok: false,
                                            error:
                                                'Recording is already running.'
                                        };
                                    }


                                    try {

                                        // ---------------------
                                        // Request microphone
                                        // ---------------------

                                        const stream =
                                            await navigator
                                                .mediaDevices
                                                .getUserMedia({
                                                    audio: true
                                                });


                                        window
                                            .memoraiAudioStream =
                                            stream;


                                        window
                                            .memoraiAudioChunks =
                                            [];


                                        // ---------------------
                                        // Select audio format
                                        // ---------------------

                                        let options = {};


                                        if (
                                            MediaRecorder
                                                .isTypeSupported(
                                                    'audio/webm;codecs=opus'
                                                )
                                        ) {

                                            options = {
                                                mimeType:
                                                    'audio/webm;codecs=opus'
                                            };

                                        }

                                        else if (
                                            MediaRecorder
                                                .isTypeSupported(
                                                    'audio/webm'
                                                )
                                        ) {

                                            options = {
                                                mimeType:
                                                    'audio/webm'
                                            };
                                        }


                                        // ---------------------
                                        // Create recorder
                                        // ---------------------

                                        const recorder =
                                            new MediaRecorder(
                                                stream,
                                                options
                                            );


                                        window
                                            .memoraiMediaRecorder =
                                            recorder;


                                        // ---------------------
                                        // Receive chunks
                                        // ---------------------

                                        recorder
                                            .ondataavailable =
                                            event => {

                                                if (
                                                    event.data
                                                    &&
                                                    event.data.size > 0
                                                ) {

                                                    window
                                                        .memoraiAudioChunks
                                                        .push(
                                                            event.data
                                                        );
                                                }
                                            };


                                        // ---------------------
                                        // Start
                                        // ---------------------

                                        recorder.start();


                                        return {
                                            ok: true
                                        };

                                    }

                                    catch (error) {

                                        return {
                                            ok: false,
                                            error:
                                                error.message
                                                ||
                                                String(error)
                                        };
                                    }

                                })();
                                """,
                                timeout=10.0,
                            )
                        )

                        # -------------------------------------
                        # START FAILED
                        # -------------------------------------

                        if (
                            not result
                            or not result.get(
                                "ok",
                                False,
                            )
                        ):

                            if result:

                                message = (
                                    result.get(
                                        "error",
                                        (
                                            "Could not access "
                                            "the microphone."
                                        ),
                                    )
                                )

                            else:

                                message = (
                                    "Could not access "
                                    "the microphone."
                                )

                            question_label.set_text(
                                (
                                    "Your spoken question "
                                    "will appear here."
                                )
                            )

                            ui.notify(
                                message,
                                type="negative",
                            )

                            return

                        # -------------------------------------
                        # RECORDING ACTIVE
                        # -------------------------------------

                        state[
                            "recording"
                        ] = True

                        recording_status.set_text(
                            "🔴 Recording... speak now"
                        )

                        question_label.set_text(
                            "Listening to your question..."
                        )

                        ui.notify(
                            (
                                "Recording started. "
                                "Speak your question."
                            ),
                            type="positive",
                        )

                    except Exception as exc:

                        state[
                            "recording"
                        ] = False

                        recording_status.set_text(
                            "🔴 Microphone error"
                        )

                        question_label.set_text(
                            (
                                "Your spoken question "
                                "will appear here."
                            )
                        )

                        ui.notify(
                            (
                                "Could not start recording: "
                                f"{exc}"
                            ),
                            type="negative",
                        )

                # =============================================
                # STOP RECORDING AND ASK MEMORAI
                # =============================================

                async def stop_and_ask() -> None:
                    """
                    Stop browser recording.

                    Send recorded audio to the FastAPI
                    endpoint.

                    Endpoint performs:

                        Faster-Whisper
                              ↓
                        transcription
                              ↓
                        MemorAiChat
                              ↓
                        RAG answer
                    """

                    if not state[
                        "recording"
                    ]:

                        ui.notify(
                            (
                                "Start recording before "
                                "pressing Stop & Ask."
                            ),
                            type="warning",
                        )

                        return

                    # Recording is no longer considered active.

                    state[
                        "recording"
                    ] = False

                    # -----------------------------------------
                    # PROCESSING UI
                    # -----------------------------------------

                    recording_status.set_text(
                        "🔵 Processing your voice..."
                    )

                    question_label.set_text(
                        "Transcribing your question..."
                    )

                    answer_label.set_text(
                        "Searching your memories..."
                    )

                    try:

                        result = (
                            await ui.run_javascript(
                                """
                                return await (async () => {

                                    const recorder =
                                        window
                                            .memoraiMediaRecorder;


                                    // -------------------------
                                    // Validate recorder
                                    // -------------------------

                                    if (!recorder) {

                                        return {
                                            ok: false,
                                            error:
                                                'No active recording was found.'
                                        };
                                    }


                                    try {

                                        // ---------------------
                                        // Wait for stop event
                                        // ---------------------

                                        const stopped =
                                            new Promise(
                                                resolve => {

                                                    recorder
                                                        .onstop =
                                                        resolve;
                                                }
                                            );


                                        if (
                                            recorder.state
                                            !== 'inactive'
                                        ) {

                                            recorder.stop();
                                        }


                                        await stopped;


                                        // ---------------------
                                        // Get audio
                                        // ---------------------

                                        const chunks =
                                            window
                                                .memoraiAudioChunks
                                            || [];


                                        if (
                                            chunks.length === 0
                                        ) {

                                            return {
                                                ok: false,
                                                error:
                                                    'No audio was recorded.'
                                            };
                                        }


                                        // ---------------------
                                        // Blob
                                        // ---------------------

                                        const mimeType =
                                            recorder.mimeType
                                            ||
                                            'audio/webm';


                                        const blob =
                                            new Blob(
                                                chunks,
                                                {
                                                    type:
                                                        mimeType
                                                }
                                            );


                                        // ---------------------
                                        // Stop microphone
                                        // ---------------------

                                        if (
                                            window
                                                .memoraiAudioStream
                                        ) {

                                            window
                                                .memoraiAudioStream
                                                .getTracks()
                                                .forEach(
                                                    track =>
                                                        track.stop()
                                                );
                                        }


                                        // ---------------------
                                        // Clear browser state
                                        // ---------------------

                                        window
                                            .memoraiAudioStream =
                                            null;


                                        window
                                            .memoraiMediaRecorder =
                                            null;


                                        window
                                            .memoraiAudioChunks =
                                            [];


                                        // ---------------------
                                        // Upload audio
                                        // ---------------------

                                        const response =
                                            await fetch(
                                                '/api/voice/question',
                                                {
                                                    method:
                                                        'POST',

                                                    headers: {
                                                        'Content-Type':
                                                            mimeType
                                                    },

                                                    body:
                                                        blob
                                                }
                                            );


                                        // ---------------------
                                        // Read server response
                                        // ---------------------

                                        const data =
                                            await response
                                                .json();


                                        return data;

                                    }

                                    catch (error) {

                                        return {
                                            ok: false,
                                            error:
                                                error.message
                                                ||
                                                String(error)
                                        };
                                    }

                                })();
                                """,
                                timeout=120.0,
                            )
                        )

                        # -------------------------------------
                        # NO RESPONSE
                        # -------------------------------------

                        if not result:

                            raise RuntimeError(
                                (
                                    "The voice server "
                                    "returned no result."
                                )
                            )

                        # -------------------------------------
                        # SERVER ERROR
                        # -------------------------------------

                        if not result.get(
                            "ok",
                            False,
                        ):

                            raise RuntimeError(
                                result.get(
                                    "error",
                                    (
                                        "Voice processing "
                                        "failed."
                                    ),
                                )
                            )

                        # -------------------------------------
                        # QUESTION
                        # -------------------------------------

                        question = (
                            result.get(
                                "question",
                                "",
                            )
                            or ""
                        ).strip()

                        # -------------------------------------
                        # ANSWER
                        # -------------------------------------

                        answer = (
                            result.get(
                                "answer",
                                "",
                            )
                            or ""
                        ).strip()

                        # -------------------------------------
                        # VALIDATE QUESTION
                        # -------------------------------------

                        if not question:

                            raise ValueError(
                                (
                                    "No understandable "
                                    "speech was detected."
                                )
                            )

                        # -------------------------------------
                        # FALLBACK ANSWER
                        # -------------------------------------

                        if not answer:

                            answer = (
                                "I could not find a reliable "
                                "answer in my memories."
                            )

                        # -------------------------------------
                        # SAVE STATE
                        # -------------------------------------

                        state[
                            "latest_question"
                        ] = question

                        state[
                            "latest_answer"
                        ] = answer

                        # -------------------------------------
                        # DISPLAY QUESTION
                        # -------------------------------------

                        question_label.set_text(
                            question
                        )

                        # -------------------------------------
                        # DISPLAY ANSWER
                        # -------------------------------------

                        answer_label.set_text(
                            answer
                        )

                        # -------------------------------------
                        # READY
                        # -------------------------------------

                        recording_status.set_text(
                            "🟢 Answer ready"
                        )

                        ui.notify(
                            (
                                "Voice question processed "
                                "successfully."
                            ),
                            type="positive",
                        )

                        # IMPORTANT:
                        #
                        # NO AUTOMATIC SPEECH HERE.
                        #
                        # Never add:
                        #
                        # speak_text(answer)
                        #
                        # User must press Speak Answer.

                    except Exception as exc:

                        # -------------------------------------
                        # CLEAR STATE
                        # -------------------------------------

                        state[
                            "latest_question"
                        ] = ""

                        state[
                            "latest_answer"
                        ] = ""

                        # -------------------------------------
                        # ERROR UI
                        # -------------------------------------

                        recording_status.set_text(
                            "🔴 Voice processing failed"
                        )

                        question_label.set_text(
                            (
                                "Your question could not "
                                "be transcribed."
                            )
                        )

                        answer_label.set_text(
                            (
                                "MemorAi could not process "
                                "the voice question."
                            )
                        )

                        ui.notify(
                            str(exc),
                            type="negative",
                        )

                # =============================================
                # CANCEL RECORDING
                # =============================================

                async def cancel_recording() -> None:
                    """
                    Cancel browser recording without
                    sending the audio to MemorAi.
                    """

                    try:

                        await ui.run_javascript(
                            """
                            return await (async () => {

                                const recorder =
                                    window
                                        .memoraiMediaRecorder;


                                // -----------------------------
                                // Stop recorder
                                // -----------------------------

                                if (
                                    recorder
                                    &&
                                    recorder.state
                                    !== 'inactive'
                                ) {

                                    recorder.stop();
                                }


                                // -----------------------------
                                // Stop microphone tracks
                                // -----------------------------

                                if (
                                    window
                                        .memoraiAudioStream
                                ) {

                                    window
                                        .memoraiAudioStream
                                        .getTracks()
                                        .forEach(
                                            track =>
                                                track.stop()
                                        );
                                }


                                // -----------------------------
                                // Clear browser state
                                // -----------------------------

                                window
                                    .memoraiAudioStream =
                                    null;


                                window
                                    .memoraiMediaRecorder =
                                    null;


                                window
                                    .memoraiAudioChunks =
                                    [];


                                return true;

                            })();
                            """,
                            timeout=5.0,
                        )

                    except Exception:

                        pass

                    # -----------------------------------------
                    # RESET STATE
                    # -----------------------------------------

                    state[
                        "recording"
                    ] = False

                    state[
                        "latest_question"
                    ] = ""

                    state[
                        "latest_answer"
                    ] = ""

                    # -----------------------------------------
                    # RESET UI
                    # -----------------------------------------

                    recording_status.set_text(
                        "⚪ Ready to record"
                    )

                    question_label.set_text(
                        (
                            "Your spoken question "
                            "will appear here."
                        )
                    )

                    answer_label.set_text(
                        (
                            "Ask a voice question "
                            "to receive an answer."
                        )
                    )

                    ui.notify(
                        "Recording cancelled."
                    )

                # =============================================
                # RECORDING BUTTONS
                # =============================================

                with ui.row().classes(
                    "w-full gap-3 flex-wrap mt-3"
                ):

                    ui.button(
                        "Start Recording",
                        icon="mic",
                        on_click=start_recording,
                    ).classes(
                        "memorai-action-button"
                    )

                    ui.button(
                        "Stop & Ask",
                        icon="stop_circle",
                        on_click=stop_and_ask,
                    ).classes(
                        "memorai-action-button"
                    )

                    ui.button(
                        "Cancel",
                        icon="close",
                        on_click=cancel_recording,
                    ).props(
                        "outline"
                    ).classes(
                        "memorai-action-button"
                    )

            # =================================================
            # MEMORAI ANSWER CARD
            # =================================================

            with ui.card().classes(
                "memorai-card w-full"
            ):

                ui.label(
                    "🧠 MemorAi Answer"
                ).classes(
                    "memorai-section-title"
                )

                # =============================================
                # ANSWER DISPLAY
                # =============================================

                answer_label = (
                    ui.label(
                        (
                            "Ask a voice question "
                            "to receive an answer."
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
                # SPEAK LATEST ANSWER
                # =============================================

                def speak_latest_answer() -> None:
                    """
                    Speak the latest MemorAi answer.

                    Only runs after the user explicitly
                    presses Speak Answer.
                    """

                    answer = (
                        state.get(
                            "latest_answer",
                            "",
                        )
                        or ""
                    ).strip()

                    if not answer:

                        ui.notify(
                            (
                                "Ask a voice question first, "
                                "then press Speak Answer."
                            ),
                            type="warning",
                        )

                        return

                    speak_text(
                        answer
                    )

                # =============================================
                # SPEECH CONTROLS
                # =============================================

                with ui.row().classes(
                    "w-full gap-3 flex-wrap mt-3"
                ):

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
                        on_click=stop_speaking,
                    ).props(
                        "outline"
                    ).classes(
                        "memorai-action-button"
                    )

            # =================================================
            # PRIVACY CARD
            # =================================================

            with ui.card().classes(
                "memorai-card w-full"
            ):

                ui.label(
                    "🔒 Voice Privacy"
                ).classes(
                    "memorai-section-title"
                )

                ui.label(
                    (
                        "Your microphone is used only after "
                        "you press Start Recording. Recording "
                        "stops when you press Stop & Ask "
                        "or Cancel."
                    )
                ).classes(
                    "text-lg"
                )