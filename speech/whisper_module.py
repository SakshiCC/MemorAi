from __future__ import annotations

import logging
from pathlib import Path

from faster_whisper import WhisperModel

from speech.transcription import (
    TranscriptionCleaner,
)


logger = logging.getLogger(__name__)


# ============================================================
# WHISPER TRANSCRIBER
# ============================================================


class WhisperTranscriber:
    """
    Local speech-to-text module for MemorAi.

    Responsibility:

        Audio file
            ↓
        Faster-Whisper
            ↓
        Clean transcription

    This module does NOT:

        - search memory
        - access MySQL
        - access ChromaDB
        - call Ollama
        - generate answers
    """

    def __init__(
        self,
        model_size: str = "base.en",
        device: str = "cpu",
        compute_type: str = "int8",
    ) -> None:
        """
        Initialize Faster-Whisper.

        Parameters
        ----------
        model_size:
            Whisper model.

            Examples:
                tiny.en
                base.en
                small.en

        device:
            cpu
            cuda

        compute_type:

            CPU:
                int8

            GPU:
                float16
        """

        self.model_size = (
            model_size
        )

        self.device = (
            device
        )

        self.compute_type = (
            compute_type
        )

        self.cleaner = (
            TranscriptionCleaner()
        )

        # ----------------------------------------------------
        # MODEL INFORMATION
        # ----------------------------------------------------

        print(
            "\nLoading Whisper model..."
        )

        print(
            f"Model: {self.model_size}"
        )

        print(
            f"Device: {self.device}"
        )

        print(
            (
                "Compute type: "
                f"{self.compute_type}"
            )
        )

        # ----------------------------------------------------
        # LOAD MODEL
        # ----------------------------------------------------

        try:

            self.model = (
                WhisperModel(
                    self.model_size,
                    device=self.device,
                    compute_type=(
                        self.compute_type
                    ),
                )
            )

        except Exception as exc:

            raise RuntimeError(
                (
                    "Failed to load "
                    f"Whisper model: {exc}"
                )
            ) from exc

        print(
            "Whisper model loaded successfully."
        )

    # ========================================================
    # TRANSCRIBE
    # ========================================================

    def transcribe(
        self,
        audio_path: str,
    ) -> str:
        """
        Convert an audio file into validated text.

        Supported input formats depend on the audio
        decoder used by Faster-Whisper.

        Browser recordings such as WebM can normally
        be handled directly.
        """

        path = Path(
            audio_path
        )

        # ----------------------------------------------------
        # VALIDATE FILE
        # ----------------------------------------------------

        if not path.exists():

            raise FileNotFoundError(
                (
                    "Audio file not found: "
                    f"{path}"
                )
            )

        if not path.is_file():

            raise ValueError(
                (
                    "Audio path is not a file: "
                    f"{path}"
                )
            )

        if path.stat().st_size == 0:

            raise ValueError(
                "Audio recording is empty."
            )

        print(
            f"\nTranscribing: {path}"
        )

        # ----------------------------------------------------
        # WHISPER
        # ----------------------------------------------------

        try:

            segments, _info = (
                self.model.transcribe(
                    str(path),
                    beam_size=5,
                    language="en",
                    vad_filter=True,
                )
            )

        except Exception as exc:

            raise RuntimeError(
                (
                    "Whisper transcription "
                    f"failed: {exc}"
                )
            ) from exc

        # ----------------------------------------------------
        # BUILD TRANSCRIPTION
        # ----------------------------------------------------

        transcription_parts = []

        for segment in segments:

            text = (
                segment.text
                .strip()
            )

            if not text:

                continue

            transcription_parts.append(
                text
            )

            print(
                (
                    f"[{segment.start:.2f}s -> "
                    f"{segment.end:.2f}s] "
                    f"{text}"
                )
            )

        raw_text = (
            " ".join(
                transcription_parts
            )
            .strip()
        )

        # ----------------------------------------------------
        # VALIDATE / CLEAN
        # ----------------------------------------------------

        result = (
            self.cleaner
            .validate(
                raw_text
            )
        )

        if not result.is_valid:

            logger.warning(
                (
                    "Invalid transcription: %s"
                ),
                result.reason,
            )

            return ""

        return (
            result.text
        )


# ============================================================
# COMMAND LINE TEST
# ============================================================


def main() -> None:

    transcriber = (
        WhisperTranscriber(
            model_size="base.en",
            device="cpu",
            compute_type="int8",
        )
    )

    audio_path = (
        "recordings/microphone_test.wav"
    )

    try:

        text = (
            transcriber.transcribe(
                audio_path
            )
        )

        print(
            "\n"
            + "=" * 50
        )

        print(
            "MemorAi Whisper Result"
        )

        print(
            "=" * 50
        )

        if text:

            print(
                f"\nTranscription:\n{text}"
            )

        else:

            print(
                "\nNo speech detected."
            )

    except Exception as exc:

        print(
            (
                "\nWhisper test failed: "
                f"{exc}"
            )
        )


if __name__ == "__main__":

    main()