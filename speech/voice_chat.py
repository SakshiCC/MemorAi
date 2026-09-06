import logging
from pathlib import Path

from speech.microphone import (
    MicrophoneRecorder,
)

from speech.transcription import (
    TranscriptionCleaner,
)


logger = logging.getLogger(__name__)


class MemorAiVoiceChat:
    """
    Voice interface for MemorAi.

    Pipeline:

        ENTER
          ↓
        ambient calibration
          ↓
        speech detection
          ↓
        recording
          ↓
        silence detection
          ↓
        Whisper
          ↓
        transcription validation
          ↓
        MemorAiChat
          ↓
        RAG answer
    """

    def __init__(
        self,
        chat,
        recorder=None,
        transcriber=None,
        cleaner=None,
    ):
        """
        Initialize the voice interface.
        """

        if chat is None:
            raise ValueError(
                "A MemorAiChat instance is required."
            )

        self.chat = chat

        if recorder is not None:

            self.recorder = (
                recorder
            )

        else:

            self.recorder = (
                MicrophoneRecorder(
                    sample_rate=16000,
                    channels=1,
                    silence_threshold=0.015,
                    silence_duration=1.2,
                    speech_timeout=5.0,
                    max_duration=15.0,
                    block_duration=0.1,
                    calibration_duration=0.8,
                    noise_multiplier=3.0,
                    minimum_threshold=0.008,
                    maximum_threshold=0.08,
                )
            )

        if transcriber is not None:

            self.transcriber = (
                transcriber
            )

        else:

            # Lazy import prevents Whisper loading
            # during fake unit tests.
            from speech.whisper_module import (
                WhisperTranscriber,
            )

            self.transcriber = (
                WhisperTranscriber(
                    model_size="base.en",
                    device="cpu",
                    compute_type="int8",
                )
            )

        if cleaner is not None:

            self.cleaner = (
                cleaner
            )

        else:

            self.cleaner = (
                TranscriptionCleaner()
            )

    def record_question(
        self,
        output_path: str = (
            "recordings/voice_question.wav"
        ),
    ) -> Path:
        """
        Record one automatically bounded spoken question.
        """

        recorded_path = (
            self.recorder
            .record_until_silence(
                output_path=output_path
            )
        )

        return Path(
            recorded_path
        )

    def transcribe_question(
        self,
        audio_path,
    ) -> str:
        """
        Convert audio into clean text.
        """

        raw_text = (
            self.transcriber
            .transcribe(
                str(audio_path)
            )
        )

        result = (
            self.cleaner
            .validate(
                raw_text
            )
        )

        if not result.is_valid:

            raise ValueError(
                result.reason
                or "No valid speech was detected."
            )

        return result.text

    def ask_question(
        self,
        question: str,
    ) -> str:
        """
        Send a clean text question into MemorAiChat.
        """

        result = (
            self.cleaner
            .validate(
                question
            )
        )

        if not result.is_valid:

            raise ValueError(
                result.reason
                or "Invalid question."
            )

        question = (
            result.text
        )

        logger.info(
            "Voice question: %s",
            question,
        )

        try:

            answer = (
                self.chat
                .answer(
                    question
                )
            )

        except Exception as exc:

            logger.exception(
                "MemorAiChat failed while "
                "processing voice question."
            )

            raise RuntimeError(
                "MemorAi could not process the "
                "question through the memory system."
            ) from exc

        if answer is None:

            return (
                "I could not find an answer "
                "for that question."
            )

        answer = (
            str(answer)
            .strip()
        )

        if not answer:

            return (
                "I could not find an answer "
                "for that question."
            )

        return answer

    def ask_from_audio(
        self,
        audio_path,
    ) -> tuple[str, str]:
        """
        Existing audio -> Whisper -> RAG.
        """

        question = (
            self.transcribe_question(
                audio_path
            )
        )

        answer = (
            self.ask_question(
                question
            )
        )

        return (
            question,
            answer,
        )

    def ask_from_microphone(
        self,
    ) -> tuple[str, str]:
        """
        Complete microphone interaction.
        """

        audio_path = (
            self.record_question()
        )

        return (
            self.ask_from_audio(
                audio_path
            )
        )

    def run(self):
        """
        Repeated push-to-speak voice interaction.
        """

        print(
            "\n"
            + "=" * 60
        )

        print(
            "MemorAi Voice Chat"
        )

        print(
            "=" * 60
        )

        print(
            "\nVoice recognition is ready."
        )

        print(
            "Room noise will be calibrated "
            "before each question."
        )

        print(
            "\nWhen calibration appears, "
            "stay quiet briefly."
        )

        print(
            "\nControls:"
        )

        print(
            "  Press ENTER to speak."
        )

        print(
            "  Type q and press ENTER to quit."
        )

        while True:

            print(
                "\n"
                + "-" * 60
            )

            try:

                command = input(
                    "\nPress ENTER to speak "
                    "or type q to quit: "
                )

                command = (
                    command
                    .strip()
                    .lower()
                )

                if command in {
                    "q",
                    "quit",
                    "exit",
                }:

                    print(
                        "\nClosing MemorAi Voice Chat."
                    )

                    break

                if command:

                    print(
                        "\nUnknown command."
                    )

                    continue

                print(
                    "\nPlease remain quiet "
                    "during calibration..."
                )

                question, answer = (
                    self.ask_from_microphone()
                )

                print(
                    "\n"
                    + "-" * 60
                )

                print(
                    "\nYou said:"
                )

                print(
                    question
                )

                print(
                    "\nMemorAi:"
                )

                print(
                    answer
                )

            except TimeoutError:

                print(
                    "\nNo speech was detected."
                )

                print(
                    "Press ENTER and try again."
                )

            except ValueError as exc:

                print(
                    "\nThe recording could not "
                    "be understood safely."
                )

                print(
                    f"Reason: {exc}"
                )

                print(
                    "Please try speaking again."
                )

            except RuntimeError as exc:

                print(
                    "\nMemorAi encountered a "
                    "processing problem."
                )

                print(
                    f"Reason: {exc}"
                )

                print(
                    "You can try another question."
                )

            except KeyboardInterrupt:

                print(
                    "\n\nVoice chat interrupted."
                )

                break

            except Exception as exc:

                logger.exception(
                    "Unexpected voice interaction failure."
                )

                print(
                    "\nAn unexpected voice error occurred."
                )

                print(
                    f"Error: {exc}"
                )

                print(
                    "You can try again without "
                    "restarting MemorAi."
                )


def main():
    """
    Run MemorAi voice chat.

        python -m speech.voice_chat
    """

    print(
        "\nInitializing MemorAi..."
    )

    try:

        from llm.chat import (
            MemorAiChat,
        )

        chat = (
            MemorAiChat()
        )

        print(
            "Initializing speech recognition..."
        )

        voice_chat = (
            MemorAiVoiceChat(
                chat=chat
            )
        )

        print(
            "MemorAi initialization complete."
        )

        voice_chat.run()

    except KeyboardInterrupt:

        print(
            "\nMemorAi stopped."
        )

    except Exception as exc:

        logger.exception(
            "Failed to initialize voice chat."
        )

        print(
            "\nFailed to initialize "
            "MemorAi Voice Chat."
        )

        print(
            f"Error: {exc}"
        )


if __name__ == "__main__":
    main()