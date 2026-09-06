from pathlib import Path

from speech.microphone import MicrophoneRecorder
from speech.transcription import TranscriptionCleaner
from speech.voice_chat import MemorAiVoiceChat


class FakeChat:
    """
    Fake MemorAiChat used for speech pipeline tests.

    No MySQL, ChromaDB, Ollama, or real RAG is used.
    """

    def __init__(self):
        self.last_question = None
        self.call_count = 0

    def answer(self, question):
        self.last_question = question
        self.call_count += 1

        return (
            "Your cell phone was last observed "
            "on the table."
        )


class FakeRecorder:
    """
    Fake microphone recorder.

    Prevents automated tests from activating
    the real microphone.
    """

    def __init__(self):
        self.call_count = 0
        self.last_output_path = None

    def record_until_silence(
        self,
        output_path="recordings/voice_question.wav",
    ):
        self.call_count += 1
        self.last_output_path = output_path

        return Path(output_path)


class FakeTranscriber:
    """
    Fake Whisper transcriber.

    Prevents the real Whisper model from loading
    during automated speech regression tests.
    """

    def __init__(
        self,
        text="Where is my cell phone?",
    ):
        self.text = text
        self.call_count = 0
        self.last_audio_path = None

    def transcribe(self, audio_path):
        self.call_count += 1
        self.last_audio_path = audio_path

        return self.text


def test_speech_pipeline():

    print("\nMemorAi Speech Pipeline Test Suite\n")

    # =====================================================
    # Microphone configuration
    # =====================================================

    recorder = MicrophoneRecorder()

    assert recorder.sample_rate == 16000
    print("PASS: Whisper-compatible sample rate")

    assert recorder.channels == 1
    print("PASS: Mono microphone configuration")

    assert recorder.silence_duration > 0
    print("PASS: Silence detection configured")

    assert recorder.speech_timeout > 0
    print("PASS: Speech timeout configured")

    assert recorder.max_duration > 0
    print("PASS: Maximum recording duration configured")

    assert recorder.calibration_duration >= 0
    print("PASS: Ambient calibration configured")

    # =====================================================
    # Transcription cleaning
    # =====================================================

    cleaner = TranscriptionCleaner()

    result = cleaner.validate(
        "   Where   is my cell phone?   "
    )

    assert result.is_valid is True
    assert result.text == "Where is my cell phone?"

    print("PASS: Valid transcription cleaned")

    # =====================================================
    # Empty speech handling
    # =====================================================

    result = cleaner.validate("")

    assert result.is_valid is False

    print("PASS: Empty speech rejected")

    # =====================================================
    # Noise label handling
    # =====================================================

    result = cleaner.validate("[Music]")

    assert result.is_valid is False

    print("PASS: Whisper noise label rejected")

    # =====================================================
    # Punctuation-only handling
    # =====================================================

    result = cleaner.validate("...")

    assert result.is_valid is False

    print("PASS: Punctuation-only output rejected")

    # =====================================================
    # Meaning preservation
    # =====================================================

    result = cleaner.validate(
        "That is my cell phone."
    )

    assert result.is_valid is True
    assert result.text == "That is my cell phone."

    print("PASS: Transcription meaning preserved")

    # =====================================================
    # Fake complete voice pipeline
    # =====================================================

    fake_chat = FakeChat()

    fake_recorder = FakeRecorder()

    fake_transcriber = FakeTranscriber(
        text="Where is my cell phone?"
    )

    voice_chat = MemorAiVoiceChat(
        chat=fake_chat,
        recorder=fake_recorder,
        transcriber=fake_transcriber,
        cleaner=cleaner,
    )

    question, answer = (
        voice_chat.ask_from_microphone()
    )

    assert fake_recorder.call_count == 1
    print("PASS: Recording stage called")

    assert fake_transcriber.call_count == 1
    print("PASS: Whisper stage called")

    assert question == "Where is my cell phone?"
    print("PASS: Spoken question preserved")

    assert (
        fake_chat.last_question
        == "Where is my cell phone?"
    )

    print("PASS: Question forwarded to MemorAiChat")

    assert (
        answer
        == (
            "Your cell phone was last observed "
            "on the table."
        )
    )

    print("PASS: MemorAi answer returned")

    # =====================================================
    # Repeated questions
    # =====================================================

    voice_chat.ask_from_microphone()
    voice_chat.ask_from_microphone()

    assert fake_chat.call_count == 3

    print("PASS: Repeated voice interactions supported")

    # =====================================================
    # Invalid Whisper result must never reach RAG
    # =====================================================

    invalid_chat = FakeChat()

    invalid_voice_chat = MemorAiVoiceChat(
        chat=invalid_chat,
        recorder=FakeRecorder(),
        transcriber=FakeTranscriber(
            text=""
        ),
    )

    rejected = False

    try:
        invalid_voice_chat.ask_from_microphone()

    except ValueError:
        rejected = True

    assert rejected is True

    assert invalid_chat.call_count == 0

    print(
        "PASS: Invalid transcription blocked before RAG"
    )

    # =====================================================
    # Non-speech must never reach RAG
    # =====================================================

    noise_chat = FakeChat()

    noise_voice_chat = MemorAiVoiceChat(
        chat=noise_chat,
        recorder=FakeRecorder(),
        transcriber=FakeTranscriber(
            text="[Noise]"
        ),
    )

    rejected = False

    try:
        noise_voice_chat.ask_from_microphone()

    except ValueError:
        rejected = True

    assert rejected is True

    assert noise_chat.call_count == 0

    print(
        "PASS: Non-speech blocked before RAG"
    )

    print(
        "\nAll MemorAi speech pipeline tests passed."
    )


if __name__ == "__main__":
    test_speech_pipeline()