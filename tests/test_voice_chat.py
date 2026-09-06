from pathlib import Path

from speech.voice_chat import (
    MemorAiVoiceChat,
)


class FakeChat:
    """
    Fake MemorAiChat.

    Prevents tests from accessing the real RAG.
    """

    def __init__(self):

        self.last_question = None
        self.call_count = 0

    def answer(
        self,
        question,
    ):

        self.last_question = (
            question
        )

        self.call_count += 1

        return (
            "Your cell phone was last observed "
            "on the table."
        )


class FakeTranscriber:
    """
    Fake Whisper implementation.
    """

    def __init__(
        self,
        text="Where is my cell phone?",
    ):

        self.text = text
        self.call_count = 0
        self.last_audio_path = None

    def transcribe(
        self,
        audio_path,
    ):

        self.call_count += 1

        self.last_audio_path = (
            audio_path
        )

        return self.text


class FakeRecorder:
    """
    Fake automatic speech recorder.

    It avoids activating the actual microphone.
    """

    def __init__(self):

        self.call_count = 0
        self.last_output_path = None

    def record_until_silence(
        self,
        output_path=(
            "recordings/voice_question.wav"
        ),
    ):

        self.call_count += 1

        self.last_output_path = (
            output_path
        )

        return Path(
            output_path
        )


def test_voice_chat():

    print(
        "\nMemorAi Voice Chat Test Suite\n"
    )

    fake_chat = (
        FakeChat()
    )

    fake_transcriber = (
        FakeTranscriber()
    )

    fake_recorder = (
        FakeRecorder()
    )

    voice_chat = (
        MemorAiVoiceChat(
            chat=fake_chat,
            recorder=fake_recorder,
            transcriber=fake_transcriber,
        )
    )

    # =========================================
    # Test 1
    # Automatic recording
    # =========================================

    audio_path = (
        voice_chat
        .record_question()
    )

    assert (
        fake_recorder.call_count
        == 1
    )

    assert (
        audio_path
        == Path(
            "recordings/"
            "voice_question.wav"
        )
    )

    print(
        "PASS: Automatic speech recording interface"
    )

    # =========================================
    # Test 2
    # Speech transcription
    # =========================================

    question = (
        voice_chat
        .transcribe_question(
            audio_path
        )
    )

    assert (
        question
        == "Where is my cell phone?"
    )

    print(
        "PASS: Speech transcription accepted"
    )

    # =========================================
    # Test 3
    # Question forwarding
    # =========================================

    answer = (
        voice_chat
        .ask_question(
            question
        )
    )

    assert (
        fake_chat.last_question
        == "Where is my cell phone?"
    )

    print(
        "PASS: Spoken question forwarded to MemorAiChat"
    )

    # =========================================
    # Test 4
    # Answer
    # =========================================

    assert (
        answer
        == (
            "Your cell phone was last observed "
            "on the table."
        )
    )

    print(
        "PASS: MemorAi answer returned"
    )

    # =========================================
    # Test 5
    # Audio -> answer
    # =========================================

    question, answer = (
        voice_chat
        .ask_from_audio(
            "fake_audio.wav"
        )
    )

    assert (
        question
        == "Where is my cell phone?"
    )

    assert answer

    print(
        "PASS: Audio-to-RAG bridge"
    )

    # =========================================
    # Test 6
    # Microphone -> answer
    # =========================================

    question, answer = (
        voice_chat
        .ask_from_microphone()
    )

    assert (
        question
        == "Where is my cell phone?"
    )

    assert answer

    print(
        "PASS: Automatic microphone-to-answer pipeline"
    )

    # =========================================
    # Test 7
    # Empty transcription
    # =========================================

    empty_voice_chat = (
        MemorAiVoiceChat(
            chat=FakeChat(),
            recorder=FakeRecorder(),
            transcriber=FakeTranscriber(
                text=""
            ),
        )
    )

    rejected = False

    try:

        empty_voice_chat.transcribe_question(
            "fake.wav"
        )

    except ValueError:

        rejected = True

    assert rejected

    print(
        "PASS: Empty transcription rejected"
    )

    # =========================================
    # Test 8
    # Noise
    # =========================================

    noise_voice_chat = (
        MemorAiVoiceChat(
            chat=FakeChat(),
            recorder=FakeRecorder(),
            transcriber=FakeTranscriber(
                text="..."
            ),
        )
    )

    rejected = False

    try:

        noise_voice_chat.transcribe_question(
            "fake.wav"
        )

    except ValueError:

        rejected = True

    assert rejected

    print(
        "PASS: Non-speech transcription rejected"
    )

    # =========================================
    # Test 9
    # Meaning preserved
    # =========================================

    meaning_chat = (
        FakeChat()
    )

    meaning_voice_chat = (
        MemorAiVoiceChat(
            chat=meaning_chat,
            recorder=FakeRecorder(),
            transcriber=FakeTranscriber(
                text="That is my cell phone."
            ),
        )
    )

    question, _ = (
        meaning_voice_chat
        .ask_from_audio(
            "fake.wav"
        )
    )

    assert (
        question
        == "That is my cell phone."
    )

    assert (
        meaning_chat.last_question
        == "That is my cell phone."
    )

    print(
        "PASS: Spoken meaning preserved"
    )

    # =========================================
    # Test 10
    # Repeated questions
    # =========================================

    repeated_chat = (
        FakeChat()
    )

    repeated_voice_chat = (
        MemorAiVoiceChat(
            chat=repeated_chat,
            recorder=FakeRecorder(),
            transcriber=FakeTranscriber(
                text="Where are my glasses?"
            ),
        )
    )

    repeated_voice_chat.ask_from_microphone()

    repeated_voice_chat.ask_from_microphone()

    repeated_voice_chat.ask_from_microphone()

    assert (
        repeated_chat.call_count
        == 3
    )

    print(
        "PASS: Repeated voice questions supported"
    )

    # =========================================
    # Test 11
    # Fake Whisper
    # =========================================

    assert isinstance(
        repeated_voice_chat.transcriber,
        FakeTranscriber,
    )

    print(
        "PASS: Tests do not load real Whisper"
    )

    print(
        "\nAll MemorAi voice chat tests passed."
    )


if __name__ == "__main__":

    test_voice_chat()