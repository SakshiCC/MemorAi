from pathlib import Path

from speech.whisper_module import WhisperTranscriber


def test_whisper_configuration():
    print("\nMemorAi Whisper Test Suite\n")

    # -----------------------------------------
    # Test 1: model configuration
    # -----------------------------------------

    transcriber = WhisperTranscriber(
        model_size="base.en",
        device="cpu",
        compute_type="int8",
    )

    assert transcriber.model_size == "base.en"
    print("PASS: Whisper model configuration")

    assert transcriber.device == "cpu"
    print("PASS: CPU inference configuration")

    assert transcriber.compute_type == "int8"
    print("PASS: int8 compute configuration")

    # -----------------------------------------
    # Test 2: recording exists
    # -----------------------------------------

    audio_path = Path(
        "recordings/microphone_test.wav"
    )

    assert audio_path.exists(), (
        f"Test recording not found: {audio_path}"
    )

    print("PASS: Test audio file exists")

    # -----------------------------------------
    # Test 3: transcription runs
    # -----------------------------------------

    text = transcriber.transcribe(
        str(audio_path)
    )

    assert isinstance(text, str)

    print("PASS: Whisper returned text")

    # -----------------------------------------
    # Test 4: speech was actually detected
    # -----------------------------------------

    assert text.strip(), (
        "Whisper returned an empty transcription."
    )

    print("PASS: Speech detected")

    print("\nTranscription:")
    print(text)

    print(
        "\nAll MemorAi Whisper basic tests passed."
    )


if __name__ == "__main__":
    test_whisper_configuration()