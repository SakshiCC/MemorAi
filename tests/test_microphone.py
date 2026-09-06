import numpy as np

from speech.microphone import (
    MicrophoneRecorder,
)


def test_microphone_configuration():

    print(
        "\nMemorAi Microphone Test Suite\n"
    )

    recorder = (
        MicrophoneRecorder()
    )

    # -------------------------------------------------
    # Basic configuration
    # -------------------------------------------------

    assert (
        recorder.sample_rate
        == 16000
    )

    print(
        "PASS: Whisper-compatible sample rate"
    )

    assert (
        recorder.channels
        == 1
    )

    print(
        "PASS: Mono microphone configuration"
    )

    assert (
        recorder.calibration_duration
        > 0
    )

    print(
        "PASS: Ambient calibration configured"
    )

    assert (
        recorder.noise_multiplier
        > 1
    )

    print(
        "PASS: Noise multiplier configured"
    )

    # -------------------------------------------------
    # Silence RMS
    # -------------------------------------------------

    silent_audio = np.zeros(
        1600,
        dtype=np.float32,
    )

    silent_rms = (
        recorder.calculate_rms(
            silent_audio
        )
    )

    assert (
        silent_rms
        == 0.0
    )

    print(
        "PASS: Silence RMS calculation"
    )

    # -------------------------------------------------
    # Simulated background noise
    # -------------------------------------------------

    background_noise = (
        np.full(
            1600,
            0.003,
            dtype=np.float32,
        )
    )

    ambient_rms = (
        recorder.calculate_rms(
            background_noise
        )
    )

    assert (
        ambient_rms
        > 0
    )

    print(
        "PASS: Ambient noise RMS calculation"
    )

    # -------------------------------------------------
    # Adaptive threshold calculation
    # -------------------------------------------------

    threshold = (
        recorder
        .calculate_adaptive_threshold(
            ambient_rms
        )
    )

    assert (
        threshold
        >= recorder.minimum_threshold
    )

    assert (
        threshold
        <= recorder.maximum_threshold
    )

    print(
        "PASS: Adaptive speech threshold calculation"
    )

    # -------------------------------------------------
    # Calibration from fake audio
    # -------------------------------------------------

    calibrated_threshold = (
        recorder
        .calibrate_from_audio(
            background_noise
        )
    )

    assert (
        recorder.last_ambient_rms
        is not None
    )

    assert (
        recorder.current_threshold
        == calibrated_threshold
    )

    print(
        "PASS: Ambient noise calibration"
    )

    # -------------------------------------------------
    # Very quiet room should not make threshold zero
    # -------------------------------------------------

    zero_threshold = (
        recorder
        .calculate_adaptive_threshold(
            0.0
        )
    )

    assert (
        zero_threshold
        == recorder.minimum_threshold
    )

    print(
        "PASS: Minimum speech threshold enforced"
    )

    # -------------------------------------------------
    # Very noisy room threshold capped safely
    # -------------------------------------------------

    high_threshold = (
        recorder
        .calculate_adaptive_threshold(
            1.0
        )
    )

    assert (
        high_threshold
        == recorder.maximum_threshold
    )

    print(
        "PASS: Maximum speech threshold enforced"
    )

    # -------------------------------------------------
    # Silence classification
    # -------------------------------------------------

    recorder.current_threshold = (
        0.01
    )

    assert (
        recorder.contains_speech(
            silent_audio
        )
        is False
    )

    print(
        "PASS: Silence correctly classified"
    )

    # -------------------------------------------------
    # Simulated speech
    # -------------------------------------------------

    simulated_speech = (
        np.full(
            1600,
            0.1,
            dtype=np.float32,
        )
    )

    assert (
        recorder.contains_speech(
            simulated_speech
        )
        is True
    )

    print(
        "PASS: Speech correctly classified"
    )

    # -------------------------------------------------
    # Empty audio
    # -------------------------------------------------

    empty_audio = np.array(
        [],
        dtype=np.float32,
    )

    assert (
        recorder.calculate_rms(
            empty_audio
        )
        == 0.0
    )

    print(
        "PASS: Empty audio safely handled"
    )

    # -------------------------------------------------
    # Devices
    # -------------------------------------------------

    devices = (
        recorder.list_devices()
    )

    assert (
        devices is not None
    )

    assert (
        len(devices)
        > 0
    )

    print(
        "PASS: Audio devices detected"
    )

    microphone = (
        recorder
        .get_default_input_device()
    )

    assert (
        microphone[
            "max_input_channels"
        ]
        > 0
    )

    print(
        "PASS: Default microphone detected"
    )

    print(
        "\nDefault microphone:"
    )

    print(
        microphone["name"]
    )

    print(
        "\nAll MemorAi microphone tests passed."
    )


if __name__ == "__main__":

    test_microphone_configuration()