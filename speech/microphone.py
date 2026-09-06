from pathlib import Path
import logging
import time

import numpy as np
import sounddevice as sd
import soundfile as sf


logger = logging.getLogger(__name__)


class MicrophoneRecorder:
    """
    Handles microphone recording for MemorAi.

    Responsibilities:
        - microphone access
        - fixed-duration recording
        - ambient-noise calibration
        - speech detection
        - silence detection
        - automatic recording stop
        - WAV file saving

    This module does NOT:
        - perform speech recognition
        - access MySQL
        - access ChromaDB
        - call Ollama
        - generate answers
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        device=None,
        silence_threshold: float = 0.015,
        silence_duration: float = 1.2,
        speech_timeout: float = 5.0,
        max_duration: float = 15.0,
        block_duration: float = 0.1,
        calibration_duration: float = 0.8,
        noise_multiplier: float = 3.0,
        minimum_threshold: float = 0.008,
        maximum_threshold: float = 0.08,
    ):
        """
        Parameters
        ----------
        sample_rate:
            Recording sample rate.

        channels:
            Number of microphone channels.

        device:
            Optional input device.

        silence_threshold:
            Default threshold used when adaptive
            calibration is unavailable.

        silence_duration:
            Required silence after speech before
            recording stops.

        speech_timeout:
            Maximum time to wait for speech to begin.

        max_duration:
            Maximum duration for one spoken question.

        block_duration:
            Audio block duration in seconds.

        calibration_duration:
            Duration used to measure ambient noise.

        noise_multiplier:
            Ambient RMS is multiplied by this value
            to create the adaptive speech threshold.

        minimum_threshold:
            Lowest allowed adaptive threshold.

        maximum_threshold:
            Highest allowed adaptive threshold.
        """

        if sample_rate <= 0:
            raise ValueError(
                "Sample rate must be greater than zero."
            )

        if channels <= 0:
            raise ValueError(
                "Channels must be greater than zero."
            )

        if silence_duration <= 0:
            raise ValueError(
                "Silence duration must be greater than zero."
            )

        if speech_timeout <= 0:
            raise ValueError(
                "Speech timeout must be greater than zero."
            )

        if max_duration <= 0:
            raise ValueError(
                "Maximum duration must be greater than zero."
            )

        if block_duration <= 0:
            raise ValueError(
                "Block duration must be greater than zero."
            )

        if calibration_duration < 0:
            raise ValueError(
                "Calibration duration cannot be negative."
            )

        if noise_multiplier <= 0:
            raise ValueError(
                "Noise multiplier must be greater than zero."
            )

        self.sample_rate = sample_rate
        self.channels = channels
        self.device = device

        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        self.speech_timeout = speech_timeout
        self.max_duration = max_duration
        self.block_duration = block_duration

        self.calibration_duration = calibration_duration
        self.noise_multiplier = noise_multiplier

        self.minimum_threshold = minimum_threshold
        self.maximum_threshold = maximum_threshold

        self.current_threshold = silence_threshold
        self.last_ambient_rms = None

    def list_devices(self):
        """
        Return all available audio devices.
        """

        return sd.query_devices()

    def get_default_input_device(self):
        """
        Return default microphone information.
        """

        default_input = sd.default.device[0]

        if default_input is None or default_input < 0:
            raise RuntimeError(
                "No default microphone input device was found."
            )

        device_info = sd.query_devices(
            default_input
        )

        if device_info["max_input_channels"] <= 0:
            raise RuntimeError(
                "The default device does not support audio input."
            )

        return device_info

    @staticmethod
    def calculate_rms(audio) -> float:
        """
        Calculate Root Mean Square audio energy.

        Returns
        -------
        float
            RMS energy.
        """

        if audio is None:
            return 0.0

        audio = np.asarray(
            audio,
            dtype=np.float32,
        )

        if audio.size == 0:
            return 0.0

        squared = np.square(
            audio,
            dtype=np.float32,
        )

        mean_square = np.mean(
            squared
        )

        if not np.isfinite(
            mean_square
        ):
            return 0.0

        return float(
            np.sqrt(
                mean_square
            )
        )

    def calculate_adaptive_threshold(
        self,
        ambient_rms: float,
    ) -> float:
        """
        Convert measured ambient noise into a speech threshold.

        The threshold is constrained so extremely quiet or
        noisy rooms cannot create unreasonable values.
        """

        if ambient_rms < 0:
            ambient_rms = 0.0

        threshold = (
            ambient_rms
            * self.noise_multiplier
        )

        threshold = max(
            threshold,
            self.minimum_threshold,
        )

        threshold = min(
            threshold,
            self.maximum_threshold,
        )

        return float(
            threshold
        )

    def contains_speech(
        self,
        audio,
        threshold=None,
    ) -> bool:
        """
        Determine whether an audio block contains
        sufficient energy to be treated as speech.
        """

        if threshold is None:
            threshold = (
                self.current_threshold
            )

        rms = self.calculate_rms(
            audio
        )

        return (
            rms >= threshold
        )

    def calibrate_from_audio(
        self,
        audio,
    ) -> float:
        """
        Calculate a speech threshold using existing
        audio data.

        Useful for tests because it does not activate
        the microphone.
        """

        ambient_rms = (
            self.calculate_rms(
                audio
            )
        )

        threshold = (
            self.calculate_adaptive_threshold(
                ambient_rms
            )
        )

        self.last_ambient_rms = (
            ambient_rms
        )

        self.current_threshold = (
            threshold
        )

        return threshold

    def calibrate_ambient_noise(
        self,
        stream,
        block_size: int,
    ) -> float:
        """
        Measure room noise using the active microphone stream.

        The user should remain quiet during calibration.

        Returns
        -------
        float
            Adaptive speech threshold.
        """

        if self.calibration_duration <= 0:
            self.current_threshold = (
                self.silence_threshold
            )

            return self.current_threshold

        number_of_blocks = max(
            1,
            int(
                self.calibration_duration
                / self.block_duration
            ),
        )

        calibration_frames = []

        print(
            "Calibrating room noise..."
        )

        for _ in range(
            number_of_blocks
        ):

            audio_block, overflowed = (
                stream.read(
                    block_size
                )
            )

            if overflowed:
                logger.warning(
                    "Microphone input overflow "
                    "during calibration."
                )

            audio_block = np.asarray(
                audio_block,
                dtype=np.float32,
            )

            calibration_frames.append(
                audio_block.copy()
            )

        if not calibration_frames:
            self.current_threshold = (
                self.silence_threshold
            )

            return self.current_threshold

        calibration_audio = (
            np.concatenate(
                calibration_frames,
                axis=0,
            )
        )

        threshold = (
            self.calibrate_from_audio(
                calibration_audio
            )
        )

        logger.info(
            "Ambient RMS: %.6f | "
            "Adaptive threshold: %.6f",
            self.last_ambient_rms,
            threshold,
        )

        print(
            "Noise calibration complete."
        )

        return threshold

    def record(
        self,
        duration: float = 5.0,
        output_path: str = (
            "recordings/test_recording.wav"
        ),
    ) -> Path:
        """
        Record for a fixed duration.

        Retained for microphone diagnostics and
        backwards compatibility.
        """

        if duration <= 0:
            raise ValueError(
                "Recording duration must be greater than zero."
            )

        output_path = Path(
            output_path
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        print(
            f"\nRecording for {duration:.1f} seconds..."
        )

        print(
            "Speak now."
        )

        try:

            audio = sd.rec(
                int(
                    duration
                    * self.sample_rate
                ),
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="float32",
                device=self.device,
            )

            sd.wait()

        except Exception as exc:
            raise RuntimeError(
                f"Microphone recording failed: {exc}"
            ) from exc

        if audio is None:
            raise RuntimeError(
                "Microphone returned no audio."
            )

        audio = np.asarray(
            audio,
            dtype=np.float32,
        )

        if audio.size == 0:
            raise RuntimeError(
                "Microphone returned an empty recording."
            )

        sf.write(
            output_path,
            audio,
            self.sample_rate,
        )

        print(
            "Recording completed."
        )

        print(
            f"Saved to: {output_path.resolve()}"
        )

        return output_path

    def record_until_silence(
        self,
        output_path: str = (
            "recordings/voice_question.wav"
        ),
    ) -> Path:
        """
        Calibrate ambient noise, wait for speech,
        record it and stop after sufficient silence.

        Raises
        ------
        TimeoutError
            If speech does not begin before timeout.

        RuntimeError
            If microphone capture fails.
        """

        output_path = Path(
            output_path
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        block_size = max(
            1,
            int(
                self.sample_rate
                * self.block_duration
            ),
        )

        frames = []

        speech_started = False

        silence_started_at = None

        try:

            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="float32",
                device=self.device,
                blocksize=block_size,
            ) as stream:

                # -----------------------------------------
                # Ambient calibration
                # -----------------------------------------

                threshold = (
                    self.calibrate_ambient_noise(
                        stream=stream,
                        block_size=block_size,
                    )
                )

                print(
                    "\nListening..."
                )

                print(
                    "Speak when you are ready."
                )

                waiting_started_at = (
                    time.monotonic()
                )

                speech_started_at = None

                while True:

                    audio_block, overflowed = (
                        stream.read(
                            block_size
                        )
                    )

                    if overflowed:
                        logger.warning(
                            "Microphone input overflow detected."
                        )

                    audio_block = np.asarray(
                        audio_block,
                        dtype=np.float32,
                    )

                    if audio_block.size == 0:
                        continue

                    now = (
                        time.monotonic()
                    )

                    is_speech = (
                        self.contains_speech(
                            audio_block,
                            threshold=threshold,
                        )
                    )

                    # -------------------------------------
                    # Waiting for speech
                    # -------------------------------------

                    if not speech_started:

                        if is_speech:

                            speech_started = True

                            speech_started_at = (
                                now
                            )

                            silence_started_at = (
                                None
                            )

                            frames.append(
                                audio_block.copy()
                            )

                            print(
                                "Speech detected..."
                            )

                        else:

                            waiting_duration = (
                                now
                                - waiting_started_at
                            )

                            if (
                                waiting_duration
                                >= self.speech_timeout
                            ):
                                raise TimeoutError(
                                    "No speech was detected "
                                    "before the timeout."
                                )

                        continue

                    # -------------------------------------
                    # Speech already started
                    # -------------------------------------

                    frames.append(
                        audio_block.copy()
                    )

                    spoken_duration = (
                        now
                        - speech_started_at
                    )

                    # -------------------------------------
                    # Maximum duration safety limit
                    # -------------------------------------

                    if (
                        spoken_duration
                        >= self.max_duration
                    ):

                        logger.warning(
                            "Maximum voice recording "
                            "duration reached."
                        )

                        print(
                            "Maximum recording duration "
                            "reached."
                        )

                        break

                    # -------------------------------------
                    # Speech block
                    # -------------------------------------

                    if is_speech:

                        silence_started_at = (
                            None
                        )

                        continue

                    # -------------------------------------
                    # Silence block
                    # -------------------------------------

                    if silence_started_at is None:

                        silence_started_at = (
                            now
                        )

                    silence_length = (
                        now
                        - silence_started_at
                    )

                    if (
                        silence_length
                        >= self.silence_duration
                    ):

                        print(
                            "Silence detected. "
                            "Stopping recording."
                        )

                        break

        except TimeoutError:
            raise

        except Exception as exc:
            raise RuntimeError(
                f"Microphone recording failed: {exc}"
            ) from exc

        if not frames:
            raise RuntimeError(
                "No speech audio was recorded."
            )

        audio = np.concatenate(
            frames,
            axis=0,
        )

        if audio.size == 0:
            raise RuntimeError(
                "Recorded speech was empty."
            )

        # ---------------------------------------------
        # Final safety check
        # ---------------------------------------------

        total_rms = (
            self.calculate_rms(
                audio
            )
        )

        if total_rms <= 0:
            raise RuntimeError(
                "Recorded audio contains no usable signal."
            )

        sf.write(
            output_path,
            audio,
            self.sample_rate,
        )

        print(
            "Recording completed."
        )

        print(
            f"Saved to: {output_path.resolve()}"
        )

        logger.info(
            "Voice recording saved to %s",
            output_path,
        )

        return output_path


def main():
    """
    Manual Phase 9 microphone test.
    """

    print(
        "\nMemorAi Adaptive Microphone Test"
    )

    print(
        "-" * 50
    )

    recorder = (
        MicrophoneRecorder()
    )

    try:

        microphone = (
            recorder
            .get_default_input_device()
        )

        print(
            "\nDefault microphone:"
        )

        print(
            microphone["name"]
        )

        print(
            "\nWhen calibration starts, "
            "remain quiet briefly."
        )

        print(
            "Then speak normally."
        )

        audio_path = (
            recorder
            .record_until_silence(
                output_path=(
                    "recordings/"
                    "adaptive_microphone_test.wav"
                )
            )
        )

        print(
            "\nAdaptive recording test succeeded."
        )

        print(
            f"Audio: {audio_path}"
        )

        print(
            f"Ambient RMS: "
            f"{recorder.last_ambient_rms:.6f}"
        )

        print(
            f"Speech threshold: "
            f"{recorder.current_threshold:.6f}"
        )

    except TimeoutError as exc:

        print(
            "\nNo speech detected."
        )

        print(
            f"Reason: {exc}"
        )

    except Exception as exc:

        print(
            "\nMicrophone test failed."
        )

        print(
            f"Error: {exc}"
        )


if __name__ == "__main__":
    main()