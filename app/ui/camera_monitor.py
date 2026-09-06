from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Callable, Optional

import cv2
import numpy as np


# ============================================================
# CAMERA STATE
# ============================================================


class CameraState(str, Enum):

    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"


# ============================================================
# LATEST DETECTION
# ============================================================


@dataclass
class LatestDetection:

    object_name: str
    confidence: float
    location: str
    detected_at: datetime


# ============================================================
# FRAME PROCESSING RESULT
# ============================================================


@dataclass
class FrameProcessingResult:

    annotated_frame: np.ndarray

    latest_detection: Optional[
        LatestDetection
    ] = None


FrameProcessor = Callable[
    [np.ndarray],
    FrameProcessingResult,
]


# ============================================================
# CAMERA MONITOR
# ============================================================


class CameraMonitor:
    """
    Low-latency MemorAi camera service.

    Architecture:

        Capture Thread
            ↓
        latest raw frame
            ↓
        Inference Thread
            ↓
        YOLO / vision adapter
            ↓
        latest annotated frame

    Important:
    - camera capture is NOT blocked by YOLO
    - old frames are dropped
    - UI always receives the newest available frame
    """

    def __init__(
        self,
        frame_processor: FrameProcessor | None = None,
        camera_index: int = 0,
        width: int = 640,
        height: int = 480,
        capture_fps: int = 30,
        inference_fps: int = 4,
    ) -> None:

        self.frame_processor = (
            frame_processor
        )

        self.camera_index = (
            camera_index
        )

        self.width = width
        self.height = height

        self.capture_fps = max(
            1,
            int(capture_fps),
        )

        self.inference_fps = max(
            1,
            int(inference_fps),
        )

        # ----------------------------------------------------
        # CAMERA
        # ----------------------------------------------------

        self._camera: (
            cv2.VideoCapture | None
        ) = None

        # ----------------------------------------------------
        # THREADS
        # ----------------------------------------------------

        self._capture_thread: (
            threading.Thread | None
        ) = None

        self._inference_thread: (
            threading.Thread | None
        ) = None

        # ----------------------------------------------------
        # EVENTS
        # ----------------------------------------------------

        self._stop_event = (
            threading.Event()
        )

        self._pause_event = (
            threading.Event()
        )

        # ----------------------------------------------------
        # LOCK
        # ----------------------------------------------------

        self._lock = (
            threading.RLock()
        )

        # ----------------------------------------------------
        # STATE
        # ----------------------------------------------------

        self._state = (
            CameraState.STOPPED
        )

        self._last_error: (
            str | None
        ) = None

        # ----------------------------------------------------
        # FRAME STORAGE
        # ----------------------------------------------------

        self._latest_raw_frame: (
            np.ndarray | None
        ) = None

        self._latest_annotated_frame: (
            np.ndarray | None
        ) = None

        # ----------------------------------------------------
        # DETECTION
        # ----------------------------------------------------

        self._latest_detection: (
            LatestDetection | None
        ) = None

        # ----------------------------------------------------
        # COUNTERS
        # ----------------------------------------------------

        self._captured_frames = 0

        self._processed_frames = 0

        self._started_at: (
            datetime | None
        ) = None

        self._last_inference_seconds: float = 0.0


    # ========================================================
    # PROPERTIES
    # ========================================================


    @property
    def state(
        self,
    ) -> CameraState:

        with self._lock:

            return self._state


    @property
    def running(
        self,
    ) -> bool:

        return (
            self.state
            == CameraState.RUNNING
        )


    @property
    def paused(
        self,
    ) -> bool:

        return (
            self.state
            == CameraState.PAUSED
        )


    @property
    def stopped(
        self,
    ) -> bool:

        return (
            self.state
            == CameraState.STOPPED
        )


    @property
    def last_error(
        self,
    ) -> str | None:

        with self._lock:

            return self._last_error


    # ========================================================
    # FRAME PROCESSOR
    # ========================================================


    def set_frame_processor(
        self,
        processor: FrameProcessor,
    ) -> None:

        self.frame_processor = processor


    # ========================================================
    # OPEN CAMERA
    # ========================================================


    def _open_camera(
        self,
    ) -> None:

        with self._lock:

            if (
                self._camera is not None
                and self._camera.isOpened()
            ):

                return

        camera = cv2.VideoCapture(
            self.camera_index,
            cv2.CAP_DSHOW,
        )

        if not camera.isOpened():

            camera.release()

            camera = cv2.VideoCapture(
                self.camera_index
            )

        if not camera.isOpened():

            camera.release()

            raise RuntimeError(
                "MemorAi could not open the webcam."
            )

        camera.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            self.width,
        )

        camera.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            self.height,
        )

        camera.set(
            cv2.CAP_PROP_FPS,
            self.capture_fps,
        )

        # Try to keep the driver buffer small.
        # Some backends may ignore this setting.
        camera.set(
            cv2.CAP_PROP_BUFFERSIZE,
            1,
        )

        with self._lock:

            self._camera = camera


    # ========================================================
    # START
    # ========================================================


    def start(
        self,
    ) -> None:

        current_state = (
            self.state
        )

        if (
            current_state
            == CameraState.RUNNING
        ):

            return

        if (
            current_state
            == CameraState.PAUSED
        ):

            self.resume()

            return

        with self._lock:

            self._state = (
                CameraState.STARTING
            )

            self._last_error = None

            self._captured_frames = 0

            self._processed_frames = 0

            self._started_at = (
                datetime.now()
            )

        self._stop_event.clear()

        self._pause_event.clear()

        try:

            self._open_camera()

        except Exception as exc:

            with self._lock:

                self._state = (
                    CameraState.ERROR
                )

                self._last_error = str(
                    exc
                )

            return

        # ----------------------------------------------------
        # CAPTURE THREAD
        # ----------------------------------------------------

        self._capture_thread = (
            threading.Thread(
                target=self._capture_loop,
                name="MemorAiCapture",
                daemon=True,
            )
        )

        # ----------------------------------------------------
        # INFERENCE THREAD
        # ----------------------------------------------------

        self._inference_thread = (
            threading.Thread(
                target=self._inference_loop,
                name="MemorAiInference",
                daemon=True,
            )
        )

        with self._lock:

            self._state = (
                CameraState.RUNNING
            )

        self._capture_thread.start()

        self._inference_thread.start()


    # ========================================================
    # PAUSE
    # ========================================================


    def pause(
        self,
    ) -> None:

        if (
            self.state
            != CameraState.RUNNING
        ):

            return

        self._pause_event.set()

        with self._lock:

            self._state = (
                CameraState.PAUSED
            )


    # ========================================================
    # RESUME
    # ========================================================


    def resume(
        self,
    ) -> None:

        if (
            self.state
            != CameraState.PAUSED
        ):

            return

        self._pause_event.clear()

        with self._lock:

            self._state = (
                CameraState.RUNNING
            )


    # ========================================================
    # STOP
    # ========================================================


    def stop(
        self,
    ) -> None:

        if (
            self.state
            == CameraState.STOPPED
        ):

            return

        self._stop_event.set()

        self._pause_event.clear()

        # ----------------------------------------------------
        # WAIT FOR THREADS
        # ----------------------------------------------------

        for thread in (
            self._capture_thread,
            self._inference_thread,
        ):

            if (
                thread is not None
                and thread.is_alive()
                and thread
                is not threading.current_thread()
            ):

                thread.join(
                    timeout=2.0
                )

        self._release_camera()

        with self._lock:

            self._capture_thread = None

            self._inference_thread = None

            self._state = (
                CameraState.STOPPED
            )


    # ========================================================
    # CAPTURE LOOP
    # ========================================================


    def _capture_loop(
        self,
    ) -> None:
        """
        Capture frames as quickly as the webcam provides them.

        This thread NEVER runs YOLO.
        """

        target_delay = (
            1.0
            / self.capture_fps
        )

        while not self._stop_event.is_set():

            if self._pause_event.is_set():

                time.sleep(
                    0.05
                )

                continue

            with self._lock:

                camera = (
                    self._camera
                )

            if camera is None:

                break

            start_time = (
                time.perf_counter()
            )

            ok, frame = (
                camera.read()
            )

            if not ok:

                time.sleep(
                    0.01
                )

                continue

            with self._lock:

                # Always replace previous frame.
                #
                # We deliberately keep only ONE latest frame.
                self._latest_raw_frame = (
                    frame
                )

                self._captured_frames += 1

            elapsed = (
                time.perf_counter()
                - start_time
            )

            sleep_time = (
                target_delay
                - elapsed
            )

            if sleep_time > 0:

                time.sleep(
                    sleep_time
                )


    # ========================================================
    # INFERENCE LOOP
    # ========================================================


    def _inference_loop(
        self,
    ) -> None:
        """
        YOLO inference loop.

        Important:
        it always processes the NEWEST camera frame.

        Frames captured while YOLO is busy are intentionally
        discarded instead of creating a backlog.
        """

        inference_interval = (
            1.0
            / self.inference_fps
        )

        while not self._stop_event.is_set():

            if self._pause_event.is_set():

                time.sleep(
                    0.05
                )

                continue

            # ------------------------------------------------
            # COPY NEWEST FRAME
            # ------------------------------------------------

            with self._lock:

                if (
                    self._latest_raw_frame
                    is None
                ):

                    frame = None

                else:

                    frame = (
                        self._latest_raw_frame.copy()
                    )

            if frame is None:

                time.sleep(
                    0.02
                )

                continue

            inference_start = (
                time.perf_counter()
            )

            try:

                if (
                    self.frame_processor
                    is not None
                ):

                    result = (
                        self.frame_processor(
                            frame
                        )
                    )

                    if not isinstance(
                        result,
                        FrameProcessingResult,
                    ):

                        raise TypeError(
                            "Frame processor must return "
                            "FrameProcessingResult."
                        )

                    annotated_frame = (
                        result.annotated_frame
                    )

                    detection = (
                        result.latest_detection
                    )

                else:

                    annotated_frame = frame

                    detection = None

                with self._lock:

                    self._latest_annotated_frame = (
                        annotated_frame
                    )

                    if detection is not None:

                        self._latest_detection = (
                            detection
                        )

                    self._processed_frames += 1

            except Exception as exc:

                with self._lock:

                    self._last_error = str(
                        exc
                    )

                print(
                    "MemorAi inference error:",
                    exc,
                )

            inference_seconds = (
                time.perf_counter()
                - inference_start
            )

            with self._lock:

                self._last_inference_seconds = (
                    inference_seconds
                )

            remaining = (
                inference_interval
                - inference_seconds
            )

            if remaining > 0:

                time.sleep(
                    remaining
                )


    # ========================================================
    # RELEASE CAMERA
    # ========================================================


    def _release_camera(
        self,
    ) -> None:

        with self._lock:

            camera = (
                self._camera
            )

            self._camera = None

        if camera is not None:

            try:

                camera.release()

            except Exception:

                pass


    # ========================================================
    # GET DISPLAY FRAME
    # ========================================================


    def get_latest_frame(
        self,
    ) -> np.ndarray | None:
        """
        Return the best frame for Streamlit.

        Priority:
        1. most recent annotated frame
        2. raw live frame while first inference is loading
        """

        with self._lock:

            # ------------------------------------------------
            # LIVE RAW FRAME
            #
            # Important:
            # raw frame is more recent than YOLO output.
            #
            # We'll overlay YOLO detections independently later.
            # ------------------------------------------------

            if (
                self._latest_raw_frame
                is not None
            ):

                frame = (
                    self._latest_raw_frame.copy()
                )

            elif (
                self._latest_annotated_frame
                is not None
            ):

                frame = (
                    self._latest_annotated_frame.copy()
                )

            else:

                return None

        return cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )


    # ========================================================
    # GET ANNOTATED FRAME
    # ========================================================


    def get_latest_annotated_frame(
        self,
    ) -> np.ndarray | None:
        """
        Used later by the dedicated Camera / Monitoring page.
        """

        with self._lock:

            if (
                self._latest_annotated_frame
                is None
            ):

                return None

            frame = (
                self._latest_annotated_frame.copy()
            )

        return cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )


    # ========================================================
    # LATEST DETECTION
    # ========================================================


    def get_latest_detection(
        self,
    ) -> LatestDetection | None:

        with self._lock:

            detection = (
                self._latest_detection
            )

            if detection is None:

                return None

            return LatestDetection(
                object_name=(
                    detection.object_name
                ),
                confidence=(
                    detection.confidence
                ),
                location=(
                    detection.location
                ),
                detected_at=(
                    detection.detected_at
                ),
            )


    # ========================================================
    # STATUS
    # ========================================================


    def get_status(
        self,
    ) -> dict:

        with self._lock:

            return {
                "state": (
                    self._state.value
                ),

                "captured_frames": (
                    self._captured_frames
                ),

                "processed_frames": (
                    self._processed_frames
                ),

                # backwards-compatible key used by home.py
                "frames_processed": (
                    self._processed_frames
                ),

                "started_at": (
                    self._started_at
                ),

                "last_error": (
                    self._last_error
                ),

                "inference_seconds": (
                    self._last_inference_seconds
                ),

                "has_raw_frame": (
                    self._latest_raw_frame
                    is not None
                ),

                "has_annotated_frame": (
                    self._latest_annotated_frame
                    is not None
                ),
            }