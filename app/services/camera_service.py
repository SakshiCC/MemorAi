from __future__ import annotations

import threading
import time

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Generator

import cv2
import numpy as np

from config import (
    CAMERA_INDEX,
    CONFIRMATION_FRAMES,
    DETECTION_COOLDOWN_SECONDS,
)

from database.db_manager import (
    initialize_database,
)

from database.medicine_repository import (
    get_all_medicine_reference_images,
)

from face.person_monitor import (
    ConfirmedPersonEncounter,
    PersonMonitor,
)

from medicine.medicine_camera_pipeline import (
    ActivePersonContext,
    ConfirmedMedicineEvent,
    MedicineCameraPipeline,
)

from vision.detection_confirmor import (
    DetectionConfirmor,
)

from vision.event_logger import (
    EventLogger,
)

from vision.location_mapper import (
    map_bbox_to_location,
)

from vision.object_detector import (
    ObjectDetector,
)

from utils.logger import (
    get_logger,
)


# ============================================================
# LOGGER
# ============================================================

logger = get_logger(
    "nicegui.camera_service",
    "vision.log",
)


# ============================================================
# CAMERA STATE
# ============================================================

class CameraState(
    str,
    Enum,
):

    STOPPED = "stopped"

    STARTING = "starting"

    RUNNING = "running"

    PAUSED = "paused"

    ERROR = "error"


# ============================================================
# OBJECT SNAPSHOT
# ============================================================

@dataclass
class DetectionSnapshot:

    object_name: str

    confidence: float

    location: str

    detected_at: datetime


# ============================================================
# OBJECT DISPLAY ANNOTATION
# ============================================================

@dataclass
class DisplayAnnotation:

    label: str

    confidence: float

    bbox: tuple[
        int,
        int,
        int,
        int,
    ]


# ============================================================
# PERSON SNAPSHOT
# ============================================================

@dataclass
class PersonSnapshot:

    person_id: int

    person_name: str

    display_name: str

    similarity: float

    detected_at: datetime

    image_path: str

    event_id: int


# ============================================================
# LIVE PERSON CONTEXT
# ============================================================

@dataclass
class LivePersonContext:

    person_id: int

    person_name: str

    display_name: str

    similarity: float

    bbox: tuple[
        int,
        int,
        int,
        int,
    ]

    observed_at: datetime

    seen_monotonic: float


# ============================================================
# FACE DISPLAY ANNOTATION
# ============================================================

@dataclass
class FaceDisplayAnnotation:

    label: str

    similarity: float

    is_known: bool

    bbox: tuple[
        int,
        int,
        int,
        int,
    ]


# ============================================================
# MEDICINE SNAPSHOT
# ============================================================

@dataclass
class MedicineSnapshot:

    event_id: int

    medicine_id: int

    medicine_name: str

    dosage: str | None

    person_id: int

    person_name: str

    medicine_similarity: float

    person_similarity: float

    action_confidence: float

    overall_confidence: float

    detected_at: datetime

    image_path: str

    status: str


# ============================================================
# MEDICINE LIVE STATUS
# ============================================================

@dataclass
class MedicineLiveStatus:

    phase: str

    message: str

    medicine_name: str | None

    medicine_similarity: float | None

    medicine_threshold: float | None

    medicine_is_known: bool

    confirmation_count: int

    confirmation_required: int

    person_name: str | None

    person_similarity: float | None

    action_confidence: float | None


# ============================================================
# CAMERA SERVICE
# ============================================================

class CameraService:

    def __init__(
        self,
        camera_index: int = CAMERA_INDEX,
        width: int = 640,
        height: int = 480,
        capture_fps: int = 30,
        inference_fps: int = 4,
        stream_fps: int = 15,
        face_inference_fps: int = 3,

        # KEEP WORKING MEDICINE RECOGNITION SPEED
        medicine_scan_fps: int = 4,

        # Hand/action checking
        medicine_action_fps: int = 6,

        enable_object_monitoring: bool = True,
        enable_person_monitoring: bool = True,
        enable_medicine_monitoring: bool = True,
    ) -> None:

        self.camera_index = int(
            camera_index
        )

        self.width = max(
            1,
            int(
                width
            ),
        )

        self.height = max(
            1,
            int(
                height
            ),
        )

        self.capture_fps = max(
            1,
            int(
                capture_fps
            ),
        )

        self.inference_fps = max(
            1,
            int(
                inference_fps
            ),
        )

        self.stream_fps = max(
            1,
            int(
                stream_fps
            ),
        )

        self.face_inference_fps = max(
            1,
            int(
                face_inference_fps
            ),
        )

        self.medicine_scan_fps = max(
            1,
            int(
                medicine_scan_fps
            ),
        )

        self.medicine_action_fps = max(
            1,
            int(
                medicine_action_fps
            ),
        )

        self.enable_object_monitoring = bool(
            enable_object_monitoring
        )

        self.enable_person_monitoring = bool(
            enable_person_monitoring
        )

        self.enable_medicine_monitoring = bool(
            enable_medicine_monitoring
        )

        initialize_database()

        # ====================================================
        # OBJECT PIPELINE
        # ====================================================

        self.detector: (
            ObjectDetector
            | None
        ) = None

        self.confirmor: (
            DetectionConfirmor
            | None
        ) = None

        self.event_logger: (
            EventLogger
            | None
        ) = None

        if self.enable_object_monitoring:

            self.detector = (
                ObjectDetector()
            )

            self.confirmor = (
                DetectionConfirmor(
                    required_frames=(
                        CONFIRMATION_FRAMES
                    ),

                    cooldown_seconds=(
                        DETECTION_COOLDOWN_SECONDS
                    ),
                )
            )

            self.event_logger = (
                EventLogger(
                    index_in_vector_memory=True
                )
            )

        # ====================================================
        # PERSON PIPELINE
        # ====================================================

        self.person_monitor: (
            PersonMonitor
            | None
        ) = None

        if self.enable_person_monitoring:

            self.person_monitor = (
                PersonMonitor()
            )

        # ====================================================
        # MEDICINE PIPELINE
        # ====================================================

        self._medicine_pipeline: (
            MedicineCameraPipeline
            | None
        ) = None

        self._medicine_pipeline_guard = (
            threading.RLock()
        )

        # ====================================================
        # CAMERA
        # ====================================================

        self._camera: (
            cv2.VideoCapture
            | None
        ) = None

        # ====================================================
        # THREADS
        # ====================================================

        self._capture_thread: (
            threading.Thread
            | None
        ) = None

        self._inference_thread: (
            threading.Thread
            | None
        ) = None

        self._face_thread: (
            threading.Thread
            | None
        ) = None

        self._medicine_thread: (
            threading.Thread
            | None
        ) = None

        # ====================================================
        # LOCKS
        # ====================================================

        self._lock = (
            threading.RLock()
        )

        self._stop_event = (
            threading.Event()
        )

        self._pause_detection_event = (
            threading.Event()
        )

        # ====================================================
        # STATE
        # ====================================================

        self._state = (
            CameraState.STOPPED
        )

        self._last_error: (
            str
            | None
        ) = None

        self._last_face_error: (
            str
            | None
        ) = None

        self._last_medicine_error: (
            str
            | None
        ) = None

        # ====================================================
        # FRAME DATA
        # ====================================================

        self._latest_frame: (
            np.ndarray
            | None
        ) = None

        self._latest_annotations: list[
            DisplayAnnotation
        ] = []

        self._latest_face_annotations: list[
            FaceDisplayAnnotation
        ] = []

        # ====================================================
        # PERSON CONTEXT
        # ====================================================

        self._current_known_person: (
            LivePersonContext
            | None
        ) = None

        # Normal person context
        self._person_context_max_age_seconds = (
            2.0
        )

        # Medicine may cover face during package scanning.
        self._medicine_person_context_max_age_seconds = (
            12.0
        )

        # A fresh action face context is useful but is no
        # longer required.
        self._medicine_action_person_context_max_age_seconds = (
            4.0
        )

        # ====================================================
        # LATEST RESULTS
        # ====================================================

        self._latest_detection: (
            DetectionSnapshot
            | None
        ) = None

        self._latest_person: (
            PersonSnapshot
            | None
        ) = None

        self._latest_medicine: (
            MedicineSnapshot
            | None
        ) = None

        # ====================================================
        # MEDICINE STATUS
        # ====================================================

        self._medicine_live_status = (
            self._empty_medicine_status()
        )

        # ====================================================
        # VERSIONS
        # ====================================================

        self._detection_version = 0

        self._person_version = 0

        self._medicine_version = 0

        # ====================================================
        # PERFORMANCE
        # ====================================================

        self._captured_frames = 0

        self._processed_frames = 0

        self._face_processed_frames = 0

        self._medicine_processed_frames = 0

        self._last_inference_seconds = 0.0

        self._last_face_inference_seconds = 0.0

        self._last_medicine_inference_seconds = 0.0


    # ========================================================
    # DEFAULT MEDICINE STATUS
    # ========================================================

    @staticmethod
    def _empty_medicine_status(
    ) -> MedicineLiveStatus:

        return MedicineLiveStatus(
            phase="idle",

            message=(
                "Waiting for a registered medicine."
            ),

            medicine_name=None,

            medicine_similarity=None,

            medicine_threshold=None,

            medicine_is_known=False,

            confirmation_count=0,

            confirmation_required=0,

            person_name=None,

            person_similarity=None,

            action_confidence=None,
        )


    # ========================================================
    # PROPERTIES
    # ========================================================

    @property
    def state(
        self,
    ) -> CameraState:

        with self._lock:

            return (
                self._state
            )


    @property
    def last_error(
        self,
    ) -> str | None:

        with self._lock:

            return (
                self._last_error
            )


    @property
    def last_face_error(
        self,
    ) -> str | None:

        with self._lock:

            return (
                self._last_face_error
            )


    @property
    def last_medicine_error(
        self,
    ) -> str | None:

        with self._lock:

            return (
                self._last_medicine_error
            )


    # ========================================================
    # LATEST FRAME
    # ========================================================

    def get_latest_frame_copy(
        self,
    ) -> np.ndarray | None:

        with self._lock:

            if (
                self._latest_frame
                is None
            ):

                return None

            return (
                self._latest_frame
                .copy()
            )


    # ========================================================
    # MEDICINE REFERENCES AVAILABLE
    # ========================================================

    @staticmethod
    def _medicine_references_available(
    ) -> bool:

        try:

            rows = (
                get_all_medicine_reference_images(
                    active_medicines_only=True
                )
            )

            return any(
                row.get(
                    "embedding"
                )
                is not None

                for row
                in rows
            )

        except Exception:

            logger.exception(
                "Could not check registered "
                "medicine references."
            )

            return False


    # ========================================================
    # MEDICINE PIPELINE
    # ========================================================

    def _get_or_create_medicine_pipeline(
        self,
    ) -> MedicineCameraPipeline | None:

        if (
            not self.enable_medicine_monitoring
        ):

            return None

        with self._medicine_pipeline_guard:

            if (
                self._medicine_pipeline
                is not None
            ):

                return (
                    self._medicine_pipeline
                )

            if (
                not self
                ._medicine_references_available()
            ):

                return None

            logger.info(
                "Initializing live Medicine "
                "Camera Pipeline..."
            )

            pipeline = (
                MedicineCameraPipeline()
            )

            self._medicine_pipeline = (
                pipeline
            )

            logger.info(
                "Live Medicine Camera "
                "Pipeline initialized."
            )

            return (
                pipeline
            )


    # ========================================================
    # RESET MEDICINE
    # ========================================================

    def _reset_medicine_pipeline(
        self,
    ) -> None:

        with self._medicine_pipeline_guard:

            if (
                self._medicine_pipeline
                is not None
            ):

                try:

                    self._medicine_pipeline.reset()

                except Exception:

                    logger.exception(
                        "Could not reset medicine pipeline."
                    )

        with self._lock:

            self._medicine_live_status = (
                self._empty_medicine_status()
            )


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

        camera = (
            cv2.VideoCapture(
                self.camera_index,
                cv2.CAP_DSHOW,
            )
        )

        if not camera.isOpened():

            camera.release()

            camera = (
                cv2.VideoCapture(
                    self.camera_index
                )
            )

        if not camera.isOpened():

            camera.release()

            raise RuntimeError(
                "Could not open MemorAi webcam."
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

        camera.set(
            cv2.CAP_PROP_BUFFERSIZE,
            1,
        )

        with self._lock:

            self._camera = (
                camera
            )


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

            self._last_error = (
                None
            )

            self._last_face_error = (
                None
            )

            self._last_medicine_error = (
                None
            )

            self._latest_frame = (
                None
            )

            self._latest_annotations = []

            self._latest_face_annotations = []

            self._current_known_person = (
                None
            )

        if (
            self.person_monitor
            is not None
        ):

            self.person_monitor.reset()

        self._reset_medicine_pipeline()

        self._stop_event.clear()

        self._pause_detection_event.clear()

        try:

            self._open_camera()

        except Exception as exc:

            with self._lock:

                self._state = (
                    CameraState.ERROR
                )

                self._last_error = (
                    str(
                        exc
                    )
                )

            logger.exception(
                "Camera startup failed."
            )

            return

        self._capture_thread = (
            threading.Thread(
                target=(
                    self._capture_loop
                ),

                name="MemorAiCapture",

                daemon=True,
            )
        )

        if (
            self.enable_object_monitoring
        ):

            self._inference_thread = (
                threading.Thread(
                    target=(
                        self._inference_loop
                    ),

                    name="MemorAiInference",

                    daemon=True,
                )
            )

        else:

            self._inference_thread = (
                None
            )

        if (
            self.enable_person_monitoring
        ):

            self._face_thread = (
                threading.Thread(
                    target=(
                        self._face_inference_loop
                    ),

                    name="MemorAiFaceInference",

                    daemon=True,
                )
            )

        else:

            self._face_thread = (
                None
            )

        if (
            self.enable_medicine_monitoring
        ):

            self._medicine_thread = (
                threading.Thread(
                    target=(
                        self._medicine_inference_loop
                    ),

                    name="MemorAiMedicineInference",

                    daemon=True,
                )
            )

        else:

            self._medicine_thread = (
                None
            )

        with self._lock:

            self._state = (
                CameraState.RUNNING
            )

        self._capture_thread.start()

        if (
            self._inference_thread
            is not None
        ):

            self._inference_thread.start()

        if (
            self._face_thread
            is not None
        ):

            self._face_thread.start()

        if (
            self._medicine_thread
            is not None
        ):

            self._medicine_thread.start()

        logger.info(
            (
                "MemorAi camera started. "
                "objects=%s people=%s medicine=%s"
            ),
            self.enable_object_monitoring,
            self.enable_person_monitoring,
            self.enable_medicine_monitoring,
        )


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

        self._pause_detection_event.set()

        if (
            self.person_monitor
            is not None
        ):

            self.person_monitor.reset()

        self._reset_medicine_pipeline()

        with self._lock:

            self._state = (
                CameraState.PAUSED
            )

            self._latest_annotations = []

            self._latest_face_annotations = []

            self._current_known_person = (
                None
            )

        logger.info(
            "MemorAi monitoring paused."
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

        if (
            self.person_monitor
            is not None
        ):

            self.person_monitor.reset()

        self._reset_medicine_pipeline()

        with self._lock:

            self._current_known_person = (
                None
            )

        self._pause_detection_event.clear()

        with self._lock:

            self._state = (
                CameraState.RUNNING
            )

        logger.info(
            "MemorAi monitoring resumed."
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

        self._pause_detection_event.clear()

        for thread in (
            self._capture_thread,
            self._inference_thread,
            self._face_thread,
            self._medicine_thread,
        ):

            if (
                thread is not None
                and thread.is_alive()
                and thread
                is not threading.current_thread()
            ):

                thread.join(
                    timeout=3.0
                )

        self._release_camera()

        if (
            self.person_monitor
            is not None
        ):

            self.person_monitor.reset()

        self._reset_medicine_pipeline()

        with self._lock:

            self._capture_thread = (
                None
            )

            self._inference_thread = (
                None
            )

            self._face_thread = (
                None
            )

            self._medicine_thread = (
                None
            )

            self._latest_frame = (
                None
            )

            self._latest_annotations = []

            self._latest_face_annotations = []

            self._current_known_person = (
                None
            )

            self._state = (
                CameraState.STOPPED
            )

        logger.info(
            "MemorAi camera stopped."
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

            self._camera = (
                None
            )

        if camera is not None:

            try:

                camera.release()

            except Exception:

                pass


    # ========================================================
    # CAPTURE LOOP
    # ========================================================

    def _capture_loop(
        self,
    ) -> None:

        delay = (
            1.0
            / self.capture_fps
        )

        while (
            not self._stop_event
            .is_set()
        ):

            started = (
                time.perf_counter()
            )

            with self._lock:

                camera = (
                    self._camera
                )

            if camera is None:

                break

            ok, frame = (
                camera.read()
            )

            if not ok:

                time.sleep(
                    0.01
                )

                continue

            with self._lock:

                self._latest_frame = (
                    frame
                )

                self._captured_frames += 1

            remaining = (
                delay
                - (
                    time.perf_counter()
                    - started
                )
            )

            if remaining > 0:

                time.sleep(
                    remaining
                )


    # ========================================================
    # OBJECT INFERENCE LOOP
    # ========================================================

    def _inference_loop(
        self,
    ) -> None:

        if (
            self.detector is None
            or self.confirmor is None
            or self.event_logger is None
        ):

            return

        delay = (
            1.0
            / self.inference_fps
        )

        while (
            not self._stop_event
            .is_set()
        ):

            if (
                self._pause_detection_event
                .is_set()
            ):

                time.sleep(
                    0.05
                )

                continue

            frame = (
                self.get_latest_frame_copy()
            )

            if frame is None:

                time.sleep(
                    0.01
                )

                continue

            inference_start = (
                time.perf_counter()
            )

            try:

                frame_height, frame_width = (
                    frame.shape[
                        :2
                    ]
                )

                detections = (
                    self.detector.detect(
                        frame
                    )
                )

                annotations: list[
                    DisplayAnnotation
                ] = []

                for detection in detections:

                    (
                        x1,
                        y1,
                        x2,
                        y2,
                    ) = (
                        detection.bbox
                    )

                    annotations.append(
                        DisplayAnnotation(
                            label=(
                                detection.label
                            ),

                            confidence=float(
                                detection.confidence
                            ),

                            bbox=(
                                int(
                                    x1
                                ),
                                int(
                                    y1
                                ),
                                int(
                                    x2
                                ),
                                int(
                                    y2
                                ),
                            ),
                        )
                    )

                with self._lock:

                    self._latest_annotations = (
                        annotations
                    )

                confirmed_detections = (
                    self.confirmor.update(
                        detections
                    )
                )

                evidence_frame = (
                    self._draw_annotations(
                        frame,
                        annotations,
                    )
                )

                for confirmed in (
                    confirmed_detections
                ):

                    detection = (
                        confirmed.detection
                    )

                    location = (
                        map_bbox_to_location(
                            detection.bbox,
                            frame_width,
                            frame_height,
                        )
                    )

                    (
                        object_event_id,
                        event_id,
                    ) = (
                        self.event_logger
                        .log_object_detection(
                            frame=(
                                evidence_frame
                            ),

                            object_name=(
                                detection.label
                            ),

                            confidence=(
                                detection.confidence
                            ),

                            bbox=(
                                detection.bbox
                            ),

                            location=(
                                location
                            ),
                        )
                    )

                    snapshot = (
                        DetectionSnapshot(
                            object_name=(
                                detection.label
                            ),

                            confidence=float(
                                detection.confidence
                            ),

                            location=(
                                location
                            ),

                            detected_at=(
                                datetime.now()
                            ),
                        )
                    )

                    with self._lock:

                        self._latest_detection = (
                            snapshot
                        )

                        self._detection_version += 1

                    logger.info(
                        (
                            "NiceGUI confirmed memory: "
                            "object=%s "
                            "object_event=%s "
                            "event=%s "
                            "version=%s"
                        ),
                        detection.label,
                        object_event_id,
                        event_id,
                        self.get_detection_version(),
                    )

                with self._lock:

                    self._processed_frames += 1

            except Exception as exc:

                logger.exception(
                    "Camera inference error."
                )

                with self._lock:

                    self._last_error = (
                        str(
                            exc
                        )
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
                delay
                - inference_seconds
            )

            if remaining > 0:

                time.sleep(
                    remaining
                )


    # ========================================================
    # UPDATE CURRENT PERSON
    # ========================================================

    def _update_live_person_context(
        self,
        observations,
    ) -> None:

        known = [
            observation

            for observation
            in observations

            if (
                observation.is_known
                and observation.person_id
                is not None
            )
        ]

        if not known:

            return

        best = max(
            known,

            key=lambda item: (
                float(
                    item.similarity
                )
            ),
        )

        context = (
            LivePersonContext(
                person_id=int(
                    best.person_id
                ),

                person_name=str(
                    best.person_name
                ),

                display_name=str(
                    best.display_name
                ),

                similarity=float(
                    best.similarity
                ),

                bbox=(
                    int(
                        best.bbox[
                            0
                        ]
                    ),
                    int(
                        best.bbox[
                            1
                        ]
                    ),
                    int(
                        best.bbox[
                            2
                        ]
                    ),
                    int(
                        best.bbox[
                            3
                        ]
                    ),
                ),

                observed_at=(
                    best.detected_at
                ),

                seen_monotonic=(
                    time.monotonic()
                ),
            )
        )

        with self._lock:

            self._current_known_person = (
                context
            )


    # ========================================================
    # CURRENT PERSON
    # ========================================================

    def _get_current_person_context(
        self,
        max_age_seconds: float | None = None,
    ) -> ActivePersonContext | None:

        with self._lock:

            current = (
                self._current_known_person
            )

            if current is None:

                return None

            age = (
                time.monotonic()
                - current.seen_monotonic
            )

            allowed_age = (
                self._person_context_max_age_seconds

                if max_age_seconds is None

                else max(
                    0.0,
                    float(
                        max_age_seconds
                    ),
                )
            )

            if age > allowed_age:

                return None

            return (
                ActivePersonContext(
                    person_id=(
                        current.person_id
                    ),

                    person_name=(
                        current.person_name
                    ),

                    display_name=(
                        current.display_name
                    ),

                    similarity=(
                        current.similarity
                    ),

                    bbox=(
                        current.bbox
                    ),

                    observed_at=(
                        current.observed_at
                    ),
                )
            )


    # ========================================================
    # FACE INFERENCE
    # ========================================================

    def _face_inference_loop(
        self,
    ) -> None:

        if (
            self.person_monitor
            is None
        ):

            return

        delay = (
            1.0
            / self.face_inference_fps
        )

        while (
            not self._stop_event
            .is_set()
        ):

            if (
                self._pause_detection_event
                .is_set()
            ):

                time.sleep(
                    0.05
                )

                continue

            frame = (
                self.get_latest_frame_copy()
            )

            if frame is None:

                time.sleep(
                    0.01
                )

                continue

            inference_start = (
                time.perf_counter()
            )

            try:

                (
                    observations,
                    confirmed_encounters,
                ) = (
                    self.person_monitor
                    .process_frame(
                        frame
                    )
                )

                face_annotations: list[
                    FaceDisplayAnnotation
                ] = []

                for observation in observations:

                    label = (
                        observation.display_name

                        if observation.is_known

                        else "Unknown"
                    )

                    face_annotations.append(
                        FaceDisplayAnnotation(
                            label=(
                                label
                            ),

                            similarity=float(
                                observation.similarity
                            ),

                            is_known=(
                                observation.is_known
                            ),

                            bbox=(
                                observation.bbox
                            ),
                        )
                    )

                self._update_live_person_context(
                    observations
                )

                with self._lock:

                    self._latest_face_annotations = (
                        face_annotations
                    )

                    self._face_processed_frames += 1

                for encounter in (
                    confirmed_encounters
                ):

                    self._publish_person_encounter(
                        encounter
                    )

            except Exception as exc:

                logger.exception(
                    "Face inference error."
                )

                with self._lock:

                    self._last_face_error = (
                        str(
                            exc
                        )
                    )

            inference_seconds = (
                time.perf_counter()
                - inference_start
            )

            with self._lock:

                self._last_face_inference_seconds = (
                    inference_seconds
                )

            remaining = (
                delay
                - inference_seconds
            )

            if remaining > 0:

                time.sleep(
                    remaining
                )


    # ========================================================
    # MEDICINE SCAN BOX
    # ========================================================

    @staticmethod
    def get_medicine_scan_bounds(
        frame_width: int,
        frame_height: int,
    ) -> tuple[
        int,
        int,
        int,
        int,
    ]:

        x1 = int(
            frame_width
            * 0.28
        )

        x2 = int(
            frame_width
            * 0.72
        )

        y1 = int(
            frame_height
            * 0.42
        )

        y2 = int(
            frame_height
            * 0.78
        )

        return (
            x1,
            y1,
            x2,
            y2,
        )


    # ========================================================
    # MEDICINE CROP
    # ========================================================

    @classmethod
    def _extract_medicine_scan_region(
        cls,
        frame: np.ndarray,
    ) -> np.ndarray:

        height, width = (
            frame.shape[
                :2
            ]
        )

        (
            x1,
            y1,
            x2,
            y2,
        ) = (
            cls.get_medicine_scan_bounds(
                frame_width=(
                    width
                ),

                frame_height=(
                    height
                ),
            )
        )

        crop = (
            frame[
                y1:y2,
                x1:x2,
            ]
        )

        if crop.size == 0:

            return (
                frame.copy()
            )

        return (
            crop.copy()
        )


    # ========================================================
    # SYNC MEDICINE STATUS
    # ========================================================

    def _sync_medicine_live_status(
        self,
        pipeline: MedicineCameraPipeline,
    ) -> None:

        status = (
            pipeline.status
        )

        with self._lock:

            self._medicine_live_status = (
                MedicineLiveStatus(
                    phase=(
                        status.phase
                    ),

                    message=(
                        status.message
                    ),

                    medicine_name=(
                        status.medicine_name
                    ),

                    medicine_similarity=(
                        status.medicine_similarity
                    ),

                    medicine_threshold=(
                        status.medicine_threshold
                    ),

                    medicine_is_known=(
                        status.medicine_is_known
                    ),

                    confirmation_count=(
                        status.confirmation_count
                    ),

                    confirmation_required=(
                        status.confirmation_required
                    ),

                    person_name=(
                        status.person_name
                    ),

                    person_similarity=(
                        status.person_similarity
                    ),

                    action_confidence=(
                        status.action_confidence
                    ),
                )
            )


    # ========================================================
    # MEDICINE LOOP
    #
    # RECOGNITION FLOW REMAINS UNCHANGED.
    #
    # ONLY ACTION MODE IS FIXED.
    # ========================================================

    def _medicine_inference_loop(
        self,
    ) -> None:

        while (
            not self._stop_event
            .is_set()
        ):

            if (
                self._pause_detection_event
                .is_set()
            ):

                time.sleep(
                    0.05
                )

                continue

            loop_start = (
                time.perf_counter()
            )

            target_delay = (
                0.50
            )

            try:

                pipeline = (
                    self._get_or_create_medicine_pipeline()
                )

                # ============================================
                # NO REGISTERED MEDICINE
                # ============================================

                if pipeline is None:

                    with self._lock:

                        self._medicine_live_status = (
                            MedicineLiveStatus(
                                phase=(
                                    "no_registered_medicine"
                                ),

                                message=(
                                    "No active registered "
                                    "medicine with visual "
                                    "references is available."
                                ),

                                medicine_name=None,

                                medicine_similarity=None,

                                medicine_threshold=None,

                                medicine_is_known=False,

                                confirmation_count=0,

                                confirmation_required=0,

                                person_name=None,

                                person_similarity=None,

                                action_confidence=None,
                            )
                        )

                    time.sleep(
                        1.0
                    )

                    continue

                frame = (
                    self.get_latest_frame_copy()
                )

                if frame is None:

                    time.sleep(
                        0.02
                    )

                    continue

                # ============================================
                # PERSON CONTEXTS
                # ============================================

                # Longer-lived person context used while the
                # medicine packet may cover part of the face.
                scan_person = (
                    self._get_current_person_context(
                        max_age_seconds=(
                            self
                            ._medicine_person_context_max_age_seconds
                        )
                    )
                )

                # Fresh face context is preferred for mouth
                # location, but is NOT mandatory.
                action_person = (
                    self._get_current_person_context(
                        max_age_seconds=(
                            self
                            ._medicine_action_person_context_max_age_seconds
                        )
                    )
                )

                # ============================================
                # ACTION MODE
                #
                # THIS IS THE FIX.
                # ============================================

                if pipeline.action_active:

                    # ------------------------------------------------
                    # Prefer the freshest face bbox.
                    #
                    # Otherwise use the longer cached scan context.
                    #
                    # If both have expired, pass None.
                    # MedicineCameraPipeline will continue with the
                    # person context stored when action tracking was
                    # armed.
                    # ------------------------------------------------

                    action_context = (
                        action_person

                        if action_person is not None

                        else scan_person
                    )

                    result = (
                        pipeline
                        .process_action_frame(
                            frame=(
                                frame
                            ),

                            person=(
                                action_context
                            ),
                        )
                    )

                    self._sync_medicine_live_status(
                        pipeline
                    )

                    if result is not None:

                        self._publish_medicine_event(
                            result
                        )

                    target_delay = (
                        1.0
                        / self.medicine_action_fps
                    )

                # ============================================
                # MEDICINE KNOWN, WAITING FOR PERSON
                # ============================================

                elif pipeline.waiting_for_person:

                    if scan_person is not None:

                        pipeline.associate_person(
                            scan_person
                        )

                    self._sync_medicine_live_status(
                        pipeline
                    )

                    target_delay = (
                        1.0
                        / self.medicine_scan_fps
                    )

                # ============================================
                # MEDICINE SCANNING
                #
                # DO NOT CHANGE.
                # ============================================

                else:

                    medicine_crop = (
                        self._extract_medicine_scan_region(
                            frame
                        )
                    )

                    (
                        observation,
                        confirmed,
                    ) = (
                        pipeline
                        .process_medicine_image(
                            medicine_image=(
                                medicine_crop
                            ),

                            person=(
                                scan_person
                            ),
                        )
                    )

                    self._sync_medicine_live_status(
                        pipeline
                    )

                    if (
                        observation.medicine_id
                        is not None
                    ):

                        logger.info(
                            (
                                "Live candidate: %s "
                                "source=%s "
                                "similarity=%.4f "
                                "dino=%.4f "
                                "ocr=%.4f "
                                "accepted=%s "
                                "frames=%s/%s"
                            ),
                            observation.medicine_name,
                            observation.recognition_source,
                            observation.similarity,
                            observation.dino_score,
                            observation.ocr_score,
                            observation.is_known,
                            observation.confirmation_count,
                            observation.confirmation_required,
                        )

                    if confirmed is not None:

                        logger.info(
                            (
                                "Live medicine package "
                                "recognized. "
                                "medicine_id=%s "
                                "medicine=%s "
                                "similarity=%.4f "
                                "person=%s"
                            ),
                            confirmed.medicine_id,
                            confirmed.medicine_name,
                            confirmed.similarity,
                            (
                                scan_person.display_name

                                if scan_person is not None

                                else "none"
                            ),
                        )

                    target_delay = (
                        1.0
                        / self.medicine_scan_fps
                    )

                with self._lock:

                    self._medicine_processed_frames += 1

            except Exception as exc:

                logger.exception(
                    "Medicine inference error."
                )

                with self._lock:

                    self._last_medicine_error = (
                        str(
                            exc
                        )
                    )

                target_delay = (
                    0.50
                )

            inference_seconds = (
                time.perf_counter()
                - loop_start
            )

            with self._lock:

                self._last_medicine_inference_seconds = (
                    inference_seconds
                )

            remaining = (
                target_delay
                - inference_seconds
            )

            if remaining > 0:

                time.sleep(
                    remaining
                )


    # ========================================================
    # PUBLISH MEDICINE EVENT
    # ========================================================

    def _publish_medicine_event(
        self,
        event: ConfirmedMedicineEvent,
    ) -> None:

        snapshot = (
            MedicineSnapshot(
                event_id=(
                    event.event_id
                ),

                medicine_id=(
                    event.medicine_id
                ),

                medicine_name=(
                    event.medicine_name
                ),

                dosage=(
                    event.dosage
                ),

                person_id=(
                    event.person_id
                ),

                person_name=(
                    event.person_name
                ),

                medicine_similarity=(
                    event.medicine_similarity
                ),

                person_similarity=(
                    event.person_similarity
                ),

                action_confidence=(
                    event.action_confidence
                ),

                overall_confidence=(
                    event.overall_confidence
                ),

                detected_at=(
                    event.observed_at
                ),

                image_path=(
                    event.evidence_image_path
                ),

                status=(
                    event.status
                ),
            )
        )

        with self._lock:

            self._latest_medicine = (
                snapshot
            )

            self._medicine_version += 1

            version = (
                self._medicine_version
            )

        logger.info(
            (
                "NiceGUI camera-confirmed medicine "
                "memory: "
                "medicine=%s "
                "medicine_id=%s "
                "person=%s "
                "event=%s "
                "confidence=%.4f "
                "version=%s"
            ),
            event.medicine_name,
            event.medicine_id,
            event.person_name,
            event.event_id,
            event.overall_confidence,
            version,
        )


    # ========================================================
    # PUBLISH PERSON
    # ========================================================

    def _publish_person_encounter(
        self,
        encounter: ConfirmedPersonEncounter,
    ) -> None:

        snapshot = (
            PersonSnapshot(
                person_id=(
                    encounter.person_id
                ),

                person_name=(
                    encounter.person_name
                ),

                display_name=(
                    encounter.display_name
                ),

                similarity=float(
                    encounter.similarity
                ),

                detected_at=(
                    encounter.detected_at
                ),

                image_path=(
                    encounter.image_path
                ),

                event_id=(
                    encounter.event_id
                ),
            )
        )

        with self._lock:

            self._latest_person = (
                snapshot
            )

            self._person_version += 1

            version = (
                self._person_version
            )

        logger.info(
            (
                "NiceGUI confirmed person memory: "
                "person=%s "
                "person_id=%s "
                "event=%s "
                "similarity=%.4f "
                "version=%s"
            ),
            encounter.display_name,
            encounter.person_id,
            encounter.event_id,
            encounter.similarity,
            version,
        )


    # ========================================================
    # DRAW OBJECT ANNOTATIONS
    # ========================================================

    @staticmethod
    def _draw_annotations(
        frame: np.ndarray,
        annotations: list[
            DisplayAnnotation
        ],
    ) -> np.ndarray:

        display_frame = (
            frame.copy()
        )

        for annotation in annotations:

            (
                x1,
                y1,
                x2,
                y2,
            ) = (
                annotation.bbox
            )

            cv2.rectangle(
                display_frame,
                (
                    x1,
                    y1,
                ),
                (
                    x2,
                    y2,
                ),
                (
                    0,
                    255,
                    0,
                ),
                2,
            )

            cv2.putText(
                display_frame,
                (
                    f"{annotation.label} "
                    f"{annotation.confidence:.2f}"
                ),
                (
                    x1,
                    max(
                        20,
                        y1 - 8,
                    ),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (
                    0,
                    255,
                    0,
                ),
                2,
            )

        return (
            display_frame
        )


    # ========================================================
    # DRAW FACE ANNOTATIONS
    # ========================================================

    @staticmethod
    def _draw_face_annotations(
        frame: np.ndarray,
        annotations: list[
            FaceDisplayAnnotation
        ],
    ) -> np.ndarray:

        display_frame = (
            frame.copy()
        )

        for annotation in annotations:

            (
                x,
                y,
                width,
                height,
            ) = (
                annotation.bbox
            )

            x2 = (
                x
                + width
            )

            y2 = (
                y
                + height
            )

            cv2.rectangle(
                display_frame,
                (
                    x,
                    y,
                ),
                (
                    x2,
                    y2,
                ),
                (
                    255,
                    255,
                    255,
                ),
                2,
            )

            label_text = (
                (
                    f"{annotation.label} "
                    f"{annotation.similarity:.2f}"
                )

                if annotation.is_known

                else "Unknown"
            )

            cv2.putText(
                display_frame,
                label_text,
                (
                    x,
                    max(
                        20,
                        y - 8,
                    ),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (
                    255,
                    255,
                    255,
                ),
                2,
            )

        return (
            display_frame
        )


    # ========================================================
    # DRAW MEDICINE BOX
    # ========================================================

    @classmethod
    def _draw_medicine_scan_region(
        cls,
        frame: np.ndarray,
    ) -> np.ndarray:

        display_frame = (
            frame.copy()
        )

        height, width = (
            display_frame.shape[
                :2
            ]
        )

        (
            x1,
            y1,
            x2,
            y2,
        ) = (
            cls.get_medicine_scan_bounds(
                frame_width=(
                    width
                ),

                frame_height=(
                    height
                ),
            )
        )

        cv2.rectangle(
            display_frame,
            (
                x1,
                y1,
            ),
            (
                x2,
                y2,
            ),
            (
                255,
                255,
                255,
            ),
            2,
        )

        cv2.putText(
            display_frame,
            (
                "MEDICINE ONLY - "
                "fill most of this box"
            ),
            (
                x1,
                max(
                    22,
                    y1 - 10,
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (
                255,
                255,
                255,
            ),
            1,
            cv2.LINE_AA,
        )

        return (
            display_frame
        )


    # ========================================================
    # STREAM FRAME
    # ========================================================

    def _get_stream_frame(
        self,
    ) -> np.ndarray | None:

        frame = (
            self.get_latest_frame_copy()
        )

        if frame is None:

            return None

        with self._lock:

            annotations = (
                list(
                    self._latest_annotations
                )
            )

            face_annotations = (
                list(
                    self._latest_face_annotations
                )
            )

            medicine_phase = (
                self._medicine_live_status.phase
            )

        if annotations:

            frame = (
                self._draw_annotations(
                    frame,
                    annotations,
                )
            )

        if face_annotations:

            frame = (
                self._draw_face_annotations(
                    frame,
                    face_annotations,
                )
            )

        if (
            self.enable_medicine_monitoring

            and medicine_phase
            not in {
                "watching_action",
                "event_saved",
            }
        ):

            frame = (
                self._draw_medicine_scan_region(
                    frame
                )
            )

        return (
            frame
        )


    # ========================================================
    # MJPEG
    # ========================================================

    def mjpeg_stream(
        self,
    ) -> Generator[
        bytes,
        None,
        None,
    ]:

        delay = (
            1.0
            / self.stream_fps
        )

        while True:

            current_state = (
                self.state
            )

            if current_state in (
                CameraState.STOPPED,
                CameraState.ERROR,
            ):

                time.sleep(
                    0.1
                )

                continue

            frame = (
                self._get_stream_frame()
            )

            if frame is None:

                time.sleep(
                    0.02
                )

                continue

            ok, encoded = (
                cv2.imencode(
                    ".jpg",
                    frame,
                    [
                        cv2.IMWRITE_JPEG_QUALITY,
                        80,
                    ],
                )
            )

            if not ok:

                time.sleep(
                    0.005
                )

                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Cache-Control: no-cache\r\n"
                b"Pragma: no-cache\r\n\r\n"
                + encoded.tobytes()
                + b"\r\n"
            )

            time.sleep(
                delay
            )


    # ========================================================
    # GET LATEST OBJECT
    # ========================================================

    def get_latest_detection(
        self,
    ) -> DetectionSnapshot | None:

        with self._lock:

            detection = (
                self._latest_detection
            )

            if detection is None:

                return None

            return (
                DetectionSnapshot(
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
            )


    # ========================================================
    # GET LATEST PERSON
    # ========================================================

    def get_latest_person(
        self,
    ) -> PersonSnapshot | None:

        with self._lock:

            person = (
                self._latest_person
            )

            if person is None:

                return None

            return (
                PersonSnapshot(
                    person_id=(
                        person.person_id
                    ),

                    person_name=(
                        person.person_name
                    ),

                    display_name=(
                        person.display_name
                    ),

                    similarity=(
                        person.similarity
                    ),

                    detected_at=(
                        person.detected_at
                    ),

                    image_path=(
                        person.image_path
                    ),

                    event_id=(
                        person.event_id
                    ),
                )
            )


    # ========================================================
    # GET LATEST MEDICINE
    # ========================================================

    def get_latest_medicine(
        self,
    ) -> MedicineSnapshot | None:

        with self._lock:

            medicine = (
                self._latest_medicine
            )

            if medicine is None:

                return None

            return (
                MedicineSnapshot(
                    event_id=(
                        medicine.event_id
                    ),

                    medicine_id=(
                        medicine.medicine_id
                    ),

                    medicine_name=(
                        medicine.medicine_name
                    ),

                    dosage=(
                        medicine.dosage
                    ),

                    person_id=(
                        medicine.person_id
                    ),

                    person_name=(
                        medicine.person_name
                    ),

                    medicine_similarity=(
                        medicine.medicine_similarity
                    ),

                    person_similarity=(
                        medicine.person_similarity
                    ),

                    action_confidence=(
                        medicine.action_confidence
                    ),

                    overall_confidence=(
                        medicine.overall_confidence
                    ),

                    detected_at=(
                        medicine.detected_at
                    ),

                    image_path=(
                        medicine.image_path
                    ),

                    status=(
                        medicine.status
                    ),
                )
            )


    # ========================================================
    # MEDICINE LIVE STATUS
    # ========================================================

    def get_medicine_live_status(
        self,
    ) -> MedicineLiveStatus:

        with self._lock:

            status = (
                self._medicine_live_status
            )

            return (
                MedicineLiveStatus(
                    phase=(
                        status.phase
                    ),

                    message=(
                        status.message
                    ),

                    medicine_name=(
                        status.medicine_name
                    ),

                    medicine_similarity=(
                        status.medicine_similarity
                    ),

                    medicine_threshold=(
                        status.medicine_threshold
                    ),

                    medicine_is_known=(
                        status.medicine_is_known
                    ),

                    confirmation_count=(
                        status.confirmation_count
                    ),

                    confirmation_required=(
                        status.confirmation_required
                    ),

                    person_name=(
                        status.person_name
                    ),

                    person_similarity=(
                        status.person_similarity
                    ),

                    action_confidence=(
                        status.action_confidence
                    ),
                )
            )


    # ========================================================
    # FACE ANNOTATIONS
    # ========================================================

    def get_face_annotations(
        self,
    ) -> list[
        FaceDisplayAnnotation
    ]:

        with self._lock:

            return (
                list(
                    self._latest_face_annotations
                )
            )


    # ========================================================
    # VERSIONS
    # ========================================================

    def get_detection_version(
        self,
    ) -> int:

        with self._lock:

            return (
                self._detection_version
            )


    def get_person_version(
        self,
    ) -> int:

        with self._lock:

            return (
                self._person_version
            )


    def get_medicine_version(
        self,
    ) -> int:

        with self._lock:

            return (
                self._medicine_version
            )


    # ========================================================
    # STATUS
    # ========================================================

    def get_status(
        self,
    ) -> dict:

        with self._lock:

            return {
                "state":
                    self._state.value,

                "captured_frames":
                    self._captured_frames,

                "processed_frames":
                    self._processed_frames,

                "face_processed_frames":
                    self._face_processed_frames,

                "medicine_processed_frames":
                    self._medicine_processed_frames,

                "inference_seconds":
                    self._last_inference_seconds,

                "face_inference_seconds":
                    self._last_face_inference_seconds,

                "medicine_inference_seconds":
                    self._last_medicine_inference_seconds,

                "last_error":
                    self._last_error,

                "last_face_error":
                    self._last_face_error,

                "last_medicine_error":
                    self._last_medicine_error,

                "detection_version":
                    self._detection_version,

                "person_version":
                    self._person_version,

                "medicine_version":
                    self._medicine_version,

                "medicine_phase":
                    self._medicine_live_status.phase,

                "medicine_message":
                    self._medicine_live_status.message,

                "medicine_candidate":
                    (
                        self
                        ._medicine_live_status
                        .medicine_name
                    ),

                "medicine_similarity":
                    (
                        self
                        ._medicine_live_status
                        .medicine_similarity
                    ),

                "medicine_threshold":
                    (
                        self
                        ._medicine_live_status
                        .medicine_threshold
                    ),

                "medicine_confirmation_count":
                    (
                        self
                        ._medicine_live_status
                        .confirmation_count
                    ),

                "medicine_confirmation_required":
                    (
                        self
                        ._medicine_live_status
                        .confirmation_required
                    ),
            }