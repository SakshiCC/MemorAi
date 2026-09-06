from __future__ import annotations

from datetime import datetime
from threading import RLock

import numpy as np

from config import (
    CONFIRMATION_FRAMES,
    DETECTION_COOLDOWN_SECONDS,
)

from database.db_manager import (
    initialize_database,
)

from utils.logger import (
    get_logger,
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

from vision.object_detection import (
    draw_center_point,
    draw_detection,
)

from vision.object_detector import (
    ObjectDetector,
)

from app.ui.camera_monitor import (
    FrameProcessingResult,
    LatestDetection,
)


logger = get_logger(
    "ui.vision_adapter",
    "vision.log",
)


# ============================================================
# VISION ADAPTER
# ============================================================


class MemorAiVisionAdapter:
    """
    Connect the Streamlit CameraMonitor to the existing
    MemorAi computer-vision and memory pipeline.

    Existing pipeline reused:

        Camera frame
            ↓
        ObjectDetector
            ↓
        Important-object filtering
            ↓
        DetectionConfirmor
            ↓
        Location mapping
            ↓
        EventLogger
            ↓
        object_events
            +
        events
            +
        ChromaDB

    This adapter does NOT open the webcam.

    CameraMonitor owns the webcam.

    This adapter only processes the frame received from
    CameraMonitor.
    """

    def __init__(
        self,
        index_in_vector_memory: bool = True,
    ) -> None:

        # ----------------------------------------------------
        # Lock
        # ----------------------------------------------------

        self._lock = RLock()

        # ----------------------------------------------------
        # Initialize database
        # ----------------------------------------------------

        logger.info(
            "Initializing MemorAi vision adapter."
        )

        initialize_database()

        # ----------------------------------------------------
        # YOLO detector
        # ----------------------------------------------------

        self.detector = ObjectDetector()

        # ----------------------------------------------------
        # Multi-frame confirmation
        # ----------------------------------------------------

        self.confirmor = DetectionConfirmor(
            required_frames=CONFIRMATION_FRAMES,
            cooldown_seconds=DETECTION_COOLDOWN_SECONDS,
        )

        # ----------------------------------------------------
        # Event logger
        # ----------------------------------------------------

        self.event_logger = EventLogger(
            index_in_vector_memory=(
                index_in_vector_memory
            )
        )

        # ----------------------------------------------------
        # Latest confirmed detection
        # ----------------------------------------------------

        self._latest_confirmed_detection: (
            LatestDetection | None
        ) = None

        logger.info(
            "MemorAi vision adapter initialized."
        )


    # ========================================================
    # PROCESS ONE CAMERA FRAME
    # ========================================================

    def process_frame(
        self,
        frame: np.ndarray,
    ) -> FrameProcessingResult:
        """
        Process one frame using the existing MemorAi pipeline.

        Returns
        -------
        FrameProcessingResult

        annotated_frame:
            Frame with bounding boxes and center points.

        latest_detection:
            Latest confirmed important detection.
            None if the current frame did not produce a
            newly confirmed event.
        """

        if frame is None:

            raise ValueError(
                "Vision adapter received an empty frame."
            )

        if frame.size == 0:

            raise ValueError(
                "Vision adapter received an invalid frame."
            )

        # CameraMonitor owns the original frame.
        # Work on a copy.
        annotated_frame = frame.copy()

        frame_height, frame_width = (
            annotated_frame.shape[:2]
        )

        # ----------------------------------------------------
        # YOLO DETECTION
        # ----------------------------------------------------

        detections = self.detector.detect(
            annotated_frame
        )

        # ----------------------------------------------------
        # DRAW CURRENT DETECTIONS
        # ----------------------------------------------------

        for detection in detections:

            draw_detection(
                annotated_frame,
                detection.label,
                detection.confidence,
                detection.bbox,
            )

            draw_center_point(
                annotated_frame,
                detection.bbox,
            )

        # ----------------------------------------------------
        # MULTI-FRAME CONFIRMATION
        # ----------------------------------------------------

        confirmed_detections = (
            self.confirmor.update(
                detections
            )
        )

        # Latest newly confirmed detection generated
        # during this frame.
        latest_for_frame = None

        # ----------------------------------------------------
        # STORE CONFIRMED DETECTIONS
        # ----------------------------------------------------

        for confirmed in confirmed_detections:

            detection = (
                confirmed.detection
            )

            # ------------------------------------------------
            # LOCATION
            # ------------------------------------------------

            location = map_bbox_to_location(
                detection.bbox,
                frame_width,
                frame_height,
            )

            # ------------------------------------------------
            # STORE MEMORY
            #
            # EventLogger already handles:
            #
            # object_events
            # events
            # evidence image
            # vector indexing
            # ------------------------------------------------

            object_event_id, event_id = (
                self.event_logger.log_object_detection(
                    frame=annotated_frame,
                    object_name=detection.label,
                    confidence=detection.confidence,
                    bbox=detection.bbox,
                    location=location,
                )
            )

            # ------------------------------------------------
            # UI latest-detection object
            # ------------------------------------------------

            ui_detection = LatestDetection(
                object_name=detection.label,
                confidence=detection.confidence,
                location=location,
                detected_at=datetime.now(),
            )

            latest_for_frame = ui_detection

            with self._lock:

                self._latest_confirmed_detection = (
                    ui_detection
                )

            logger.info(
                (
                    "Streamlit confirmed detection stored: "
                    "object=%s "
                    "confidence=%.3f "
                    "location=%s "
                    "object_event=%s "
                    "event=%s"
                ),
                detection.label,
                detection.confidence,
                location,
                object_event_id,
                event_id,
            )

        # ----------------------------------------------------
        # RETURN FRAME TO CAMERA MONITOR
        # ----------------------------------------------------

        return FrameProcessingResult(
            annotated_frame=annotated_frame,
            latest_detection=latest_for_frame,
        )


    # ========================================================
    # GET LAST CONFIRMED DETECTION
    # ========================================================

    def get_latest_confirmed_detection(
        self,
    ) -> LatestDetection | None:
        """
        Return latest confirmed detection even if the current
        frame did not create a new memory event.
        """

        with self._lock:

            detection = (
                self._latest_confirmed_detection
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
    # RESET CONFIRMATION STATE
    # ========================================================

    def reset_confirmation_state(
        self,
    ) -> None:
        """
        Reset DetectionConfirmor if that class exposes a reset
        method.

        Useful later when monitoring is intentionally reset.
        """

        reset_method = getattr(
            self.confirmor,
            "reset",
            None,
        )

        if callable(reset_method):

            reset_method()

            logger.info(
                "Detection confirmation state reset."
            )