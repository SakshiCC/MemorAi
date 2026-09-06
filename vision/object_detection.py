from __future__ import annotations

import cv2

from config import (
    CAMERA_INDEX,
    CONFIRMATION_FRAMES,
    DETECTION_COOLDOWN_SECONDS,
)

from database.db_manager import (
    initialize_database,
)

from utils.constants import EXIT_KEYS
from utils.logger import get_logger

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


logger = get_logger(
    "vision.pipeline",
    "vision.log",
)


# ============================================================
# DRAW DETECTION BOX
# ============================================================

def draw_detection(
    frame,
    label: str,
    confidence: float,
    bbox: tuple[int, int, int, int],
) -> None:

    x1, y1, x2, y2 = bbox

    # Bounding box
    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2,
    )

    # Label
    cv2.putText(
        frame,
        f"{label} {confidence:.2f}",
        (
            x1,
            max(
                20,
                y1 - 8,
            ),
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )


# ============================================================
# DRAW CENTER POINT
# ============================================================

def draw_center_point(
    frame,
    bbox: tuple[int, int, int, int],
) -> tuple[int, int]:

    x1, y1, x2, y2 = bbox

    center_x = int(
        (x1 + x2) / 2
    )

    center_y = int(
        (y1 + y2) / 2
    )

    cv2.circle(
        frame,
        (
            center_x,
            center_y,
        ),
        5,
        (0, 0, 255),
        -1,
    )

    return (
        center_x,
        center_y,
    )


# ============================================================
# MAIN WEBCAM PIPELINE
# ============================================================

def run_camera() -> None:

    # --------------------------------------------------------
    # DATABASE
    # --------------------------------------------------------

    initialize_database()


    # --------------------------------------------------------
    # YOLO MODEL
    # --------------------------------------------------------

    detector = ObjectDetector()


    # --------------------------------------------------------
    # DETECTION CONFIRMATION
    # --------------------------------------------------------

    confirmor = DetectionConfirmor(
        required_frames=CONFIRMATION_FRAMES,
        cooldown_seconds=DETECTION_COOLDOWN_SECONDS,
    )


    # --------------------------------------------------------
    # EVENT LOGGER
    # --------------------------------------------------------

    event_logger = EventLogger(
        index_in_vector_memory=True
    )


    # --------------------------------------------------------
    # OPEN CAMERA
    # --------------------------------------------------------

    camera = cv2.VideoCapture(
        CAMERA_INDEX
    )


    if not camera.isOpened():

        raise RuntimeError(
            (
                "Could not open webcam "
                f"at index {CAMERA_INDEX}."
            )
        )


    logger.info(
        "MemorAi webcam pipeline started."
    )


    try:

        while True:

            # ------------------------------------------------
            # READ FRAME
            # ------------------------------------------------

            ok, frame = camera.read()

            if not ok:

                logger.warning(
                    "Could not read webcam frame."
                )

                break


            frame_height, frame_width = (
                frame.shape[:2]
            )


            # ------------------------------------------------
            # YOLO DETECTION
            # ------------------------------------------------

            detections = detector.detect(
                frame
            )


            # ------------------------------------------------
            # DRAW ALL YOLO DETECTIONS
            # ------------------------------------------------

            for detection in detections:

                draw_detection(
                    frame,
                    detection.label,
                    detection.confidence,
                    detection.bbox,
                )

                draw_center_point(
                    frame,
                    detection.bbox,
                )


            # ------------------------------------------------
            # CONFIRM DETECTIONS
            # ------------------------------------------------

            confirmed_detections = (
                confirmor.update(
                    detections
                )
            )


            # ------------------------------------------------
            # STORE CONFIRMED DETECTIONS
            # ------------------------------------------------

            for confirmed in confirmed_detections:

                detection = (
                    confirmed.detection
                )


                # --------------------------------------------
                # LOCATION / REGION
                # --------------------------------------------

                location = (
                    map_bbox_to_location(
                        detection.bbox,
                        frame_width,
                        frame_height,
                    )
                )


                # --------------------------------------------
                # SAVE:
                #
                # object_events
                # +
                # events
                # +
                # ChromaDB
                # --------------------------------------------

                object_event_id, event_id = (
                    event_logger.log_object_detection(
                        frame=frame,
                        object_name=detection.label,
                        confidence=detection.confidence,
                        bbox=detection.bbox,
                        location=location,
                    )
                )


                logger.info(
                    (
                        "Confirmed detection stored: "
                        "object_event=%s "
                        "event=%s"
                    ),
                    object_event_id,
                    event_id,
                )


            # ------------------------------------------------
            # DISPLAY CAMERA
            # ------------------------------------------------

            cv2.imshow(
                "MemorAi Object Detection",
                frame,
            )


            # ------------------------------------------------
            # EXIT
            # ------------------------------------------------

            key = (
                cv2.waitKey(1)
                & 0xFF
            )


            if key in EXIT_KEYS:

                break


    finally:

        camera.release()

        cv2.destroyAllWindows()

        logger.info(
            "MemorAi webcam pipeline stopped."
        )


# ============================================================
# PROGRAM ENTRY
# ============================================================

if __name__ == "__main__":

    run_camera()