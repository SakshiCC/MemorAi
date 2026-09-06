from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO

from config import IMPORTANT_OBJECTS, YOLO_CONFIDENCE, YOLO_MODEL_PATH
from utils.helpers import normalize_object_name
from utils.logger import get_logger

logger = get_logger("vision.detector", "vision.log")


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]


class ObjectDetector:
    def __init__(
        self,
        model_path: str = str(YOLO_MODEL_PATH),
        confidence: float = YOLO_CONFIDENCE,
        important_objects: set[str] | None = None,
    ) -> None:
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.important_objects = (
            {normalize_object_name(x) for x in important_objects}
            if important_objects is not None
            else set(IMPORTANT_OBJECTS)
        )

        logger.info("YOLO model loaded from %s", model_path)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            verbose=False,
        )

        detections: list[Detection] = []

        for result in results:
            if result.boxes is None:
                continue

            names = result.names

            for box in result.boxes:
                class_id = int(box.cls.item())
                raw_label = str(names[class_id])
                label = normalize_object_name(raw_label)

                if (
                    self.important_objects
                    and label not in self.important_objects
                ):
                    continue

                confidence = float(box.conf.item())
                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0].tolist(),
                )

                detections.append(
                    Detection(
                        label=label,
                        confidence=confidence,
                        bbox=(x1, y1, x2, y2),
                    )
                )

        return detections
