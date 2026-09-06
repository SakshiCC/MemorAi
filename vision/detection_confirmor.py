from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from time import monotonic

from vision.object_detector import Detection


@dataclass(frozen=True)
class ConfirmedDetection:
    detection: Detection
    consecutive_frames: int


class DetectionConfirmor:
    """
    Confirm a label only after it appears in several consecutive frames,
    and suppress repeated logging for a configurable cooldown period.
    """

    def __init__(
        self,
        required_frames: int = 4,
        cooldown_seconds: float = 8.0,
    ) -> None:
        self.required_frames = max(1, int(required_frames))
        self.cooldown_seconds = max(0.0, float(cooldown_seconds))

        self._counts: dict[str, int] = defaultdict(int)
        self._last_detection: dict[str, Detection] = {}
        self._last_logged_at: dict[str, float] = {}

    def update(
        self,
        detections: list[Detection],
    ) -> list[ConfirmedDetection]:
        current_labels = {detection.label for detection in detections}

        # Reset labels that disappeared in this frame.
        for label in list(self._counts):
            if label not in current_labels:
                self._counts[label] = 0

        # Keep the highest-confidence detection for each label.
        best_by_label: dict[str, Detection] = {}

        for detection in detections:
            previous = best_by_label.get(detection.label)

            if (
                previous is None
                or detection.confidence > previous.confidence
            ):
                best_by_label[detection.label] = detection

        confirmed: list[ConfirmedDetection] = []
        now = monotonic()

        for label, detection in best_by_label.items():
            self._counts[label] += 1
            self._last_detection[label] = detection

            if self._counts[label] < self.required_frames:
                continue

            last_logged = self._last_logged_at.get(label)

            if (
                last_logged is not None
                and now - last_logged < self.cooldown_seconds
            ):
                continue

            self._last_logged_at[label] = now

            confirmed.append(
                ConfirmedDetection(
                    detection=detection,
                    consecutive_frames=self._counts[label],
                )
            )

        return confirmed
