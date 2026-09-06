from __future__ import annotations

from dataclasses import dataclass

from pathlib import Path

import time

import cv2
import mediapipe as mp
import numpy as np

from config import (
    MEDICINE_HAND_DETECTION_CONFIDENCE,
    MEDICINE_HAND_MODEL_PATH,
    MEDICINE_HAND_PRESENCE_CONFIDENCE,
    MEDICINE_HAND_TRACKING_CONFIDENCE,
)


# ============================================================
# LANDMARK INDEXES
# ============================================================

WRIST = 0

THUMB_TIP = 4

INDEX_FINGER_TIP = 8

MIDDLE_FINGER_TIP = 12

RING_FINGER_TIP = 16

PINKY_TIP = 20


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class HandPoint:

    x: float

    y: float

    z: float


@dataclass
class HandObservation:
    """
    One detected hand.

    Coordinates x/y are normalized to [0, 1].
    """

    handedness: str

    handedness_score: float

    landmarks: list[
        HandPoint
    ]

    focus_x: float

    focus_y: float


# ============================================================
# HAND LANDMARK DETECTOR
# ============================================================

class HandLandmarkDetector:
    """
    MediaPipe Hand Landmarker wrapper.

    Uses VIDEO mode so MediaPipe can reuse tracking
    information between sequential frames.

    The detector returns normalized hand coordinates.

    It does not perform any medication-action reasoning.
    """

    def __init__(
        self,
        model_path: (
            str
            | Path
            | None
        ) = None,
        max_hands: int = 2,
        detection_confidence: float = (
            MEDICINE_HAND_DETECTION_CONFIDENCE
        ),
        presence_confidence: float = (
            MEDICINE_HAND_PRESENCE_CONFIDENCE
        ),
        tracking_confidence: float = (
            MEDICINE_HAND_TRACKING_CONFIDENCE
        ),
    ) -> None:

        self.model_path = Path(
            model_path
            or MEDICINE_HAND_MODEL_PATH
        )

        if not self.model_path.exists():

            raise FileNotFoundError(
                "MediaPipe Hand Landmarker model "
                "was not found:\n"
                f"{self.model_path}\n\n"
                "Run:\n"
                "python -m medicine.download_hand_model"
            )

        self.max_hands = max(
            1,
            int(
                max_hands
            ),
        )

        self.detection_confidence = (
            self._validate_probability(
                detection_confidence,
                "detection_confidence",
            )
        )

        self.presence_confidence = (
            self._validate_probability(
                presence_confidence,
                "presence_confidence",
            )
        )

        self.tracking_confidence = (
            self._validate_probability(
                tracking_confidence,
                "tracking_confidence",
            )
        )

        # ----------------------------------------------------
        # MEDIAPIPE OPTIONS
        # ----------------------------------------------------

        base_options = (
            mp.tasks.BaseOptions(
                model_asset_path=str(
                    self.model_path
                )
            )
        )

        options = (
            mp.tasks.vision.HandLandmarkerOptions(
                base_options=(
                    base_options
                ),
                running_mode=(
                    mp.tasks.vision.RunningMode.VIDEO
                ),
                num_hands=(
                    self.max_hands
                ),
                min_hand_detection_confidence=(
                    self.detection_confidence
                ),
                min_hand_presence_confidence=(
                    self.presence_confidence
                ),
                min_tracking_confidence=(
                    self.tracking_confidence
                ),
            )
        )

        self._landmarker = (
            mp.tasks.vision.HandLandmarker
            .create_from_options(
                options
            )
        )

        # ----------------------------------------------------
        # VIDEO TIMESTAMP
        # ----------------------------------------------------

        self._last_timestamp_ms = 0

    # ========================================================
    # VALIDATION
    # ========================================================

    @staticmethod
    def _validate_probability(
        value: float,
        field_name: str,
    ) -> float:

        number = float(
            value
        )

        if not (
            0.0
            <= number
            <= 1.0
        ):

            raise ValueError(
                f"{field_name} must be "
                "between 0.0 and 1.0."
            )

        return number

    # ========================================================
    # TIMESTAMP
    # ========================================================

    def _next_timestamp_ms(
        self,
    ) -> int:

        timestamp = int(
            time.monotonic()
            * 1000
        )

        # MediaPipe VIDEO mode requires increasing
        # timestamps.

        if (
            timestamp
            <= self._last_timestamp_ms
        ):

            timestamp = (
                self._last_timestamp_ms
                + 1
            )

        self._last_timestamp_ms = (
            timestamp
        )

        return timestamp

    # ========================================================
    # FOCUS POINT
    # ========================================================

    @staticmethod
    def _calculate_focus_point(
        landmarks: list[
            HandPoint
        ],
    ) -> tuple[
        float,
        float,
    ]:
        """
        Use finger tips as the functional hand point.

        For a medicine-taking action, finger tips are more
        useful than wrist position when measuring proximity
        to the mouth area.
        """

        required_indexes = (
            THUMB_TIP,
            INDEX_FINGER_TIP,
            MIDDLE_FINGER_TIP,
        )

        selected = [
            landmarks[
                index
            ]
            for index
            in required_indexes
            if index
            < len(
                landmarks
            )
        ]

        if not selected:

            raise ValueError(
                "Hand landmarks are incomplete."
            )

        focus_x = (
            sum(
                point.x
                for point
                in selected
            )
            / len(
                selected
            )
        )

        focus_y = (
            sum(
                point.y
                for point
                in selected
            )
            / len(
                selected
            )
        )

        return (
            float(
                focus_x
            ),
            float(
                focus_y
            ),
        )

    # ========================================================
    # DETECT
    # ========================================================

    def detect(
        self,
        frame: np.ndarray,
    ) -> list[
        HandObservation
    ]:

        if frame is None:

            raise ValueError(
                "Hand detection frame "
                "cannot be None."
            )

        if not isinstance(
            frame,
            np.ndarray,
        ):

            raise TypeError(
                "Hand detection frame must "
                "be a NumPy array."
            )

        if frame.size == 0:

            raise ValueError(
                "Hand detection frame "
                "cannot be empty."
            )

        if (
            frame.ndim != 3
            or frame.shape[2] != 3
        ):

            raise ValueError(
                "Expected a BGR color frame."
            )

        # ----------------------------------------------------
        # BGR -> RGB
        # ----------------------------------------------------

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        rgb = np.ascontiguousarray(
            rgb
        )

        mp_image = mp.Image(
            image_format=(
                mp.ImageFormat.SRGB
            ),
            data=rgb,
        )

        result = (
            self._landmarker
            .detect_for_video(
                mp_image,
                self._next_timestamp_ms(),
            )
        )

        observations: list[
            HandObservation
        ] = []

        hand_landmarks = (
            result.hand_landmarks
            or []
        )

        handedness = (
            result.handedness
            or []
        )

        for index, landmark_group in enumerate(
            hand_landmarks
        ):

            points: list[
                HandPoint
            ] = []

            for landmark in landmark_group:

                points.append(
                    HandPoint(
                        x=float(
                            landmark.x
                        ),
                        y=float(
                            landmark.y
                        ),
                        z=float(
                            landmark.z
                        ),
                    )
                )

            if not points:

                continue

            (
                focus_x,
                focus_y,
            ) = (
                self._calculate_focus_point(
                    points
                )
            )

            hand_label = (
                "Unknown"
            )

            hand_score = (
                0.0
            )

            if (
                index
                < len(
                    handedness
                )
                and handedness[
                    index
                ]
            ):

                category = (
                    handedness[
                        index
                    ][
                        0
                    ]
                )

                hand_label = str(
                    category.category_name
                    or "Unknown"
                )

                hand_score = float(
                    category.score
                    or 0.0
                )

            observations.append(
                HandObservation(
                    handedness=(
                        hand_label
                    ),
                    handedness_score=(
                        hand_score
                    ),
                    landmarks=(
                        points
                    ),
                    focus_x=(
                        focus_x
                    ),
                    focus_y=(
                        focus_y
                    ),
                )
            )

        return observations

    # ========================================================
    # CLOSE
    # ========================================================

    def close(
        self,
    ) -> None:

        landmarker = getattr(
            self,
            "_landmarker",
            None,
        )

        if landmarker is not None:

            try:

                landmarker.close()

            except Exception:

                pass

    # ========================================================
    # DESTRUCTOR
    # ========================================================

    def __del__(
        self,
    ) -> None:

        try:

            self.close()

        except Exception:

            pass