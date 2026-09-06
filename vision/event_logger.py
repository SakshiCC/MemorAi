from __future__ import annotations

from datetime import datetime

import numpy as np

from database.event_repository import (
    get_event_by_id,
    save_event,
    save_object_event,
)

from database.vector_memory import VectorMemory

from utils.file_manager import save_evidence_image
from utils.logger import get_logger


logger = get_logger(
    "vision.event_logger",
    "vision.log",
)


class EventLogger:

    def __init__(
        self,
        index_in_vector_memory: bool = True,
    ) -> None:

        if index_in_vector_memory:

            self.vector_memory = VectorMemory()

        else:

            self.vector_memory = None


    # ========================================================
    # LOG CONFIRMED YOLO OBJECT
    # ========================================================

    def log_object_detection(
        self,
        frame: np.ndarray,
        object_name: str,
        confidence: float,
        bbox: tuple[int, int, int, int],
        location: str,
    ) -> tuple[int, int]:
        """
        Save one confirmed object detection.

        Two database records are created:

        1. object_events
           Detailed computer-vision record.

        2. events
           High-level memory record.

        The high-level event is then indexed into ChromaDB.

        Returns:
            (object_event_id, event_id)
        """

        # ----------------------------------------------------
        # TIMESTAMP
        # ----------------------------------------------------

        detected_at = datetime.now()


        # ----------------------------------------------------
        # SAVE ONE EVIDENCE IMAGE
        # ----------------------------------------------------

        image_path = save_evidence_image(
            frame,
            object_name,
        )


        # ----------------------------------------------------
        # BUILD HUMAN-READABLE MEMORY
        # ----------------------------------------------------

        memory_text = (
            f"MemorAi observed {object_name} "
            f"in the {location} "
            f"at "
            f"{detected_at.strftime('%Y-%m-%d %H:%M:%S')}."
        )


        # ----------------------------------------------------
        # 1. SAVE DETAILED VISION RECORD
        # ----------------------------------------------------

        object_event_id = save_object_event(
            object_name=object_name,
            confidence=confidence,
            bbox=bbox,
            region=location,
            detected_at=detected_at,
            image_path=image_path,
            memory_text=memory_text,
        )


        logger.info(
            (
                "Saved object_event id=%s "
                "object=%s "
                "confidence=%.3f "
                "bbox=%s "
                "region=%s"
            ),
            object_event_id,
            object_name,
            confidence,
            bbox,
            location,
        )


        # ----------------------------------------------------
        # 2. SAVE HIGH-LEVEL MEMORY EVENT
        # ----------------------------------------------------

        event_id = save_event(
            object_name=object_name,
            confidence=confidence,
            location=location,
            detected_at=detected_at,
            image_path=image_path,
            memory_text=memory_text,
        )


        logger.info(
            (
                "Saved memory event id=%s "
                "object=%s "
                "confidence=%.3f "
                "location=%s"
            ),
            event_id,
            object_name,
            confidence,
            location,
        )


        # ----------------------------------------------------
        # 3. INDEX HIGH-LEVEL EVENT INTO CHROMADB
        # ----------------------------------------------------

        if self.vector_memory is not None:

            event = get_event_by_id(
                event_id
            )

            if event:

                try:

                    self.vector_memory.add_event(
                        event
                    )

                    logger.info(
                        (
                            "Indexed event id=%s "
                            "into ChromaDB."
                        ),
                        event_id,
                    )

                except Exception:

                    # MySQL remains authoritative.
                    #
                    # If vector indexing fails, we can later run:
                    #
                    # python -m database.index_existing_memories

                    logger.exception(
                        (
                            "Event %s was saved to MySQL "
                            "but failed ChromaDB indexing."
                        ),
                        event_id,
                    )


        return (
            object_event_id,
            event_id,
        )