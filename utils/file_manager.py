from __future__ import annotations

from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from config import EVIDENCE_DIR


def ensure_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_evidence_image(
    frame: np.ndarray,
    object_name: str,
    directory: str | Path = EVIDENCE_DIR,
) -> str:
    """Save one evidence frame and return its absolute path."""
    target_dir = ensure_directory(directory)

    safe_name = "".join(
        char if char.isalnum() else "_"
        for char in object_name.strip().lower()
    ).strip("_") or "object"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_path = target_dir / f"{safe_name}_{timestamp}.jpg"

    if not cv2.imwrite(str(output_path), frame):
        raise IOError(f"Could not save evidence image: {output_path}")

    return str(output_path.resolve())
