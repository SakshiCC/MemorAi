
from __future__ import annotations

from pathlib import Path

import requests

from config import BASE_DIR


# ============================================================
# PATHS
# ============================================================

FACE_MODEL_DIR = (
    BASE_DIR
    / "models"
    / "face"
)

FACE_MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# MODEL INFORMATION
# ============================================================

MODELS = {
    "yunet": {
        "filename": (
            "face_detection_yunet_2023mar.onnx"
        ),
        "url": (
            "https://github.com/opencv/opencv_zoo/"
            "raw/refs/heads/main/models/"
            "face_detection_yunet/"
            "face_detection_yunet_2023mar.onnx"
        ),
    },
    "sface": {
        "filename": (
            "face_recognition_sface_2021dec.onnx"
        ),
        "url": (
            "https://github.com/opencv/opencv_zoo/"
            "raw/refs/heads/main/models/"
            "face_recognition_sface/"
            "face_recognition_sface_2021dec.onnx"
        ),
    },
}


# ============================================================
# DOWNLOAD HELPER
# ============================================================

def download_file(
    url: str,
    destination: Path,
) -> None:

    if destination.exists():

        size_mb = (
            destination.stat().st_size
            / (1024 * 1024)
        )

        if size_mb > 0.1:

            print(
                f"Already available: "
                f"{destination.name}"
            )

            return

    print(
        f"Downloading "
        f"{destination.name}..."
    )

    response = requests.get(
        url,
        stream=True,
        timeout=120,
    )

    response.raise_for_status()

    with destination.open("wb") as file:

        for chunk in response.iter_content(
            chunk_size=1024 * 1024
        ):

            if chunk:
                file.write(chunk)

    print(
        f"Downloaded: "
        f"{destination}"
    )


# ============================================================
# MAIN DOWNLOAD
# ============================================================

def download_face_models() -> None:

    print()
    print(
        "MemorAi Face Model Downloader"
    )
    print("=" * 40)

    for model in MODELS.values():

        destination = (
            FACE_MODEL_DIR
            / model["filename"]
        )

        download_file(
            model["url"],
            destination,
        )

    print()
    print(
        "All face models are ready."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    download_face_models()