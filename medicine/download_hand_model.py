from __future__ import annotations

from pathlib import Path

import requests

from config import (
    MEDICINE_HAND_MODEL_PATH,
)


# ============================================================
# MODEL URL
# ============================================================

HAND_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/"
    "mediapipe-models/"
    "hand_landmarker/"
    "hand_landmarker/"
    "float16/"
    "latest/"
    "hand_landmarker.task"
)


# ============================================================
# DOWNLOAD
# ============================================================

def download_hand_landmarker_model(
    destination: Path | None = None,
) -> Path:

    if destination is None:

        destination = (
            MEDICINE_HAND_MODEL_PATH
        )

    destination = Path(
        destination
    )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # ALREADY AVAILABLE
    # --------------------------------------------------------

    if (
        destination.exists()
        and destination.is_file()
        and destination.stat().st_size
        > 100_000
    ):

        print(
            "Hand Landmarker model already available:"
        )

        print(
            destination
        )

        return destination

    print(
        "Downloading MediaPipe Hand Landmarker model..."
    )

    response = requests.get(
        HAND_LANDMARKER_MODEL_URL,
        stream=True,
        timeout=120,
    )

    response.raise_for_status()

    temporary_path = (
        destination.with_suffix(
            destination.suffix
            + ".download"
        )
    )

    try:

        with temporary_path.open(
            "wb"
        ) as file:

            for chunk in response.iter_content(
                chunk_size=1024 * 1024
            ):

                if chunk:

                    file.write(
                        chunk
                    )

        if (
            not temporary_path.exists()
            or temporary_path.stat().st_size
            <= 100_000
        ):

            raise RuntimeError(
                "Downloaded Hand Landmarker "
                "model appears to be invalid."
            )

        temporary_path.replace(
            destination
        )

    finally:

        if temporary_path.exists():

            try:

                temporary_path.unlink()

            except OSError:

                pass

    print(
        "Hand Landmarker model downloaded:"
    )

    print(
        destination
    )

    return destination


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    download_hand_landmarker_model()