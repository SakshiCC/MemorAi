from __future__ import annotations

from pathlib import Path

import cv2

from medicine.medicine_image_processor import (
    MedicineImageProcessor,
)


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR = (
    Path(
        "tests"
    )
    / "output"
    / "medicine_preprocessing"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CAMERA INDEX
# ============================================================

CAMERA_INDEX = 0


# ============================================================
# MEDICINE SCAN REGION
# ============================================================

def get_medicine_scan_bounds(
    frame_width: int,
    frame_height: int,
) -> tuple[
    int,
    int,
    int,
    int,
]:
    """
    Same scan region currently used by CameraService.

    Width:
        28% -> 72%

    Height:
        42% -> 78%
    """

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


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print()
    print(
        "MemorAi Medicine Image Processing Test"
    )
    print(
        "======================================"
    )
    print()
    print(
        "Place the medicine package inside "
        "the white medicine box."
    )
    print()
    print(
        "Press C to capture and preprocess."
    )
    print(
        "Press Q to quit."
    )
    print()

    camera = cv2.VideoCapture(
        CAMERA_INDEX,
        cv2.CAP_DSHOW,
    )

    if not camera.isOpened():

        camera.release()

        camera = cv2.VideoCapture(
            CAMERA_INDEX
        )

    if not camera.isOpened():

        raise RuntimeError(
            "Could not open webcam."
        )

    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        640,
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        480,
    )

    processor = (
        MedicineImageProcessor(
            upscale_factor=2.0
        )
    )

    try:

        while True:

            ok, frame = (
                camera.read()
            )

            if not ok:

                continue

            height, width = (
                frame.shape[:2]
            )

            (
                x1,
                y1,
                x2,
                y2,
            ) = (
                get_medicine_scan_bounds(
                    frame_width=width,
                    frame_height=height,
                )
            )

            display = (
                frame.copy()
            )

            cv2.rectangle(
                display,
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
                display,
                (
                    "MEDICINE ONLY - "
                    "fill most of this box"
                ),
                (
                    x1,
                    max(
                        25,
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

            cv2.putText(
                display,
                "C = capture | Q = quit",
                (
                    20,
                    35,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (
                    255,
                    255,
                    255,
                ),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(
                (
                    "MemorAi Medicine "
                    "Image Processing Test"
                ),
                display,
            )

            key = (
                cv2.waitKey(
                    1
                )
                & 0xFF
            )

            # ================================================
            # QUIT
            # ================================================

            if key in (
                ord(
                    "q"
                ),
                ord(
                    "Q"
                ),
            ):

                break

            # ================================================
            # CAPTURE
            # ================================================

            if key in (
                ord(
                    "c"
                ),
                ord(
                    "C"
                ),
            ):

                crop = (
                    frame[
                        y1:y2,
                        x1:x2,
                    ]
                    .copy()
                )

                if crop.size == 0:

                    print(
                        "ERROR: medicine crop is empty."
                    )

                    continue

                print()
                print(
                    "Processing medicine crop..."
                )

                processed = (
                    processor.process(
                        crop
                    )
                )

                outputs = {

                    "01_original.jpg":
                        processed.original,

                    "02_enlarged.jpg":
                        processed.enlarged,

                    "03_gray.jpg":
                        processed.gray,

                    "04_clahe.jpg":
                        processed.clahe,

                    "05_denoised.jpg":
                        processed.denoised,

                    "06_sharpened.jpg":
                        processed.sharpened,

                    "07_adaptive_threshold.jpg":
                        processed.adaptive_threshold,

                    "08_otsu_threshold.jpg":
                        processed.otsu_threshold,

                    "09_inverted_threshold.jpg":
                        processed.inverted_threshold,
                }

                for (
                    filename,
                    image,
                ) in outputs.items():

                    output_path = (
                        OUTPUT_DIR
                        / filename
                    )

                    success = (
                        cv2.imwrite(
                            str(
                                output_path
                            ),
                            image,
                        )
                    )

                    if not success:

                        raise RuntimeError(
                            (
                                "Could not save "
                                f"{output_path}"
                            )
                        )

                print(
                    "PASS: preprocessing completed."
                )

                print()
                print(
                    "Images saved to:"
                )

                print(
                    OUTPUT_DIR.resolve()
                )

                print()
                print(
                    "Generated:"
                )

                for filename in outputs:

                    print(
                        f"  {filename}"
                    )

                print()

                # --------------------------------------------
                # SHOW IMPORTANT OUTPUTS
                # --------------------------------------------

                cv2.imshow(
                    "1 - Original Crop",
                    processed.original,
                )

                cv2.imshow(
                    "2 - Enlarged",
                    processed.enlarged,
                )

                cv2.imshow(
                    "3 - CLAHE",
                    processed.clahe,
                )

                cv2.imshow(
                    "4 - Sharpened",
                    processed.sharpened,
                )

                cv2.imshow(
                    "5 - Adaptive Threshold",
                    processed.adaptive_threshold,
                )

                cv2.imshow(
                    "6 - Otsu Threshold",
                    processed.otsu_threshold,
                )

    finally:

        camera.release()

        cv2.destroyAllWindows()

        print()
        print(
            "Medicine preprocessing test stopped."
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()