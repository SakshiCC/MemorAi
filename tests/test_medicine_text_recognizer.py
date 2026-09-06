from __future__ import annotations

import time

import cv2

from medicine.medicine_text_recognizer import (
    MedicineTextRecognizer,
)


# ============================================================
# CAMERA
# ============================================================

CAMERA_INDEX = 0


# ============================================================
# TEST MEDICINE
# ============================================================

MEDICINE_NAME = "omez"

DOSAGE = "20mg"

# This text was visibly present on your real Omez foil.
GENERIC_NAME = "omeprazole"


# ============================================================
# MEDICINE REGION
# ============================================================

def medicine_region(
    frame,
) -> tuple[
    int,
    int,
    int,
    int,
]:

    height, width = (
        frame.shape[:2]
    )

    x1 = int(
        width
        * 0.28
    )

    x2 = int(
        width
        * 0.72
    )

    y1 = int(
        height
        * 0.42
    )

    y2 = int(
        height
        * 0.78
    )

    return (
        x1,
        y1,
        x2,
        y2,
    )


# ============================================================
# PRINT RESULT
# ============================================================

def print_result(
    result,
    total_time: float,
) -> None:

    print()
    print(
        "=" * 60
    )

    print(
        "OCR RESULT"
    )

    print(
        "=" * 60
    )

    if not result.ocr_items:

        print(
            "No readable OCR text detected."
        )

    else:

        print()
        print(
            "Detected OCR items:"
        )

        for item in result.ocr_items:

            print(
                (
                    f"  [{item.source_variant}] "
                    f"{item.text!r} "
                    f"confidence="
                    f"{item.confidence:.3f}"
                )
            )

    print()

    print(
        "Combined detected text:"
    )

    print(
        (
            result.detected_text
            if result.detected_text
            else "(none)"
        )
    )

    print()

    print(
        (
            "Name score:       "
            f"{result.name_score:.3f}"
        )
    )

    print(
        (
            "Dosage score:     "
            f"{result.dosage_score:.3f}"
        )
    )

    print(
        (
            "Combined score:   "
            f"{result.combined_score:.3f}"
        )
    )

    print(
        (
            "Matched alias:    "
            f"{result.matched_alias}"
        )
    )

    print(
        (
            "OCR processing:   "
            f"{result.processing_seconds:.3f}s"
        )
    )

    print(
        (
            "Total test time:  "
            f"{total_time:.3f}s"
        )
    )

    print()

    if (
        result.has_strong_name_evidence
    ):

        print(
            "RESULT: STRONG MEDICINE TEXT MATCH ✓"
        )

    elif (
        result.has_name_evidence
    ):

        print(
            "RESULT: MEDICINE TEXT MATCH ✓"
        )

    else:

        print(
            "RESULT: TEXT MATCH NOT STRONG ENOUGH"
        )

    print(
        "=" * 60
    )

    print()


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print()
    print(
        "MemorAi EasyOCR Medicine Test"
    )

    print(
        "=============================="
    )

    print()

    print(
        "Expected:"
    )

    print(
        f"Medicine: {MEDICINE_NAME}"
    )

    print(
        f"Generic:  {GENERIC_NAME}"
    )

    print(
        f"Dosage:   {DOSAGE}"
    )

    print()

    print(
        "Loading EasyOCR..."
    )

    print(
        "The first run may download OCR model files."
    )

    print()

    recognizer = (
        MedicineTextRecognizer(
            gpu=False
        )
    )

    print(
        "EasyOCR ready."
    )

    print()

    print(
        "Instructions:"
    )

    print(
        "1. Place the package inside the white box."
    )

    print(
        "2. Press C to perform OCR."
    )

    print(
        "3. Test Omez first."
    )

    print(
        "4. Then put Feviquick in the SAME box."
    )

    print(
        "5. Press C again."
    )

    print(
        "6. Press Q to quit."
    )

    print()

    camera = (
        cv2.VideoCapture(
            CAMERA_INDEX,
            cv2.CAP_DSHOW,
        )
    )

    if not camera.isOpened():

        camera.release()

        camera = (
            cv2.VideoCapture(
                CAMERA_INDEX
            )
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

    try:

        while True:

            ok, frame = (
                camera.read()
            )

            if not ok:

                continue

            (
                x1,
                y1,
                x2,
                y2,
            ) = (
                medicine_region(
                    frame
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
                    "FILL MOST OF THIS BOX"
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
                "C = OCR | Q = quit",
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
                "MemorAi Medicine OCR Test",
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
            # OCR
            # ================================================

            if key not in (
                ord(
                    "c"
                ),
                ord(
                    "C"
                ),
            ):

                continue

            crop = (
                frame[
                    y1:y2,
                    x1:x2,
                ]
                .copy()
            )

            if crop.size == 0:

                print(
                    "Medicine crop is empty."
                )

                continue

            print()
            print(
                "Running OCR..."
            )

            started = (
                time.perf_counter()
            )

            result = (
                recognizer.match_medicine(
                    image=(
                        crop
                    ),

                    medicine_name=(
                        MEDICINE_NAME
                    ),

                    dosage=(
                        DOSAGE
                    ),

                    generic_name=(
                        GENERIC_NAME
                    ),
                )
            )

            total_time = (
                time.perf_counter()
                - started
            )

            print_result(
                result=(
                    result
                ),

                total_time=(
                    total_time
                ),
            )

    finally:

        camera.release()

        cv2.destroyAllWindows()

        print()
        print(
            "Medicine OCR test stopped."
        )


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":

    main()