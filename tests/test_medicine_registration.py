from __future__ import annotations

from pathlib import Path

from shutil import rmtree

from tempfile import TemporaryDirectory

from datetime import time

from uuid import uuid4

import cv2
import numpy as np

from database.db_manager import (
    execute,
    initialize_database,
)

from database.medicine_repository import (
    get_medicine_by_id,
    get_medicine_reference_images,
    get_medicine_schedules,
)

from medicine.medicine_recognizer import (
    MedicineRecognitionEngine,
)

from medicine.medicine_registration import (
    MedicineRegistrationService,
)


# ============================================================
# TEST RESULT
# ============================================================

def print_test_result(
    test_name: str,
    passed: bool,
) -> None:

    status = (
        "PASS"
        if passed
        else "FAIL"
    )

    print(
        f"{status}: {test_name}"
    )


# ============================================================
# CREATE PACKAGE SAMPLE
# ============================================================

def create_package_sample(
    path: Path,
    sample_number: int,
) -> None:

    image = np.full(
        (
            400,
            600,
            3,
        ),
        255,
        dtype=np.uint8,
    )

    offset = (
        sample_number
        * 3
    )

    cv2.rectangle(
        image,
        (
            60 + offset,
            80,
        ),
        (
            540 - offset,
            320,
        ),
        (
            30,
            30,
            30,
        ),
        4,
    )

    cv2.putText(
        image,
        "MEMORAI TEST",
        (
            110,
            180,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (
            0,
            0,
            0,
        ),
        3,
        cv2.LINE_AA,
    )

    cv2.putText(
        image,
        "500 mg",
        (
            210,
            245,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (
            0,
            0,
            0,
        ),
        2,
        cv2.LINE_AA,
    )

    success = cv2.imwrite(
        str(
            path
        ),
        image,
    )

    if not success:

        raise RuntimeError(
            "Could not create medicine "
            "registration test image."
        )


# ============================================================
# DATABASE CLEANUP
# ============================================================

def cleanup_medicine(
    medicine_id: int | None,
) -> None:

    if medicine_id is None:

        return

    execute(
        """
        DELETE FROM medicines
        WHERE id = %s
        """,
        (
            medicine_id,
        ),
    )


# ============================================================
# TEST
# ============================================================

def test_medicine_registration() -> None:

    print(
        "\n"
        "MemorAi Medicine Registration Test Suite\n"
    )

    initialize_database()

    print(
        "Loading medicine visual model..."
    )

    recognizer = (
        MedicineRecognitionEngine()
    )

    registration = (
        MedicineRegistrationService(
            recognizer=recognizer,
            required_samples=5,
        )
    )

    print(
        "Medicine visual model loaded.\n"
    )

    medicine_id: int | None = None

    stored_directory: (
        Path
        | None
    ) = None

    unique_suffix = (
        uuid4()
        .hex[:8]
    )

    medicine_name = (
        "Registration Test Medicine "
        f"{unique_suffix}"
    )

    try:

        with TemporaryDirectory() as temp_dir:

            temp_path = Path(
                temp_dir
            )

            sample_paths: list[
                Path
            ] = []

            for index in range(
                1,
                6,
            ):

                sample_path = (
                    temp_path
                    / (
                        f"sample_"
                        f"{index:02d}.jpg"
                    )
                )

                create_package_sample(
                    sample_path,
                    index,
                )

                sample_paths.append(
                    sample_path
                )

            # ================================================
            # REGISTER
            # ================================================

            result = (
                registration
                .register_medicine(
                    medicine_name=medicine_name,
                    dosage="500 mg",
                    schedule_time=time(
                        8,
                        0,
                    ),
                    sample_paths=sample_paths,
                    schedule_label="Morning",
                    instructions=(
                        "Medicine registration test."
                    ),
                )
            )

            medicine_id = (
                result.medicine_id
            )

            assert (
                medicine_id
                > 0
            )

            assert (
                result.sample_count
                == 5
            )

            assert (
                len(
                    result.samples
                )
                == 5
            )

            print_test_result(
                "Register medicine with images",
                True,
            )

            # ================================================
            # STORED MEDICINE
            # ================================================

            medicine = (
                get_medicine_by_id(
                    medicine_id
                )
            )

            assert (
                medicine
                is not None
            )

            assert (
                medicine[
                    "medicine_name"
                ]
                == medicine_name
            )

            assert (
                medicine[
                    "dosage"
                ]
                == "500 mg"
            )

            print_test_result(
                "Medicine stored in MySQL",
                True,
            )

            # ================================================
            # REFERENCE IMAGES
            # ================================================

            references = (
                get_medicine_reference_images(
                    medicine_id
                )
            )

            assert (
                len(
                    references
                )
                == 5
            )

            assert all(
                row[
                    "embedding"
                ]
                is not None
                for row
                in references
            )

            assert all(
                row[
                    "embedding_model"
                ]
                == recognizer.model_name
                for row
                in references
            )

            print_test_result(
                "Store five visual embeddings",
                True,
            )

            # ================================================
            # PRIMARY IMAGE
            # ================================================

            primary = [
                row
                for row
                in references
                if bool(
                    row[
                        "is_primary"
                    ]
                )
            ]

            assert (
                len(
                    primary
                )
                == 1
            )

            print_test_result(
                "Primary medicine image",
                True,
            )

            # ================================================
            # FILES COPIED
            # ================================================

            for sample in result.samples:

                assert (
                    sample.stored_path
                    .exists()
                )

            stored_directory = (
                result.samples[
                    0
                ]
                .stored_path
                .parent
            )

            print_test_result(
                "Medicine registration images saved",
                True,
            )

            # ================================================
            # PRIMARY SCHEDULE
            # ================================================

            schedules = (
                get_medicine_schedules(
                    medicine_id,
                    active_only=True,
                )
            )

            assert (
                len(
                    schedules
                )
                >= 1
            )

            print_test_result(
                "Medicine schedule stored",
                True,
            )

            # ================================================
            # RECOGNIZE ONE REGISTERED SAMPLE
            # ================================================

            match = (
                recognizer
                .recognize_image(
                    result.samples[
                        0
                    ]
                    .stored_path
                )
            )

            assert (
                match.is_known
            )

            assert (
                match.medicine_id
                == medicine_id
            )

            print_test_result(
                "Recognize registered package",
                True,
            )

        print(
            "\n"
            "All medicine registration tests passed."
            "\n"
        )

    finally:

        cleanup_medicine(
            medicine_id
        )

        if (
            stored_directory
            is not None
            and stored_directory.exists()
        ):

            rmtree(
                stored_directory,
                ignore_errors=True,
            )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    test_medicine_registration()