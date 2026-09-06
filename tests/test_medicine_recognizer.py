from __future__ import annotations

from datetime import time

from pathlib import Path

from tempfile import TemporaryDirectory

from uuid import uuid4

import cv2
import numpy as np

from database.db_manager import (
    execute,
    initialize_database,
)

from database.medicine_repository import (
    add_medicine_reference_image,
    create_medicine,
)

from medicine.medicine_recognizer import (
    MedicineRecognitionEngine,
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
# CREATE TEST IMAGE
# ============================================================

def create_test_package_image(
    path: Path,
    text: str,
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

    cv2.rectangle(
        image,
        (
            60,
            80,
        ),
        (
            540,
            320,
        ),
        (
            40,
            40,
            40,
        ),
        4,
    )

    cv2.putText(
        image,
        text,
        (
            110,
            210,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.3,
        (
            0,
            0,
            0,
        ),
        3,
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
            "recognizer test image."
        )


# ============================================================
# CLEANUP
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

def test_medicine_recognizer() -> None:

    print(
        "\n"
        "MemorAi Medicine Recognizer Test Suite\n"
    )

    initialize_database()

    print(
        "Loading medicine visual model..."
    )

    recognizer = (
        MedicineRecognitionEngine()
    )

    print(
        "Medicine visual model loaded.\n"
    )

    medicine_id: int | None = None

    unique_suffix = (
        uuid4()
        .hex[:8]
    )

    medicine_name = (
        "Recognizer Test Medicine "
        f"{unique_suffix}"
    )

    try:

        with TemporaryDirectory() as temp_dir:

            temp_path = Path(
                temp_dir
            )

            image_path = (
                temp_path
                / "medicine.jpg"
            )

            create_test_package_image(
                image_path,
                "MEMORAI 500",
            )

            # ================================================
            # EMBEDDING
            # ================================================

            embedding = (
                recognizer
                .generate_embedding(
                    image_path
                )
            )

            assert isinstance(
                embedding,
                np.ndarray,
            )

            assert (
                embedding.ndim
                == 1
            )

            assert (
                embedding.size
                > 0
            )

            assert np.all(
                np.isfinite(
                    embedding
                )
            )

            print_test_result(
                "Generate medicine embedding",
                True,
            )

            # ================================================
            # NORMALIZATION
            # ================================================

            norm = float(
                np.linalg.norm(
                    embedding
                )
            )

            assert abs(
                norm - 1.0
            ) < 0.001

            print_test_result(
                "Medicine embedding normalization",
                True,
            )

            # ================================================
            # SERIALIZATION
            # ================================================

            serialized = (
                recognizer
                .serialize_embedding(
                    embedding
                )
            )

            restored = (
                recognizer
                .deserialize_embedding(
                    serialized
                )
            )

            assert (
                restored.shape
                == embedding.shape
            )

            assert np.allclose(
                restored,
                embedding,
                atol=1e-6,
            )

            print_test_result(
                "Embedding serialization",
                True,
            )

            # ================================================
            # SELF SIMILARITY
            # ================================================

            similarity = (
                recognizer
                .compare_embeddings(
                    embedding,
                    restored,
                )
            )

            assert (
                similarity
                > 0.999
            )

            print_test_result(
                "Embedding cosine similarity",
                True,
            )

            # ================================================
            # DATABASE MEDICINE
            # ================================================

            medicine_id = (
                create_medicine(
                    medicine_name=medicine_name,
                    dosage="500 mg",
                    schedule_time=time(
                        8,
                        0,
                    ),
                    schedule_label="Morning",
                )
            )

            reference_id = (
                add_medicine_reference_image(
                    medicine_id=medicine_id,
                    image_path=str(
                        image_path
                    ),
                    embedding=serialized,
                    embedding_model=(
                        recognizer
                        .model_name
                    ),
                    is_primary=True,
                )
            )

            assert (
                reference_id
                > 0
            )

            print_test_result(
                "Store visual embedding",
                True,
            )

            # ================================================
            # MATCH STORED MEDICINE
            # ================================================

            match = (
                recognizer
                .recognize_image(
                    image_path
                )
            )

            assert (
                match.is_known
            )

            assert (
                match.medicine_id
                == medicine_id
            )

            assert (
                match.medicine_name
                == medicine_name
            )

            assert (
                match.similarity
                > 0.999
            )

            print_test_result(
                "Recognize registered medicine",
                True,
            )

        print(
            "\n"
            "All medicine recognizer tests passed."
            "\n"
        )

    finally:

        cleanup_medicine(
            medicine_id
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    test_medicine_recognizer()