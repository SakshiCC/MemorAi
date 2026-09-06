from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from face.face_recognition import (
    FaceRecognitionEngine,
)


# ============================================================
# TEST IMAGE
# ============================================================

TEST_IMAGE_PATH = (
    Path("data")
    / "temp"
    / "face_test.jpeg"
)


# ============================================================
# HELPERS
# ============================================================

def print_pass(
    name: str,
) -> None:

    print(
        f"PASS: {name}"
    )


# ============================================================
# MODEL INITIALIZATION
# ============================================================

def test_model_initialization(
    engine: FaceRecognitionEngine,
) -> None:

    assert engine.detector is not None

    assert engine.recognizer is not None

    print_pass(
        "YuNet and SFace initialization"
    )


# ============================================================
# FACE DETECTION
# ============================================================

def test_face_detection(
    engine: FaceRecognitionEngine,
    image: np.ndarray,
):

    faces = engine.detect_faces(
        image
    )

    assert faces, (
    "No face was detected in "
    "face_test.jpeg."
)

    print(
        f"Detected faces: {len(faces)}"
    )

    for index, face in enumerate(
        faces,
        start=1,
    ):

        print(
            f"Face {index}: "
            f"bbox={face.bbox}, "
            f"confidence="
            f"{face.confidence:.4f}"
        )

    print_pass(
        "YuNet face detection"
    )

    return faces


# ============================================================
# EMBEDDING
# ============================================================

def test_embedding_generation(
    engine: FaceRecognitionEngine,
    image: np.ndarray,
    face,
) -> np.ndarray:

    embedding = (
        engine.generate_embedding(
            image,
            face,
        )
    )

    assert isinstance(
        embedding,
        np.ndarray,
    )

    assert embedding.ndim == 1

    assert embedding.size > 0

    assert np.all(
        np.isfinite(
            embedding
        )
    )

    print(
        f"Embedding dimension: "
        f"{embedding.size}"
    )

    print(
        "First 5 embedding values:"
    )

    print(
        embedding[:5]
    )

    print_pass(
        "SFace embedding generation"
    )

    return embedding


# ============================================================
# SELF SIMILARITY
# ============================================================

def test_self_similarity(
    engine: FaceRecognitionEngine,
    embedding: np.ndarray,
) -> None:

    similarity = (
        engine.compare_embeddings(
            embedding,
            embedding,
        )
    )

    print(
        f"Self similarity: "
        f"{similarity:.6f}"
    )

    assert similarity > 0.99

    print_pass(
        "SFace similarity comparison"
    )


# ============================================================
# FULL TEST
# ============================================================

def test_face_recognition() -> None:

    print()
    print(
        "MemorAi Face Recognition Test Suite"
    )

    print("=" * 50)

    # --------------------------------------------------------
    # TEST IMAGE
    # --------------------------------------------------------

    if not TEST_IMAGE_PATH.exists():

        raise FileNotFoundError(
            "\nTest image not found.\n\n"
            "Please place a clear face photo at:\n"
            f"{TEST_IMAGE_PATH}\n"
        )

    image = cv2.imread(
        str(
            TEST_IMAGE_PATH
        )
    )

    if image is None:

        raise RuntimeError(
            "Could not read test face image."
        )

    print(
        f"Image size: "
        f"{image.shape[1]}x"
        f"{image.shape[0]}"
    )

    # --------------------------------------------------------
    # ENGINE
    # --------------------------------------------------------

    engine = (
        FaceRecognitionEngine()
    )

    test_model_initialization(
        engine
    )

    # --------------------------------------------------------
    # FACE DETECTION
    # --------------------------------------------------------

    faces = test_face_detection(
        engine,
        image,
    )

    # --------------------------------------------------------
    # USE MOST CONFIDENT FACE
    # --------------------------------------------------------

    best_face = max(
        faces,
        key=lambda face: (
            face.confidence
        ),
    )

    # --------------------------------------------------------
    # EMBEDDING
    # --------------------------------------------------------

    embedding = (
        test_embedding_generation(
            engine,
            image,
            best_face,
        )
    )

    # --------------------------------------------------------
    # SIMILARITY
    # --------------------------------------------------------

    test_self_similarity(
        engine,
        embedding,
    )

    print()
    print(
        "All MemorAi face recognition "
        "tests passed."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    test_face_recognition()