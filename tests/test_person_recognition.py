from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import cv2

from config import (
    FACE_RECOGNITION_COSINE_THRESHOLD,
)
from database.person_repository import (
    deactivate_person,
    get_person_by_id,
)
from face.face_recognition import (
    DetectedFace,
    FaceMatch,
    FaceRecognitionEngine,
)
from face.person_registration import (
    PersonRegistrationService,
)


# ============================================================
# TEST PATHS
# ============================================================

TEST_DATA_DIR = (
    Path("data")
    / "people"
    / "registration_test"
)


# ------------------------------------------------------------
# REGISTRATION IMAGES
# ------------------------------------------------------------

REGISTRATION_IMAGES = [
    TEST_DATA_DIR / "sample1.jpeg",
    TEST_DATA_DIR / "sample2.jpeg",
    TEST_DATA_DIR / "sample3.jpeg",
    TEST_DATA_DIR / "sample4.jpeg",
    TEST_DATA_DIR / "sample5.jpeg",
]


# ------------------------------------------------------------
# UNSEEN IMAGES OF SAME PERSON
# ------------------------------------------------------------

KNOWN_TEST_IMAGES = [
    TEST_DATA_DIR / "sample6.jpeg",
    TEST_DATA_DIR / "sample7.jpeg",
]


# ------------------------------------------------------------
# DIFFERENT PERSON
# ------------------------------------------------------------

UNKNOWN_TEST_IMAGE = (
    TEST_DATA_DIR
    / "unknown_person.jpeg"
)


# ============================================================
# HELPERS
# ============================================================

def print_pass(
    test_name: str,
) -> None:

    print(
        f"PASS: {test_name}"
    )


def validate_test_images() -> None:
    """
    Make sure all required test images exist.
    """

    required_images = (
        REGISTRATION_IMAGES
        + KNOWN_TEST_IMAGES
        + [
            UNKNOWN_TEST_IMAGE,
        ]
    )

    missing_images = [
        path
        for path in required_images
        if not path.exists()
    ]

    if missing_images:

        missing_text = "\n".join(
            str(path)
            for path in missing_images
        )

        raise FileNotFoundError(
            "The following recognition test "
            "images are missing:\n"
            f"{missing_text}"
        )

    print_pass(
        "Recognition test images available"
    )


# ============================================================
# LOAD IMAGE
# ============================================================

def load_test_image(
    image_path: Path,
):
    """
    Load one image using OpenCV.
    """

    image = cv2.imread(
        str(
            image_path
        )
    )

    if image is None:

        raise RuntimeError(
            "Could not read test image: "
            f"{image_path}"
        )

    return image


# ============================================================
# EXTRACT SINGLE FACE
# ============================================================

def extract_single_face(
    engine: FaceRecognitionEngine,
    image_path: Path,
) -> DetectedFace:
    """
    Detect exactly one face and generate its embedding.
    """

    image = load_test_image(
        image_path
    )

    faces = engine.detect_faces(
        image
    )

    if not faces:

        raise AssertionError(
            "No face detected in test image: "
            f"{image_path}"
        )

    if len(faces) != 1:

        raise AssertionError(
            "Expected exactly one face in "
            f"{image_path}, but detected "
            f"{len(faces)}."
        )

    face = faces[0]

    embedding = (
        engine.generate_embedding(
            image,
            face,
        )
    )

    assert embedding is not None

    assert embedding.shape == (128,)

    return face


# ============================================================
# RECOGNIZE IMAGE
# ============================================================

def recognize_image(
    engine: FaceRecognitionEngine,
    image_path: Path,
) -> FaceMatch:
    """
    Run complete recognition for one image.
    """

    face = extract_single_face(
        engine,
        image_path,
    )

    result = engine.recognize_face(
        face
    )

    return result


# ============================================================
# KNOWN PERSON TEST
# ============================================================

def test_known_person(
    engine: FaceRecognitionEngine,
    image_path: Path,
    expected_person_id: int,
    expected_display_name: str,
) -> float:
    """
    Test an unseen photo of the registered person.
    """

    result = recognize_image(
        engine,
        image_path,
    )

    print()
    print(
        f"Known-person test: "
        f"{image_path.name}"
    )

    print(
        f"Recognized as: "
        f"{result.display_name}"
    )

    print(
        f"Similarity: "
        f"{result.similarity:.6f}"
    )

    print(
        f"Known: "
        f"{result.is_known}"
    )

    assert result.is_known, (
        f"{image_path.name} should have been "
        "recognized as the registered person."
    )

    assert result.person_id == (
        expected_person_id
    ), (
        "Face matched the wrong person."
    )

    assert result.display_name == (
        expected_display_name
    )

    print_pass(
        f"Known recognition - "
        f"{image_path.name}"
    )

    return float(
        result.similarity
    )


# ============================================================
# UNKNOWN PERSON TEST
# ============================================================

def test_unknown_person(
    engine: FaceRecognitionEngine,
    image_path: Path,
    registered_person_id: int,
) -> float:
    """
    A different person should not be accepted as
    the registered identity.
    """

    result = recognize_image(
        engine,
        image_path,
    )

    print()
    print(
        f"Unknown-person test: "
        f"{image_path.name}"
    )

    print(
        f"Recognition result: "
        f"{result.display_name}"
    )

    print(
        f"Best similarity: "
        f"{result.similarity:.6f}"
    )

    print(
        f"Known: "
        f"{result.is_known}"
    )

    assert not (
        result.is_known
        and result.person_id
        == registered_person_id
    ), (
        "Different person was incorrectly "
        "recognized as the registered person. "
        "The recognition threshold may need "
        "calibration."
    )

    assert not result.is_known, (
        "Different person was classified as "
        "a known identity."
    )

    assert result.person_id is None

    assert (
        result.display_name
        == "Unknown"
    )

    print_pass(
        "Unknown person rejection"
    )

    return float(
        result.similarity
    )


# ============================================================
# SIMILARITY SEPARATION TEST
# ============================================================

def test_similarity_separation(
    known_similarities: list[float],
    unknown_similarity: float,
) -> None:
    """
    The same person's images should be more similar
    than an unrelated person's image.

    This provides useful information for later
    threshold calibration.
    """

    assert known_similarities

    minimum_known = min(
        known_similarities
    )

    average_known = (
        sum(
            known_similarities
        )
        / len(
            known_similarities
        )
    )

    print()
    print(
        "Similarity analysis"
    )

    print(
        "-" * 50
    )

    print(
        f"Configured threshold: "
        f"{FACE_RECOGNITION_COSINE_THRESHOLD:.6f}"
    )

    print(
        f"Minimum known similarity: "
        f"{minimum_known:.6f}"
    )

    print(
        f"Average known similarity: "
        f"{average_known:.6f}"
    )

    print(
        f"Unknown similarity: "
        f"{unknown_similarity:.6f}"
    )

    separation = (
        minimum_known
        - unknown_similarity
    )

    print(
        f"Similarity separation: "
        f"{separation:.6f}"
    )

    assert average_known > (
        unknown_similarity
    ), (
        "Known-person similarity should be "
        "higher than the unrelated person's "
        "similarity."
    )

    print_pass(
        "Known/unknown similarity separation"
    )


# ============================================================
# FULL RECOGNITION TEST
# ============================================================

def test_person_recognition() -> None:

    print()
    print(
        "MemorAi Person Recognition Test Suite"
    )

    print(
        "=" * 50
    )

    # --------------------------------------------------------
    # VERIFY IMAGES
    # --------------------------------------------------------

    validate_test_images()

    # --------------------------------------------------------
    # INITIALIZE FACE ENGINE
    # --------------------------------------------------------

    engine = (
        FaceRecognitionEngine()
    )

    print_pass(
        "Face recognition engine initialization"
    )

    print(
        f"Recognition threshold: "
        f"{engine.recognition_threshold:.6f}"
    )

    # --------------------------------------------------------
    # CREATE UNIQUE TEMPORARY IDENTITY
    # --------------------------------------------------------

    suffix = (
        uuid4().hex[:8]
    )

    internal_name = (
        f"recognition_test_{suffix}"
    )

    display_name = (
        "Recognition Test Person"
    )

    # --------------------------------------------------------
    # REGISTRATION SERVICE
    # --------------------------------------------------------

    registration_service = (
        PersonRegistrationService(
            face_engine=engine,
            required_samples=5,
        )
    )

    # --------------------------------------------------------
    # REGISTER USING SAMPLE 1-5
    # --------------------------------------------------------

    registration_result = (
        registration_service.register_person(
            name=internal_name,
            display_name=display_name,
            sample_paths=REGISTRATION_IMAGES,
            notes=(
                "Temporary identity for "
                "MemorAi recognition testing."
            ),
        )
    )

    person_id = (
        registration_result.person_id
    )

    print()
    print(
        "Temporary test identity registered."
    )

    print(
        f"Person ID: {person_id}"
    )

    print(
        f"Display name: "
        f"{display_name}"
    )

    print(
        f"Registration samples: "
        f"{registration_result.sample_count}"
    )

    print_pass(
        "Temporary recognition identity registration"
    )

    try:

        # ----------------------------------------------------
        # CONFIRM ACTIVE
        # ----------------------------------------------------

        person = get_person_by_id(
            person_id
        )

        assert person is not None

        assert bool(
            person["is_active"]
        )

        print_pass(
            "Recognition identity active"
        )

        # ----------------------------------------------------
        # TEST SAMPLE 6
        # ----------------------------------------------------

        known_similarity_1 = (
            test_known_person(
                engine=engine,
                image_path=(
                    KNOWN_TEST_IMAGES[0]
                ),
                expected_person_id=(
                    person_id
                ),
                expected_display_name=(
                    display_name
                ),
            )
        )

        # ----------------------------------------------------
        # TEST SAMPLE 7
        # ----------------------------------------------------

        known_similarity_2 = (
            test_known_person(
                engine=engine,
                image_path=(
                    KNOWN_TEST_IMAGES[1]
                ),
                expected_person_id=(
                    person_id
                ),
                expected_display_name=(
                    display_name
                ),
            )
        )

        # ----------------------------------------------------
        # TEST UNRELATED FACE
        # ----------------------------------------------------

        unknown_similarity = (
            test_unknown_person(
                engine=engine,
                image_path=(
                    UNKNOWN_TEST_IMAGE
                ),
                registered_person_id=(
                    person_id
                ),
            )
        )

        # ----------------------------------------------------
        # ANALYZE SCORES
        # ----------------------------------------------------

        test_similarity_separation(
            known_similarities=[
                known_similarity_1,
                known_similarity_2,
            ],
            unknown_similarity=(
                unknown_similarity
            ),
        )

    finally:

        # ----------------------------------------------------
        # ALWAYS DEACTIVATE TEMPORARY IDENTITY
        # ----------------------------------------------------

        deactivate_person(
            person_id
        )

        person = get_person_by_id(
            person_id
        )

        assert person is not None

        assert not bool(
            person["is_active"]
        )

        print()
        print_pass(
            "Temporary identity deactivated"
        )

    # --------------------------------------------------------
    # FINISHED
    # --------------------------------------------------------

    print()
    print(
        "All MemorAi person recognition "
        "tests passed."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    test_person_recognition()