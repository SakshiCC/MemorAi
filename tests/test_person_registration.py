from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from database.person_repository import (
    count_face_embeddings,
    deactivate_person,
    get_face_embeddings,
    get_person_by_id,
)
from face.person_registration import (
    PersonRegistrationService,
)


# ============================================================
# TEST CONFIGURATION
# ============================================================

TEST_SAMPLE_DIR = (
    Path("data")
    / "people"
    / "registration_test"
)

TEST_SAMPLE_PATHS = [
    TEST_SAMPLE_DIR
    / "sample1.jpeg",

    TEST_SAMPLE_DIR
    / "sample2.jpeg",

    TEST_SAMPLE_DIR
    / "sample3.jpeg",

    TEST_SAMPLE_DIR
    / "sample4.jpeg",

    TEST_SAMPLE_DIR
    / "sample5.jpeg",
]


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

    missing = [
        path
        for path in TEST_SAMPLE_PATHS
        if not path.exists()
    ]

    if missing:

        missing_text = "\n".join(
            str(path)
            for path in missing
        )

        raise FileNotFoundError(
            "Registration test images are missing:\n"
            f"{missing_text}"
        )

    print_pass(
        "Registration test images available"
    )


# ============================================================
# REGISTRATION TEST
# ============================================================

def test_person_registration() -> None:

    print()
    print(
        "MemorAi Person Registration Test Suite"
    )

    print("=" * 50)

    # --------------------------------------------------------
    # CHECK IMAGES
    # --------------------------------------------------------

    validate_test_images()

    # --------------------------------------------------------
    # CREATE UNIQUE TEST IDENTITY
    # --------------------------------------------------------

    suffix = uuid4().hex[:8]

    test_name = (
        f"registration_test_{suffix}"
    )

    display_name = (
        "Registration Test Person"
    )

    # --------------------------------------------------------
    # CREATE SERVICE
    # --------------------------------------------------------

    service = (
        PersonRegistrationService(
            required_samples=5
        )
    )

    print_pass(
        "Registration service initialization"
    )

    # --------------------------------------------------------
    # REGISTER
    # --------------------------------------------------------

    result = (
        service.register_person(
            name=test_name,
            display_name=display_name,
            sample_paths=(
                TEST_SAMPLE_PATHS
            ),
            notes=(
                "Temporary MemorAi "
                "registration test."
            ),
        )
    )

    assert result.person_id > 0

    assert (
        result.sample_count
        == 5
    )

    assert (
        result.display_name
        == display_name
    )

    print_pass(
        "Person registration"
    )

    # --------------------------------------------------------
    # DATABASE PERSON
    # --------------------------------------------------------

    person = get_person_by_id(
        result.person_id
    )

    assert person is not None

    assert bool(
        person["is_active"]
    )

    print_pass(
        "Registered person stored in MySQL"
    )

    # --------------------------------------------------------
    # EMBEDDINGS
    # --------------------------------------------------------

    embedding_count = (
        count_face_embeddings(
            result.person_id
        )
    )

    assert (
        embedding_count
        == 5
    )

    print_pass(
        "Five face embeddings stored"
    )

    embeddings = (
        get_face_embeddings(
            result.person_id
        )
    )

    assert len(
        embeddings
    ) == 5

    for embedding_row in embeddings:

        embedding = (
            embedding_row[
                "embedding"
            ]
        )

        assert (
            embedding.shape
            == (128,)
        )

    print_pass(
        "Stored embeddings are valid"
    )

    # --------------------------------------------------------
    # REGISTRATION IMAGES
    # --------------------------------------------------------

    assert len(
        result.samples
    ) == 5

    for sample in result.samples:

        assert (
            sample.stored_path.exists()
        )

        assert (
            sample.confidence > 0.0
        )

        print(
            "Stored sample:"
            f" {sample.stored_path}"
            f" | confidence="
            f"{sample.confidence:.4f}"
        )

    print_pass(
        "Registration images stored"
    )

    # --------------------------------------------------------
    # CLEANUP TEST IDENTITY
    # --------------------------------------------------------

    deactivate_person(
        result.person_id
    )

    person = get_person_by_id(
        result.person_id
    )

    assert person is not None

    assert not bool(
        person["is_active"]
    )

    print_pass(
        "Test identity deactivated"
    )

    # --------------------------------------------------------
    # FINISHED
    # --------------------------------------------------------

    print()
    print(
        "All MemorAi person registration "
        "tests passed."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    test_person_registration()