from __future__ import annotations

from uuid import uuid4

from database.db_manager import (
    initialize_database,
)
from database.person_repository import (
    create_person,
    deactivate_person,
    get_person_by_id,
    update_person,
)


def print_pass(
    name: str,
) -> None:

    print(
        f"PASS: {name}"
    )


def test_person_relation() -> None:

    print()
    print(
        "MemorAi Person Relation Test Suite"
    )

    print(
        "=" * 50
    )

    initialize_database()

    suffix = (
        uuid4().hex[:8]
    )

    test_name = (
        f"relation_test_{suffix}"
    )

    # --------------------------------------------------------
    # CREATE PERSON WITH RELATION
    # --------------------------------------------------------

    person_id = create_person(
        name=(
            test_name
        ),
        display_name=(
            "Relation Test Person"
        ),
        relation=(
            "Daughter"
        ),
        notes=(
            "Temporary relation test."
        ),
    )

    assert person_id > 0

    print_pass(
        "Person created with relation"
    )

    # --------------------------------------------------------
    # READ RELATION
    # --------------------------------------------------------

    person = get_person_by_id(
        person_id
    )

    assert person is not None

    assert (
        person[
            "relation"
        ]
        == "Daughter"
    )

    print(
        f"Stored relation: "
        f"{person['relation']}"
    )

    print_pass(
        "Relation retrieved from MySQL"
    )

    # --------------------------------------------------------
    # UPDATE RELATION
    # --------------------------------------------------------

    update_person(
        person_id=(
            person_id
        ),
        relation=(
            "Friend"
        ),
    )

    updated = get_person_by_id(
        person_id
    )

    assert updated is not None

    assert (
        updated[
            "relation"
        ]
        == "Friend"
    )

    print(
        f"Updated relation: "
        f"{updated['relation']}"
    )

    print_pass(
        "Relation update"
    )

    # --------------------------------------------------------
    # DEACTIVATE TEST PERSON
    # --------------------------------------------------------

    deactivate_person(
        person_id
    )

    print_pass(
        "Temporary identity deactivated"
    )

    print()
    print(
        "All MemorAi person relation "
        "tests passed."
    )


if __name__ == "__main__":

    test_person_relation()