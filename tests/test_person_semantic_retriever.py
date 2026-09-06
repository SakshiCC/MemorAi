from __future__ import annotations

import uuid

from datetime import datetime

from database.person_vector_memory import (
    PersonVectorMemory,
)

from memory.person_retriever import (
    PersonMemoryRetriever,
)


# ============================================================
# OUTPUT
# ============================================================

def print_test_result(
    test_name: str,
) -> None:

    print(
        f"PASS: {test_name}"
    )


# ============================================================
# TEST DATA
# ============================================================

class SemanticPersonTestData:

    def __init__(
        self,
    ) -> None:

        suffix = (
            uuid.uuid4()
            .hex[:8]
        )

        numeric_suffix = int(
            suffix[:5],
            16,
        )

        self.person_id = (
            9_500_000
            + numeric_suffix
        )

        self.event_id = (
            8_500_000
            + numeric_suffix
        )

        self.person_name = (
            f"Semantic Visitor {suffix}"
        )

        self.relation = (
            "Family Friend"
        )

        self.detected_at = (
            datetime.now()
            .replace(
                microsecond=0
            )
        )

        # ----------------------------------------------------
        # UNIQUE SEMANTIC PHRASE
        # ----------------------------------------------------
        #
        # We deliberately use distinctive wording so the
        # temporary vector memory can be reliably found even
        # when the user's Chroma collection already contains
        # real person encounters.

        self.semantic_phrase = (
            "stopped by for a "
            "family afternoon visit"
        )

    # ========================================================
    # EVENT
    # ========================================================

    def event(
        self,
    ) -> dict:

        readable_time = (
            self.detected_at.strftime(
                "%d %b %Y at %I:%M %p"
            )
        )

        return {
            "id": (
                self.event_id
            ),

            "person_id": (
                self.person_id
            ),

            "person_name": (
                self.person_name
            ),

            "relation": (
                self.relation
            ),

            "similarity": (
                0.91
            ),

            "detected_at": (
                self.detected_at
            ),

            "image_path": (
                None
            ),

            "memory_text": (
                f"MemorAi remembers that "
                f"{self.person_name}, "
                f"the user's "
                f"{self.relation}, "
                f"{self.semantic_phrase} "
                f"on {readable_time}."
            ),
        }


# ============================================================
# EXACT VECTOR INSERTION TEST
# ============================================================

def test_temporary_memory_exists(
    vector_memory: PersonVectorMemory,
    data: SemanticPersonTestData,
) -> None:

    record = (
        vector_memory
        .get_person_event(
            data.event_id
        )
    )

    assert record is not None, (
        "Temporary person vector "
        "memory was not stored."
    )

    assert (
        record[
            "memory_id"
        ]
        == (
            f"person_event_"
            f"{data.event_id}"
        )
    )

    metadata = (
        record[
            "metadata"
        ]
    )

    assert (
        int(
            metadata[
                "person_id"
            ]
        )
        == data.person_id
    )

    assert (
        metadata[
            "person_name"
        ]
        == data.person_name
    )

    print_test_result(
        "Temporary semantic person memory"
    )


# ============================================================
# PERSON QUESTION DETECTION
# ============================================================

def test_semantic_person_question_detection(
    retriever: PersonMemoryRetriever,
) -> None:

    question = (
        "Do you remember someone "
        "who stopped by?"
    )

    assert (
        retriever.is_person_question(
            question
        )
        is True
    )

    print_test_result(
        "Semantic person question detection"
    )


# ============================================================
# DIRECT PERSON SEMANTIC SEARCH
# ============================================================

def test_direct_semantic_search(
    retriever: PersonMemoryRetriever,
    data: SemanticPersonTestData,
) -> None:
    """
    First prove that PersonMemorySearch itself can retrieve
    the temporary vector memory.
    """

    results = (
        retriever
        .semantic_search
        .search(
            query=(
                "family afternoon visit "
                "stopped by"
            ),
            top_k=20,
        )
    )

    assert results, (
        "Direct person semantic search "
        "returned no memories."
    )

    matching_results = [
        result
        for result in results
        if int(
            result.metadata.get(
                "person_event_id",
                -1,
            )
        )
        == data.event_id
    ]

    assert matching_results, (
        "PersonMemorySearch could not "
        "retrieve the temporary person "
        "vector memory."
    )

    result = (
        matching_results[
            0
        ]
    )

    assert (
        int(
            result.metadata[
                "person_id"
            ]
        )
        == data.person_id
    )

    print_test_result(
        "Direct person semantic search"
    )


# ============================================================
# RETRIEVER SEMANTIC FALLBACK
# ============================================================

def test_semantic_fallback(
    retriever: PersonMemoryRetriever,
    data: SemanticPersonTestData,
) -> None:
    """
    Now prove that PersonMemoryRetriever reaches its semantic
    fallback and returns the temporary Chroma memory.
    """

    question = (
        "Do you remember someone "
        "who stopped by for a "
        "family afternoon visit?"
    )

    assert (
        retriever.is_person_question(
            question
        )
        is True
    )

    results = (
        retriever.retrieve(
            question,
            top_k=20,
        )
    )

    assert results, (
        "PersonMemoryRetriever semantic "
        "fallback returned no memories."
    )

    matching_results = [
        result
        for result in results
        if int(
            result.get(
                "person_event_id",
                -1,
            )
        )
        == data.event_id
    ]

    assert matching_results, (
        "The temporary semantic person "
        "encounter was not returned by "
        "PersonMemoryRetriever."
    )

    result = (
        matching_results[
            0
        ]
    )

    assert (
        int(
            result[
                "person_id"
            ]
        )
        == data.person_id
    )

    assert (
        result[
            "person_name"
        ]
        == data.person_name
    )

    assert (
        result[
            "relation"
        ]
        == data.relation
    )

    assert (
        result[
            "memory_type"
        ]
        == "person"
    )

    assert (
        result[
            "event_type"
        ]
        == "person_encounter"
    )

    assert (
        result.get(
            "memory_text"
        )
    )

    assert (
        data.semantic_phrase
        in result[
            "memory_text"
        ]
    )

    print_test_result(
        "Person semantic Chroma fallback"
    )


# ============================================================
# STRUCTURED QUESTION REGRESSION
# ============================================================

def test_structured_question_regression(
    retriever: PersonMemoryRetriever,
) -> None:

    question = (
        "Who was the last person I saw?"
    )

    assert (
        retriever.is_person_question(
            question
        )
        is True
    )

    results = (
        retriever.retrieve(
            question,
            top_k=1,
        )
    )

    assert isinstance(
        results,
        list,
    )

    print_test_result(
        "Structured person retrieval regression"
    )


# ============================================================
# UNKNOWN PERSON PROTECTION
# ============================================================

def test_unknown_person_protection(
    retriever: PersonMemoryRetriever,
) -> None:

    unknown_name = (
        "PersonWhoDefinitelyDoesNotExist"
    )

    question = (
        "When did I last see "
        f"{unknown_name}?"
    )

    assert (
        retriever.is_person_question(
            question
        )
        is True
    )

    results = (
        retriever.retrieve(
            question
        )
    )

    assert (
        results
        == []
    ), (
        "Unknown proper name incorrectly "
        "matched another person's "
        "semantic memory."
    )

    print_test_result(
        "Unknown person semantic protection"
    )


# ============================================================
# OBJECT ROUTING REGRESSION
# ============================================================

def test_object_question_not_person_routed(
    retriever: PersonMemoryRetriever,
) -> None:

    question = (
        "Where is my cell phone?"
    )

    assert (
        retriever.is_person_question(
            question
        )
        is False
    )

    results = (
        retriever.retrieve(
            question
        )
    )

    assert (
        results
        == []
    )

    print_test_result(
        "Object question not person routed"
    )


# ============================================================
# CLEANUP
# ============================================================

def cleanup(
    vector_memory: PersonVectorMemory,
    data: SemanticPersonTestData,
) -> None:

    try:

        vector_memory.delete_person_event(
            data.event_id
        )

    except Exception:

        pass


# ============================================================
# RUN
# ============================================================

def test_person_semantic_retriever(
) -> None:

    print(
        "\nMemorAi Person Semantic "
        "Retriever Test Suite\n"
    )

    vector_memory = (
        PersonVectorMemory()
    )

    data = (
        SemanticPersonTestData()
    )

    try:

        # ----------------------------------------------------
        # INSERT TEMPORARY VECTOR MEMORY
        # ----------------------------------------------------

        vector_memory.add_person_event(
            data.event()
        )

        # ----------------------------------------------------
        # VERIFY EXACT INSERTION
        # ----------------------------------------------------

        test_temporary_memory_exists(
            vector_memory,
            data,
        )

        # ----------------------------------------------------
        # CREATE RETRIEVER
        # ----------------------------------------------------

        retriever = (
            PersonMemoryRetriever()
        )

        # ----------------------------------------------------
        # RUN TESTS
        # ----------------------------------------------------

        test_semantic_person_question_detection(
            retriever
        )

        test_direct_semantic_search(
            retriever,
            data,
        )

        test_semantic_fallback(
            retriever,
            data,
        )

        test_structured_question_regression(
            retriever
        )

        test_unknown_person_protection(
            retriever
        )

        test_object_question_not_person_routed(
            retriever
        )

        print(
            "\nAll MemorAi person semantic "
            "retriever tests passed."
        )

    finally:

        cleanup(
            vector_memory,
            data,
        )

        print(
            "\nTemporary semantic person "
            "memory cleaned."
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    test_person_semantic_retriever()