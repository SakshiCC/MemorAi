from __future__ import annotations

from datetime import (
    datetime,
    timedelta,
)

import uuid

from database.conversation_vector_memory import (
    ConversationVectorMemory,
)


# ============================================================
# HELPER
# ============================================================

def print_test_result(
    test_name: str,
) -> None:

    print(
        f"PASS: {test_name}"
    )


# ============================================================
# MAIN TEST
# ============================================================

def test_conversation_vector_memory() -> None:

    print(
        (
            "\nMemorAi Conversation "
            "Vector Memory Test Suite\n"
        )
    )

    vector_memory = (
        ConversationVectorMemory()
    )

    # ========================================================
    # UNIQUE IDS
    # ========================================================

    unique_base = int(
        uuid.uuid4().int
        % 1_000_000_000
    )

    appointment_id = (
        unique_base
    )

    shopping_id = (
        unique_base
        + 1
    )

    created_ids = [
        appointment_id,
        shopping_id,
    ]

    now = (
        datetime.now()
        .replace(
            microsecond=0
        )
    )

    try:

        # ====================================================
        # APPOINTMENT CONVERSATION
        # ====================================================

        appointment_conversation = {
            "id": (
                appointment_id
            ),

            "title": (
                "Doctor appointment"
            ),

            "participant_text": (
                "Sakshi"
            ),

            "full_transcript": (
                "Sakshi told me that my doctor "
                "appointment is on Monday at 10 AM. "
                "I need to bring my medical reports "
                "and arrive fifteen minutes early."
            ),

            "summary": (
                "Doctor appointment is Monday at "
                "10 AM. Medical reports should be "
                "brought and arrival should be "
                "fifteen minutes early."
            ),

            "status": (
                "completed"
            ),

            "started_at": (
                now
            ),

            "ended_at": (
                now
                + timedelta(
                    minutes=2
                )
            ),
        }

        vector_memory.add_conversation(
            conversation=(
                appointment_conversation
            ),

            importance_score=(
                1.0
            ),

            matched_categories=(
                "appointment",
                "schedule",
                "task",
            ),
        )

        stored = (
            vector_memory
            .get_conversation(
                appointment_id
            )
        )

        assert (
            stored
            is not None
        )

        assert (
            stored[
                "memory_id"
            ]
            == (
                "conversation_"
                f"{appointment_id}"
            )
        )

        assert (
            "doctor appointment"
            in stored[
                "text"
            ].lower()
        )

        assert (
            stored[
                "metadata"
            ][
                "memory_type"
            ]
            == "conversation"
        )

        assert (
            int(
                stored[
                    "metadata"
                ][
                    "conversation_id"
                ]
            )
            == appointment_id
        )

        assert (
            stored[
                "metadata"
            ][
                "participant_text"
            ]
            == "Sakshi"
        )

        print_test_result(
            (
                "Important conversation "
                "added to vector memory"
            )
        )

        # ====================================================
        # SHOPPING CONVERSATION
        # ====================================================

        shopping_conversation = {
            "id": (
                shopping_id
            ),

            "title": (
                "Shopping reminder"
            ),

            "participant_text": (
                "Rahul"
            ),

            "full_transcript": (
                "Rahul reminded me to buy milk "
                "and bread this evening before "
                "the shops close."
            ),

            "summary": (
                "Buy milk and bread "
                "this evening."
            ),

            "status": (
                "completed"
            ),

            "started_at": (
                now
                + timedelta(
                    minutes=5
                )
            ),

            "ended_at": (
                now
                + timedelta(
                    minutes=6
                )
            ),
        }

        vector_memory.add_conversation(
            conversation=(
                shopping_conversation
            ),

            importance_score=(
                0.82
            ),

            matched_categories=(
                "reminder",
                "task",
                "schedule",
            ),
        )

        print_test_result(
            (
                "Second conversation "
                "added to vector memory"
            )
        )

        # ====================================================
        # SEMANTIC APPOINTMENT SEARCH
        # ====================================================

        appointment_search = (
            vector_memory.search(
                query=(
                    "What do I need to bring "
                    "to my doctor appointment?"
                ),
                n_results=5,
            )
        )

        appointment_ids = (
            appointment_search.get(
                "ids",
                [[]],
            )[0]
        )

        assert (
            (
                "conversation_"
                f"{appointment_id}"
            )
            in appointment_ids
        )

        print_test_result(
            (
                "Semantic doctor "
                "appointment search"
            )
        )

        # ====================================================
        # SEMANTIC SHOPPING SEARCH
        # ====================================================

        shopping_search = (
            vector_memory.search(
                query=(
                    "What groceries was I "
                    "reminded to purchase?"
                ),
                n_results=5,
            )
        )

        shopping_ids = (
            shopping_search.get(
                "ids",
                [[]],
            )[0]
        )

        assert (
            (
                "conversation_"
                f"{shopping_id}"
            )
            in shopping_ids
        )

        print_test_result(
            (
                "Semantic shopping "
                "conversation search"
            )
        )

        # ====================================================
        # PARTICIPANT FILTER
        # ====================================================

        sakshi_search = (
            vector_memory.search(
                query=(
                    "What was discussed?"
                ),
                n_results=5,
                participant_text=(
                    "Sakshi"
                ),
            )
        )

        sakshi_ids = (
            sakshi_search.get(
                "ids",
                [[]],
            )[0]
        )

        assert (
            (
                "conversation_"
                f"{appointment_id}"
            )
            in sakshi_ids
        )

        assert (
            (
                "conversation_"
                f"{shopping_id}"
            )
            not in sakshi_ids
        )

        print_test_result(
            "Conversation participant filtering"
        )

        # ====================================================
        # METADATA
        # ====================================================

        metadata = (
            stored[
                "metadata"
            ]
        )

        assert (
            float(
                metadata[
                    "importance_score"
                ]
            )
            == 1.0
        )

        assert (
            "appointment"
            in str(
                metadata[
                    "importance_categories"
                ]
            )
        )

        assert (
            metadata[
                "event_type"
            ]
            == "important_conversation"
        )

        print_test_result(
            "Conversation vector metadata"
        )

        # ====================================================
        # REJECT INCOMPLETE CONVERSATION
        # ====================================================

        incomplete = {
            "id": (
                unique_base
                + 100
            ),

            "status": (
                "recording"
            ),

            "full_transcript": (
                "This should not be indexed."
            ),
        }

        try:

            vector_memory.add_conversation(
                incomplete
            )

        except ValueError:

            print_test_result(
                (
                    "Incomplete conversation "
                    "vector protection"
                )
            )

        else:

            raise AssertionError(
                (
                    "An incomplete conversation "
                    "was incorrectly indexed."
                )
            )

        # ====================================================
        # EMPTY SEARCH
        # ====================================================

        try:

            vector_memory.search(
                query=""
            )

        except ValueError:

            print_test_result(
                "Empty conversation search protection"
            )

        else:

            raise AssertionError(
                (
                    "Empty conversation query "
                    "was incorrectly accepted."
                )
            )

    finally:

        # ====================================================
        # CLEANUP
        # ====================================================

        for conversation_id in (
            created_ids
        ):

            try:

                vector_memory.delete_conversation(
                    conversation_id
                )

            except Exception:

                pass

    print(
        (
            "\nAll MemorAi Conversation "
            "Vector Memory tests passed."
        )
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    test_conversation_vector_memory()