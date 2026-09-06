from database.vector_memory import VectorMemory


def main() -> None:
    memory = VectorMemory()

    memory.add_memory(
        memory_id="manual_test_1",
        text="MemorAi observed a bottle in the center area.",
        metadata={
            "object_name": "bottle",
            "location": "center area",
            "detected_at": "2026-08-07T12:00:00",
            "event_type": "test",
        },
    )

    results = memory.search(
        "Where was the bottle?",
        n_results=3,
    )

    documents = results.get("documents", [[]])[0]

    assert documents, "No vector search result was returned."
    assert "bottle" in documents[0].lower()

    print("Vector memory test passed.")
    print(documents[0])


if __name__ == "__main__":
    main()
