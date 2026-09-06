from database.db_manager import initialize_database
from database.event_repository import get_recent_events
from database.vector_memory import VectorMemory


def main() -> None:
    initialize_database()

    vector_memory = VectorMemory()

    # Change the limit if your database already contains more than 10,000 events.
    events = get_recent_events(limit=10_000)

    if not events:
        print("No MySQL events found to index.")
        return

    indexed = 0

    for event in reversed(events):
        vector_memory.add_event(event)
        indexed += 1

    print(f"Indexed {indexed} events into ChromaDB.")
    print(f"Vector collection count: {vector_memory.count()}")


if __name__ == "__main__":
    main()
