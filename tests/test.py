from database.db_manager import initialize_database
from database.event_repository import get_recent_events
from database.vector_memory import VectorMemory


def main() -> None:
    initialize_database()

    vector_memory = VectorMemory()

    print("MemorAi basic integration check")
    print(f"Recent MySQL events: {len(get_recent_events(limit=5))}")
    print(f"ChromaDB memories: {vector_memory.count()}")
    print("Basic integration check completed.")


if __name__ == "__main__":
    main()
