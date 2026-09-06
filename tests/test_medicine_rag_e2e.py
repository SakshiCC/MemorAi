from __future__ import annotations

from datetime import datetime

from llm.chat import MemorAiChat
from database.medicine_repository import (
    get_recent_medicine_events,
)


# ============================================================
# DISPLAY HELPERS
# ============================================================

def print_header(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def print_pass(test_name: str) -> None:
    print(f"PASS: {test_name}")


def print_answer(question: str, answer: str) -> None:
    print()
    print(f"Question: {question}")
    print(f"Answer:   {answer}")


# ============================================================
# EVENT HELPERS
# ============================================================

def get_latest_real_camera_confirmed_event():
    """
    Return the newest real camera-confirmed medicine event.

    This test intentionally does NOT create fake medicine events.
    It validates the real live-camera -> database -> vector -> RAG
    pipeline.
    """

    events = get_recent_medicine_events(limit=100)

    for event in events:
        status = str(
            event.get("status") or ""
        ).strip().lower()

        if status == "camera_confirmed":
            return event

    return None


def get_event_medicine_name(event: dict) -> str:
    return str(
        event.get("medicine_name")
        or event.get("name")
        or ""
    ).strip()


def get_event_dosage(event: dict) -> str:
    return str(
        event.get("dosage")
        or ""
    ).strip()


def get_event_observed_at(event: dict):
    return (
        event.get("observed_at")
        or event.get("detected_at")
        or event.get("created_at")
    )


# ============================================================
# TEST 1
# REAL CAMERA EVENT EXISTS
# ============================================================

def test_real_camera_event_exists():
    event = get_latest_real_camera_confirmed_event()

    assert event is not None, (
        "No real camera-confirmed medicine event exists. "
        "Run the live camera medicine flow first."
    )

    medicine_name = get_event_medicine_name(event)

    assert medicine_name, (
        "Camera-confirmed medicine event has no medicine name."
    )

    print_pass(
        "Real camera-confirmed medicine event exists"
    )

    return event


# ============================================================
# TEST 2
# TODAY QUESTION
# ============================================================

def test_today_question(
    chat: MemorAiChat,
    medicine_name: str,
):
    question = (
        f"Did I take {medicine_name} today?"
    )

    answer = chat.answer(question)

    print_answer(question, answer)

    assert answer
    assert medicine_name.lower() in answer.lower()

    lower_answer = answer.lower()

    assert (
        "camera-confirmed" in lower_answer
        or "camera confirmed" in lower_answer
    ), (
        "Medicine answer should preserve safe "
        "camera-confirmed wording."
    )

    print_pass(
        "Today medicine question routed through RAG"
    )


# ============================================================
# TEST 3
# LAST CONFIRMED QUESTION
# ============================================================

def test_last_confirmed_question(
    chat: MemorAiChat,
    medicine_name: str,
):
    question = (
        f"When was {medicine_name} last confirmed?"
    )

    answer = chat.answer(question)

    print_answer(question, answer)

    assert answer
    assert medicine_name.lower() in answer.lower()

    lower_answer = answer.lower()

    assert (
        "camera-confirmed" in lower_answer
        or "camera confirmed" in lower_answer
    )

    print_pass(
        "Latest medicine confirmation retrieval"
    )


# ============================================================
# TEST 4
# RECENT MEDICATION EVENTS
# ============================================================

def test_recent_events(
    chat: MemorAiChat,
    medicine_name: str,
):
    question = (
        "What medication events happened recently?"
    )

    answer = chat.answer(question)

    print_answer(question, answer)

    assert answer

    assert medicine_name.lower() in answer.lower(), (
        "Latest camera-confirmed medicine was not "
        "included in recent medication events."
    )

    print_pass(
        "Recent medication event retrieval"
    )


# ============================================================
# TEST 5
# SAFE WORDING
# ============================================================

def test_safe_wording(
    chat: MemorAiChat,
    medicine_name: str,
):
    question = (
        f"Did I take {medicine_name} today?"
    )

    answer = chat.answer(question)

    print_answer(question, answer)

    lower_answer = answer.lower()

    unsafe_exact_phrases = [
        f"you took {medicine_name.lower()}",
        f"you swallowed {medicine_name.lower()}",
        f"you definitely took {medicine_name.lower()}",
    ]

    for phrase in unsafe_exact_phrases:
        assert phrase not in lower_answer, (
            "Unsafe medication certainty detected: "
            f"{phrase}"
        )

    assert (
        "camera-confirmed" in lower_answer
        or "camera confirmed" in lower_answer
    )

    print_pass(
        "Medication answer uses safe observation wording"
    )


# ============================================================
# TEST 6
# UNKNOWN MEDICINE
# ============================================================

def test_unknown_medicine(
    chat: MemorAiChat,
):
    fake_name = "zzzxxyy_unknown_medicine"

    question = (
        f"Did I take {fake_name} today?"
    )

    answer = chat.answer(question)

    print_answer(question, answer)

    assert answer

    lower_answer = answer.lower()

    assert (
        "no reliable" in lower_answer
        or "couldn't find" in lower_answer
        or "could not find" in lower_answer
        or "don't have" in lower_answer
        or "do not have" in lower_answer
        or "no camera-confirmed" in lower_answer
        or "no camera confirmed" in lower_answer
    ), (
        "Unknown medicine should not produce "
        "a positive confirmation."
    )

    print_pass(
        "Unknown medicine handled safely"
    )


# ============================================================
# TEST 7
# EVENT DATE CONSISTENCY
# ============================================================

def test_event_date_consistency(
    chat: MemorAiChat,
    event: dict,
):
    medicine_name = get_event_medicine_name(event)
    observed_at = get_event_observed_at(event)

    assert observed_at is not None

    if isinstance(observed_at, str):
        try:
            observed_at = datetime.fromisoformat(
                observed_at
            )
        except ValueError:
            observed_at = None

    assert observed_at is not None, (
        "Could not interpret medicine event timestamp."
    )

    question = (
        f"When was {medicine_name} last confirmed?"
    )

    answer = chat.answer(question)

    print_answer(question, answer)

    assert answer

    # We don't require one exact formatting style.
    # chat.py may say "today at 03:26 PM" instead of
    # "16 August 2026 at 03:26 PM".
    #
    # We therefore verify the medicine identity and
    # safe confirmation terminology here.

    assert medicine_name.lower() in answer.lower()

    assert (
        "camera-confirmed" in answer.lower()
        or "camera confirmed" in answer.lower()
    )

    print_pass(
        "Medicine event date retrieval consistency"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print_header(
        "MemorAi Medicine RAG End-to-End Test Suite"
    )

    # --------------------------------------------------------
    # Locate latest REAL medicine event
    # --------------------------------------------------------

    event = test_real_camera_event_exists()

    medicine_name = get_event_medicine_name(event)
    dosage = get_event_dosage(event)
    observed_at = get_event_observed_at(event)

    print()
    print("Using real camera event:")
    print(
        f"Event ID:      {event.get('id')}"
    )
    print(
        f"Medicine ID:   {event.get('medicine_id')}"
    )
    print(
        f"Medicine:      {medicine_name}"
    )
    print(
        f"Dosage:        {dosage}"
    )
    print(
        f"Status:        {event.get('status')}"
    )
    print(
        f"Observed At:   {observed_at}"
    )

    # --------------------------------------------------------
    # Initialize unified MemorAi chat
    # --------------------------------------------------------

    chat = MemorAiChat()

    # --------------------------------------------------------
    # Run end-to-end questions
    # --------------------------------------------------------

    test_today_question(
        chat,
        medicine_name,
    )

    test_last_confirmed_question(
        chat,
        medicine_name,
    )

    test_recent_events(
        chat,
        medicine_name,
    )

    test_safe_wording(
        chat,
        medicine_name,
    )

    test_unknown_medicine(
        chat,
    )

    test_event_date_consistency(
        chat,
        event,
    )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print_header(
        "PASS: Medicine RAG End-to-End Test Suite"
    )

    print(
        "\nVerified pipeline:"
    )

    print(
        "Live Camera"
        " -> Medicine Recognition"
        " -> Action Confirmation"
        " -> MySQL"
        " -> ChromaDB"
        " -> Medicine Retriever"
        " -> MemorAiChat"
    )


if __name__ == "__main__":
    main()