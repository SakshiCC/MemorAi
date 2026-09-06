from __future__ import annotations

from datetime import (
    date,
    timedelta,
)

from llm.chat import (
    FALLBACK_ANSWER,
    MemorAiChat,
)

from memory.retriever import (
    MemoryRetriever,
)

from database.event_repository import (
    get_events_for_date,
    get_latest_event_for_object,
)


# ============================================================
# HELPER
# ============================================================

def print_test_result(
    test_name: str,
    passed: bool,
) -> None:

    status = (
        "PASS"
        if passed
        else "FAIL"
    )

    print(
        f"{status}: {test_name}"
    )


# ============================================================
# HELPER
# GET LATEST PHONE MEMORY
# ============================================================

def get_latest_phone_memory() -> dict:
    """
    Return the newest cell-phone memory stored
    in the events table.

    The tests use the real latest memory instead
    of hard-coding an old location or timestamp.
    """

    memory = (
        get_latest_event_for_object(
            "cell phone"
        )
    )

    assert (
        memory is not None
    ), (
        "No cell phone memory exists "
        "in the database."
    )

    return memory


# ============================================================
# TEST 1
# EMPTY QUESTION
# ============================================================

def test_empty_question() -> None:

    chat = (
        MemorAiChat()
    )

    result = (
        chat.answer(
            ""
        )
    )

    expected = (
        "Please ask a memory-related question."
    )

    assert (
        result == expected
    )

    print_test_result(
        "Empty question handling",
        True,
    )


# ============================================================
# TEST 2
# UNKNOWN OBJECT
# ============================================================

def test_unknown_object() -> None:

    chat = (
        MemorAiChat()
    )

    result = (
        chat.answer(
            "Where are my car keys?"
        )
    )

    assert (
        result
        == FALLBACK_ANSWER
    )

    print_test_result(
        "Unknown object fallback",
        True,
    )


# ============================================================
# TEST 3
# LATEST CELL PHONE RETRIEVAL
# ============================================================

def test_latest_phone_retrieval() -> None:

    retriever = (
        MemoryRetriever()
    )

    memories = (
        retriever.retrieve(
            "Where is my cell phone?"
        )
    )

    assert memories

    latest = (
        memories[0]
    )

    assert (
        latest.get(
            "object_name"
        )
        == "cell phone"
    )

    assert (
        latest.get(
            "location"
        )
        is not None
    )

    assert (
        latest.get(
            "detected_at"
        )
        is not None
    )

    # --------------------------------------------------------
    # COMPARE RETRIEVER WITH ACTUAL DATABASE LATEST MEMORY
    # --------------------------------------------------------

    database_latest = (
        get_latest_phone_memory()
    )

    assert (
        latest.get(
            "location"
        )
        == database_latest.get(
            "location"
        )
    )

    assert (
        latest.get(
            "detected_at"
        )
        == database_latest.get(
            "detected_at"
        )
    )

    print_test_result(
        "Latest cell phone retrieval",
        True,
    )


# ============================================================
# TEST 4
# LOCATION ANSWER
# ============================================================

def test_location_answer() -> None:

    chat = (
        MemorAiChat()
    )

    latest_memory = (
        get_latest_phone_memory()
    )

    expected_location = (
        str(
            latest_memory.get(
                "location"
            )
            or ""
        )
        .strip()
        .lower()
    )

    assert expected_location

    result = (
        chat.answer(
            "Where is my cell phone?"
        )
    )

    lowered = (
        result.lower()
    )

    # --------------------------------------------------------
    # BASIC ANSWER CONTENT
    # --------------------------------------------------------

    assert (
        "cell phone"
        in lowered
    )

    assert (
        "last observed"
        in lowered
    )

    # --------------------------------------------------------
    # DYNAMIC LOCATION
    # --------------------------------------------------------

    assert (
        expected_location
        in lowered
    )

    # --------------------------------------------------------
    # SAFE WORDING
    # --------------------------------------------------------

    assert not (
        lowered.startswith(
            "your cell phone is"
        )
    )

    print_test_result(
        "Location answer",
        True,
    )


# ============================================================
# TEST 5
# TIME ANSWER
# ============================================================

def test_time_answer() -> None:

    chat = (
        MemorAiChat()
    )

    latest_memory = (
        get_latest_phone_memory()
    )

    expected_location = (
        str(
            latest_memory.get(
                "location"
            )
            or ""
        )
        .strip()
        .lower()
    )

    detected_at = (
        latest_memory.get(
            "detected_at"
        )
    )

    assert (
        detected_at
        is not None
    )

    expected_time = (
        detected_at.strftime(
            "%I:%M %p"
        )
        .lower()
    )

    expected_day = str(
        detected_at.day
    )

    expected_year = str(
        detected_at.year
    )

    expected_month = (
        detected_at.strftime(
            "%B"
        )
        .lower()
    )

    result = (
        chat.answer(
            (
                "When did I last see "
                "my cell phone?"
            )
        )
    )

    lowered = (
        result.lower()
    )

    assert (
        "cell phone"
        in lowered
    )

    assert (
        "last observed"
        in lowered
    )

    assert (
        expected_location
        in lowered
    )

    assert (
        expected_time
        in lowered
    )

    assert (
        expected_day
        in lowered
    )

    assert (
        expected_month
        in lowered
    )

    assert (
        expected_year
        in lowered
    )

    print_test_result(
        "Latest time answer",
        True,
    )


# ============================================================
# TEST 6
# TODAY QUERY
# ============================================================

def test_today_query() -> None:

    chat = (
        MemorAiChat()
    )

    today_events = (
        get_events_for_date(
            date.today(),
            "cell phone",
        )
    )

    result = (
        chat.answer(
            "Did I see my cell phone today?"
        )
    )

    # --------------------------------------------------------
    # DATABASE HAS NO TODAY EVENT
    # --------------------------------------------------------

    if not today_events:

        assert (
            result
            == FALLBACK_ANSWER
        )

        print_test_result(
            "Today no-result fallback",
            True,
        )

        return

    # --------------------------------------------------------
    # DATABASE HAS TODAY EVENT
    # --------------------------------------------------------

    lowered = (
        result.lower()
    )

    assert (
        lowered.startswith(
            "yes."
        )
    )

    assert (
        "cell phone"
        in lowered
    )

    assert (
        "today"
        in lowered
    )

    latest_today = (
        today_events[0]
    )

    detected_at = (
        latest_today.get(
            "detected_at"
        )
    )

    if detected_at:

        expected_time = (
            detected_at.strftime(
                "%I:%M %p"
            )
            .lower()
        )

        assert (
            expected_time
            in lowered
        )

    print_test_result(
        "Today query",
        True,
    )


# ============================================================
# TEST 7
# YESTERDAY QUERY
# ============================================================

def test_yesterday_query() -> None:

    chat = (
        MemorAiChat()
    )

    yesterday = (
        date.today()
        - timedelta(
            days=1
        )
    )

    yesterday_events = (
        get_events_for_date(
            yesterday,
            "cell phone",
        )
    )

    result = (
        chat.answer(
            (
                "Did I see my cell "
                "phone yesterday?"
            )
        )
    )

    # --------------------------------------------------------
    # NO DATABASE RECORD YESTERDAY
    # --------------------------------------------------------

    if not yesterday_events:

        assert (
            result
            == FALLBACK_ANSWER
        )

        print_test_result(
            "Yesterday no-result fallback",
            True,
        )

        return

    # --------------------------------------------------------
    # DATABASE DOES HAVE YESTERDAY RECORD
    # --------------------------------------------------------

    lowered = (
        result.lower()
    )

    assert (
        lowered.startswith(
            "yes."
        )
    )

    assert (
        "cell phone"
        in lowered
    )

    assert (
        "yesterday"
        in lowered
    )

    latest_yesterday = (
        yesterday_events[0]
    )

    detected_at = (
        latest_yesterday.get(
            "detected_at"
        )
    )

    if detected_at:

        expected_time = (
            detected_at.strftime(
                "%I:%M %p"
            )
            .lower()
        )

        assert (
            expected_time
            in lowered
        )

    expected_location = (
        str(
            latest_yesterday.get(
                "location"
            )
            or ""
        )
        .strip()
        .lower()
    )

    if expected_location:

        assert (
            expected_location
            in lowered
        )

    print_test_result(
        "Yesterday query",
        True,
    )


# ============================================================
# TEST 8
# RECENT OBJECTS
# ============================================================

def test_recent_objects() -> None:

    chat = (
        MemorAiChat()
    )

    result = (
        chat.answer(
            (
                "What objects were "
                "detected recently?"
            )
        )
    )

    lowered = (
        result.lower()
    )

    assert (
        "recently"
        in lowered
    )

    assert (
        "cell phone"
        in lowered
    )

    assert (
        "book"
        in lowered
    )

    print_test_result(
        "Recent object summary",
        True,
    )


# ============================================================
# TEST 9
# SAFE CURRENT LOCATION WORDING
# ============================================================

def test_safe_location_wording() -> None:

    chat = (
        MemorAiChat()
    )

    latest_memory = (
        get_latest_phone_memory()
    )

    expected_location = (
        str(
            latest_memory.get(
                "location"
            )
            or ""
        )
        .strip()
        .lower()
    )

    result = (
        chat.answer(
            "Where is my cell phone right now?"
        )
    )

    lowered = (
        result.lower()
    )

    assert (
        "last observed"
        in lowered
    )

    assert (
        expected_location
        in lowered
    )

    assert not (
        lowered.startswith(
            "your cell phone is"
        )
    )

    print_test_result(
        "Safe current-location wording",
        True,
    )


# ============================================================
# RUN ALL TESTS
# ============================================================

def main() -> None:

    print(
        "=" * 60
    )

    print(
        "MemorAi RAG Test Suite"
    )

    print(
        "=" * 60
    )

    tests = [

        test_empty_question,

        test_unknown_object,

        test_latest_phone_retrieval,

        test_location_answer,

        test_time_answer,

        test_today_query,

        test_yesterday_query,

        test_recent_objects,

        test_safe_location_wording,

    ]

    passed = 0

    failed = 0

    for test_function in tests:

        try:

            test_function()

            passed += 1

        except AssertionError as error:

            failed += 1

            print_test_result(
                test_function.__name__,
                False,
            )

            if str(
                error
            ):

                print(
                    (
                        "Reason: "
                        f"{error}"
                    )
                )

        except Exception as error:

            failed += 1

            print(
                (
                    "ERROR: "
                    f"{test_function.__name__}"
                )
            )

            print(
                (
                    "Reason: "
                    f"{error}"
                )
            )

    print()

    print(
        "=" * 60
    )

    print(
        f"Passed: {passed}"
    )

    print(
        f"Failed: {failed}"
    )

    print(
        "=" * 60
    )

    if failed == 0:

        print(
            (
                "All MemorAi "
                "RAG tests passed."
            )
        )

    else:

        print(
            (
                "Some MemorAi "
                "RAG tests failed."
            )
        )


if __name__ == "__main__":

    main()