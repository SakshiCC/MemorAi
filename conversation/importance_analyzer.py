from __future__ import annotations

import re

from dataclasses import (
    dataclass,
)


# ============================================================
# RESULT
# ============================================================

@dataclass(frozen=True)
class ConversationImportanceResult:

    is_important: bool

    score: float

    reasons: tuple[str, ...]

    matched_categories: tuple[str, ...]


# ============================================================
# ANALYZER
# ============================================================

class ConversationImportanceAnalyzer:
    """
    Determine whether a conversation contains information
    that is useful as long-term memory.

    MemorAi should remember useful information such as:

        appointments
        medicines
        reminders
        plans
        promises / commitments
        important dates / times
        where items were placed
        visitors / meetings
        tasks
        important instructions

    Casual conversation should normally be discarded.

    Important design rule:

        Short information can still be important.

    Examples:

        "Buy milk tomorrow."

        "My passport is in the drawer."

        "Rahul will visit tomorrow."

    These should not be discarded merely because they are
    short sentences.
    """

    # ========================================================
    # CATEGORY PATTERNS
    # ========================================================

    CATEGORY_PATTERNS = {

        # ----------------------------------------------------
        # APPOINTMENTS
        # ----------------------------------------------------

        "appointment": (
            r"\bappointment\b",
            r"\bdoctor\b",
            r"\bhospital\b",
            r"\bclinic\b",
            r"\bcheck[- ]?up\b",
            r"\bmeeting\b",
        ),

        # ----------------------------------------------------
        # MEDICINES
        # ----------------------------------------------------

        "medicine": (
            r"\bmedicine\b",
            r"\bmedication\b",
            r"\btablet\b",
            r"\bpill\b",
            r"\bdose\b",
            r"\bprescription\b",
            r"\bcapsule\b",
            r"\btake .* after\b",
            r"\btake .* before\b",
        ),

        # ----------------------------------------------------
        # REMINDERS
        # ----------------------------------------------------

        "reminder": (
            r"\bremember\b",
            r"\bremind\b",
            r"\bdon't forget\b",
            r"\bdo not forget\b",
            r"\bneed to\b",
            r"\bmust\b",
        ),

        # ----------------------------------------------------
        # COMMITMENTS / PROMISES / PLANS
        # ----------------------------------------------------

        "commitment": (
            r"\bi will\b",
            r"\bi'll\b",
            r"\bwe will\b",
            r"\bwe'll\b",
            r"\bhe will\b",
            r"\bshe will\b",
            r"\bthey will\b",
            r"\bgoing to\b",
            r"\bpromise\b",
            r"\bpromised\b",
            r"\bpick .* up\b",
            r"\bcome over\b",
            r"\bvisit\b",
        ),

        # ----------------------------------------------------
        # LOCATION MEMORIES
        # ----------------------------------------------------

        "location_memory": (
            r"\bkept\b",
            r"\bkeep\b",
            r"\bput\b",
            r"\bleft\b",
            r"\bstored\b",
            r"\bplaced\b",
            r"\bin the drawer\b",
            r"\bon the table\b",
            r"\bon the desk\b",
            r"\bin the cupboard\b",
            r"\bin the cabinet\b",
            r"\bin the bag\b",
            r"\bin the wardrobe\b",
            r"\bon the shelf\b",
        ),

        # ----------------------------------------------------
        # TASKS
        # ----------------------------------------------------

        "task": (
            r"\bbuy\b",
            r"\bbring\b",
            r"\bcall\b",
            r"\bpay\b",
            r"\bcollect\b",
            r"\bsend\b",
            r"\bsubmit\b",
            r"\bpurchase\b",
            r"\bpick up\b",
            r"\bbook\b",
            r"\brenew\b",
            r"\breturn\b",
        ),

        # ----------------------------------------------------
        # SCHEDULE / RELATIVE DATE
        # ----------------------------------------------------

        "schedule": (
            r"\btomorrow\b",
            r"\btoday\b",
            r"\btonight\b",
            r"\bthis evening\b",
            r"\bthis morning\b",
            r"\bthis afternoon\b",
            r"\bnext week\b",
            r"\bnext month\b",
            r"\bnext year\b",
            r"\bmonday\b",
            r"\btuesday\b",
            r"\bwednesday\b",
            r"\bthursday\b",
            r"\bfriday\b",
            r"\bsaturday\b",
            r"\bsunday\b",
        ),

        # ----------------------------------------------------
        # IMPORTANT DATES
        # ----------------------------------------------------

        "important_date": (
            r"\bbirthday\b",
            r"\banniversary\b",
            r"\bdeadline\b",
            r"\bdue date\b",
            r"\bexpiry date\b",
            r"\bexpiration date\b",
            r"\brenewal date\b",
        ),
    }


    # ========================================================
    # TIME / DATE PATTERNS
    # ========================================================

    TIME_PATTERNS = (

        r"\b\d{1,2}:\d{2}\s*(?:am|pm)?\b",

        r"\b\d{1,2}\s*(?:am|pm)\b",

        (
            r"\bat\s+"
            r"(?:one|two|three|four|five|six|seven|"
            r"eight|nine|ten|eleven|twelve)"
            r"(?:\s+thirty)?\b"
        ),
    )


    DATE_PATTERNS = (

        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",

        (
            r"\b\d{1,2}(?:st|nd|rd|th)?\s+"
            r"(?:january|february|march|april|may|june|"
            r"july|august|september|october|november|december)"
            r"(?:\s+\d{4})?\b"
        ),

        (
            r"\b(?:january|february|march|april|may|june|"
            r"july|august|september|october|november|december)"
            r"\s+\d{1,2}(?:st|nd|rd|th)?"
            r"(?:,\s*\d{4})?\b"
        ),
    )


    # ========================================================
    # CASUAL PHRASES
    # ========================================================

    CASUAL_PATTERNS = (

        r"^\s*hello[.!]?\s*$",

        r"^\s*hi[.!]?\s*$",

        r"^\s*hey[.!]?\s*$",

        r"^\s*how are you[?!.]?\s*$",

        r"^\s*i am fine[.!]?\s*$",

        r"^\s*i'm fine[.!]?\s*$",

        r"^\s*i am good[.!]?\s*$",

        r"^\s*i'm good[.!]?\s*$",

        r"^\s*good morning[.!]?\s*$",

        r"^\s*good afternoon[.!]?\s*$",

        r"^\s*good evening[.!]?\s*$",

        r"^\s*good night[.!]?\s*$",

        r"^\s*see you[.!]?\s*$",

        r"^\s*okay[.!]?\s*$",

        r"^\s*ok[.!]?\s*$",

        r"^\s*thanks?[.!]?\s*$",

        r"^\s*thank you[.!]?\s*$",

        r"^\s*yes[.!]?\s*$",

        r"^\s*no[.!]?\s*$",

        r"^\s*bye[.!]?\s*$",

        r"^\s*goodbye[.!]?\s*$",
    )


    # ========================================================
    # INITIALIZE
    # ========================================================

    def __init__(
        self,
        threshold: float = 0.60,
    ) -> None:

        value = float(
            threshold
        )

        if not (
            0.0
            <= value
            <= 1.0
        ):

            raise ValueError(
                (
                    "Importance threshold must "
                    "be between 0 and 1."
                )
            )

        self.threshold = (
            value
        )


    # ========================================================
    # PATTERN HELPER
    # ========================================================

    @staticmethod
    def _matches_any(
        text: str,
        patterns: tuple[str, ...],
    ) -> bool:

        return any(
            re.search(
                pattern,
                text,
                flags=re.IGNORECASE,
            )
            is not None

            for pattern in patterns
        )


    # ========================================================
    # ANALYZE
    # ========================================================

    def analyze(
        self,
        transcript: str,
    ) -> ConversationImportanceResult:

        cleaned = (
            str(
                transcript
                or ""
            )
            .strip()
        )

        # ====================================================
        # EMPTY
        # ====================================================

        if not cleaned:

            return (
                ConversationImportanceResult(
                    is_important=False,

                    score=0.0,

                    reasons=(
                        "The transcript is empty.",
                    ),

                    matched_categories=(),
                )
            )

        lowered = (
            cleaned.lower()
        )

        # ====================================================
        # CLEAR CASUAL SPEECH
        # ====================================================

        if self._matches_any(
            lowered,
            self.CASUAL_PATTERNS,
        ):

            return (
                ConversationImportanceResult(
                    is_important=False,

                    score=0.05,

                    reasons=(
                        (
                            "The transcript appears "
                            "to contain only casual speech."
                        ),
                    ),

                    matched_categories=(),
                )
            )

        score = 0.0

        reasons: list[str] = []

        categories: list[str] = []

        # ====================================================
        # SEMANTIC CATEGORIES
        # ====================================================

        for (
            category,
            patterns,
        ) in self.CATEGORY_PATTERNS.items():

            if self._matches_any(
                lowered,
                patterns,
            ):

                categories.append(
                    category
                )

                score += 0.18

        # ====================================================
        # TIME
        # ====================================================

        has_time = (
            self._matches_any(
                lowered,
                self.TIME_PATTERNS,
            )
        )

        if has_time:

            score += 0.18

            reasons.append(
                (
                    "The conversation contains "
                    "a specific time."
                )
            )

        # ====================================================
        # DATE
        # ====================================================

        has_date = (
            self._matches_any(
                lowered,
                self.DATE_PATTERNS,
            )
        )

        if has_date:

            score += 0.18

            reasons.append(
                (
                    "The conversation contains "
                    "a specific date."
                )
            )

        # ====================================================
        # MULTIPLE USEFUL SIGNALS
        # ====================================================

        if len(
            categories
        ) >= 2:

            score += 0.18

            reasons.append(
                (
                    "The conversation contains "
                    "multiple useful memory signals."
                )
            )

        if len(
            categories
        ) >= 3:

            score += 0.10

        # ====================================================
        # TARGETED SHORT-MEMORY BONUSES
        #
        # These fix useful short memories without lowering
        # the global importance threshold.
        # ====================================================

        category_set = set(
            categories
        )

        # ----------------------------------------------------
        # TASK + SCHEDULE
        #
        # Examples:
        #
        # Buy milk tomorrow.
        # Pay the electricity bill tomorrow.
        # ----------------------------------------------------

        if {
            "task",
            "schedule",
        }.issubset(
            category_set
        ):

            score += 0.08

            reasons.append(
                (
                    "A scheduled task was detected."
                )
            )

        # ----------------------------------------------------
        # COMMITMENT + SCHEDULE
        #
        # Example:
        #
        # Rahul will visit tomorrow.
        # ----------------------------------------------------

        if {
            "commitment",
            "schedule",
        }.issubset(
            category_set
        ):

            score += 0.08

            reasons.append(
                (
                    "A scheduled commitment or "
                    "visit was detected."
                )
            )

        # ----------------------------------------------------
        # LOCATION MEMORY
        #
        # Location facts are useful even when very short.
        #
        # Examples:
        #
        # My passport is in the drawer.
        # I kept my keys in the cupboard.
        # ----------------------------------------------------

        if (
            "location_memory"
            in category_set
        ):

            score = max(
                score,
                0.65,
            )

            reasons.append(
                (
                    "A useful object-location "
                    "memory was detected."
                )
            )

        # ----------------------------------------------------
        # IMPORTANT DATE
        #
        # Birthdays / anniversaries / deadlines are valuable
        # even if the sentence is short.
        # ----------------------------------------------------

        if (
            "important_date"
            in category_set
        ):

            score = max(
                score,
                0.65,
            )

            reasons.append(
                (
                    "An important date "
                    "was detected."
                )
            )

        # ====================================================
        # DETAIL LEVEL
        # ====================================================

        word_count = len(
            cleaned.split()
        )

        if word_count >= 15:

            score += 0.08

        if word_count >= 30:

            score += 0.05

        # ====================================================
        # CAP SCORE
        # ====================================================

        score = min(
            round(
                score,
                3,
            ),
            1.0,
        )

        # ====================================================
        # CATEGORY REASONS
        # ====================================================

        if categories:

            reasons.insert(
                0,
                (
                    "Useful memory categories detected: "
                    + ", ".join(
                        categories
                    )
                    + "."
                ),
            )

        if not reasons:

            reasons.append(
                (
                    "No strong long-term memory "
                    "signals were detected."
                )
            )

        # ====================================================
        # RESULT
        # ====================================================

        return (
            ConversationImportanceResult(
                is_important=(
                    score
                    >= self.threshold
                ),

                score=(
                    score
                ),

                reasons=tuple(
                    reasons
                ),

                matched_categories=tuple(
                    categories
                ),
            )
        )