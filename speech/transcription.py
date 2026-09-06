import re
from dataclasses import dataclass


@dataclass
class TranscriptionResult:
    """
    Result returned by TranscriptionCleaner.
    """

    text: str
    is_valid: bool
    reason: str | None = None


class TranscriptionCleaner:
    """
    Safely clean and validate Whisper text before it
    reaches MemorAiChat.

    The cleaner must not guess or rewrite the user's
    intended meaning.
    """

    NON_SPEECH_PATTERNS = {
        "",
        ".",
        "..",
        "...",
        "?",
        "!",
        "[silence]",
        "(silence)",
        "silence",
        "[music]",
        "(music)",
        "music",
        "[noise]",
        "(noise)",
        "noise",
        "[background noise]",
        "(background noise)",
        "background noise",
        "[applause]",
        "(applause)",
        "[laughter]",
        "(laughter)",
    }

    def __init__(
        self,
        min_characters: int = 2,
        max_characters: int = 500,
        max_repeated_word_count: int = 8,
    ):
        self.min_characters = (
            min_characters
        )

        self.max_characters = (
            max_characters
        )

        self.max_repeated_word_count = (
            max_repeated_word_count
        )

    def clean(
        self,
        text: str | None,
    ) -> str:
        """
        Perform safe formatting cleanup.

        Does not alter semantic meaning.
        """

        if text is None:
            return ""

        text = str(
            text
        )

        # Replace tabs/newlines/repeated whitespace.
        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        text = (
            text.strip()
        )

        # Remove spaces before punctuation.
        text = re.sub(
            r"\s+([?.!,;:])",
            r"\1",
            text,
        )

        return text

    @staticmethod
    def _normalize_for_validation(
        text: str,
    ) -> str:
        """
        Produce a lowercase comparison form.

        This is used only for validation.
        The returned user text is not replaced with it.
        """

        normalized = (
            text.lower().strip()
        )

        normalized = (
            normalized
            .replace("♪", "")
            .strip()
        )

        return normalized

    def _has_excessive_repetition(
        self,
        text: str,
    ) -> bool:
        """
        Detect obvious repeated-word hallucination.

        Example:
            "the the the the the the..."
        """

        words = re.findall(
            r"[A-Za-z0-9']+",
            text.lower(),
        )

        if not words:
            return False

        current_word = None
        current_count = 0

        for word in words:

            if word == current_word:
                current_count += 1

            else:
                current_word = word
                current_count = 1

            if (
                current_count
                >= self.max_repeated_word_count
            ):
                return True

        return False

    def validate(
        self,
        text: str | None,
    ) -> TranscriptionResult:
        """
        Clean and validate Whisper output.
        """

        cleaned = (
            self.clean(
                text
            )
        )

        if not cleaned:

            return TranscriptionResult(
                text="",
                is_valid=False,
                reason="No speech was detected.",
            )

        validation_text = (
            self._normalize_for_validation(
                cleaned
            )
        )

        if (
            validation_text
            in self.NON_SPEECH_PATTERNS
        ):

            return TranscriptionResult(
                text=cleaned,
                is_valid=False,
                reason=(
                    "Only silence or non-speech "
                    "audio was detected."
                ),
            )

        if (
            len(cleaned)
            < self.min_characters
        ):

            return TranscriptionResult(
                text=cleaned,
                is_valid=False,
                reason=(
                    "The transcription is too short."
                ),
            )

        if (
            len(cleaned)
            > self.max_characters
        ):

            return TranscriptionResult(
                text=cleaned,
                is_valid=False,
                reason=(
                    "The transcription is "
                    "unexpectedly long."
                ),
            )

        if not re.search(
            r"[A-Za-z0-9]",
            cleaned,
        ):

            return TranscriptionResult(
                text=cleaned,
                is_valid=False,
                reason=(
                    "No meaningful speech was detected."
                ),
            )

        if (
            self._has_excessive_repetition(
                cleaned
            )
        ):

            return TranscriptionResult(
                text=cleaned,
                is_valid=False,
                reason=(
                    "The transcription contains "
                    "excessive repeated words."
                ),
            )

        return TranscriptionResult(
            text=cleaned,
            is_valid=True,
            reason=None,
        )


def main():
    cleaner = (
        TranscriptionCleaner()
    )

    examples = [
        "   Where   is my cellphone?   ",
        "",
        "...",
        "[Music]",
        "Where are my glasses?",
        "the the the the the the the the the",
    ]

    print(
        "\nMemorAi Transcription Validation Test"
    )

    print(
        "-" * 50
    )

    for raw_text in examples:

        result = (
            cleaner.validate(
                raw_text
            )
        )

        print(
            f"\nRaw:     {raw_text!r}"
        )

        print(
            f"Cleaned: {result.text!r}"
        )

        print(
            f"Valid:   {result.is_valid}"
        )

        print(
            f"Reason:  {result.reason}"
        )


if __name__ == "__main__":
    main()