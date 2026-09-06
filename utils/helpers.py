from __future__ import annotations

import re
from datetime import datetime
from typing import Any


def utc_safe_iso(value: datetime | str | None) -> str:
    if value is None:
        return datetime.now().isoformat(timespec="seconds")

    if isinstance(value, datetime):
        return value.isoformat(timespec="seconds")

    return str(value)


def normalize_object_name(name: str) -> str:
    cleaned = re.sub(r"\s+", " ", name.strip().lower())

    aliases = {
        "phone": "cell phone",
        "mobile": "cell phone",
        "mobile phone": "cell phone",
        "spectacles": "glasses",
        "spectacle": "glasses",
    }

    return aliases.get(cleaned, cleaned)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
