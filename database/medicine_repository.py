from __future__ import annotations

from datetime import (
    date,
    datetime,
    time,
)

from typing import Any, Iterable

from database.db_manager import (
    execute,
    fetch_all,
    fetch_one,
)


# ============================================================
# CONSTANTS
# ============================================================

STATUS_CAMERA_CONFIRMED = "camera_confirmed"
STATUS_LOW_CONFIDENCE = "low_confidence"
STATUS_UNCONFIRMED = "unconfirmed"
STATUS_MISSED = "missed"

CONFIRMATION_CAMERA_ASSISTED = "camera_assisted"
EVENT_CAMERA_CONFIRMATION = "camera_confirmation"


# ============================================================
# HELPERS
# ============================================================

def normalize_medicine_name(
    medicine_name: str,
) -> str:

    cleaned = " ".join(
        medicine_name
        .strip()
        .split()
    )

    if not cleaned:

        raise ValueError(
            "Medicine name cannot be empty."
        )

    return cleaned


def normalize_optional_text(
    value: str | None,
) -> str | None:

    if value is None:

        return None

    cleaned = " ".join(
        value
        .strip()
        .split()
    )

    return (
        cleaned
        if cleaned
        else None
    )


# ============================================================
# OCR KEYWORDS
# ============================================================

def normalize_ocr_keywords(
    keywords: (
        str
        | Iterable[str]
        | None
    ),
) -> str | None:

    if keywords is None:

        return None

    if isinstance(
        keywords,
        str,
    ):

        values = (
            keywords
            .replace(
                "\n",
                ",",
            )
            .split(
                ","
            )
        )

    else:

        values = list(
            keywords
        )

    cleaned_values: list[str] = []

    seen: set[str] = set()

    for value in values:

        cleaned = " ".join(
            str(
                value
            )
            .strip()
            .split()
        )

        if not cleaned:

            continue

        key = (
            cleaned.lower()
        )

        if key in seen:

            continue

        seen.add(
            key
        )

        cleaned_values.append(
            cleaned
        )

    if not cleaned_values:

        return None

    return ", ".join(
        cleaned_values
    )


def parse_ocr_keywords(
    value: str | None,
) -> list[str]:

    if not value:

        return []

    return [
        cleaned
        for cleaned
        in (
            item.strip()
            for item
            in str(
                value
            ).split(
                ","
            )
        )
        if cleaned
    ]


def validate_probability(
    value: float | None,
    field_name: str,
) -> float | None:

    if value is None:

        return None

    number = float(
        value
    )

    if not (
        0.0
        <= number
        <= 1.0
    ):

        raise ValueError(
            f"{field_name} must be between "
            "0.0 and 1.0."
        )

    return number


# ============================================================
# MEDICINE LOOKUP
# ============================================================

def get_medicine_by_id(
    medicine_id: int,
) -> dict[str, Any] | None:

    return fetch_one(
        """
        SELECT
            id,
            medicine_name,
            dosage,
            ocr_keywords,
            instructions,
            schedule_time,
            schedule_label,
            is_active,
            created_at,
            updated_at
        FROM medicines
        WHERE id = %s
        LIMIT 1
        """,
        (
            medicine_id,
        ),
    )


def get_medicine_by_name(
    medicine_name: str,
) -> dict[str, Any] | None:

    cleaned_name = (
        normalize_medicine_name(
            medicine_name
        )
    )

    return fetch_one(
        """
        SELECT
            id,
            medicine_name,
            dosage,
            ocr_keywords,
            instructions,
            schedule_time,
            schedule_label,
            is_active,
            created_at,
            updated_at
        FROM medicines
        WHERE LOWER(medicine_name) = LOWER(%s)
        ORDER BY id ASC
        LIMIT 1
        """,
        (
            cleaned_name,
        ),
    )


# ============================================================
# CREATE MEDICINE
# ============================================================

def create_medicine(
    medicine_name: str,
    dosage: str | None,
    schedule_time: time,
    schedule_label: str | None = None,
    instructions: str | None = None,
    ocr_keywords: (
        str
        | Iterable[str]
        | None
    ) = None,
) -> int:

    cleaned_name = (
        normalize_medicine_name(
            medicine_name
        )
    )

    cleaned_dosage = (
        normalize_optional_text(
            dosage
        )
    )

    cleaned_keywords = (
        normalize_ocr_keywords(
            ocr_keywords
        )
    )

    cleaned_label = (
        normalize_optional_text(
            schedule_label
        )
    )

    cleaned_instructions = (
        normalize_optional_text(
            instructions
        )
    )

    medicine_id = execute(
        """
        INSERT INTO medicines (
            medicine_name,
            dosage,
            ocr_keywords,
            instructions,
            schedule_time,
            schedule_label,
            is_active
        )
        VALUES (
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            TRUE
        )
        """,
        (
            cleaned_name,
            cleaned_dosage,
            cleaned_keywords,
            cleaned_instructions,
            schedule_time,
            cleaned_label,
        ),
    )

    if medicine_id <= 0:

        raise RuntimeError(
            "Failed to create medicine."
        )

    add_medicine_schedule(
        medicine_id=medicine_id,
        schedule_time=schedule_time,
        schedule_label=cleaned_label,
    )

    return medicine_id


# ============================================================
# ALL MEDICINES
# ============================================================

def get_all_medicines(
    active_only: bool = True,
) -> list[dict[str, Any]]:

    where = (
        "WHERE is_active = TRUE"
        if active_only
        else ""
    )

    return fetch_all(
        f"""
        SELECT
            id,
            medicine_name,
            dosage,
            ocr_keywords,
            instructions,
            schedule_time,
            schedule_label,
            is_active,
            created_at,
            updated_at
        FROM medicines
        {where}
        ORDER BY
            medicine_name ASC,
            id ASC
        """
    )


# ============================================================
# UPDATE MEDICINE
# ============================================================

def update_medicine(
    medicine_id: int,
    medicine_name: str | None = None,
    dosage: str | None = None,
    instructions: str | None = None,
    ocr_keywords: (
        str
        | Iterable[str]
        | None
    ) = None,
) -> None:

    medicine = (
        get_medicine_by_id(
            medicine_id
        )
    )

    if medicine is None:

        raise ValueError(
            "Medicine ID does not exist: "
            f"{medicine_id}"
        )

    final_name = (
        normalize_medicine_name(
            medicine_name
        )
        if medicine_name is not None
        else medicine[
            "medicine_name"
        ]
    )

    final_dosage = (
        normalize_optional_text(
            dosage
        )
        if dosage is not None
        else medicine[
            "dosage"
        ]
    )

    final_instructions = (
        normalize_optional_text(
            instructions
        )
        if instructions is not None
        else medicine[
            "instructions"
        ]
    )

    final_keywords = (
        normalize_ocr_keywords(
            ocr_keywords
        )
        if ocr_keywords is not None
        else medicine.get(
            "ocr_keywords"
        )
    )

    execute(
        """
        UPDATE medicines
        SET
            medicine_name = %s,
            dosage = %s,
            ocr_keywords = %s,
            instructions = %s
        WHERE id = %s
        """,
        (
            final_name,
            final_dosage,
            final_keywords,
            final_instructions,
            medicine_id,
        ),
    )


def update_medicine_ocr_keywords(
    medicine_id: int,
    ocr_keywords: (
        str
        | Iterable[str]
        | None
    ),
) -> None:

    medicine = (
        get_medicine_by_id(
            medicine_id
        )
    )

    if medicine is None:

        raise ValueError(
            "Medicine ID does not exist: "
            f"{medicine_id}"
        )

    execute(
        """
        UPDATE medicines
        SET ocr_keywords = %s
        WHERE id = %s
        """,
        (
            normalize_ocr_keywords(
                ocr_keywords
            ),
            medicine_id,
        ),
    )


# ============================================================
# MEDICINE ACTIVE / INACTIVE
# ============================================================

def deactivate_medicine(
    medicine_id: int,
) -> None:

    if (
        get_medicine_by_id(
            medicine_id
        )
        is None
    ):

        raise ValueError(
            "Medicine ID does not exist: "
            f"{medicine_id}"
        )

    execute(
        """
        UPDATE medicines
        SET is_active = FALSE
        WHERE id = %s
        """,
        (
            medicine_id,
        ),
    )


def reactivate_medicine(
    medicine_id: int,
) -> None:

    if (
        get_medicine_by_id(
            medicine_id
        )
        is None
    ):

        raise ValueError(
            "Medicine ID does not exist: "
            f"{medicine_id}"
        )

    execute(
        """
        UPDATE medicines
        SET is_active = TRUE
        WHERE id = %s
        """,
        (
            medicine_id,
        ),
    )


# ============================================================
# MEDICINE SCHEDULES
# ============================================================

def add_medicine_schedule(
    medicine_id: int,
    schedule_time: time,
    schedule_label: str | None = None,
) -> int:

    if (
        get_medicine_by_id(
            medicine_id
        )
        is None
    ):

        raise ValueError(
            "Medicine ID does not exist: "
            f"{medicine_id}"
        )

    schedule_id = execute(
        """
        INSERT INTO medicine_schedules (
            medicine_id,
            schedule_time,
            schedule_label,
            is_active
        )
        VALUES (
            %s,
            %s,
            %s,
            TRUE
        )
        """,
        (
            medicine_id,
            schedule_time,
            normalize_optional_text(
                schedule_label
            ),
        ),
    )

    if schedule_id <= 0:

        raise RuntimeError(
            "Failed to create medicine schedule."
        )

    return schedule_id


def get_medicine_schedules(
    medicine_id: int,
    active_only: bool = True,
) -> list[dict[str, Any]]:

    extra = (
        "AND is_active = TRUE"
        if active_only
        else ""
    )

    return fetch_all(
        f"""
        SELECT
            id,
            medicine_id,
            schedule_time,
            schedule_label,
            is_active,
            created_at
        FROM medicine_schedules
        WHERE medicine_id = %s
        {extra}
        ORDER BY
            schedule_time ASC,
            id ASC
        """,
        (
            medicine_id,
        ),
    )


def get_schedule_by_id(
    schedule_id: int,
) -> dict[str, Any] | None:

    return fetch_one(
        """
        SELECT
            id,
            medicine_id,
            schedule_time,
            schedule_label,
            is_active,
            created_at
        FROM medicine_schedules
        WHERE id = %s
        LIMIT 1
        """,
        (
            schedule_id,
        ),
    )


def deactivate_medicine_schedule(
    schedule_id: int,
) -> None:

    if (
        get_schedule_by_id(
            schedule_id
        )
        is None
    ):

        raise ValueError(
            "Medicine schedule ID does not exist: "
            f"{schedule_id}"
        )

    execute(
        """
        UPDATE medicine_schedules
        SET is_active = FALSE
        WHERE id = %s
        """,
        (
            schedule_id,
        ),
    )


def reactivate_medicine_schedule(
    schedule_id: int,
) -> None:

    if (
        get_schedule_by_id(
            schedule_id
        )
        is None
    ):

        raise ValueError(
            "Medicine schedule ID does not exist: "
            f"{schedule_id}"
        )

    execute(
        """
        UPDATE medicine_schedules
        SET is_active = TRUE
        WHERE id = %s
        """,
        (
            schedule_id,
        ),
    )


# ============================================================
# REFERENCE IMAGES
# ============================================================

def add_medicine_reference_image(
    medicine_id: int,
    image_path: str,
    embedding: bytes | None = None,
    embedding_model: str | None = None,
    is_primary: bool = False,
) -> int:

    if (
        get_medicine_by_id(
            medicine_id
        )
        is None
    ):

        raise ValueError(
            "Medicine ID does not exist: "
            f"{medicine_id}"
        )

    cleaned_path = (
        image_path.strip()
    )

    if not cleaned_path:

        raise ValueError(
            "Medicine reference image path "
            "cannot be empty."
        )

    image_id = execute(
        """
        INSERT INTO medicine_reference_images (
            medicine_id,
            image_path,
            embedding,
            embedding_model,
            is_primary
        )
        VALUES (
            %s,
            %s,
            %s,
            %s,
            %s
        )
        """,
        (
            medicine_id,
            cleaned_path,
            embedding,
            normalize_optional_text(
                embedding_model
            ),
            bool(
                is_primary
            ),
        ),
    )

    if image_id <= 0:

        raise RuntimeError(
            "Failed to save medicine "
            "reference image."
        )

    return image_id


def get_medicine_reference_images(
    medicine_id: int,
) -> list[dict[str, Any]]:

    return fetch_all(
        """
        SELECT
            id,
            medicine_id,
            image_path,
            embedding,
            embedding_model,
            is_primary,
            created_at
        FROM medicine_reference_images
        WHERE medicine_id = %s
        ORDER BY
            is_primary DESC,
            id ASC
        """,
        (
            medicine_id,
        ),
    )


def get_all_medicine_reference_images(
    active_medicines_only: bool = True,
) -> list[dict[str, Any]]:

    where = (
        "WHERE m.is_active = TRUE"
        if active_medicines_only
        else ""
    )

    return fetch_all(
        f"""
        SELECT
            mri.id,
            mri.medicine_id,
            mri.image_path,
            mri.embedding,
            mri.embedding_model,
            mri.is_primary,
            mri.created_at,

            m.medicine_name,
            m.dosage,
            m.ocr_keywords

        FROM medicine_reference_images AS mri

        INNER JOIN medicines AS m
            ON m.id = mri.medicine_id

        {where}

        ORDER BY
            m.medicine_name ASC,
            mri.is_primary DESC,
            mri.id ASC
        """
    )


def set_primary_reference_image(
    medicine_id: int,
    reference_image_id: int,
) -> None:

    if (
        get_medicine_by_id(
            medicine_id
        )
        is None
    ):

        raise ValueError(
            "Medicine ID does not exist."
        )

    image = fetch_one(
        """
        SELECT id
        FROM medicine_reference_images
        WHERE id = %s
          AND medicine_id = %s
        LIMIT 1
        """,
        (
            reference_image_id,
            medicine_id,
        ),
    )

    if image is None:

        raise ValueError(
            "Medicine reference image "
            "does not exist."
        )

    execute(
        """
        UPDATE medicine_reference_images
        SET is_primary = FALSE
        WHERE medicine_id = %s
        """,
        (
            medicine_id,
        ),
    )

    execute(
        """
        UPDATE medicine_reference_images
        SET is_primary = TRUE
        WHERE id = %s
          AND medicine_id = %s
        """,
        (
            reference_image_id,
            medicine_id,
        ),
    )


def delete_medicine_reference_image(
    reference_image_id: int,
) -> None:

    execute(
        """
        DELETE FROM medicine_reference_images
        WHERE id = %s
        """,
        (
            reference_image_id,
        ),
    )


# ============================================================
# MEDICINE MEMORY TEXT
# ============================================================

def create_medicine_memory_text(
    medicine_name: str,
    dosage: str | None,
    observed_at: datetime,
    status: str,
) -> str:

    readable_time = (
        observed_at.strftime(
            "%d %b %Y at %I:%M %p"
        )
    )

    medicine_description = (
        medicine_name
    )

    if dosage:

        medicine_description += (
            f" {dosage}"
        )

    if status == STATUS_CAMERA_CONFIRMED:

        return (
            "MemorAi observed a camera-confirmed "
            f"medication event for "
            f"{medicine_description} "
            f"on {readable_time}."
        )

    if status == STATUS_LOW_CONFIDENCE:

        return (
            "MemorAi observed possible medication "
            f"activity involving "
            f"{medicine_description} "
            f"on {readable_time}, but the visual "
            "evidence was not strong enough for "
            "camera confirmation."
        )

    if status == STATUS_MISSED:

        return (
            "MemorAi has no confirmed medication "
            f"event for {medicine_description} "
            "for the scheduled dose associated "
            f"with {readable_time}."
        )

    return (
        "MemorAi has an unconfirmed medication "
        f"event involving "
        f"{medicine_description} "
        f"on {readable_time}."
    )


# ============================================================
# SAVE MEDICINE EVENT
# ============================================================

def save_medicine_event(
    medicine_id: int,
    observed_at: datetime,
    status: str = STATUS_CAMERA_CONFIRMED,
    person_id: int | None = None,
    confidence: float | None = None,
    medicine_similarity: float | None = None,
    person_similarity: float | None = None,
    action_confidence: float | None = None,
    evidence_image_path: str | None = None,
    scheduled_for: datetime | None = None,
    confirmation_method: str = CONFIRMATION_CAMERA_ASSISTED,
    event_type: str = EVENT_CAMERA_CONFIRMATION,
    memory_text: str | None = None,
) -> int:

    medicine = (
        get_medicine_by_id(
            medicine_id
        )
    )

    if medicine is None:

        raise ValueError(
            "Medicine ID does not exist: "
            f"{medicine_id}"
        )

    allowed_statuses = {
        STATUS_CAMERA_CONFIRMED,
        STATUS_LOW_CONFIDENCE,
        STATUS_UNCONFIRMED,
        STATUS_MISSED,
    }

    if status not in allowed_statuses:

        raise ValueError(
            "Unsupported medicine event status: "
            f"{status}"
        )

    confidence_value = (
        validate_probability(
            confidence,
            "confidence",
        )
    )

    medicine_similarity_value = (
        validate_probability(
            medicine_similarity,
            "medicine_similarity",
        )
    )

    person_similarity_value = (
        validate_probability(
            person_similarity,
            "person_similarity",
        )
    )

    action_confidence_value = (
        validate_probability(
            action_confidence,
            "action_confidence",
        )
    )

    if memory_text is None:

        memory_text = (
            create_medicine_memory_text(
                medicine_name=(
                    medicine[
                        "medicine_name"
                    ]
                ),
                dosage=(
                    medicine[
                        "dosage"
                    ]
                ),
                observed_at=observed_at,
                status=status,
            )
        )

    event_id = execute(
        """
        INSERT INTO medicine_events (
            medicine_id,
            person_id,
            event_type,
            status,
            confidence,
            medicine_similarity,
            person_similarity,
            action_confidence,
            evidence_image_path,
            observed_at,
            scheduled_for,
            confirmation_method,
            memory_text
        )
        VALUES (
            %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s
        )
        """,
        (
            medicine_id,
            person_id,
            event_type,
            status,
            confidence_value,
            medicine_similarity_value,
            person_similarity_value,
            action_confidence_value,
            evidence_image_path,
            observed_at,
            scheduled_for,
            confirmation_method,
            memory_text,
        ),
    )

    if event_id <= 0:

        raise RuntimeError(
            "Failed to save medicine event."
        )

    return event_id


# ============================================================
# EVENT QUERIES
# ============================================================

def get_medicine_event_by_id(
    event_id: int,
) -> dict[str, Any] | None:

    return fetch_one(
        """
        SELECT
            me.*,
            m.medicine_name,
            m.dosage,
            m.ocr_keywords,
            m.instructions,
            p.display_name AS person_name,
            p.relation AS person_relation
        FROM medicine_events AS me
        INNER JOIN medicines AS m
            ON m.id = me.medicine_id
        LEFT JOIN registered_people AS p
            ON p.id = me.person_id
        WHERE me.id = %s
        LIMIT 1
        """,
        (
            event_id,
        ),
    )


def get_latest_medicine_event(
    medicine_id: int,
    confirmed_only: bool = False,
) -> dict[str, Any] | None:

    status_clause = (
        "AND me.status = %s"
        if confirmed_only
        else ""
    )

    params: tuple[Any, ...] = (
        (
            medicine_id,
            STATUS_CAMERA_CONFIRMED,
        )
        if confirmed_only
        else (
            medicine_id,
        )
    )

    return fetch_one(
        f"""
        SELECT
            me.*,
            m.medicine_name,
            m.dosage,
            m.ocr_keywords,
            m.instructions,
            p.display_name AS person_name,
            p.relation AS person_relation
        FROM medicine_events AS me
        INNER JOIN medicines AS m
            ON m.id = me.medicine_id
        LEFT JOIN registered_people AS p
            ON p.id = me.person_id
        WHERE me.medicine_id = %s
        {status_clause}
        ORDER BY
            me.observed_at DESC,
            me.id DESC
        LIMIT 1
        """,
        params,
    )


def get_latest_medicine_event_by_name(
    medicine_name: str,
    confirmed_only: bool = False,
) -> dict[str, Any] | None:

    medicine = (
        get_medicine_by_name(
            medicine_name
        )
    )

    if medicine is None:

        return None

    return get_latest_medicine_event(
        medicine_id=int(
            medicine[
                "id"
            ]
        ),
        confirmed_only=confirmed_only,
    )


def get_medicine_events_between(
    start_time: datetime,
    end_time: datetime,
    medicine_id: int | None = None,
    confirmed_only: bool = False,
) -> list[dict[str, Any]]:

    if start_time > end_time:

        raise ValueError(
            "start_time cannot be after end_time."
        )

    conditions = [
        "me.observed_at >= %s",
        "me.observed_at <= %s",
    ]

    params: list[Any] = [
        start_time,
        end_time,
    ]

    if medicine_id is not None:

        conditions.append(
            "me.medicine_id = %s"
        )

        params.append(
            medicine_id
        )

    if confirmed_only:

        conditions.append(
            "me.status = %s"
        )

        params.append(
            STATUS_CAMERA_CONFIRMED
        )

    where_clause = " AND ".join(
        conditions
    )

    return fetch_all(
        f"""
        SELECT
            me.*,
            m.medicine_name,
            m.dosage,
            m.ocr_keywords,
            m.instructions,
            p.display_name AS person_name,
            p.relation AS person_relation
        FROM medicine_events AS me
        INNER JOIN medicines AS m
            ON m.id = me.medicine_id
        LEFT JOIN registered_people AS p
            ON p.id = me.person_id
        WHERE {where_clause}
        ORDER BY
            me.observed_at DESC,
            me.id DESC
        """,
        params,
    )


def get_medicine_events_for_date(
    target_date: date,
    medicine_id: int | None = None,
    confirmed_only: bool = False,
) -> list[dict[str, Any]]:

    return get_medicine_events_between(
        start_time=datetime.combine(
            target_date,
            time.min,
        ),
        end_time=datetime.combine(
            target_date,
            time.max,
        ),
        medicine_id=medicine_id,
        confirmed_only=confirmed_only,
    )


def get_today_medicine_events(
    medicine_id: int | None = None,
    confirmed_only: bool = False,
) -> list[dict[str, Any]]:

    return get_medicine_events_for_date(
        target_date=date.today(),
        medicine_id=medicine_id,
        confirmed_only=confirmed_only,
    )


def get_recent_medicine_events(
    limit: int = 20,
    confirmed_only: bool = False,
) -> list[dict[str, Any]]:

    safe_limit = max(
        1,
        min(
            int(
                limit
            ),
            10000,
        ),
    )

    where = (
        "WHERE me.status = %s"
        if confirmed_only
        else ""
    )

    params = (
        (
            STATUS_CAMERA_CONFIRMED,
        )
        if confirmed_only
        else ()
    )

    return fetch_all(
        f"""
        SELECT
            me.*,
            m.medicine_name,
            m.dosage,
            m.ocr_keywords,
            m.instructions,
            p.display_name AS person_name,
            p.relation AS person_relation
        FROM medicine_events AS me
        INNER JOIN medicines AS m
            ON m.id = me.medicine_id
        LEFT JOIN registered_people AS p
            ON p.id = me.person_id
        {where}
        ORDER BY
            me.observed_at DESC,
            me.id DESC
        LIMIT {safe_limit}
        """,
        params,
    )


# ============================================================
# ACTIVE MEDICINE SCHEDULES
# ============================================================

def get_all_active_schedules(
) -> list[dict[str, Any]]:

    return fetch_all(
        """
        SELECT
            ms.id AS schedule_id,
            ms.medicine_id,
            ms.schedule_time,
            ms.schedule_label,
            ms.is_active AS schedule_active,
            m.medicine_name,
            m.dosage,
            m.ocr_keywords,
            m.instructions,
            m.is_active AS medicine_active
        FROM medicine_schedules AS ms
        INNER JOIN medicines AS m
            ON m.id = ms.medicine_id
        WHERE ms.is_active = TRUE
          AND m.is_active = TRUE
        ORDER BY
            ms.schedule_time ASC,
            m.medicine_name ASC
        """
    )