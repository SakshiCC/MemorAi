from __future__ import annotations

from typing import Any, Iterable

from mysql.connector import Error

from database.mysql_connection import (
    database_connection,
)

from utils.logger import get_logger


# ============================================================
# LOGGER
# ============================================================

logger = get_logger(
    "database.manager",
    "database.log",
)


# ============================================================
# OBJECT MEMORY EVENTS
# ============================================================

EVENTS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS events (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,

    event_type VARCHAR(50)
        NOT NULL
        DEFAULT 'object_detected',

    object_name VARCHAR(120)
        NOT NULL,

    confidence DECIMAL(6,5)
        NULL,

    location VARCHAR(120)
        NULL,

    detected_at DATETIME(6)
        NOT NULL,

    image_path VARCHAR(1024)
        NULL,

    memory_text TEXT
        NULL,

    created_at TIMESTAMP
        NOT NULL
        DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (id),

    INDEX idx_events_object_time (
        object_name,
        detected_at
    ),

    INDEX idx_events_detected_at (
        detected_at
    ),

    INDEX idx_events_location (
        location
    )
)
ENGINE=InnoDB
DEFAULT CHARSET=utf8mb4
COLLATE=utf8mb4_unicode_ci;
"""


# ============================================================
# REGISTERED PEOPLE
# ============================================================

REGISTERED_PEOPLE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS registered_people (
    id BIGINT UNSIGNED
        NOT NULL
        AUTO_INCREMENT,

    name VARCHAR(120)
        NOT NULL,

    display_name VARCHAR(120)
        NULL,

    relation VARCHAR(120)
        NULL,

    profile_image_path VARCHAR(1024)
        NULL,

    notes VARCHAR(500)
        NULL,

    is_active BOOLEAN
        NOT NULL
        DEFAULT TRUE,

    created_at TIMESTAMP
        NOT NULL
        DEFAULT CURRENT_TIMESTAMP,

    updated_at TIMESTAMP
        NOT NULL
        DEFAULT CURRENT_TIMESTAMP
        ON UPDATE CURRENT_TIMESTAMP,

    PRIMARY KEY (id),

    UNIQUE KEY uq_registered_people_name (
        name
    ),

    INDEX idx_registered_people_active (
        is_active
    ),

    INDEX idx_registered_people_relation (
        relation
    )
)
ENGINE=InnoDB
DEFAULT CHARSET=utf8mb4
COLLATE=utf8mb4_unicode_ci;
"""


# ============================================================
# PERSON FACE EMBEDDINGS
# ============================================================

PERSON_FACE_EMBEDDINGS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS person_face_embeddings (
    id BIGINT UNSIGNED
        NOT NULL
        AUTO_INCREMENT,

    person_id BIGINT UNSIGNED
        NOT NULL,

    embedding BLOB
        NOT NULL,

    embedding_dim SMALLINT UNSIGNED
        NOT NULL,

    embedding_dtype VARCHAR(20)
        NOT NULL
        DEFAULT 'float32',

    sample_image_path VARCHAR(1024)
        NULL,

    created_at TIMESTAMP
        NOT NULL
        DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (id),

    INDEX idx_face_embeddings_person (
        person_id
    ),

    CONSTRAINT fk_face_embedding_person
        FOREIGN KEY (
            person_id
        )
        REFERENCES registered_people (
            id
        )
        ON DELETE CASCADE
        ON UPDATE CASCADE
)
ENGINE=InnoDB
DEFAULT CHARSET=utf8mb4
COLLATE=utf8mb4_unicode_ci;
"""


# ============================================================
# PERSON EVENTS
# ============================================================

PERSON_EVENTS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS person_events (
    id BIGINT UNSIGNED
        NOT NULL
        AUTO_INCREMENT,

    person_id BIGINT UNSIGNED
        NOT NULL,

    person_name VARCHAR(120)
        NOT NULL,

    similarity DECIMAL(7,6)
        NULL,

    image_path VARCHAR(1024)
        NULL,

    detected_at DATETIME(6)
        NOT NULL,

    memory_text TEXT
        NULL,

    created_at TIMESTAMP
        NOT NULL
        DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (id),

    INDEX idx_person_events_person_time (
        person_id,
        detected_at
    ),

    INDEX idx_person_events_name_time (
        person_name,
        detected_at
    ),

    INDEX idx_person_events_detected_at (
        detected_at
    ),

    CONSTRAINT fk_person_event_person
        FOREIGN KEY (
            person_id
        )
        REFERENCES registered_people (
            id
        )
        ON DELETE RESTRICT
        ON UPDATE CASCADE
)
ENGINE=InnoDB
DEFAULT CHARSET=utf8mb4
COLLATE=utf8mb4_unicode_ci;
"""


# ============================================================
# CONVERSATION SESSIONS
# ============================================================

CONVERSATION_SESSIONS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS conversation_sessions (
    id BIGINT UNSIGNED
        NOT NULL
        AUTO_INCREMENT,

    title VARCHAR(255)
        NULL,

    participant_text VARCHAR(500)
        NULL,

    full_transcript LONGTEXT
        NULL,

    summary TEXT
        NULL,

    status VARCHAR(30)
        NOT NULL
        DEFAULT 'recording',

    started_at DATETIME(6)
        NOT NULL,

    ended_at DATETIME(6)
        NULL,

    created_at TIMESTAMP
        NOT NULL
        DEFAULT CURRENT_TIMESTAMP,

    updated_at TIMESTAMP
        NOT NULL
        DEFAULT CURRENT_TIMESTAMP
        ON UPDATE CURRENT_TIMESTAMP,

    PRIMARY KEY (id),

    INDEX idx_conversation_started_at (
        started_at
    ),

    INDEX idx_conversation_ended_at (
        ended_at
    ),

    INDEX idx_conversation_status (
        status
    )
)
ENGINE=InnoDB
DEFAULT CHARSET=utf8mb4
COLLATE=utf8mb4_unicode_ci;
"""


# ============================================================
# CONVERSATION SEGMENTS
# ============================================================

CONVERSATION_SEGMENTS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS conversation_segments (
    id BIGINT UNSIGNED
        NOT NULL
        AUTO_INCREMENT,

    conversation_id BIGINT UNSIGNED
        NOT NULL,

    sequence_no INT UNSIGNED
        NOT NULL,

    transcript TEXT
        NOT NULL,

    started_at DATETIME(6)
        NULL,

    ended_at DATETIME(6)
        NULL,

    created_at TIMESTAMP
        NOT NULL
        DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (id),

    UNIQUE KEY uq_conversation_segment_sequence (
        conversation_id,
        sequence_no
    ),

    INDEX idx_conversation_segments_session (
        conversation_id
    ),

    CONSTRAINT fk_conversation_segment_session
        FOREIGN KEY (
            conversation_id
        )
        REFERENCES conversation_sessions (
            id
        )
        ON DELETE CASCADE
        ON UPDATE CASCADE
)
ENGINE=InnoDB
DEFAULT CHARSET=utf8mb4
COLLATE=utf8mb4_unicode_ci;
"""


# ============================================================
# CONVERSATION SETTINGS
# ============================================================

CONVERSATION_SETTINGS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS conversation_settings (
    id TINYINT UNSIGNED
        NOT NULL,

    memory_enabled BOOLEAN
        NOT NULL
        DEFAULT FALSE,

    importance_threshold DECIMAL(4,3)
        NOT NULL
        DEFAULT 0.600,

    updated_at TIMESTAMP
        NOT NULL
        DEFAULT CURRENT_TIMESTAMP
        ON UPDATE CURRENT_TIMESTAMP,

    PRIMARY KEY (id)
)
ENGINE=InnoDB
DEFAULT CHARSET=utf8mb4
COLLATE=utf8mb4_unicode_ci;
"""


# ============================================================
# MEDICINES
# ============================================================

MEDICINES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS medicines (
    id INT
        NOT NULL
        AUTO_INCREMENT,

    medicine_name VARCHAR(150)
        NOT NULL,

    dosage VARCHAR(100)
        NULL,

    ocr_keywords TEXT
        NULL,

    instructions VARCHAR(255)
        NULL,

    schedule_time TIME
        NOT NULL,

    schedule_label VARCHAR(50)
        NULL,

    is_active BOOLEAN
        NOT NULL
        DEFAULT TRUE,

    created_at DATETIME
        NOT NULL
        DEFAULT CURRENT_TIMESTAMP,

    updated_at DATETIME
        NOT NULL
        DEFAULT CURRENT_TIMESTAMP
        ON UPDATE CURRENT_TIMESTAMP,

    PRIMARY KEY (id),

    INDEX idx_medicines_name (
        medicine_name
    ),

    INDEX idx_medicines_active (
        is_active
    ),

    INDEX idx_medicines_schedule_time (
        schedule_time
    )
)
ENGINE=InnoDB
DEFAULT CHARSET=utf8mb4
COLLATE=utf8mb4_unicode_ci;
"""


# ============================================================
# MEDICINE SCHEDULES
# ============================================================

MEDICINE_SCHEDULES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS medicine_schedules (
    id INT
        NOT NULL
        AUTO_INCREMENT,

    medicine_id INT
        NOT NULL,

    schedule_time TIME
        NOT NULL,

    schedule_label VARCHAR(50)
        NULL,

    is_active BOOLEAN
        NOT NULL
        DEFAULT TRUE,

    created_at DATETIME
        NOT NULL
        DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (id),

    INDEX idx_schedule_medicine (
        medicine_id
    ),

    INDEX idx_schedule_time (
        schedule_time
    ),

    CONSTRAINT fk_schedule_medicine
        FOREIGN KEY (
            medicine_id
        )
        REFERENCES medicines (
            id
        )
        ON DELETE CASCADE
)
ENGINE=InnoDB
DEFAULT CHARSET=utf8mb4
COLLATE=utf8mb4_unicode_ci;
"""


# ============================================================
# MEDICINE REFERENCE IMAGES
# ============================================================

MEDICINE_REFERENCE_IMAGES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS medicine_reference_images (
    id INT
        NOT NULL
        AUTO_INCREMENT,

    medicine_id INT
        NOT NULL,

    image_path VARCHAR(500)
        NOT NULL,

    embedding LONGBLOB
        NULL,

    embedding_model VARCHAR(100)
        NULL,

    is_primary BOOLEAN
        NOT NULL
        DEFAULT FALSE,

    created_at DATETIME
        NOT NULL
        DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (id),

    INDEX idx_reference_medicine_id (
        medicine_id
    ),

    CONSTRAINT fk_medicine_reference_image
        FOREIGN KEY (
            medicine_id
        )
        REFERENCES medicines (
            id
        )
        ON DELETE CASCADE
)
ENGINE=InnoDB
DEFAULT CHARSET=utf8mb4
COLLATE=utf8mb4_unicode_ci;
"""


# ============================================================
# MEDICINE EVENTS
# ============================================================

MEDICINE_EVENTS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS medicine_events (
    id INT
        NOT NULL
        AUTO_INCREMENT,

    medicine_id INT
        NOT NULL,

    person_id BIGINT UNSIGNED
        NULL,

    event_type VARCHAR(50)
        NOT NULL
        DEFAULT 'camera_confirmation',

    status VARCHAR(50)
        NOT NULL
        DEFAULT 'camera_confirmed',

    confidence DECIMAL(6,5)
        NULL,

    medicine_similarity DECIMAL(6,5)
        NULL,

    person_similarity DECIMAL(6,5)
        NULL,

    action_confidence DECIMAL(6,5)
        NULL,

    evidence_image_path VARCHAR(500)
        NULL,

    observed_at DATETIME
        NOT NULL
        DEFAULT CURRENT_TIMESTAMP,

    scheduled_for DATETIME
        NULL,

    confirmation_method VARCHAR(50)
        NOT NULL
        DEFAULT 'camera_assisted',

    memory_text TEXT
        NULL,

    created_at DATETIME
        NOT NULL
        DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (id),

    INDEX idx_medicine_events_medicine (
        medicine_id
    ),

    INDEX idx_medicine_events_person (
        person_id
    ),

    INDEX idx_medicine_events_observed_at (
        observed_at
    ),

    INDEX idx_medicine_events_status (
        status
    ),

    CONSTRAINT fk_medicine_event_medicine
        FOREIGN KEY (
            medicine_id
        )
        REFERENCES medicines (
            id
        )
        ON DELETE CASCADE,

    CONSTRAINT fk_medicine_event_person
        FOREIGN KEY (
            person_id
        )
        REFERENCES registered_people (
            id
        )
        ON DELETE SET NULL
)
ENGINE=InnoDB
DEFAULT CHARSET=utf8mb4
COLLATE=utf8mb4_unicode_ci;
"""


# ============================================================
# SCHEMA HELPERS
# ============================================================

def _column_exists(
    cursor,
    table_name: str,
    column_name: str,
) -> bool:

    cursor.execute(
        """
        SELECT COUNT(*)
        FROM information_schema.COLUMNS
        WHERE TABLE_SCHEMA = DATABASE()
          AND TABLE_NAME = %s
          AND COLUMN_NAME = %s
        """,
        (
            table_name,
            column_name,
        ),
    )

    result = cursor.fetchone()

    return bool(
        result
        and int(
            result[0]
        ) > 0
    )


def _ensure_people_schema(
    cursor,
) -> None:

    if not _column_exists(
        cursor,
        "registered_people",
        "relation",
    ):

        cursor.execute(
            """
            ALTER TABLE registered_people
            ADD COLUMN relation VARCHAR(120) NULL
            AFTER display_name
            """
        )

        logger.info(
            "Added relation column to registered_people."
        )

    if not _column_exists(
        cursor,
        "registered_people",
        "profile_image_path",
    ):

        cursor.execute(
            """
            ALTER TABLE registered_people
            ADD COLUMN profile_image_path VARCHAR(1024) NULL
            AFTER relation
            """
        )

        logger.info(
            "Added profile_image_path column "
            "to registered_people."
        )


# ============================================================
# ENSURE MEDICINE OCR COLUMN
# ============================================================

def _ensure_medicine_schema(
    cursor,
) -> None:

    if not _column_exists(
        cursor,
        "medicines",
        "ocr_keywords",
    ):

        cursor.execute(
            """
            ALTER TABLE medicines
            ADD COLUMN ocr_keywords TEXT NULL
            AFTER dosage
            """
        )

        logger.info(
            "Added ocr_keywords column to medicines."
        )


def _ensure_default_conversation_settings(
    cursor,
) -> None:

    cursor.execute(
        """
        INSERT INTO conversation_settings (
            id,
            memory_enabled,
            importance_threshold
        )
        VALUES (
            1,
            FALSE,
            0.600
        )
        ON DUPLICATE KEY UPDATE
            id = id
        """
    )


# ============================================================
# INITIALIZE DATABASE
# ============================================================

def initialize_database() -> None:

    table_queries = (
        (
            "events",
            EVENTS_TABLE_SQL,
        ),
        (
            "registered_people",
            REGISTERED_PEOPLE_TABLE_SQL,
        ),
        (
            "person_face_embeddings",
            PERSON_FACE_EMBEDDINGS_TABLE_SQL,
        ),
        (
            "person_events",
            PERSON_EVENTS_TABLE_SQL,
        ),
        (
            "conversation_sessions",
            CONVERSATION_SESSIONS_TABLE_SQL,
        ),
        (
            "conversation_segments",
            CONVERSATION_SEGMENTS_TABLE_SQL,
        ),
        (
            "conversation_settings",
            CONVERSATION_SETTINGS_TABLE_SQL,
        ),
        (
            "medicines",
            MEDICINES_TABLE_SQL,
        ),
        (
            "medicine_schedules",
            MEDICINE_SCHEDULES_TABLE_SQL,
        ),
        (
            "medicine_reference_images",
            MEDICINE_REFERENCE_IMAGES_TABLE_SQL,
        ),
        (
            "medicine_events",
            MEDICINE_EVENTS_TABLE_SQL,
        ),
    )

    with database_connection() as connection:

        cursor = connection.cursor()

        try:

            for (
                table_name,
                query,
            ) in table_queries:

                cursor.execute(
                    query
                )

                logger.info(
                    "Database table checked: %s",
                    table_name,
                )

            _ensure_people_schema(
                cursor
            )

            _ensure_medicine_schema(
                cursor
            )

            _ensure_default_conversation_settings(
                cursor
            )

            connection.commit()

            logger.info(
                "Database tables initialized."
            )

        except Error:

            connection.rollback()

            logger.exception(
                "Database initialization failed."
            )

            raise

        finally:

            cursor.close()


# ============================================================
# GENERIC WRITE
# ============================================================

def execute(
    query: str,
    params: Iterable[Any] | None = None,
) -> int:

    with database_connection() as connection:

        cursor = connection.cursor()

        try:

            cursor.execute(
                query,
                tuple(
                    params or ()
                ),
            )

            connection.commit()

            return int(
                cursor.lastrowid or 0
            )

        except Error:

            connection.rollback()

            logger.exception(
                "Database write failed."
            )

            raise

        finally:

            cursor.close()


# ============================================================
# FETCH ALL
# ============================================================

def fetch_all(
    query: str,
    params: Iterable[Any] | None = None,
) -> list[dict[str, Any]]:

    with database_connection() as connection:

        cursor = connection.cursor(
            dictionary=True,
        )

        try:

            cursor.execute(
                query,
                tuple(
                    params or ()
                ),
            )

            return list(
                cursor.fetchall()
            )  # type: ignore

        finally:

            cursor.close()


# ============================================================
# FETCH ONE
# ============================================================

def fetch_one(
    query: str,
    params: Iterable[Any] | None = None,
) -> dict[str, Any] | None:

    rows = fetch_all(
        query,
        params,
    )

    return (
        rows[0]
        if rows
        else None
    )