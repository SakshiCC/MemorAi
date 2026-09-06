from __future__ import annotations

from database.db_manager import (
    initialize_database,
)

from database.medicine_repository import (
    get_medicine_by_name,
    update_medicine_ocr_keywords,
)


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print()
    print(
        "MemorAi Medicine OCR Keyword Backfill"
    )

    print(
        "====================================="
    )

    initialize_database()

    medicine = (
        get_medicine_by_name(
            "omez"
        )
    )

    if medicine is None:

        raise RuntimeError(
            "Omez is not registered."
        )

    medicine_id = int(
        medicine[
            "id"
        ]
    )

    keywords = [
        "omez",
        "omeprazole",
    ]

    update_medicine_ocr_keywords(
        medicine_id=medicine_id,
        ocr_keywords=keywords,
    )

    updated = (
        get_medicine_by_name(
            "omez"
        )
    )

    print()
    print(
        "PASS"
    )

    print(
        "Medicine:",
        updated[
            "medicine_name"
        ],
    )

    print(
        "Dosage:",
        updated[
            "dosage"
        ],
    )

    print(
        "OCR keywords:",
        updated[
            "ocr_keywords"
        ],
    )


if __name__ == "__main__":

    main()