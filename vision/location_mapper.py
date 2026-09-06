from __future__ import annotations


def map_bbox_to_location(
    bbox: tuple[int, int, int, int],
    frame_width: int,
    frame_height: int,
) -> str:
    """
    Map the center of a detection to a coarse 3x3 screen location.

    This is intentionally an observed camera-region description, not a
    real-world room location such as 'table' or 'bed'.
    """
    x1, y1, x2, y2 = bbox

    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    horizontal_third = frame_width / 3
    vertical_third = frame_height / 3

    if center_x < horizontal_third:
        horizontal = "left"
    elif center_x < horizontal_third * 2:
        horizontal = "center"
    else:
        horizontal = "right"

    if center_y < vertical_third:
        vertical = "top"
    elif center_y < vertical_third * 2:
        vertical = "middle"
    else:
        vertical = "bottom"

    if horizontal == "center" and vertical == "middle":
        return "center area"

    return f"{vertical}-{horizontal} area"
