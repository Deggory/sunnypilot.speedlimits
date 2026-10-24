"""What an "Other sign" box is (``sign_type``): the kinds the review offers, the same ids as the page's
"Kind" select (web/src/review/catalog.ts SIGN_KIND_GROUPS).

Some give a limit without a number (town entry, motorway, expressway), some carry numbers that are no
speed (weight, height, width, minimum speed): the model has to know them apart from the speed signs.
Direction signs say where they point. Only a box of family ``other_sign`` has a kind; ``None`` means not
said (older reviews, or not chosen).
"""
from __future__ import annotations

SIGN_TYPES = (
    # Danger
    'pedestrian_crossing', 'children', 'traffic_queue', 'roadworks', 'danger_other',
    # Priority
    'give_way', 'stop', 'main_road', 'main_road_end',
    # Bans and other numbers
    'no_overtaking', 'no_overtaking_end', 'no_entry', 'minimum_speed', 'weight_limit', 'height_limit',
    'width_limit',
    # Roads and places
    'town_entry', 'town_exit', 'motorway_start', 'motorway_end', 'expressway_start', 'expressway_end',
    'roundabout', 'one_way',
    # Directions
    'direction_left', 'direction_right', 'direction_straight', 'direction_several',
    'other',
)


def sign_type_problem(detection: dict) -> str | None:
    """Why the sign_type of a box is not allowed, or None when it is."""
    value = detection.get('sign_type')
    if value is None:
        return None
    if value not in SIGN_TYPES:
        return 'sign_type must be one of: ' + ', '.join(SIGN_TYPES) + ', or null'
    if detection.get('family') != 'other_sign':
        return 'sign_type is only for boxes of family other_sign'
    return None
