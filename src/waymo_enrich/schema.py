"""Label schema for driving-scene enrichment.

Kept deliberately small. Fields marked SCORED have ground truth derivable from Waymo Open
Dataset v2 components; the rest are enrichment-only (unscored).
"""
from __future__ import annotations

from typing import Any, Optional

ENUMS: dict[str, list[str]] = {
    "time_of_day": ["day", "dawn_dusk", "night"],
    "weather": ["dry", "rain"],
    "road_type": ["highway", "urban_street", "residential", "parking_lot", "other"],
    "ego_motion": ["stopped", "accelerating", "braking", "cruising"],
}
BOOLS = ["pedestrian_visible", "cyclist_visible"]

# Fields compared against Waymo-derived ground truth.
SCORED = ["time_of_day", "weather", "pedestrian_visible", "cyclist_visible", "ego_motion"]
# ego_motion ground truth comes from the same velocity signal we may feed the model (use_ego),
# so it is excluded from scoring in that configuration to avoid a circular result.
EGO_DERIVED = {"ego_motion"}
ALL_FIELDS = list(ENUMS) + BOOLS

SCHEMA_PROMPT = """Fields (use exactly these keys and allowed values):
- time_of_day: one of day | dawn_dusk | night
- weather: one of dry | rain
- road_type: one of highway | urban_street | residential | parking_lot | other
- pedestrian_visible: true or false (any pedestrian clearly visible in this frame)
- cyclist_visible: true or false (any cyclist or rider clearly visible in this frame)
- ego_motion: one of stopped | accelerating | braking | cruising (the camera vehicle's motion)
- notable_event: a short phrase for anything safety-relevant happening now, or null"""

_ALIASES = {
    "dawn/dusk": "dawn_dusk", "dusk": "dawn_dusk", "dawn": "dawn_dusk", "sunny": "dry", "clear": "dry",
    "rainy": "rain", "urban": "urban_street", "street": "urban_street", "city": "urban_street",
    "residential_street": "residential", "slowing": "braking", "decelerating": "braking",
    "speeding_up": "accelerating", "steady": "cruising", "moving": "cruising", "driving": "cruising",
}


def _norm_enum(field: str, value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    v = value.strip().lower().replace(" ", "_").replace("-", "_")
    v = _ALIASES.get(v, v)
    return v if v in ENUMS[field] else None


def _norm_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "yes", "1"}:
            return True
        if v in {"false", "no", "0"}:
            return False
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    return None


def normalize(raw: Any) -> dict[str, Any]:
    """Coerce a model output dict to the schema. Invalid values become None (never guessed)."""
    raw = raw if isinstance(raw, dict) else {}
    out: dict[str, Any] = {f: _norm_enum(f, raw.get(f)) for f in ENUMS}
    for f in BOOLS:
        out[f] = _norm_bool(raw.get(f))
    event = raw.get("notable_event")
    out["notable_event"] = event.strip() if isinstance(event, str) and event.strip().lower() not in {"", "none", "null"} else None
    return out
