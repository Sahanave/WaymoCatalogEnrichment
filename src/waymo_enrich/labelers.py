"""The two labelers compared in this project.

per_frame (A): each keyframe is labeled on its own.
rolling   (C): each keyframe is labeled with the previous frame image(s), the model's own previous
               state, and optionally ego speed. Same schema, same model; only the context differs.

Pipeline shape follows the retail blueprint: analyze -> categorize into a fixed schema -> normalize.
Outputs are constrained to the schema by `schema.normalize` (invalid values become None, never guessed).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional

from .clips import Clip
from .llm import image_part, parse_llm_json
from .schema import SCHEMA_PROMPT, normalize

logger = logging.getLogger("waymo_enrich.labelers")

SYSTEM = (
    "You label frames from a front-facing car camera for an autonomous-driving dataset. "
    "Describe only what is visible. Never invent objects. Return ONLY one JSON object, no markdown."
)


@dataclass
class Settings:
    mode: str = "per_frame"       # per_frame | rolling
    prev_frames: int = 1          # rolling: how many previous keyframe images to include (0 = state text only)
    use_ego: bool = False         # rolling: feed ego speed (m/s) to the model
    refresh_every: int = 0        # rolling: drop carried state every K frames (0 = never) to limit error carry-over
    retries: int = 2
    base_dir: Optional[str] = None  # temporal: run dir of per_frame labels (A) to compare each frame against


def _per_frame_parts(frame) -> list[dict]:
    text = (
        f"{SCHEMA_PROMPT}\n\nLabel this single frame. ego_motion is hard to judge from one frame; give your best guess.\n"
        'Return JSON with keys: time_of_day, weather, road_type, pedestrian_visible, cyclist_visible, ego_motion, notable_event.'
    )
    return [{"type": "text", "text": "FRAME:"}, image_part(frame.image_bytes()), {"type": "text", "text": text}]


def _rolling_parts(clip: Clip, k: int, state: Optional[dict], s: Settings) -> list[dict]:
    cur = clip.frames[k]
    parts: list[dict] = []
    for j in range(max(0, k - s.prev_frames), k):
        prev = clip.frames[j]
        parts.append({"type": "text", "text": f"PREVIOUS FRAME (t={prev.t_s:.1f}s):"})
        parts.append(image_part(prev.image_bytes()))
    parts.append({"type": "text", "text": f"CURRENT FRAME (t={cur.t_s:.1f}s):"})
    parts.append(image_part(cur.image_bytes()))

    lines = [SCHEMA_PROMPT, ""]
    if state is None:
        lines.append("PREVIOUS STATE: none (first frame or refreshed). Label the current frame.")
    else:
        lines.append("PREVIOUS STATE (your labels for the previous frame):")
        lines.append(json.dumps(state, ensure_ascii=False))
        lines.append(
            "Keep each previous value unless the current frame (compared with the previous frame) gives clear evidence "
            "that it changed. Use the frame-to-frame change to judge ego_motion (stopped / accelerating / braking / cruising)."
        )
    if s.use_ego and cur.speed_mps is not None:
        prev_speed = clip.frames[k - 1].speed_mps if k > 0 else None
        ego = f"EGO SPEED now: {cur.speed_mps:.1f} m/s"
        if prev_speed is not None:
            ego += f"; one keyframe ago: {prev_speed:.1f} m/s"
        lines.append(ego + ".")
    lines.append(
        "Return JSON with keys: time_of_day, weather, road_type, pedestrian_visible, cyclist_visible, ego_motion, "
        'notable_event, changed (list of field names whose value differs from the previous state).'
    )
    parts.append({"type": "text", "text": "\n".join(lines)})
    return parts


def _trace_input(parts: list[dict], clip: Clip, k: int, n_prev: int) -> dict:
    """What a model was given: frame indexes for images, and the text parts verbatim."""
    n_img = sum(1 for p in parts if p.get("type") == "image_url")
    first = max(0, k - n_prev) if n_img > 1 else k
    return {"images": list(range(first, first + n_img)) if n_img else [],
            "text": "\n".join(p["text"] for p in parts if p.get("type") == "text")}


# Fields a single frame can answer. A (per_frame) owns these; the temporal run only says whether they changed.
TEMPORAL_FIELDS = ["time_of_day", "weather", "road_type", "pedestrian_visible", "cyclist_visible"]
TEMPORAL_SYSTEM = (
    "You compare two consecutive frames from a front-facing car camera for an autonomous-driving dataset. "
    "Judge only what changed. Describe only what is visible. Return ONLY one JSON object, no markdown."
)


def load_base(base_dir: str, clip_id: str) -> dict[int, dict]:
    """Per-frame labels (run A) for a clip, keyed by frame index."""
    path = Path(base_dir) / f"{clip_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"no base labels for clip {clip_id} in {base_dir} (run label --mode per_frame first)")
    return {f["i"]: f for f in json.loads(path.read_text())["frames"]}


def _temporal_parts(clip: Clip, k: int, prev_labels: Optional[dict]) -> list[dict]:
    cur = clip.frames[k]
    parts: list[dict] = []
    if k > 0:
        parts += [{"type": "text", "text": f"PREVIOUS FRAME (t={clip.frames[k - 1].t_s:.1f}s):"},
                  image_part(clip.frames[k - 1].image_bytes())]
    parts += [{"type": "text", "text": f"CURRENT FRAME (t={cur.t_s:.1f}s):"}, image_part(cur.image_bytes())]
    lines = [SCHEMA_PROMPT, ""]
    if k == 0:
        lines.append("There is no previous frame, so changed must be an empty list. Give your best guess for ego_motion.")
    else:
        shown = json.dumps(prev_labels, ensure_ascii=False) if prev_labels else "unavailable"
        lines += [f"LABELS FOR THE PREVIOUS FRAME (from a single-frame labeler): {shown}", "",
                  "Which of these labels no longer hold for the CURRENT frame because the scene changed? List a field in "
                  "changed ONLY with clear visual evidence in the two frames (a pedestrian appearing or leaving, rain starting "
                  "or stopping, a new road type). Small shifts in camera perspective are not changes. Also judge ego_motion "
                  "from the change between the frames."]
    lines.append(f"Fields you may list in changed: {', '.join(TEMPORAL_FIELDS)}.")
    lines.append("Return JSON with keys: changed (list of field names), ego_motion, change_note (short phrase describing "
                 "what changed, or null).")
    parts.append({"type": "text", "text": "\n".join(lines)})
    return parts


def _label_temporal(clip: Clip, client, settings: Settings) -> dict[str, Any]:
    if not settings.base_dir:
        raise ValueError("temporal mode needs base_dir (the per_frame run to compare against)")
    base = load_base(settings.base_dir, clip.clip_id)
    frames_out: list[dict[str, Any]] = []
    for k, frame in enumerate(clip.frames):
        prev = base.get(clip.frames[k - 1].i) if k > 0 else None
        prev_labels = None
        if prev and prev.get("parse_ok"):
            prev_labels = {f: prev["pred"].get(f) for f in TEMPORAL_FIELDS}
        parts = _temporal_parts(clip, k, prev_labels)
        parsed, raw, err = _call(client, parts, settings.retries, TEMPORAL_SYSTEM)
        changed: list[str] = []
        pred = normalize(None)
        if parsed is not None:
            c = parsed.get("changed")
            changed = [f for f in (c if isinstance(c, list) else []) if f in TEMPORAL_FIELDS] if k > 0 else []
            note = parsed.get("change_note")
            pred = normalize({"ego_motion": parsed.get("ego_motion"), "notable_event": note})
        frames_out.append({
            "i": frame.i, "t_s": frame.t_s, "pred": pred, "changed": changed,
            "parse_ok": parsed is not None, "error": err, "raw": raw[:1500],
            "trace": {"vlm": {"input": _trace_input(parts, clip, k, 1), "output": raw}},
        })
    return {"clip_id": clip.clip_id, "settings": asdict(settings), "frames": frames_out}


def _call(client, parts: list[dict], retries: int, system: str = SYSTEM) -> tuple[Optional[dict], str, Optional[str]]:
    """Returns (parsed_or_None, raw_text, error_or_None)."""
    raw, err = "", None
    for attempt in range(retries + 1):
        try:
            raw = client.chat(parts, system=system)
        except Exception as exc:  # network/auth/etc: record and retry
            err = f"{type(exc).__name__}: {exc}"
            logger.warning("VLM call failed (attempt %d): %s", attempt + 1, err)
            continue
        parsed = parse_llm_json(raw)
        if parsed is not None:
            return parsed, raw, None
        err = "unparseable JSON"
    return None, raw, err


def label_clip(clip: Clip, client, settings: Settings) -> dict[str, Any]:
    if settings.mode == "temporal":
        return _label_temporal(clip, client, settings)
    frames_out: list[dict[str, Any]] = []
    state: Optional[dict] = None
    for k, frame in enumerate(clip.frames):
        if settings.mode == "per_frame":
            parts = _per_frame_parts(frame)
        elif settings.mode == "rolling":
            if settings.refresh_every and k % settings.refresh_every == 0:
                state = None
            parts = _rolling_parts(clip, k, state, settings)
        else:
            raise ValueError(f"unknown mode {settings.mode!r}")

        parsed, raw, err = _call(client, parts, settings.retries)
        pred = normalize(parsed)
        changed = parsed.get("changed") if isinstance(parsed, dict) and isinstance(parsed.get("changed"), list) else []
        if parsed is not None and settings.mode == "rolling":
            state = {k_: v for k_, v in pred.items() if k_ != "notable_event"}  # carry labels, not free text
        frames_out.append({
            "i": frame.i, "t_s": frame.t_s, "pred": pred, "changed": changed,
            "parse_ok": parsed is not None, "error": err, "raw": raw[:1500],
            "trace": {"vlm": {"input": _trace_input(parts, clip, k, settings.prev_frames), "output": raw}},
        })
    return {"clip_id": clip.clip_id, "settings": asdict(settings), "frames": frames_out}
