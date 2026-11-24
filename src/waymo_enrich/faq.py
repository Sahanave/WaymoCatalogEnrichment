"""FAQ over the label catalog: a text-only LLM answers questions about a clip from the A (per_frame) and C (temporal) labels.

The LLM never sees images, so it can only answer what the catalog says. Questions with a checkable answer are graded
against Waymo-derived ground truth.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from .clips import Clip, load_clip
from .labelers import _call
from .metrics import load_run

SYSTEM = (
    "You answer questions about a driving clip using ONLY the catalog text provided, which lists one line per second. "
    "You cannot see the video. If the catalog does not say, answer that it is unknown. Return ONLY one JSON object, no markdown."
)

# kind: how the structured value is checked. first_t: first second a field is true (null if never). bool: yes/no. text: not graded.
QUESTIONS: list[dict[str, str]] = [
    {"id": "ped_first", "kind": "first_t", "field": "pedestrian_visible", "key": "t_s",
     "q": "At what time in seconds does a pedestrian first become visible? Use null if never."},
    {"id": "cyc_first", "kind": "first_t", "field": "cyclist_visible", "key": "t_s",
     "q": "At what time in seconds does a cyclist first become visible? Use null if never."},
    {"id": "scene_changed", "kind": "bool", "key": "changed",
     "q": "Did the weather or the time of day change during the clip? changed is true or false."},
    {"id": "stopped", "kind": "bool", "key": "stopped",
     "q": "Was the camera vehicle stopped at any point? stopped is true or false."},
    {"id": "summary", "kind": "text", "key": None,
     "q": "Summarize the clip in two sentences."},
]


def build_catalog(clip: Clip, a_run: dict[str, Any], c_run: dict[str, Any]) -> str:
    """One line per keyframe from A (scene attributes) and C (changes, ego motion, change note)."""
    a = {f["i"]: f for f in a_run["frames"]}
    c = {f["i"]: f for f in c_run["frames"]}
    yn = lambda v: "unknown" if v is None else ("yes" if v else "no")
    lines = []
    for fr in clip.frames:
        pa, fc = a.get(fr.i), c.get(fr.i)
        p = (pa or {}).get("pred", {})
        cp = (fc or {}).get("pred", {})
        line = (f"t={fr.t_s:.0f}s | time_of_day={p.get('time_of_day')}, weather={p.get('weather')}, road={p.get('road_type')}, "
                f"pedestrian={yn(p.get('pedestrian_visible'))}, cyclist={yn(p.get('cyclist_visible'))} | "
                f"ego_motion={cp.get('ego_motion')}")
        if fc and fc.get("changed"):
            line += f" | changed: {', '.join(fc['changed'])}"
        if cp.get("notable_event"):
            line += f" | note: {cp['notable_event']}"
        lines.append(line)
    return "\n".join(lines)


def ground_truth(clip: Clip, q: dict[str, str]) -> Any:
    g = [fr.gt for fr in clip.frames]
    if q["kind"] == "first_t":
        return next((fr.t_s for fr in clip.frames if fr.gt.get(q["field"]) is True), None)
    if q["id"] == "scene_changed":
        return any(len({x.get(f) for x in g if x.get(f) is not None}) > 1 for f in ("weather", "time_of_day"))
    if q["id"] == "stopped":
        return any(x.get("ego_motion") == "stopped" for x in g)
    return None


def _correct(q: dict[str, str], value: Any, gt: Any) -> Optional[bool]:
    if q["kind"] == "text":
        return None
    if q["kind"] == "first_t":
        if value is None or gt is None:
            return value is None and gt is None
        try:
            return abs(float(value) - float(gt)) <= 0.5
        except (TypeError, ValueError):
            return False
    return value is gt if isinstance(value, bool) else False


def faq_clip(clip: Clip, a_run: dict[str, Any], c_run: dict[str, Any], llm) -> dict[str, Any]:
    from .llm import parse_llm_json  # noqa: F401  (used via _call)

    catalog = build_catalog(clip, a_run, c_run)
    items = []
    for q in QUESTIONS:
        keys = f'"answer" (one or two sentences citing times)' + (f', "{q["key"]}"' if q["key"] else "")
        prompt = f"CATALOG:\n{catalog}\n\nQUESTION: {q['q']}\nReturn JSON with keys: {keys}."
        parsed, raw, err = _call(llm, [{"type": "text", "text": prompt}], 2, SYSTEM)
        value = parsed.get(q["key"]) if parsed and q["key"] else None
        gt = ground_truth(clip, q)
        items.append({"id": q["id"], "question": q["q"], "input": prompt, "output": raw,
                      "answer": (parsed or {}).get("answer"), "value": value, "gt": gt,
                      "correct": _correct(q, value, gt) if parsed else (False if q["kind"] != "text" else None),
                      "parse_ok": parsed is not None})
    return {"clip_id": clip.clip_id, "catalog": catalog, "items": items}


def run_faq(a_dir: str, c_dir: str, clips_root: str, clip_suffixes: list[str], out_dir: str, llm) -> list[str]:
    a = {r["clip_id"]: r for r in load_run(a_dir)}
    c = {r["clip_id"]: r for r in load_run(c_dir)}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    done = []
    for cid in a:
        if clip_suffixes and not any(cid.endswith(s) for s in clip_suffixes):
            continue
        if cid not in c:
            continue
        res = faq_clip(load_clip(Path(clips_root) / cid), a[cid], c[cid], llm)
        (out / f"{cid}.json").write_text(json.dumps(res, indent=1))
        done.append(cid)
    return done
