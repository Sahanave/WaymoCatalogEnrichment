"""Scoring: accuracy vs Waymo-derived ground truth, label stability (flip rate), parse failures."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .clips import load_clip
from .schema import BOOLS, EGO_DERIVED, SCORED


def load_run(run_dir: str | Path) -> list[dict[str, Any]]:
    return [json.loads(p.read_text()) for p in sorted(Path(run_dir).glob("*.json")) if p.name != "scores.json"]


def scored_fields(settings: dict[str, Any]) -> list[str]:
    if settings.get("mode") == "temporal":
        return ["ego_motion"]  # the other labels belong to the per_frame run; temporal is scored on changes
    use_ego = settings.get("mode") == "rolling" and settings.get("use_ego")
    return [f for f in SCORED if not (use_ego and f in EGO_DERIVED)]


def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f1}


def _score_changes(results: list[dict[str, Any]], clips_root: Path) -> dict[str, Any]:
    """Change detection: did the model flag a field as changed exactly when Waymo's label changed between frames?"""
    from .labelers import TEMPORAL_FIELDS

    fields = [f for f in TEMPORAL_FIELDS if f in SCORED]
    cm = {f: [0, 0, 0, 0] for f in fields}  # tp, fp, fn, tn
    for res in results:
        gt_by_i = {fr.i: fr.gt for fr in load_clip(clips_root / res["clip_id"]).frames}
        frames = res["frames"]
        for a, b in zip(frames, frames[1:]):
            if not b["parse_ok"]:
                # an unparseable frame flags nothing, so any real change counts as missed
                flagged: set[str] = set()
            else:
                flagged = set(b.get("changed") or [])
            for f in fields:
                ga, gb = gt_by_i.get(a["i"], {}).get(f), gt_by_i.get(b["i"], {}).get(f)
                if ga is None or gb is None:
                    continue
                real, said = ga != gb, f in flagged
                cm[f][0 if real and said else 1 if said else 2 if real else 3] += 1
    out = {}
    for f, (tp, fp, fn, tn) in cm.items():
        out[f] = {"pairs": tp + fp + fn + tn, "gt_changes": tp + fn, "flagged": tp + fp, "tp": tp, "fp": fp, "fn": fn,
                  **_prf(tp, fp, fn)}
    return out


def score_run(run_dir: str | Path, clips_root: str | Path) -> dict[str, Any]:
    results = load_run(run_dir)
    if not results:
        raise FileNotFoundError(f"no clip results in {run_dir}")
    settings = results[0]["settings"]
    fields = scored_fields(settings)
    clips_root = Path(clips_root)

    correct = {f: 0 for f in fields}
    total = {f: 0 for f in fields}
    conf = {f: [0, 0, 0] for f in BOOLS if f in fields}  # tp, fp, fn
    pred_flips = {f: 0 for f in fields}
    pred_pairs = {f: 0 for f in fields}
    gt_flips = {f: 0 for f in fields}
    gt_pairs = {f: 0 for f in fields}
    n_frames = parse_fail = 0

    for res in results:
        clip = load_clip(clips_root / res["clip_id"])
        gt_by_i = {fr.i: fr.gt for fr in clip.frames}
        frames = res["frames"]
        n_frames += len(frames)
        parse_fail += sum(1 for fr in frames if not fr["parse_ok"])
        for f in fields:
            for fr in frames:
                g = gt_by_i.get(fr["i"], {}).get(f)
                if g is None:
                    continue
                p = fr["pred"].get(f)
                total[f] += 1
                correct[f] += int(p == g)
                if f in conf:
                    if p is True and g is True:
                        conf[f][0] += 1
                    elif p is True and g is False:
                        conf[f][1] += 1
                    elif p is not True and g is True:
                        conf[f][2] += 1
            for a, b in zip(frames, frames[1:]):
                pa, pb = a["pred"].get(f), b["pred"].get(f)
                if pa is not None and pb is not None:
                    pred_pairs[f] += 1
                    pred_flips[f] += int(pa != pb)
                ga, gb = gt_by_i.get(a["i"], {}).get(f), gt_by_i.get(b["i"], {}).get(f)
                if ga is not None and gb is not None:
                    gt_pairs[f] += 1
                    gt_flips[f] += int(ga != gb)

    per_field: dict[str, Any] = {}
    for f in fields:
        entry: dict[str, Any] = {
            "n": total[f],
            "accuracy": correct[f] / total[f] if total[f] else None,
            "pred_flip_rate": pred_flips[f] / pred_pairs[f] if pred_pairs[f] else None,
            "gt_flip_rate": gt_flips[f] / gt_pairs[f] if gt_pairs[f] else None,
        }
        if f in conf and total[f] and sum(conf[f]):
            entry.update(_prf(*conf[f]))
        per_field[f] = entry

    changes = _score_changes(results, clips_root) if settings.get("mode") == "temporal" else None
    return {
        "changes": changes,
        "run": Path(run_dir).name, "settings": settings, "n_clips": len(results), "n_frames": n_frames,
        "parse_fail_rate": parse_fail / n_frames if n_frames else None, "fields": per_field,
    }


def format_table(scores: list[dict[str, Any]]) -> str:
    """Markdown comparison table, one column per run."""
    fields = list(dict.fromkeys(f for s in scores for f in s["fields"]))
    header = "| metric | " + " | ".join(s["run"] for s in scores) + " |"
    sep = "|---|" + "---|" * len(scores)
    rows = [header, sep]

    def cell(v: Any, pct: bool = True) -> str:
        return "n/a" if v is None else (f"{v:.1%}" if pct else f"{v}")

    for f in fields:
        rows.append(f"| {f} accuracy | " + " | ".join(cell(s["fields"].get(f, {}).get("accuracy")) for s in scores) + " |")
    for f in fields:
        rows.append(f"| {f} flip rate (pred / gt) | " + " | ".join(
            f"{cell(s['fields'][f]['pred_flip_rate'])} / {cell(s['fields'][f]['gt_flip_rate'])}" if f in s["fields"] else "n/a"
            for s in scores) + " |")
    for f in BOOLS:
        if any(f in s["fields"] for s in scores):
            rows.append(f"| {f} F1 | " + " | ".join(cell(s["fields"].get(f, {}).get("f1")) for s in scores) + " |")
    for f in dict.fromkeys(f for s in scores for f in (s.get("changes") or {})):
        rows.append(f"| {f} change detection (P / R / F1; real changes) | " + " | ".join(
            (lambda c: "n/a" if not c else "no real changes" if not c["gt_changes"] and not c["flagged"] else
                       f"{c['precision']:.0%} / {c['recall']:.0%} / {c['f1']:.0%}; {c['gt_changes']} real, {c['flagged']} flagged")(
                (s.get("changes") or {}).get(f)) for s in scores) + " |")
    rows.append("| parse failures | " + " | ".join(cell(s["parse_fail_rate"]) for s in scores) + " |")
    rows.append("| frames scored | " + " | ".join(cell(s["n_frames"], pct=False) for s in scores) + " |")
    return "\n".join(rows)
