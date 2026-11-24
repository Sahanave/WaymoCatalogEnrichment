"""Static HTML report: summary table plus per-clip timelines (GT vs each run, mismatches in red)."""
from __future__ import annotations

import html
import os
from pathlib import Path
from typing import Any

from .clips import load_clip
from .metrics import format_table, load_run, scored_fields

CSS = """
body{font-family:system-ui,sans-serif;margin:24px;color:#1a1a1a} table{border-collapse:collapse;margin:12px 0}
td,th{border:1px solid #ddd;padding:4px 8px;font-size:13px;vertical-align:top} th{background:#f4f4f4}
.bad{background:#fde2e2} .ok{background:#e6f4ea} .na{color:#999} img{width:160px;display:block}
pre{background:#f7f7f7;padding:12px;overflow:auto} h2{margin-top:32px}
"""


def _cell(pred: Any, gt: Any) -> str:
    shown = "—" if pred is None else html.escape(str(pred))
    if gt is None:
        return f'<td class="na">{shown}</td>'
    return f'<td class="{"ok" if pred == gt else "bad"}">{shown}</td>'


def build_report(run_dirs: list[str | Path], scores: list[dict[str, Any]], clips_root: str | Path,
                 out_path: str | Path, max_clips: int = 5) -> Path:
    out_path = Path(out_path)
    clips_root = Path(clips_root)
    runs = {Path(d).name: {r["clip_id"]: r for r in load_run(d)} for d in run_dirs}
    settings = {Path(d).name: load_run(d)[0]["settings"] for d in run_dirs}
    names = list(runs)
    clip_ids = list(dict.fromkeys(cid for r in runs.values() for cid in r))[:max_clips]

    parts = [f"<html><head><meta charset='utf-8'><style>{CSS}</style><title>Waymo scenario enrichment</title></head><body>",
             "<h1>Temporal scenario labeling: per-frame vs rolling state</h1>",
             "<p>Ground truth is derived from Waymo Open Dataset v2 (stats, camera boxes, ego velocity). "
             "Red cells disagree with ground truth; grey cells have no ground truth.</p>",
             "<h2>Results</h2><pre>" + html.escape(format_table(scores)) + "</pre>"]

    for cid in clip_ids:
        clip = load_clip(clips_root / cid)
        gt_by_i = {f.i: f.gt for f in clip.frames}
        fields = list(dict.fromkeys(f for n in names for f in scored_fields(settings[n])))
        parts.append(f"<h2>Clip {html.escape(cid)}</h2><table><tr><th>frame</th><th>image</th>")
        for f in fields:
            parts.append(f"<th>{f}<br>GT</th>" + "".join(f"<th>{html.escape(n)}</th>" for n in names))
        parts.append("<th>notable event (last run)</th></tr>")
        for fr in clip.frames:
            img = os.path.relpath(fr.image_path, out_path.parent.resolve())
            row = [f"<tr><td>{fr.t_s:.0f}s</td><td><img src='{html.escape(img)}'></td>"]
            for f in fields:
                g = gt_by_i.get(fr.i, {}).get(f)
                row.append(f"<td>{'—' if g is None else html.escape(str(g))}</td>")
                for n in names:
                    rec = next((x for x in runs[n].get(cid, {}).get("frames", []) if x["i"] == fr.i), None)
                    row.append(_cell(rec["pred"].get(f) if rec else None, g) if f in scored_fields(settings[n]) else "<td class='na'>n/a</td>")
            last = next((x for x in runs[names[-1]].get(cid, {}).get("frames", []) if x["i"] == fr.i), None)
            ev = last["pred"].get("notable_event") if last else None
            row.append(f"<td>{html.escape(ev) if ev else ''}</td></tr>")
            parts.append("".join(row))
        parts.append("</table>")
    parts.append("</body></html>")
    out_path.write_text("\n".join(parts), encoding="utf-8")
    return out_path
