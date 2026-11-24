"""Interactive single-file HTML viewer: scorecards, a frame viewer, and a timeline of GT vs each run.

No server needed: open the file in a browser. Images are referenced by relative path, so keep the report
next to (or above) the clips directory.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .clips import load_clip
from .metrics import load_run, scored_fields
from .schema import BOOLS


def _data(run_dirs, scores, clips_root: Path, out_dir: Path, max_clips: int) -> dict[str, Any]:
    names = [Path(d).name for d in run_dirs]
    runs = {n: {r["clip_id"]: r for r in load_run(d)} for n, d in zip(names, run_dirs)}
    settings = {n: load_run(d)[0]["settings"] for n, d in zip(names, run_dirs)}
    clip_ids = list(dict.fromkeys(cid for r in runs.values() for cid in r))[:max_clips]
    fields = list(dict.fromkeys(f for n in names for f in scored_fields(settings[n])))
    extra = ["road_type"] if any("road_type" in (x["pred"] or {}) for n in names for r in runs[n].values()
                                 for x in r.get("frames", [])) and "road_type" not in fields else []
    clips = []
    for cid in clip_ids:
        clip = load_clip(clips_root / cid)
        frames = []
        for fr in clip.frames:
            entry = {"i": fr.i, "t": round(fr.t_s, 1), "gt": fr.gt, "img": os.path.relpath(fr.image_path, out_dir.resolve()),
                     "speed": getattr(fr, "speed_mps", None), "runs": {}}
            for n in names:
                rec = next((x for x in runs[n].get(cid, {}).get("frames", []) if x["i"] == fr.i), None)
                if rec:
                    entry["runs"][n] = {"pred": rec["pred"], "changed": rec.get("changed", []), "ok": rec.get("parse_ok", True)}
            frames.append(entry)
        clips.append({"id": cid, "frames": frames})
    return {"runs": names, "scored": {n: scored_fields(settings[n]) for n in names}, "fields": fields + extra,
            "bools": list(BOOLS), "scores": scores, "clips": clips}


def build_report(run_dirs: list[str | Path], scores: list[dict[str, Any]], clips_root: str | Path,
                 out_path: str | Path, max_clips: int = 5) -> Path:
    out_path = Path(out_path)
    data = _data(run_dirs, scores, Path(clips_root), out_path.parent, max_clips)
    payload = json.dumps(data, default=str).replace("</", "<\\/")
    out_path.write_text(TEMPLATE.replace("/*__DATA__*/null", payload), encoding="utf-8")
    return out_path


TEMPLATE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>WaymoEnrichmentCatalogue</title><style>
:root{--ink:#101826;--paper:#F6F4EE;--card:#FBFAF6;--line:#CFCABB;--mute:#4A5568;--teal:#127A63;--tealbg:#D8EFE8;--bad:#F4C7C3;--badt:#8C1D18;--ok:#D7EFDD}
*{box-sizing:border-box}body{margin:0;font-family:Inter,system-ui,-apple-system,"Segoe UI",Roboto,Arial,sans-serif;background:var(--paper);color:#14213D}
header{background:var(--ink);color:#F6F4EE;padding:18px 28px;display:flex;gap:20px;align-items:baseline;flex-wrap:wrap}
header h1{margin:0;font-size:22px;color:#6FD6BC}header p{margin:0;color:#B8C4D6;font-size:14px}
main{padding:20px 28px;display:flex;flex-direction:column;gap:18px;max-width:1500px;margin:auto}
.tiles{display:flex;gap:14px;flex-wrap:wrap}.tile{flex:1 1 300px;background:var(--card);border:2px solid var(--line);border-radius:14px;padding:14px 18px}
.tile.best{border-color:var(--teal)}.tile h3{margin:0 0 8px;font-size:15px;color:var(--mute)}
.kv{display:flex;justify-content:space-between;font-size:14px;padding:3px 0;border-bottom:1px solid #E6E2D6}.kv b{font-variant-numeric:tabular-nums}
.big{font-size:34px;font-weight:700;color:var(--teal);margin:2px 0 6px}
.bar{display:flex;gap:10px;align-items:center;flex-wrap:wrap}select,button{font:inherit;padding:6px 10px;border-radius:8px;border:1px solid var(--line);background:var(--card)}
.view{display:flex;gap:18px;flex-wrap:wrap}.imgbox{flex:1 1 520px;background:#000;border-radius:14px;overflow:hidden;position:relative;min-height:240px}
.imgbox img{width:100%;display:block}.imgbox .stamp{position:absolute;left:10px;top:10px;background:rgba(16,24,38,.8);color:#fff;padding:4px 10px;border-radius:8px;font-size:13px}
.panel{flex:1 1 420px;background:var(--card);border:2px solid var(--line);border-radius:14px;padding:12px 16px}
table{width:100%;border-collapse:collapse;font-size:14px}th,td{padding:6px 8px;border-bottom:1px solid #E6E2D6;text-align:left}th{color:var(--mute);font-weight:600}
td.ok{background:var(--ok)}td.bad{background:var(--bad);color:var(--badt);font-weight:600}td.na{color:#8a93a3}
.chg{display:inline-block;background:var(--tealbg);color:var(--teal);border-radius:999px;padding:1px 8px;font-size:12px;margin:2px 4px 0 0}
.tl{background:var(--card);border:2px solid var(--line);border-radius:14px;padding:12px 16px;overflow-x:auto}
.tl h3{margin:0 0 8px;font-size:15px;color:var(--mute)}.row{display:flex;align-items:center;gap:8px;margin:2px 0}
.lab{flex:0 0 190px;font-size:12px;color:var(--mute);white-space:nowrap}.cells{display:flex;gap:2px}
.c{width:26px;height:18px;border-radius:3px;cursor:pointer;border:2px solid transparent}.c.bad{border-color:#C0392B}.c.sel{outline:3px solid #14213D;outline-offset:1px}
.c.flip::after{content:"";}.grp{margin-top:8px;font-size:12px;font-weight:700;color:#14213D}
details{background:var(--card);border:2px solid var(--line);border-radius:14px;padding:10px 16px}summary{cursor:pointer;font-weight:600}
.hint{font-size:12px;color:#8a93a3}
</style></head><body>
<header><h1>WaymoEnrichmentCatalogue</h1><p>Do driving-scene labels improve when the VLM remembers what came before? Per-frame vs rolling state, scored against Waymo ground truth.</p></header>
<main>
<div class="tiles" id="tiles"></div>
<div class="bar"><label>Clip <select id="clip"></select></label><button id="prev">&larr; frame</button><button id="next">frame &rarr;</button><span class="hint">Arrow keys also step through frames. Click any timeline cell to jump.</span></div>
<div class="view"><div class="imgbox"><img id="img" alt="front camera frame"><div class="stamp" id="stamp"></div></div><div class="panel" id="panel"></div></div>
<div class="tl" id="tl"></div>
<details><summary>Full scorecard</summary><div id="full"></div></details>
</main>
<script>
const D=/*__DATA__*/null;
const PAL=["#6FD6BC","#F2C14E","#7A9CC6","#E58F8F","#B79AD6","#9CCB6E","#E6A25E","#8CC8D6"];
const colorOf={};const col=v=>{if(v===null||v===undefined)return"#E3E0D5";const k=String(v);if(!(k in colorOf))colorOf[k]=PAL[Object.keys(colorOf).length%PAL.length];return colorOf[k]};
const pct=v=>v==null?"n/a":(v*100).toFixed(1)+"%";
let ci=0,fi=0;
const mean=a=>{a=a.filter(x=>x!=null);return a.length?a.reduce((s,x)=>s+x,0)/a.length:null};
function tiles(){const el=document.getElementById("tiles");el.innerHTML="";
 const rows=D.scores.map(s=>({s,acc:mean(Object.values(s.fields).map(f=>f.accuracy)),flip:mean(Object.values(s.fields).map(f=>f.pred_flip_rate)),gflip:mean(Object.values(s.fields).map(f=>f.gt_flip_rate))}));
 const best=Math.max(...rows.map(r=>r.acc??-1));
 rows.forEach(({s,acc,flip,gflip})=>{const t=document.createElement("div");t.className="tile"+(acc===best?" best":"");
  const ped=s.fields.pedestrian_visible,cyc=s.fields.cyclist_visible;
  t.innerHTML=`<h3>${s.run} &middot; ${s.n_frames} frames, ${s.n_clips} clips</h3><div class="big">${pct(acc)}</div><div class="hint" style="margin-bottom:6px">mean accuracy vs Waymo</div>
  <div class="kv"><span>pedestrian F1</span><b>${pct(ped&&ped.f1)}</b></div><div class="kv"><span>cyclist F1</span><b>${pct(cyc&&cyc.f1)}</b></div>
  <div class="kv"><span>flip rate (model / truth)</span><b>${pct(flip)} / ${pct(gflip)}</b></div><div class="kv"><span>parse failures</span><b>${pct(s.parse_fail_rate)}</b></div>`;el.appendChild(t)});}
function full(){const fs=[...new Set(D.scores.flatMap(s=>Object.keys(s.fields)))];
 let h="<table><tr><th>metric</th>"+D.scores.map(s=>`<th>${s.run}</th>`).join("")+"</tr>";
 fs.forEach(f=>{h+=`<tr><td>${f} accuracy</td>`+D.scores.map(s=>`<td>${pct(s.fields[f]&&s.fields[f].accuracy)}</td>`).join("")+"</tr>"});
 fs.forEach(f=>{h+=`<tr><td>${f} flip rate (model / truth)</td>`+D.scores.map(s=>`<td>${s.fields[f]?pct(s.fields[f].pred_flip_rate)+" / "+pct(s.fields[f].gt_flip_rate):"n/a"}</td>`).join("")+"</tr>"});
 document.getElementById("full").innerHTML=h+"</table>"}
const clip=()=>D.clips[ci],frame=()=>clip().frames[fi];
function show(){const f=frame();document.getElementById("img").src=f.img;
 document.getElementById("stamp").textContent=`${clip().id.slice(0,10)} · ${f.t}s · frame ${fi+1}/${clip().frames.length}`+(f.speed!=null?` · ${(+f.speed).toFixed(1)} m/s`:"");
 let h="<table><tr><th>field</th><th>Waymo</th>"+D.runs.map(r=>`<th>${r}</th>`).join("")+"</tr>";
 D.fields.forEach(fl=>{const g=f.gt[fl];h+=`<tr><td>${fl}</td><td>${g==null?"–":g}</td>`;
  D.runs.forEach(r=>{const rec=f.runs[r];const v=rec?rec.pred[fl]:null;const scored=(D.scored[r]||[]).includes(fl);
   const cls=!scored||g==null?"na":(v===g?"ok":"bad");h+=`<td class="${cls}">${v==null?"–":v}${!scored&&g!=null?" (unscored)":""}</td>`});h+="</tr>"});
 h+="</table>";
 D.runs.forEach(r=>{const rec=f.runs[r];if(!rec)return;const ch=(rec.changed||[]).map(c=>`<span class="chg">${c}</span>`).join("");const ev=rec.pred.notable_event;
  h+=`<div style="margin-top:10px;font-size:13px"><b>${r}</b>${ev?` &middot; ${ev}`:""}${ch?`<div>changed: ${ch}</div>`:""}${rec.ok===false?'<div style="color:#8C1D18">parse failed</div>':""}</div>`});
 document.getElementById("panel").innerHTML=h;
 document.querySelectorAll(".c").forEach(e=>e.classList.toggle("sel",+e.dataset.i===fi))}
function timeline(){const el=document.getElementById("tl");el.innerHTML="<h3>Timeline: colour = label value, red outline = disagrees with Waymo</h3>";
 D.fields.forEach(fl=>{const g=document.createElement("div");g.className="grp";g.textContent=fl;el.appendChild(g);
  const lines=[["Waymo",null],...D.runs.map(r=>[r,r])];
  lines.forEach(([lab,r])=>{const row=document.createElement("div");row.className="row";row.innerHTML=`<div class="lab">${lab}</div>`;const cells=document.createElement("div");cells.className="cells";
   clip().frames.forEach((f,i)=>{const v=r?(f.runs[r]?f.runs[r].pred[fl]:null):f.gt[fl];const gt=f.gt[fl];const c=document.createElement("div");c.className="c";c.dataset.i=i;
    c.style.background=col(v);c.title=`${f.t}s ${lab}: ${v==null?"–":v}`;if(r&&gt!=null&&v!==gt&&(D.scored[r]||[]).includes(fl))c.classList.add("bad");
    c.onclick=()=>{fi=i;show()};cells.appendChild(c)});row.appendChild(cells);el.appendChild(row)})})}
function init(){const sel=document.getElementById("clip");D.clips.forEach((c,i)=>{const o=document.createElement("option");o.value=i;o.textContent=c.id.slice(0,16);sel.appendChild(o)});
 sel.onchange=()=>{ci=+sel.value;fi=0;timeline();show()};
 const step=d=>{fi=Math.min(clip().frames.length-1,Math.max(0,fi+d));show()};
 document.getElementById("prev").onclick=()=>step(-1);document.getElementById("next").onclick=()=>step(1);
 addEventListener("keydown",e=>{if(e.key==="ArrowRight")step(1);if(e.key==="ArrowLeft")step(-1)});
 tiles();full();timeline();show()}
init();
</script></body></html>
"""
