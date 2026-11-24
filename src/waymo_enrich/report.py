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


def _data(run_dirs, scores, clips_root: Path, out_dir: Path, max_clips: int, clip_suffixes=(), faq_dir=None) -> dict[str, Any]:
    names = [Path(d).name for d in run_dirs]
    runs = {n: {r["clip_id"]: r for r in load_run(d)} for n, d in zip(names, run_dirs)}
    settings = {n: load_run(d)[0]["settings"] for n, d in zip(names, run_dirs)}
    clip_ids = list(dict.fromkeys(cid for r in runs.values() for cid in r))
    if clip_suffixes:
        clip_ids = [c for c in clip_ids if any(c.endswith(s) for s in clip_suffixes)]
    clip_ids = clip_ids[:max_clips]
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
                    entry["runs"][n] = {"pred": rec["pred"], "changed": rec.get("changed", []), "ok": rec.get("parse_ok", True),
                                       "trace": rec.get("trace")}
            frames.append(entry)
        faq = None
        if faq_dir and (Path(faq_dir) / f"{cid}.json").exists():
            faq = json.loads((Path(faq_dir) / f"{cid}.json").read_text())
        clips.append({"id": cid, "frames": frames, "faq": faq})
    return {"runs": names, "scored": {n: scored_fields(settings[n]) for n in names}, "fields": fields + extra,
            "bools": list(BOOLS), "settings": settings, "scores": scores, "clips": clips}


def build_report(run_dirs: list[str | Path], scores: list[dict[str, Any]], clips_root: str | Path,
                 out_path: str | Path, max_clips: int = 5, clip_suffixes=(), faq_dir=None) -> Path:
    out_path = Path(out_path)
    data = _data(run_dirs, scores, Path(clips_root), out_path.parent, max_clips, clip_suffixes, faq_dir)
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
.step.llm{border-color:#7A9CC6}
.legend #legend{font-size:13px;line-height:1.5;display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:4px 28px;margin-top:8px}
.legend h4{margin:8px 0 4px;font-size:13px;color:var(--teal)}.legend ul{margin:0;padding-left:18px}.legend li{margin:2px 0}
.sw{display:inline-block;width:12px;height:12px;border-radius:3px;vertical-align:-1px;margin-right:4px}
.trace{background:var(--card);border:2px solid var(--line);border-radius:14px;padding:12px 16px}
.trace h3{margin:0 0 8px;font-size:15px;color:var(--mute)}.step{border-left:4px solid var(--teal);padding:4px 12px;margin:8px 0}
.step h4{margin:0 0 4px;font-size:13px}.step pre{white-space:pre-wrap;word-break:break-word;margin:4px 0;font:12px/1.4 ui-monospace,Menlo,monospace;background:#F1EFE6;border-radius:8px;padding:8px}
.step .out{background:var(--tealbg)}.thumbs{display:flex;gap:6px;margin:4px 0}.thumbs img{height:70px;border-radius:6px;border:1px solid var(--line)}
.step summary{font-weight:500;font-size:12px;color:var(--mute)}
</style></head><body>
<header><h1>WaymoEnrichmentCatalogue</h1><p>Three ways of reading the same driving clips, each pulling out different information. Scored against Waymo ground truth where it exists.</p></header>
<main>
<div class="tiles" id="tiles"></div>
<details open class="legend"><summary>How to read this</summary><div id="legend"></div></details>
<div class="bar"><label>Clip <select id="clip"></select></label><button id="prev">&larr; frame</button><button id="next">frame &rarr;</button><span class="hint">Arrow keys also step through frames.</span></div>
<div class="view"><div class="imgbox"><img id="img" alt="front camera frame"><div class="stamp" id="stamp"></div></div><div class="panel" id="panel"></div></div>
<div class="trace" id="faq"></div>
<div class="trace" id="trace"></div>
<details><summary>Full scorecard</summary><div id="full"></div></details>
</main>
<script>
const D=/*__DATA__*/null;
const PAL=["#6FD6BC","#F2C14E","#7A9CC6","#E58F8F","#B79AD6","#9CCB6E","#E6A25E","#8CC8D6"];
const colorOf={};const col=v=>{if(v===null||v===undefined)return"#E3E0D5";const k=String(v);if(!(k in colorOf))colorOf[k]=PAL[Object.keys(colorOf).length%PAL.length];return colorOf[k]};
const pct=v=>v==null?"n/a":(v*100).toFixed(1)+"%";
let ci=0,fi=0;
const mean=a=>{a=a.filter(x=>x!=null);return a.length?a.reduce((s,x)=>s+x,0)/a.length:null};
function chg(s){if(!s.changes)return"";const v=Object.values(s.changes);const tp=v.reduce((a,c)=>a+c.tp,0),fp=v.reduce((a,c)=>a+c.fp,0),fn=v.reduce((a,c)=>a+c.fn,0);
 const p=tp+fp?tp/(tp+fp):null,r=tp+fn?tp/(tp+fn):null,f=p&&r?2*p*r/(p+r):null;
 return `<div class="kv"><span>real changes / flagged</span><b>${tp+fn} / ${tp+fp}</b></div><div class="kv"><span>change precision / recall</span><b>${pct(p)} / ${pct(r)}</b></div><div class="kv"><span>change F1 (all fields)</span><b>${tp+fn?pct(f??0):"no real changes"}</b></div>`}
function tiles(){const el=document.getElementById("tiles");el.innerHTML="";
 const rows=D.scores.map(s=>({s,acc:mean(Object.values(s.fields).map(f=>f.accuracy)),flip:mean(Object.values(s.fields).map(f=>f.pred_flip_rate)),gflip:mean(Object.values(s.fields).map(f=>f.gt_flip_rate))}));
 const best=Math.max(...rows.map(r=>r.acc??-1));
 rows.forEach(({s,acc,flip,gflip})=>{const t=document.createElement("div");t.className="tile"+"";
  const ped=s.fields.pedestrian_visible,cyc=s.fields.cyclist_visible;
  t.innerHTML=`<h3>${s.run} &middot; ${s.n_frames} frames, ${s.n_clips} clips</h3><div class="big">${pct(acc)}</div><div class="hint" style="margin-bottom:6px">mean accuracy vs Waymo</div>
  <div class="kv"><span>pedestrian F1</span><b>${pct(ped&&ped.f1)}</b></div><div class="kv"><span>cyclist F1</span><b>${pct(cyc&&cyc.f1)}</b></div>
  <div class="kv"><span>flip rate (model / truth)</span><b>${pct(flip)} / ${pct(gflip)}</b></div>${chg(s)}<div class="kv"><span>parse failures</span><b>${pct(s.parse_fail_rate)}</b></div>`;el.appendChild(t)});}
function runDesc(r){const s=(D.settings||{})[r]||{};
 if(s.mode==="temporal")return ["Reads two consecutive frames plus the per_frame labels for the previous frame, and says which of those labels no longer hold (changed), what changed (note), and the ego_motion. It does not relabel the scene and does not use its own earlier output, so errors cannot carry forward.",
  "Everything a single frame can answer comes from per_frame. This run adds only what needs two frames: whether the scene changed, what changed, and how the vehicle is moving."];const ego=s.use_ego?" Ego speed is given (ego_motion unscored).":"";
 if(s.mode==="per_frame")return ["Single VLM call per frame. Sees only the current frame.","The scene as a single moment: lighting, weather, road type, pedestrians and cyclists, with no influence from earlier frames. Useful on its own and as an independent view of each frame."];
 if(s.mode==="rolling")return [`Single VLM call per frame. Sees the previous ${s.prev_frames} and current frame plus its own previous labels.`+ego,
  "Information that needs time: the previous frame shows how the vehicle is moving, and the previous labels keep slow-changing fields like weather and time of day steady. The catch is that a wrong label can be carried forward."];
 return ["",""]}
function legend(){document.getElementById("legend").innerHTML=`
 <div><h4>Runs</h4><ul>${D.runs.map(r=>{const [d,w]=runDesc(r);return `<li><b>${r}</b>: ${d}<br><i>What it adds:</i> ${w}</li>`}).join("")}</ul>
 <h4>Ground truth</h4><ul><li>Waymo-derived: stats (time of day, weather), front-camera 2D boxes (pedestrian, cyclist), ego velocity (ego_motion).</li>
 <li>Scored fields: time_of_day, weather, pedestrian_visible, cyclist_visible, ego_motion. road_type and notable_event are enrichment only.</li>
 <li>Division of work: single-frame attributes come from <b>per_frame</b>. The <b>temporal</b> run is scored on <b>change detection</b> (did it flag a field exactly when Waymo's label changed between the two frames) and on ego_motion.</li></ul></div>
 <div><h4>Scorecard</h4><ul><li><b>Big %</b>: mean accuracy over the scored fields. Different runs extract different things, so this is not a ranking.</li>
 <li><b>F1</b>: balance of precision (when it says visible, is it right) and recall (does it catch the real ones). Used for pedestrian and cyclist because they are rare.</li>
 <li><b>Flip rate (model / truth)</b>: share of adjacent keyframe pairs where the label changes. Close to the truth is good; higher means flicker, lower means too sticky.</li>
 <li><b>Change detection</b> (temporal run): precision = of the changes it flagged, how many were real; recall = of the real changes, how many it caught. Real changes are rare, so check the counts.</li>
 <li><b>Parse failures</b>: frames with no usable JSON. They count as wrong in accuracy.</li></ul>
 <h4>Frame viewer</h4><ul><li><span class="sw" style="background:var(--ok)"></span>model matches Waymo. <span class="sw" style="background:var(--bad)"></span>model disagrees.</li>
 <li>grey / "(unscored)": no ground truth, or a field that is not scored.</li>
 <li><span class="chg">changed</span> fields the model says it changed from the previous frame.</li>
 <li>Input/output panel: the frames the VLM was given, its full prompt ("text input") and its raw output (green box).</li>
 <li>Clip FAQ (blue bar): a text-only LLM answers fixed questions from the A + C label catalog. It never sees the images. Answers with a checkable value are compared with Waymo.</li></ul></div>`}
function full(){const fs=[...new Set(D.scores.flatMap(s=>Object.keys(s.fields)))];
 let h="<table><tr><th>metric</th>"+D.scores.map(s=>`<th>${s.run}</th>`).join("")+"</tr>";
 fs.forEach(f=>{h+=`<tr><td>${f} accuracy</td>`+D.scores.map(s=>`<td>${pct(s.fields[f]&&s.fields[f].accuracy)}</td>`).join("")+"</tr>"});
 fs.forEach(f=>{h+=`<tr><td>${f} flip rate (model / truth)</td>`+D.scores.map(s=>`<td>${s.fields[f]?pct(s.fields[f].pred_flip_rate)+" / "+pct(s.fields[f].gt_flip_rate):"n/a"}</td>`).join("")+"</tr>"});
 document.getElementById("full").innerHTML=h+"</table>"}
const clip=()=>D.clips[ci],frame=()=>clip().frames[fi];
function gtChanged(){const f=frame(),p=fi>0?clip().frames[fi-1]:null;if(!p)return[];return Object.keys(f.gt).filter(k=>["time_of_day","weather","pedestrian_visible","cyclist_visible"].includes(k)&&f.gt[k]!=null&&p.gt[k]!=null&&f.gt[k]!==p.gt[k])}
function show(){const f=frame();document.getElementById("img").src=f.img;
 document.getElementById("stamp").textContent=`${clip().id.slice(0,10)} · ${f.t}s · frame ${fi+1}/${clip().frames.length}`+(f.speed!=null?` · ${(+f.speed).toFixed(1)} m/s`:"");
 let h="<table><tr><th>field</th><th>Waymo</th>"+D.runs.map(r=>`<th>${r}</th>`).join("")+"</tr>";
 D.fields.forEach(fl=>{const g=f.gt[fl];h+=`<tr><td>${fl}</td><td>${g==null?"–":g}</td>`;
  D.runs.forEach(r=>{const rec=f.runs[r];const v=rec?rec.pred[fl]:null;const scored=(D.scored[r]||[]).includes(fl);
   const cls=!scored||g==null?"na":(v===g?"ok":"bad");h+=`<td class="${cls}">${v==null?"–":v}${!scored&&g!=null?" (unscored)":""}</td>`});h+="</tr>"});
 h+="</table>";
 const gc=gtChanged();h+=`<div style="margin-top:8px;font-size:13px"><b>Waymo</b>: ${fi===0?"first frame":gc.length?"changed since previous frame: "+gc.map(c=>`<span class="chg">${c}</span>`).join(""):"no label changed since previous frame"}</div>`;
 D.runs.forEach(r=>{const rec=f.runs[r];if(!rec)return;const ch=(rec.changed||[]).map(c=>`<span class="chg">${c}</span>`).join("");const ev=rec.pred.notable_event;
  h+=`<div style="margin-top:10px;font-size:13px"><b>${r}</b>${ev?` &middot; ${ev}`:""}${ch?`<div>flagged as changed: ${ch}</div>`:""}${rec.changed&&(D.settings[r]||{}).mode==="temporal"&&!ch?'<div class="hint">flagged no changes</div>':""}${rec.ok===false?'<div style="color:#8C1D18">parse failed</div>':""}</div>`});
 document.getElementById("panel").innerHTML=h;faq();trace(f);
 document.querySelectorAll(".c").forEach(e=>e.classList.toggle("sel",+e.dataset.i===fi))}
const esc=t=>String(t==null?"":t).replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
function step(kind,title,s){const imgs=(s.input.images||[]).map(i=>`<figure style="margin:0"><img src="${clip().frames[i].img}"><figcaption class="hint">frame ${i+1} (${clip().frames[i].t}s)</figcaption></figure>`).join("");
 return `<div class="step ${kind}"><h4>${title}</h4>${imgs?`<div class="hint">image input</div><div class="thumbs">${imgs}</div>`:`<div class="hint">no image input (text only)</div>`}
 <details><summary>text input</summary><pre>${esc(s.input.text)}</pre></details><div class="hint">output</div><pre class="out">${esc(s.output)}</pre></div>`}
function faq(){const el=document.getElementById("faq"),q=clip().faq;if(!q){el.style.display="none";return}el.style.display="";
 const ok=q.items.filter(i=>i.correct!==null);const n=ok.filter(i=>i.correct).length;
 const badge=i=>i.correct===null?"":i.correct?'<span class="chg">matches Waymo</span>':'<span class="chg" style="background:var(--bad);color:var(--badt)">differs from Waymo</span>';
 const val=v=>v===null||v===undefined?"never / unknown":String(v);
 el.innerHTML=`<h3>Clip FAQ: a text-only LLM answers from the A + C label catalog (${n}/${ok.length} checkable answers match Waymo)</h3>
 <details><summary>Catalog the LLM was given (no images)</summary><pre style="white-space:pre-wrap;font:12px/1.4 ui-monospace,Menlo,monospace;background:#F1EFE6;border-radius:8px;padding:8px">${esc(q.catalog)}</pre></details>`+
 q.items.map(i=>`<div class="step llm"><h4>${esc(i.question)}</h4><pre class="out">${esc(i.answer==null?"(no parsable answer)":i.answer)}</pre>
  ${i.correct===null?"":`<div class="hint">LLM value: <b>${val(i.value)}</b> &middot; Waymo: <b>${val(i.gt)}</b> ${badge(i)}</div>`}
  <details><summary>text input and raw output</summary><pre>${esc(i.input)}\n\n--- output ---\n${esc(i.output)}</pre></details></div>`).join("")}
function trace(f){let h="<h3>What the VLM was given and what it said (this frame)</h3>";let any=false;
 D.runs.forEach(r=>{const rec=f.runs[r];if(!rec||!rec.trace)return;any=true;h+=`<div class="grp">${r}</div>`;
  if(rec.trace.vlm)h+=step("vlm","VLM: sees the frames, returns JSON",rec.trace.vlm)});
 document.getElementById("trace").innerHTML=any?h:"<h3>No input/output trace in these runs (re-run label to record it)</h3>"}
function init(){const sel=document.getElementById("clip");D.clips.forEach((c,i)=>{const o=document.createElement("option");o.value=i;o.textContent=c.id.slice(0,16);sel.appendChild(o)});
 sel.onchange=()=>{ci=+sel.value;fi=0;show()};
 const step=d=>{fi=Math.min(clip().frames.length-1,Math.max(0,fi+d));show()};
 document.getElementById("prev").onclick=()=>step(-1);document.getElementById("next").onclick=()=>step(1);
 addEventListener("keydown",e=>{if(e.key==="ArrowRight")step(1);if(e.key==="ArrowLeft")step(-1)});
 tiles();legend();full();show()}
init();
</script></body></html>
"""
