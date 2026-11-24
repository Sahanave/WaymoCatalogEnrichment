"""CLI: label clips with a VLM, score runs against Waymo-derived ground truth, build the report.

  python -m waymo_enrich label --clips data/clips --mode per_frame --out runs/A_per_frame
  python -m waymo_enrich label --clips data/clips --mode rolling --prev-frames 1 --out runs/C_rolling
  python -m waymo_enrich score runs/A_per_frame runs/C_rolling --clips data/clips --report report.html
"""
from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv

from .clips import list_clips
from .labelers import Settings, label_clip
from .llm import MockClient, VLMClient, load_config
from .metrics import format_table, score_run
from .report import build_report


def cmd_label(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    client = MockClient() if args.mock else VLMClient(cfg)
    if args.mode == "temporal" and not args.base:
        raise SystemExit("--mode temporal needs --base <per_frame run dir>")
    llm_client = None
    if args.commentary:
        if args.mode != "rolling":
            raise SystemExit("--commentary only applies to --mode rolling")
        llm_client = client if args.mock else VLMClient(cfg, section="llm")
    settings = Settings(
        mode=args.mode, prev_frames=args.prev_frames, use_ego=args.use_ego,
        refresh_every=args.refresh_every, commentary=args.commentary, base_dir=args.base, retries=cfg["vlm"].get("retries", 2),
    )
    clips = list_clips(args.clips, args.limit)
    if not clips:
        raise SystemExit(f"no clips found in {args.clips} (run scripts/extract_waymo.py first)")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def work(clip):
        target = out / f"{clip.clip_id}.json"
        if target.exists() and not args.overwrite:
            return clip.clip_id, "skipped (exists)"
        res = label_clip(clip, client, settings, llm_client)
        target.write_text(json.dumps(res, indent=1))
        bad = sum(1 for f in res["frames"] if not f["parse_ok"])
        return clip.clip_id, f"{len(res['frames'])} frames, {bad} parse failures"

    # Frames inside a rolling clip are sequential (state dependency); parallelism is across clips.
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for cid, msg in ex.map(work, clips):
            print(f"{cid}: {msg}")


def cmd_extract(args: argparse.Namespace) -> None:
    from .waymo_v2 import extract_segment

    ego = load_config(args.config)["ego_motion"]
    for seg in args.segment:
        d = extract_segment(args.root, seg, args.out, args.stride, args.width,
                            ego["stopped_speed_mps"], ego["accel_threshold_mps2"])
        print(f"{seg} -> {d}")


def cmd_score(args: argparse.Namespace) -> None:
    scores = [score_run(r, args.clips) for r in args.runs]
    for r, s in zip(args.runs, scores):
        (Path(r) / "scores.json").write_text(json.dumps(s, indent=1))
    table = format_table(scores)
    print(table)
    if args.markdown:
        Path(args.markdown).write_text(table + "\n")
    if args.report:
        path = build_report(args.runs, scores, args.clips, args.report, max_clips=args.report_clips)
        print(f"\nreport: {path}")


def cmd_models(args) -> None:
    from openai import OpenAI
    from .llm import load_config, resolve_api_key
    v = load_config(args.config)["vlm"]
    client = OpenAI(base_url=v["base_url"], api_key=resolve_api_key(v["base_url"]))
    for m in sorted(x.id for x in client.models.list().data):
        print(m)


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    logging.basicConfig(level=logging.WARNING)
    p = argparse.ArgumentParser(prog="waymo_enrich")
    sub = p.add_subparsers(dest="cmd", required=True)

    l = sub.add_parser("label", help="label clips with a VLM")
    l.add_argument("--clips", default="data/clips")
    l.add_argument("--mode", choices=["per_frame", "rolling", "temporal"], required=True)
    l.add_argument("--base", default=None, help="temporal: per_frame run dir (A) whose labels each frame is compared against")
    l.add_argument("--out", required=True)
    l.add_argument("--prev-frames", type=int, default=1, help="rolling: previous keyframe images to include")
    l.add_argument("--use-ego", action="store_true", help="rolling: give the model ego speed (ego_motion then unscored)")
    l.add_argument("--refresh-every", type=int, default=0, help="rolling: drop carried state every K frames")
    l.add_argument("--commentary", action="store_true",
                   help="rolling: VLM writes a running commentary, then a text-only LLM (config llm:) structures it into labels")
    l.add_argument("--limit", type=int, default=None, help="only the first N clips")
    l.add_argument("--workers", type=int, default=4)
    l.add_argument("--overwrite", action="store_true")
    l.add_argument("--mock", action="store_true", help="offline dry run with a fake VLM")
    l.add_argument("--config", default=None)
    l.set_defaults(fn=cmd_label)

    e = sub.add_parser("extract", help="extract keyframes + ground truth from Waymo v2 parquet files")
    e.add_argument("--root", required=True, help="dir containing camera_image/, stats/ (and camera_box/) subdirs")
    e.add_argument("--segment", action="append", required=True, help="segment context name (repeatable)")
    e.add_argument("--out", default="data/clips")
    e.add_argument("--stride", type=int, default=10, help="keep every Nth frame (10 = 1 Hz)")
    e.add_argument("--width", type=int, default=768)
    e.add_argument("--config", default=None)
    e.set_defaults(fn=cmd_extract)

    s = sub.add_parser("score", help="score one or more runs")
    s.add_argument("runs", nargs="+")
    s.add_argument("--clips", default="data/clips")
    s.add_argument("--report", default=None, help="write an HTML report here")
    s.add_argument("--report-clips", type=int, default=5)
    s.add_argument("--markdown", default=None, help="also write the table to this file")
    s.set_defaults(fn=cmd_score)

    m = sub.add_parser("models", help="list model ids served by the configured endpoint")
    m.add_argument("--config", default=None)
    m.set_defaults(fn=cmd_models)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
