"""Clip format on disk (written by scripts/extract_waymo.py, read by everything else).

data/clips/<clip_id>/meta.json
data/clips/<clip_id>/frames/000.jpg ...
meta.json: {"clip_id": str, "stride": int, "frames": [
    {"i": 0, "t_s": 0.0, "image": "frames/000.jpg", "speed_mps": 4.2, "accel_mps2": 0.1,
     "gt": {"time_of_day": "day", "weather": "dry", "pedestrian_visible": false,
            "cyclist_visible": false, "ego_motion": "cruising"}}]}
Any gt value may be null (no ground truth available for that frame/field).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass
class Frame:
    i: int
    t_s: float
    image_path: Path
    speed_mps: Optional[float]
    accel_mps2: Optional[float]
    gt: dict[str, Any]

    def image_bytes(self) -> bytes:
        return self.image_path.read_bytes()


@dataclass
class Clip:
    clip_id: str
    path: Path
    frames: list[Frame]


def load_clip(path: str | Path) -> Clip:
    path = Path(path)
    meta = json.loads((path / "meta.json").read_text())
    frames = [
        Frame(
            i=f["i"], t_s=f["t_s"], image_path=path / f["image"],
            speed_mps=f.get("speed_mps"), accel_mps2=f.get("accel_mps2"), gt=f.get("gt") or {},
        )
        for f in meta["frames"]
    ]
    return Clip(clip_id=meta["clip_id"], path=path, frames=frames)


def list_clips(root: str | Path, limit: Optional[int] = None) -> list[Clip]:
    root = Path(root)
    dirs = sorted(p for p in root.iterdir() if (p / "meta.json").exists())
    if limit:
        dirs = dirs[:limit]
    return [load_clip(d) for d in dirs]
