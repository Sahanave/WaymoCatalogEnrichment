"""Extract keyframes + ground truth from Waymo Open Dataset v2 (Perception) parquet files.

Reads, per segment (one parquet file per component, as laid out in gs://waymo_open_dataset_v_2_0_0/<split>/<component>/):
  camera_image/<segment>.parquet  front camera JPEG + ego (SDC) linear/angular velocity
  stats/<segment>.parquet         time_of_day, weather (per frame)
  camera_box/<segment>.parquet    optional: 2D labels, used for pedestrian/cyclist presence in the front camera

Column names were checked against the official v2 tutorial/component definitions
(waymo-research/waymo-open-dataset: src/waymo_open_dataset/v2). No TensorFlow needed.
"""
from __future__ import annotations

import io
import json
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from PIL import Image

CAMERA_FRONT = 1          # camera_image.CameraName.FRONT
TYPE_PEDESTRIAN = 2       # box.BoxType
TYPE_CYCLIST = 4

K_TS = "key.frame_timestamp_micros"
K_CAM = "key.camera_name"
C_IMAGE = "[CameraImageComponent].image"
C_VEL = "[CameraImageComponent].velocity.linear_velocity.{}"
C_YAW = "[CameraImageComponent].velocity.angular_velocity.z"
C_TOD = "[StatsComponent].time_of_day"
C_WEATHER = "[StatsComponent].weather"
C_BOXTYPE = "[CameraBoxComponent].type"

_TOD = {"day": "day", "dawn/dusk": "dawn_dusk", "night": "night"}
_WEATHER = {"sunny": "dry", "rain": "rain"}


def derive_ego_motion(speed: float, accel: Optional[float], stopped_speed: float = 0.5, accel_thr: float = 1.0) -> str:
    if speed < stopped_speed:
        return "stopped"
    if accel is not None and accel < -accel_thr:
        return "braking"
    if accel is not None and accel > accel_thr:
        return "accelerating"
    return "cruising"


def _read(path: Path, columns: list[str], front_only: bool) -> pd.DataFrame:
    if front_only:
        try:
            return pd.read_parquet(path, columns=columns, filters=[(K_CAM, "==", CAMERA_FRONT)])
        except Exception:  # older pyarrow / column-name quirks: fall back to filtering in pandas
            pass
    df = pd.read_parquet(path, columns=columns + ([K_CAM] if front_only and K_CAM not in columns else []))
    return df[df[K_CAM] == CAMERA_FRONT] if front_only else df


def extract_segment(root: str | Path, segment: str, out_dir: str | Path, stride: int = 10, width: int = 768,
                    stopped_speed: float = 0.5, accel_thr: float = 1.0) -> Path:
    """Write one clip directory (see clips.py for the format). stride=10 keeps 1 frame per second."""
    root, out_dir = Path(root), Path(out_dir)
    img = _read(root / "camera_image" / f"{segment}.parquet",
                [K_TS, K_CAM, C_IMAGE, C_VEL.format("x"), C_VEL.format("y"), C_VEL.format("z"), C_YAW], True)
    img = img.sort_values(K_TS).reset_index(drop=True)
    if img.empty:
        raise ValueError(f"no front-camera rows in {segment}")

    ts = img[K_TS].to_numpy(dtype=np.int64)
    vel = img[[C_VEL.format(a) for a in "xyz"]].to_numpy(dtype=float)
    speed = np.linalg.norm(vel, axis=1)
    n = len(img)
    # Acceleration over a +/-0.5 s window (5 frames at 10 Hz).
    accel = np.zeros(n)
    for i in range(n):
        lo, hi = max(0, i - 5), min(n - 1, i + 5)
        dt = (ts[hi] - ts[lo]) / 1e6
        accel[i] = (speed[hi] - speed[lo]) / dt if dt > 0 else 0.0

    stats = pd.read_parquet(root / "stats" / f"{segment}.parquet", columns=[K_TS, C_TOD, C_WEATHER]).set_index(K_TS)

    ped: dict[int, int] = {}
    cyc: dict[int, int] = {}
    have_boxes = (root / "camera_box" / f"{segment}.parquet").exists()
    if have_boxes:
        boxes = _read(root / "camera_box" / f"{segment}.parquet", [K_TS, K_CAM, C_BOXTYPE], True)
        ped = boxes[boxes[C_BOXTYPE] == TYPE_PEDESTRIAN].groupby(K_TS).size().to_dict()
        cyc = boxes[boxes[C_BOXTYPE] == TYPE_CYCLIST].groupby(K_TS).size().to_dict()

    clip_id = segment[:16]
    clip_dir = out_dir / clip_id
    (clip_dir / "frames").mkdir(parents=True, exist_ok=True)
    frames: list[dict[str, Any]] = []
    for k, i in enumerate(range(0, n, stride)):
        im = Image.open(io.BytesIO(img.loc[i, C_IMAGE])).convert("RGB")
        if im.width > width:
            im = im.resize((width, round(im.height * width / im.width)))
        rel = f"frames/{k:03d}.jpg"
        im.save(clip_dir / rel, quality=85)
        t = int(ts[i])
        row = stats.loc[t] if t in stats.index else None
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        frames.append({
            "i": k, "t_s": round((t - int(ts[0])) / 1e6, 2), "image": rel,
            "speed_mps": round(float(speed[i]), 3), "accel_mps2": round(float(accel[i]), 3),
            "gt": {
                "time_of_day": _TOD.get(str(row[C_TOD]).strip().lower()) if row is not None else None,
                "weather": _WEATHER.get(str(row[C_WEATHER]).strip().lower()) if row is not None else None,
                "pedestrian_visible": (ped.get(t, 0) > 0) if have_boxes else None,
                "cyclist_visible": (cyc.get(t, 0) > 0) if have_boxes else None,
                "ego_motion": derive_ego_motion(float(speed[i]), float(accel[i]), stopped_speed, accel_thr),
            },
        })
    (clip_dir / "meta.json").write_text(json.dumps(
        {"clip_id": clip_id, "segment": segment, "stride": stride, "frames": frames}, indent=1))
    return clip_dir
