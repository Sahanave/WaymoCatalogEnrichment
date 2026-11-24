import io

import pandas as pd
import pytest
from PIL import Image

from waymo_enrich import waymo_v2 as w

SEGMENT = "synthetic_segment_0001_abcdef"


def jpeg(color: int, size=(64, 48)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (color, color, color)).save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def v2_root(tmp_path):
    """A tiny synthetic Waymo-v2-shaped dataset: 30 frames at 10 Hz, ego brakes to a stop, one pedestrian frame."""
    root = tmp_path / "v2"
    for sub in ("camera_image", "stats", "camera_box"):
        (root / sub).mkdir(parents=True)
    n = 30
    ts = [1_000_000 + i * 100_000 for i in range(n)]
    speed = [5.0] * 10 + [5.0 - 0.5 * k for k in range(10)] + [0.0] * 10  # cruise, brake, stopped
    rows = []
    for i in range(n):
        for cam in (w.CAMERA_FRONT, 2):  # camera 2 must be filtered out
            rows.append({
                w.K_TS: ts[i], w.K_CAM: cam, w.C_IMAGE: jpeg(10 * i if cam == 1 else 255),
                w.C_VEL.format("x"): speed[i], w.C_VEL.format("y"): 0.0, w.C_VEL.format("z"): 0.0, w.C_YAW: 0.0,
            })
    pd.DataFrame(rows).to_parquet(root / "camera_image" / f"{SEGMENT}.parquet")
    pd.DataFrame({w.K_TS: ts, w.C_TOD: ["Day"] * n, w.C_WEATHER: ["Sunny"] * n}).to_parquet(
        root / "stats" / f"{SEGMENT}.parquet")
    pd.DataFrame([
        {w.K_TS: ts[10], w.K_CAM: w.CAMERA_FRONT, w.C_BOXTYPE: w.TYPE_PEDESTRIAN},
        {w.K_TS: ts[10], w.K_CAM: w.CAMERA_FRONT, w.C_BOXTYPE: 1},
        {w.K_TS: ts[20], w.K_CAM: 2, w.C_BOXTYPE: w.TYPE_PEDESTRIAN},  # other camera: must not count
    ]).to_parquet(root / "camera_box" / f"{SEGMENT}.parquet")
    return root


@pytest.fixture
def clips_root(v2_root, tmp_path):
    out = tmp_path / "clips"
    w.extract_segment(v2_root, SEGMENT, out, stride=10, width=32)
    return out
