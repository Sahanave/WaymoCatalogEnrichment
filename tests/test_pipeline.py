import json

from waymo_enrich import waymo_v2 as w
from waymo_enrich.clips import list_clips
from waymo_enrich.labelers import Settings, label_clip
from waymo_enrich.llm import MockClient, parse_llm_json
from waymo_enrich.metrics import format_table, score_run
from waymo_enrich.report import build_report
from waymo_enrich.schema import normalize


def test_parse_llm_json_variants():
    assert parse_llm_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_llm_json('<think>hmm {x}</think>{"a": 2} // note') == {"a": 2}
    assert parse_llm_json("no json here") is None
    assert parse_llm_json("[1, 2]") is None


def test_normalize_coerces_and_never_guesses():
    out = normalize({"time_of_day": "Dawn/Dusk", "weather": "SUNNY", "pedestrian_visible": "true",
                     "cyclist_visible": 0, "ego_motion": "slowing", "road_type": "moon", "notable_event": "none"})
    assert out["time_of_day"] == "dawn_dusk" and out["weather"] == "dry"
    assert out["pedestrian_visible"] is True and out["cyclist_visible"] is False
    assert out["ego_motion"] == "braking"
    assert out["road_type"] is None and out["notable_event"] is None
    assert all(v is None for v in normalize("garbage").values())


def test_derive_ego_motion():
    assert w.derive_ego_motion(0.1, 0.0) == "stopped"
    assert w.derive_ego_motion(5.0, -2.0) == "braking"
    assert w.derive_ego_motion(5.0, 2.0) == "accelerating"
    assert w.derive_ego_motion(5.0, 0.2) == "cruising"


def test_extract_synthetic_segment(clips_root):
    clips = list_clips(clips_root)
    assert len(clips) == 1
    frames = clips[0].frames
    assert [f.t_s for f in frames] == [0.0, 1.0, 2.0]  # 1 Hz keyframes
    assert [f.gt["ego_motion"] for f in frames] == ["cruising", "braking", "stopped"]
    # pedestrian only on the frame-10 keyframe, front camera only (camera-2 box at frame 20 ignored)
    assert [f.gt["pedestrian_visible"] for f in frames] == [False, True, False]
    assert all(f.gt["time_of_day"] == "day" and f.gt["weather"] == "dry" for f in frames)
    assert all(f.gt["cyclist_visible"] is False for f in frames)
    assert frames[0].image_bytes()[:2] == b"\xff\xd8"  # JPEG


def _scripted(labels):
    """Mock VLM returning labels[call_index] as JSON."""
    it = iter(labels)
    return MockClient(lambda parts, system: json.dumps(next(it)))


GOOD = {"time_of_day": "day", "weather": "dry", "road_type": "urban_street", "pedestrian_visible": False,
        "cyclist_visible": False, "ego_motion": "cruising", "notable_event": None}


def test_rolling_passes_state_prev_frame_and_ego(clips_root):
    clip = list_clips(clips_root)[0]
    client = _scripted([GOOD, {**GOOD, "ego_motion": "braking"}, {**GOOD, "ego_motion": "stopped"}])
    label_clip(clip, client, Settings(mode="rolling", prev_frames=1, use_ego=True))
    n_images = lambda parts: sum(p["type"] == "image_url" for p in parts)
    text = lambda parts: " ".join(p["text"] for p in parts if p["type"] == "text")
    assert [n_images(c[0]) for c in client.calls] == [1, 2, 2]
    assert "PREVIOUS STATE: none" in text(client.calls[0][0])
    assert '"ego_motion": "cruising"' in text(client.calls[1][0])  # state carried from call 1
    assert '"notable_event"' not in text(client.calls[1][0]).split("PREVIOUS STATE")[1].split("Keep each")[0]
    assert "EGO SPEED now" in text(client.calls[1][0])


def test_per_frame_is_independent(clips_root):
    clip = list_clips(clips_root)[0]
    client = _scripted([GOOD] * 3)
    label_clip(clip, client, Settings(mode="per_frame"))
    for parts, _ in client.calls:
        assert sum(p["type"] == "image_url" for p in parts) == 1
        assert not any("PREVIOUS" in p.get("text", "") for p in parts)


def test_refresh_drops_state(clips_root):
    clip = list_clips(clips_root)[0]
    client = _scripted([GOOD] * 3)
    label_clip(clip, client, Settings(mode="rolling", refresh_every=2))
    texts = [" ".join(p["text"] for p in c[0] if p["type"] == "text") for c in client.calls]
    assert "PREVIOUS STATE: none" in texts[0] and "PREVIOUS STATE: none" not in texts[1]
    assert "PREVIOUS STATE: none" in texts[2]  # k=2 is a refresh point


def test_garbage_output_is_recorded_not_guessed(clips_root):
    clip = list_clips(clips_root)[0]
    client = MockClient(lambda parts, system: "I cannot help with that")
    res = label_clip(clip, client, Settings(mode="rolling", retries=1))
    assert all(not f["parse_ok"] for f in res["frames"])
    assert all(v is None for v in res["frames"][0]["pred"].values())
    assert len(client.calls) == 3 * 2  # 1 try + 1 retry per frame


def _write_run(tmp_path, name, clip_id, settings, per_frame_preds):
    run = tmp_path / name
    run.mkdir()
    frames = [{"i": i, "t_s": float(i), "pred": normalize(p), "changed": [], "parse_ok": True, "error": None, "raw": ""}
              for i, p in enumerate(per_frame_preds)]
    (run / f"{clip_id}.json").write_text(json.dumps({"clip_id": clip_id, "settings": settings, "frames": frames}))
    return run


def test_scoring_accuracy_flips_and_ego_exclusion(clips_root, tmp_path):
    clip_id = list_clips(clips_root)[0].clip_id
    # A: flickers weather, misses the pedestrian, wrong ego_motion on frame 2
    a = [{**GOOD}, {**GOOD, "weather": "rain", "ego_motion": "braking"}, {**GOOD, "ego_motion": "cruising"}]
    # C: stable and correct everywhere, finds the pedestrian
    c = [{**GOOD}, {**GOOD, "pedestrian_visible": True, "ego_motion": "braking"}, {**GOOD, "ego_motion": "stopped"}]
    run_a = _write_run(tmp_path, "A", clip_id, {"mode": "per_frame"}, a)
    run_c = _write_run(tmp_path, "C", clip_id, {"mode": "rolling", "use_ego": False}, c)
    sa, sc = score_run(run_a, clips_root), score_run(run_c, clips_root)

    assert sa["fields"]["weather"]["accuracy"] == 2 / 3
    assert sa["fields"]["weather"]["pred_flip_rate"] == 1.0 and sa["fields"]["weather"]["gt_flip_rate"] == 0.0
    assert sc["fields"]["weather"]["accuracy"] == 1.0 and sc["fields"]["weather"]["pred_flip_rate"] == 0.0
    assert sa["fields"]["pedestrian_visible"]["recall"] == 0.0
    assert sc["fields"]["pedestrian_visible"]["f1"] == 1.0
    assert sa["fields"]["ego_motion"]["accuracy"] == 2 / 3 and sc["fields"]["ego_motion"]["accuracy"] == 1.0

    # With ego speed fed to the model, ego_motion must not be scored (circular).
    run_e = _write_run(tmp_path, "E", clip_id, {"mode": "rolling", "use_ego": True}, c)
    se = score_run(run_e, clips_root)
    assert "ego_motion" not in se["fields"]
    table = format_table([sa, sc, se])
    assert "weather accuracy" in table and "n/a" in table


def test_report_builds(clips_root, tmp_path):
    clip_id = list_clips(clips_root)[0].clip_id
    run = _write_run(tmp_path, "A", clip_id, {"mode": "per_frame"}, [GOOD, GOOD, GOOD])
    out = build_report([run], [score_run(run, clips_root)], clips_root, tmp_path / "report.html")
    html = out.read_text()
    assert clip_id in html and 'class="bad"' in html and "weather accuracy" in html
