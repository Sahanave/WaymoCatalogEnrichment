# WaymoCatalogEnrichment

Temporal scenario enrichment for driving data. It takes the **analyze → categorize → quality-check** pattern of the
[NVIDIA Retail Catalog Enrichment blueprint](https://build.nvidia.com/nvidia/retail-catalog-enrichment) and applies it to
the [Waymo Open Dataset](https://waymo.com/open/about/): a VLM tags keyframes of driving clips with scenario labels so
the data is searchable and easier to train on.

**The question this repo answers:** do labels get better if the VLM sees *what happened before* the frame?

| Variant | What the VLM sees per keyframe |
|---|---|
| **A `per_frame`** | the current frame only |
| **C `rolling`** | previous frame image(s) + its own previous labels (+ ego speed with `--use-ego`) |

Same model, same schema, same prompts' field definitions; only the context differs.

## Labels

`time_of_day`, `weather`, `pedestrian_visible`, `cyclist_visible`, `ego_motion` are **scored** against ground truth derived
from Waymo v2 (stats component, front-camera 2D boxes, ego velocity). `road_type` and `notable_event` are enrichment only.
With `--use-ego`, `ego_motion` is not scored, because its ground truth comes from the same velocity signal the model was given.

## Metrics

Per-field accuracy, precision/recall/F1 for pedestrian and cyclist presence, **flip rate** (how often a label changes between
adjacent keyframes, next to how often the ground truth changes), and parse-failure rate. Flip rate is meaningful on its own:
a stable label that matches reality has a flip rate near the ground-truth flip rate.

## Quickstart

```bash
pip install -e ".[dev]"
cp .env.example .env            # add NEBIUS_API_KEY
pytest                          # 10 offline tests, no key or data needed

# 1. Get 10-15 Perception segments (needs Waymo terms accepted + gcloud auth)
./scripts/download_waymo.sh list
./scripts/download_waymo.sh get <segment_id> <segment_id> ...

# 2. Extract 1 Hz front-camera keyframes + ground truth
python -m waymo_enrich extract --root data/waymo_v2/training --segment <segment_id> --out data/clips

# 3. Label with both variants
python -m waymo_enrich label --mode per_frame --out runs/A_per_frame
python -m waymo_enrich label --mode rolling   --out runs/C_rolling
python -m waymo_enrich label --mode rolling --use-ego --out runs/C_rolling_ego   # optional

# 4. Score and build the side-by-side report
python -m waymo_enrich score runs/A_per_frame runs/C_rolling --report report.html
```

Add `--mock` to `label` for an offline dry run with a fake VLM.

## What is and isn't verified

- **Tested:** pipeline logic (state carry-over, previous-frame images, retries, refresh, scoring, report) with a mock VLM
  and a synthetic Waymo-v2-shaped dataset.
- **Not yet tested:** a real VLM endpoint, and real Waymo files. Column names in `waymo_v2.py` were checked against the
  official v2 component definitions and tutorial, but run the extractor on one real segment first and eyeball `meta.json`.
- **Assumption to confirm:** a front-camera frame with no pedestrian box means no pedestrian (not "unlabeled"). If your
  segments have unlabeled frames, pedestrian/cyclist scores will be too optimistic.
- **Model:** `config.yaml` uses the Nemotron omni VLM id from the retail blueprint. Confirm your endpoint serves it, or set
  `VLM_MODEL` / `VLM_BASE_URL`. If JSON parse failures are high, the blueprint's fix is a two-step flow (short free-text
  observation, then a second call that structures it into JSON).

## Known limits

- Rolling state can carry a wrong label forward. `--refresh-every K` drops the state every K frames; compare against 0.
- The Motion split only ships camera embeddings, not video. Use the Perception split.
- Waymo data is **non-commercial**. `data/` and `runs/` are git-ignored; do not commit them.

## Layout

```
src/waymo_enrich/  schema.py  llm.py  labelers.py  clips.py  waymo_v2.py  metrics.py  report.py  run.py
scripts/download_waymo.sh
docs/plan.html   5-slide plan and architecture deck (open in a browser, arrow keys; print to PDF)
tests/
```

Portions adapted from the NVIDIA blueprint (Apache-2.0); see `NOTICE`.
