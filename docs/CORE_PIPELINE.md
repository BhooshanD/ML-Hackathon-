# PathSnap core pipeline (candidates → features → classifier → HMM/Viterbi)

```
src/pathsnap/
  io.py        read .track/.nodes/.arcs/.route, clean track, discover partitions
  geo.py       per-trip UTM projection (metres), bearings
  network.py   arc geometry, KD-tree candidate search, Dijkstra (cached), adjacency
  labels.py    monotone DP alignment of the .route arc sequence to GPS points
  features.py  candidate table + features (FEATURE_COLS)
  hmm.py       emissions, transitions, Viterbi, gap/break handling, route filling
  metrics.py   Top-1, candidate miss rate, route continuity, route P/R/F1
  pipeline.py  PathSnapPipeline.predict() / score(), predict_trip()  <- API entry point
scripts/
  build_features.py   -> data/features/{train,student_test}_{obs,cands}.parquet
  candidate_sweep.py  -> experiments/candidates/sweep.csv   (justifies radius / k)
  train_baseline.py   -> logistic-regression placeholder bundle (no longer the default)
  export_model_bundle.py -> untuned LightGBM bundle on the pipeline feature table
  compare_models.py / tune_lightgbm.py / plot_models.py   model comparison, Optuna, figures
  eda_checks.py / make_trip_map.py                        extra EDA checks, report maps
  eval_decoder.py     -> experiments/decoder/*.csv, artifacts/hmm_config.json
  predict_trip.py     CLI for one trip (same code path as the API)
tests/test_core.py    synthetic-network tests
api/
  main.py             FastAPI app: GET /trips, GET /predict/{trip_id}, request logging
  real_pipeline.py    adapter: pathsnap pipeline -> the response shape both UIs read
  dummy_pipeline.py   old synthetic placeholder (unused)
```

## Run
```bash
uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python scripts/build_features.py          # ~10 s
.venv/bin/python scripts/compare_models.py          # ~4 min, all models
.venv/bin/python scripts/tune_lightgbm.py --trials 40   # ~30 min -> artifacts/model.joblib
.venv/bin/python scripts/eval_decoder.py            # ~4 min, tunes decoder on train OOF
.venv/bin/python scripts/predict_trip.py 00000000
.venv/bin/python -m pytest tests -q

# Laptop 1 (API). Serves the evaluator trips if a folder named eval*/evaluator* exists
# under the data root, otherwise student_test. Override with PATHSNAP_PARTITION /
# PATHSNAP_DATA_ROOT / PATHSNAP_ARTIFACTS.
cd api && ../.venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
# Laptop 2 (UI): open ui/index.html, or  API_URL=http://<laptop1-ip>:8000 streamlit run ui/app.py
```

## Current results (student-test, mean over 10 trips)
Deployed model: LightGBM, Bayesian-tuned (`artifacts/model.joblib`, name `lightgbm_tuned`).

| Method | Top-1 | Route continuity | Route F1 |
|---|---|---|---|
| Nearest arc | 0.562 | 0.576 | 0.633 |
| LightGBM (tuned), classifier only | 0.927 | 0.927 | 0.871 |
| HMM, distance-only emissions | 0.914 | 0.995 | 0.970 |
| LightGBM (tuned) + HMM | **0.944** | 0.995 | 0.977 |

Sources: `experiments/decoder/student_test_comparison.csv`, per trip in
`per_trip_student_test.csv`. The HMM lowers Top-1 on trip 65 (use it for the "where HMM
did not help" discussion; map: `artifacts/maps/report_ready_trip_00000065.html`).

## Model experiments (all on `data/features/*_cands.parquet`)
| Script | Output |
|---|---|
| `scripts/compare_models.py` | `experiments/models/model_comparison.csv`, `cv_folds_all.csv`: 6 families + tuned row, 5-fold GroupKFold, student-test classifier-only and +HMM |
| `scripts/tune_lightgbm.py` | `experiments/tuning/trials.csv` (40 Optuna TPE trials), `best_params.json`, `tuned_vs_untuned.csv` |
| `scripts/plot_models.py` | `artifacts/figures/model_*.png`, `tuning_history.png`, `tuned_vs_untuned.png`, `mlp_training_history.png`, `decoder_per_trip.png` |
| `scripts/eda_checks.py` | `artifacts/eda/*` additions and three EDA figures |
| `scripts/make_trip_map.py` | report maps + prediction JSON with the current model |

Findings to state in the report:
- LightGBM has the best cross-validation macro F1 (0.945 untuned, 0.948 tuned); Random
  Forest and the MLP are within about one fold standard deviation (0.017).
- Tuning: CV Top-1 0.916 -> 0.920, CV macro F1 0.945 -> 0.948, student-test Top-1
  0.913 -> 0.916 (pooled); log loss slightly worse (0.081 -> 0.087 CV); inference about
  3x slower.
- After the HMM the gain disappears: on the training trips (out-of-fold) tuned and
  untuned both reach 0.937 Top-1; on student-test 0.944 vs 0.947
  (`experiments/tuning/end_to_end_tuned_vs_untuned.csv`). The decoder corrects most of
  the errors that tuning removes.
- `class_weight="balanced"` leaves Top-1 unchanged and worsens log loss (0.107 vs 0.083).
- In `model_comparison.csv` the HMM parameters are held fixed at the deployed values
  (tuned once, for the deployed model), not re-tuned per model.

`models.ipynb` and `models/LightGBM.joblib` are from an earlier run on a separate
`candidates.parquet` (11 `f_*` features) whose generating code is not in this repo. They
are superseded by the scripts above and are not loaded by the API.

## Contracts

**Person 2 — feature table** `data/features/train_cands.parquet`, one row per
(GPS observation, candidate arc):
- features: `pathsnap.features.FEATURE_COLS`
- target: `label` (1 = this candidate is the reference arc)
- groups for CV: `trip_id` (use GroupKFold, never split a trip)
- **train only on** `trip_label_ok == 1 & label_reliable == 1` (bad reference labels,
  see Data quality). Evaluate Top-1 per observation: argmax of `predict_proba` within
  each `(trip_id, obs_idx)` vs `label`.
- Hand back: `joblib.dump({"model": est, "feature_cols": FEATURE_COLS, "name": "..."},
  "artifacts/model.joblib")`, `est` must have `predict_proba` and be sklearn-clonable
  (eval_decoder refits it per fold). Then rerun `eval_decoder.py`.

**Person 3 — API**
```python
from pathsnap.pipeline import PathSnapPipeline, predict_trip
PIPE = PathSnapPipeline.from_artifacts("artifacts")       # load once at startup
result = predict_trip(trip_id, data_root, pipeline=PIPE)  # per request
```
`result` keys: `trip_id, model, n_obs, points[{lon,lat,t,arc_id}], pred_arcs,
classifier_only_arcs, route_arcs, route_geojson, hmm_breaks, timings,
metrics{hmm{...}, classifier_only{...}}`. It contains **no reference route**: ground
truth is only read inside `score()` on the server, after prediction.

**Person 4 — maps**: `route_geojson` (predicted) and `points` can go straight into
Folium; the reference route for report maps = `trip.route` arcs →
`net.arc_lonlat(arc_id)`.

## Design decisions (for the report / viva)
- **Projection**: trips span the globe, so each is projected into the UTM zone of its
  own centroid; all distances are metres.
- **Arcs are directed**: 92–100 % of arcs have a reverse twin (two-way road = 2 arcs,
  one-way = 1); the median trip is 95 % twins, but 26 of 90 trips are below 90 % and
  trip 80 is 35 %. Routes are contiguous (`to` of arc i == `from` of arc i+1) except
  for 14 steps across 11 trips (`artifacts/eda/route_validation.csv`).
  Consequence: the nearest arc is right only ~56 % of the time (twins are equidistant);
  heading features and the directed HMM resolve direction.
- **Labels**: `.route` is an arc sequence, not per-point labels. `labels.align_route`
  finds the monotone point→route-position assignment minimising total distance (DP,
  O(T·L)); monotonicity handles out-and-back routes.
- **Candidates**: all arcs within `radius` (50 m), max `k` (8), via KD-tree over points
  sampled every 10 m along arcs (exact projection afterwards). If none in radius, the 3
  nearest arcs are used (`forced` feature).
- **Direction features**: in slow, jittery stretches the short-window GPS heading is
  noise and the HMM picked the reverse twin arc (all of trip 7's 14 % error). Added a
  10-point-baseline heading and `progress_short/long` = signed distance travelled along
  the candidate arc between an earlier and a later point. LightGBM CV Top-1 0.892 -> 0.916.
- **HMM**: state = (arc, offset); emission = classifier probability normalised per
  observation (+ optional Gaussian distance term); transition = Exponential on
  |driving distance − GPS distance| (Newson & Krumm 2009), driving distance from
  Dijkstra on the directed graph. Unreachable / implied speed > 50 m/s → impossible.
  Up to 30 m backward jitter on the same arc is allowed but penalised (stopped
  vehicles). If no transition is feasible the chain is cut and restarted (`hmm_breaks`).
  Larger time gaps widen the Dijkstra search radius; `gap_beta_scale` loosens β.
- **Decoder tuning**: grid search on the 80 training trips with out-of-fold
  probabilities (in-sample ones are over-confident); reported on student-test.

## Data quality findings
- Trip 54: GPS is a median 84 m from any arc (shifted track / missing roads).
- Trip 33: GPS is on roads, but the reference route is a median 122 m away for most
  of the trip (route does not match track).
- Rule: an observation's label is unreliable if it is > 50 m from its aligned route arc;
  a trip is excluded from training/tuning if its median is > 30 m. Excluded: 33, 54.
  Student-test: 0.14 % unreliable observations, none excluded.
