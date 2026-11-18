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
  train_baseline.py   -> artifacts/model.joblib  (PLACEHOLDER until Person 2's model)
  eval_decoder.py     -> experiments/decoder/*.csv, artifacts/hmm_config.json
  predict_trip.py     CLI for one trip (same code path as the API)
tests/test_core.py    synthetic-network tests
```

## Run
```bash
uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python scripts/build_features.py          # ~10 s
.venv/bin/python scripts/train_baseline.py          # or Person 2's model.joblib
.venv/bin/python scripts/eval_decoder.py            # ~4 min, tunes decoder on train OOF
.venv/bin/python scripts/predict_trip.py 00000000
.venv/bin/python -m pytest tests -q
```

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
  one-way = 1), every route is contiguous (`to` of arc i == `from` of arc i+1).
  Consequence: the nearest arc is right only ~56 % of the time (twins are equidistant);
  heading features and the directed HMM resolve direction.
- **Labels**: `.route` is an arc sequence, not per-point labels. `labels.align_route`
  finds the monotone point→route-position assignment minimising total distance (DP,
  O(T·L)); monotonicity handles out-and-back routes.
- **Candidates**: all arcs within `radius` (50 m), max `k` (8), via KD-tree over points
  sampled every 10 m along arcs (exact projection afterwards). If none in radius, the 3
  nearest arcs are used (`forced` feature).
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
