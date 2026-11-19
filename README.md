# PathSnap: Map-Matching and Route Reconstruction

UE24CS352A Machine Learning Hackathon, PES University

Team: Basawaraj Panchal (PES1UG24AM350), Bhooshan Deshpande (PES1UG24AM351),
Bhumika Praveen Patil (PES1UG24AM352), Chandana O N (PES1UG24AM353)

Repository: https://github.com/BhooshanD/ML-Hackathon-
Report: `PathSnap_Report.pdf`

PathSnap matches noisy GPS points to the road network. A LightGBM classifier scores the candidate
road segments near each GPS point, and an HMM decoded with Viterbi turns those scores into a
connected route. A FastAPI server (laptop 1) runs the pipeline, and a browser UI (laptop 2) shows
the result.

## Folder layout

| Folder | Contents |
|---|---|
| `src/pathsnap/` | Core pipeline: reading files (`io.py`), UTM projection (`geo.py`), road network and candidate search (`network.py`), label alignment (`labels.py`), features (`features.py`), HMM and Viterbi (`hmm.py`), metrics (`metrics.py`), end-to-end pipeline (`pipeline.py`), Folium maps (`viz.py`) |
| `scripts/` | Feature building, candidate sweep, model comparison, Optuna tuning, decoder evaluation, plots, maps, single-trip prediction |
| `experiments/` | EDA notebook and every result CSV/JSON used in the report (models, tuning, decoder, candidates) |
| `artifacts/` | Deployed model (`model.joblib`), pipeline and HMM configs, EDA tables, figures, Folium maps, saved predictions |
| `api/` | FastAPI backend (`main.py`, `real_pipeline.py`) |
| `ui/` | Evaluator UI: `index.html` (Leaflet, used in the demo) and `app.py` (Streamlit version) |
| `tests/` | Unit tests on a synthetic road network |
| `docs/` | Pipeline design notes and the data-quality / EDA write-up |

The raw dataset (`train/`, `student_test/`, `evaluator_test/`) is not included in this zip.
Place those folders next to `src/` before running anything.

## Install

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

## Reproduce the results

```bash
python scripts/build_features.py      # candidate + feature tables -> data/features/
python scripts/candidate_sweep.py     # radius / k choice -> experiments/candidates/
python scripts/compare_models.py      # 6 model families, GroupKFold -> experiments/models/
python scripts/tune_lightgbm.py       # Optuna TPE, 40 trials -> experiments/tuning/
python scripts/eval_decoder.py        # HMM decoder tuning + comparison -> experiments/decoder/
python scripts/plot_models.py         # report figures -> artifacts/figures/
python scripts/make_trip_map.py       # Folium maps -> artifacts/maps/
python -m pytest tests -q
```

## Run the demo (two laptops)

Laptop 1 (model + API):

```bash
cd api
uvicorn main:app --host 0.0.0.0 --port 8000
```

`real_pipeline.py` reads these optional environment variables: `PATHSNAP_DATA_ROOT` (folder with the
partition folders, default: repo root), `PATHSNAP_PARTITION` (`evaluator`, `student_test` or `train`)
and `PATHSNAP_ARTIFACTS` (default: `artifacts`). Every request is logged in this terminal.

Laptop 2 (UI): open `ui/index.html` through a local web server, for example
`python -m http.server 5500` inside `ui/`, then browse to
`http://localhost:5500/?api=http://<laptop-1-IP>:8000`. Both laptops must be on the same network.

Endpoints: `GET /trips` lists the trips; `GET /predict/{trip_id}` returns the predicted route
(GeoJSON) and accuracy metrics. The reference route is read only on the server, after prediction.
