"""Real PathSnap pipeline behind the API (replaces dummy_pipeline).

Same two functions the API imports - list_trip_ids() and predict_trip(trip_id) - and the
same response shape both UIs already read:
    {trip_id, gps: [[lat, lon], ...], predicted_route: GeoJSON, accuracy, metrics: {...},
     placeholder: False}

Configuration (environment variables, all optional):
    PATHSNAP_DATA_ROOT   folder containing the partition folders   (default: repo root)
    PATHSNAP_PARTITION   evaluator | student_test | train           (default: evaluator if
                         such trips exist under the data root, otherwise student_test)
    PATHSNAP_ARTIFACTS   folder with model.joblib / hmm_config.json (default: artifacts)

Ground truth never leaves this process: prediction runs on a copy of the trip loaded
WITHOUT its .route file, the route is read afterwards only to compute the scalar
metrics, and the response contains no reference arcs.
"""
from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathsnap.io import discover_trips, read_trip          # noqa: E402
from pathsnap.pipeline import PathSnapPipeline             # noqa: E402

DATA_ROOT = Path(os.environ.get("PATHSNAP_DATA_ROOT", ROOT))
ARTIFACTS = Path(os.environ.get("PATHSNAP_ARTIFACTS", ROOT / "artifacts"))


@lru_cache(maxsize=1)
def _index():
    idx = discover_trips(DATA_ROOT)
    idx = idx[idx.partition.notna()]
    part = os.environ.get("PATHSNAP_PARTITION")
    if not part:
        part = "evaluator" if (idx.partition == "evaluator").any() else "student_test"
    idx = idx[idx.partition == part]
    if idx.empty:
        raise RuntimeError(f"no '{part}' trips found under {DATA_ROOT}")
    return part, idx.set_index("trip_id")


@lru_cache(maxsize=1)
def _pipeline() -> PathSnapPipeline:
    pipe = PathSnapPipeline.from_artifacts(ARTIFACTS)
    if pipe.bundle is None:
        raise RuntimeError(f"no model.joblib in {ARTIFACTS}")
    return pipe


def partition() -> str:
    return _index()[0]


def list_trip_ids() -> list[str]:
    return sorted(_index()[1].index)


def _merge_lines(segments: list[list[list[float]]]) -> list[list[list[float]]]:
    """Join consecutive arc segments that share an endpoint into longer polylines."""
    lines: list[list[list[float]]] = []
    for a, b in segments:
        if lines and lines[-1][-1] == a:
            lines[-1].append(b)
        else:
            lines.append([a, b])
    return lines


def predict_trip(trip_id: str) -> dict:
    part, idx = _index()
    row = idx.loc[trip_id]
    pipe = _pipeline()

    result = pipe.predict(read_trip(row.prefix, trip_id, part, load_route=False))
    scored = pipe.score(read_trip(row.prefix, trip_id, part, load_route=True), result)

    hmm, clf = scored.get("hmm", {}), scored.get("classifier_only", {})
    metrics = {
        "top1_accuracy_hmm": hmm.get("top1_accuracy"),
        "top1_accuracy_classifier_only": clf.get("top1_accuracy"),
        "route_continuity": hmm.get("route_continuity"),
        "route_continuity_classifier_only": clf.get("route_continuity"),
        "route_f1": hmm.get("route_f1"),
        "route_precision": hmm.get("route_precision"),
        "route_recall": hmm.get("route_recall"),
        "candidate_miss_rate": hmm.get("candidate_miss_rate"),
        "hmm_chain_breaks": float(len(result["hmm_breaks"])),
        "matched_arcs": float(len(result["route_arcs"])),
        "inference_seconds": result["timings"]["total_s"],
    }
    return {
        "trip_id": trip_id,
        "partition": part,
        "model": result["model"],
        "gps": [[p["lat"], p["lon"]] for p in result["points"]],
        "predicted_route": {
            "type": "Feature", "properties": {"name": "predicted_route"},
            "geometry": {"type": "MultiLineString",
                         "coordinates": _merge_lines(
                             result["route_geojson"]["geometry"]["coordinates"])}},
        "pred_arcs": result["pred_arcs"],
        "accuracy": hmm.get("top1_accuracy"),
        "metrics": {k: v for k, v in metrics.items() if v is not None},
        "placeholder": False,
    }
