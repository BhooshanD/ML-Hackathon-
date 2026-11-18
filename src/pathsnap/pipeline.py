"""End-to-end inference: raw trip -> candidates -> classifier -> HMM/Viterbi -> route.

`predict()` never reads `trip.route`; ground truth is only touched in `score()`, which
runs server-side and returns scalar metrics. This separation is the leakage guarantee
that the API (Person 3) relies on.

Model bundle contract (saved by Person 2 with joblib):
    {"model": <fitted estimator with predict_proba>, "feature_cols": [...], "name": str}
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np

from .config import Config, HMMConfig
from .features import build_trip_frame
from .hmm import (TripLattice, classifier_only, decode, emission_logprob, fill_route,
                  precompute_transitions)
from .io import Trip, discover_trips, read_trip
from .metrics import trip_metrics
from .network import build_network


def predict_proba(bundle: dict | None, cands) -> np.ndarray | None:
    if bundle is None:
        return None
    X = cands[bundle["feature_cols"]].to_numpy(dtype=float)
    return bundle["model"].predict_proba(X)[:, 1]


class PathSnapPipeline:
    def __init__(self, cfg: Config | None = None, bundle: dict | None = None):
        self.cfg = cfg or Config()
        self.bundle = bundle

    @classmethod
    def from_artifacts(cls, artifacts_dir: str | Path = "artifacts",
                       model_file: str = "model.joblib") -> "PathSnapPipeline":
        d = Path(artifacts_dir)
        cfg = Config.load(d / "config.json") if (d / "config.json").exists() else Config()
        if (d / "hmm_config.json").exists():
            cfg.hmm = HMMConfig(**json.loads((d / "hmm_config.json").read_text()))
        bundle = joblib.load(d / model_file) if (d / model_file).exists() else None
        return cls(cfg, bundle)

    # ---------------------------------------------------------------- predict
    def predict(self, trip: Trip) -> dict:
        timings = {}
        t0 = time.perf_counter()
        net = build_network(trip, self.cfg.undirected, self.cfg.cand.sample_step,
                            self.cfg.hmm.cache_limit)
        obs, cands = build_trip_frame(trip, net, self.cfg, with_labels=False)
        timings["candidates_features_s"] = time.perf_counter() - t0

        t1 = time.perf_counter()
        proba = predict_proba(self.bundle, cands)
        timings["classifier_s"] = time.perf_counter() - t1

        t2 = time.perf_counter()
        lat = TripLattice.from_frames(obs, cands)
        D_list, gps, dt = precompute_transitions(net, lat, self.cfg.hmm)
        log_em = emission_logprob(lat, proba, self.cfg.hmm)
        hmm_arc, _, breaks = decode(lat, log_em, D_list, gps, dt, self.cfg.hmm)
        timings["hmm_s"] = time.perf_counter() - t2
        timings["total_s"] = time.perf_counter() - t0

        route = fill_route(net, hmm_arc)
        return {
            "trip_id": trip.trip_id,
            "model": self.bundle.get("name", "model") if self.bundle else "distance-only",
            "n_obs": len(obs),
            "points": [{"lon": float(a), "lat": float(b), "t": float(c), "arc_id": int(d)}
                       for a, b, c, d in zip(obs.lon, obs.lat, obs.t, hmm_arc)],
            "pred_arcs": hmm_arc.tolist(),
            "classifier_only_arcs": classifier_only(lat, proba).tolist(),
            "route_arcs": route,
            "route_geojson": {
                "type": "Feature", "properties": {"name": "predicted_route"},
                "geometry": {"type": "MultiLineString",
                             "coordinates": [net.arc_lonlat(a) for a in route]}},
            "hmm_breaks": breaks,
            "timings": timings,
            "_net": net, "_gps": gps,   # internal, stripped by `public()`
        }

    # ------------------------------------------------------------------ score
    def score(self, trip: Trip, result: dict) -> dict:
        """Server-side only: compare a prediction with the trip's reference route."""
        if trip.route is None:
            return {}
        net = result["_net"]
        obs, _ = build_trip_frame(trip, net, self.cfg, with_labels=True)
        label_arc = obs.label_arc.to_numpy()
        lat_cands = obs.label_in_cands.to_numpy()
        hmm = trip_metrics(net, np.array(result["pred_arcs"]), label_arc, lat_cands,
                           trip.route, result["_gps"], np.array(result["route_arcs"]))
        clf = trip_metrics(net, np.array(result["classifier_only_arcs"]), label_arc, lat_cands,
                           trip.route, result["_gps"],
                           np.array(fill_route(net, np.array(result["classifier_only_arcs"]))))
        return {"hmm": hmm, "classifier_only": clf}

    @staticmethod
    def public(result: dict) -> dict:
        return {k: v for k, v in result.items() if not k.startswith("_")}


def predict_trip(trip_id: str, data_root: str | Path = ".", artifacts_dir: str | Path = "artifacts",
                 splits_file=None, pipeline: PathSnapPipeline | None = None) -> dict:
    """Convenience entry point for the API: predict + score one trip by id."""
    idx = discover_trips(data_root, splits_file)
    row = idx[idx.trip_id == str(trip_id)]
    if row.empty:
        raise KeyError(f"unknown trip id {trip_id}")
    r = row.iloc[0]
    pipe = pipeline or PathSnapPipeline.from_artifacts(artifacts_dir)
    # prediction sees a copy of the trip WITHOUT the route
    trip_no_gt = read_trip(r.prefix, r.trip_id, r.partition, load_route=False)
    result = pipe.predict(trip_no_gt)
    trip_gt = read_trip(r.prefix, r.trip_id, r.partition, load_route=True)
    result["metrics"] = pipe.score(trip_gt, result)
    return pipe.public(result)
