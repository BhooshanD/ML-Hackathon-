"""Tune the HMM decoder and compare classifier-only vs classifier + HMM/Viterbi.

    python scripts/eval_decoder.py --model artifacts/model.joblib

Leakage-safe protocol
  * Decoder parameters are tuned on the 80 TRAINING trips using OUT-OF-FOLD classifier
    probabilities (GroupKFold by trip). In-sample probabilities would be over-confident
    and would bias the decoder towards trusting the classifier too much.
  * The final classifier (fitted on all training trips) + tuned decoder are then
    reported on the 10 STUDENT-TEST trips. The evaluator partition is never touched.

Outputs (experiments/decoder/):
  tuning_trials.csv     every decoder configuration tried and its OOF objective
  student_test_comparison.csv / per_trip_student_test.csv
and artifacts/hmm_config.json with the best decoder parameters.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

from pathsnap.config import Config
from pathsnap.hmm import (TripLattice, classifier_only, decode, emission_logprob, fill_route,
                          precompute_transitions)
from pathsnap.io import discover_trips, read_trip
from pathsnap.metrics import trip_metrics
from pathsnap.network import build_network

GRID_CLS = {"w_cls": [1.0], "w_dist": [0.0, 0.25, 1.0], "sigma": [10.0],
            "beta": [1.0, 2.5, 5.0, 10.0, 20.0], "gap_beta_scale": [0.0, 0.2, 1.0]}
GRID_DIST = {"sigma": [2.0, 3.0, 4.0, 7.0, 10.0], "beta": [1.0, 2.5, 5.0, 10.0, 20.0],
             "gap_beta_scale": [0.0, 0.2, 1.0]}


def grid(g):
    keys = list(g)
    return [dict(zip(keys, v)) for v in itertools.product(*g.values())]


def prepare(prefix, tid, part, obs, cands, cfg):
    """Everything that does not depend on tuned parameters: network, lattice, route dists."""
    trip = read_trip(prefix, tid, part)
    net = build_network(trip, cfg.undirected, cfg.cand.sample_step, cfg.hmm.cache_limit)
    lat = TripLattice.from_frames(obs, cands)
    D, gps, dt = precompute_transitions(net, lat, cfg.hmm)
    return {"trip_id": tid, "net": net, "lat": lat, "D": D, "gps": gps, "dt": dt,
            "route": trip.route, "label_arc": obs.label_arc.to_numpy(),
            "label_in_cands": obs.label_in_cands.to_numpy(),
            "reliable": obs.label_reliable.to_numpy().astype(bool),
            "trip_ok": bool(obs.trip_label_ok.iloc[0])}


def run(T, proba, hcfg, mode, full=False):
    """mode: nearest | classifier | hmm_dist | hmm_cls. Returns metrics dict."""
    lat = T["lat"]
    t0 = time.perf_counter()
    if mode == "nearest":
        arcs = classifier_only(lat, None)
    elif mode == "classifier":
        arcs = classifier_only(lat, proba)
    else:
        le = emission_logprob(lat, proba if mode == "hmm_cls" else None, hcfg)
        arcs, _, breaks = decode(lat, le, T["D"], T["gps"], T["dt"], hcfg)
    dec_s = time.perf_counter() - t0
    if not full:
        r = T["reliable"]   # tuning objective ignores observations with bad reference labels
        return {"top1_accuracy": float(np.mean(arcs[r] == T["label_arc"][r]))}
    m = trip_metrics(T["net"], arcs, T["label_arc"], T["label_in_cands"], T["route"],
                     T["gps"], np.array(fill_route(T["net"], arcs)))
    m["decode_s"] = dec_s
    return m


def tune(trips, probas, base, g, mode):
    trials = []
    for i, params in enumerate(grid(g)):
        h = replace(base, **params)
        accs = [run(T, probas.get(T["trip_id"]), h, mode)["top1_accuracy"] for T in trips]
        trials.append({"trial": i, "mode": mode, **params,
                       "mean_top1": float(np.mean(accs)), "std_top1": float(np.std(accs))})
    df = pd.DataFrame(trials).sort_values("mean_top1", ascending=False)
    best = {k: float(df.iloc[0][k]) for k in g}
    return df, replace(base, **best)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=".")
    ap.add_argument("--splits", default=None)
    ap.add_argument("--features", default="data/features")
    ap.add_argument("--model", default="artifacts/model.joblib")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--out", default="experiments/decoder")
    ap.add_argument("--artifacts", default="artifacts", help="where hmm_config.json is written")
    args = ap.parse_args()

    f = Path(args.features)
    cfg = Config.load(f / "config.json") if (f / "config.json").exists() else Config()
    bundle = joblib.load(args.model)
    cols = bundle["feature_cols"]
    idx = discover_trips(args.data_root, args.splits).set_index("trip_id")

    data = {}
    for part in ("train", "student_test"):
        obs = pd.read_parquet(f / f"{part}_obs.parquet")
        c = pd.read_parquet(f / f"{part}_cands.parquet")
        data[part] = (obs, c)

    # --- out-of-fold probabilities on training trips (for decoder tuning)
    obs_tr, c_tr = data["train"]
    X, y, groups = c_tr[cols].to_numpy(float), c_tr.label.to_numpy(), c_tr.trip_id.to_numpy()
    oof = np.zeros(len(c_tr))
    good = ((c_tr.trip_label_ok == 1) & (c_tr.label_reliable == 1)).to_numpy()
    for k, (a, b) in enumerate(GroupKFold(n_splits=args.folds).split(X, y, groups)):
        a = a[good[a]]                       # fit on reliable labels only
        m = clone(bundle["model"]).fit(X[a], y[a])
        oof[b] = m.predict_proba(X[b])[:, 1]
        print(f"OOF fold {k + 1}/{args.folds} done")
    c_tr = c_tr.assign(p=oof)
    obs_te, c_te = data["student_test"]
    c_te = c_te.assign(p=bundle["model"].predict_proba(c_te[cols].to_numpy(float))[:, 1])

    def prep_all(obs, c, part):
        jobs = []
        for tid, cg in c.groupby("trip_id", sort=False):
            og = obs[obs.trip_id == tid]
            jobs.append(delayed(prepare)(idx.loc[tid, "prefix"], tid, part, og,
                                         cg.reset_index(drop=True), cfg))
        trips = Parallel(n_jobs=-1)(jobs)
        probas = {tid: cg.p.to_numpy() for tid, cg in c.groupby("trip_id", sort=False)}
        return trips, probas

    print("precomputing lattices / route distances ...")
    tr_trips, tr_p = prep_all(obs_tr, c_tr, "train")
    tr_trips = [T for T in tr_trips if T["trip_ok"]]   # excluded trips: see build_features
    te_trips, te_p = prep_all(obs_te, c_te, "student_test")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print("tuning decoder (classifier emissions, OOF on train) ...")
    t_cls, h_cls = tune(tr_trips, tr_p, cfg.hmm, GRID_CLS, "hmm_cls")
    print("tuning decoder (distance-only emissions, train) ...")
    t_dist, h_dist = tune(tr_trips, {}, cfg.hmm, GRID_DIST, "hmm_dist")
    pd.concat([t_cls, t_dist]).to_csv(out / "tuning_trials.csv", index=False)

    rows = []
    for name, mode, h in [("nearest_arc", "nearest", cfg.hmm),
                          (f"{bundle['name']} (classifier only)", "classifier", cfg.hmm),
                          ("HMM distance-only (tuned)", "hmm_dist", h_dist),
                          (f"{bundle['name']} + HMM (tuned)", "hmm_cls", h_cls)]:
        for T in te_trips:
            rows.append({"method": name, "trip_id": T["trip_id"],
                         **run(T, te_p.get(T["trip_id"]), h, mode, full=True)})
    per_trip = pd.DataFrame(rows)
    per_trip.to_csv(out / "per_trip_student_test.csv", index=False)
    summary = per_trip.drop(columns="trip_id").groupby("method", sort=False).mean()
    summary.to_csv(out / "student_test_comparison.csv")

    art = Path(args.artifacts)
    art.mkdir(parents=True, exist_ok=True)
    (art / "hmm_config.json").write_text(json.dumps(asdict(h_cls), indent=2))
    cfg.save(art / "config.json")
    print("\nbest decoder (classifier):", {k: float(getattr(h_cls, k)) for k in GRID_CLS})
    print("best decoder (distance)  :", {k: float(getattr(h_dist, k)) for k in GRID_DIST})
    print("\nSTUDENT-TEST (mean over trips)")
    print(summary.round(4).to_string())


if __name__ == "__main__":
    main()
