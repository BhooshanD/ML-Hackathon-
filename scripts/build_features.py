"""Build the candidate/feature tables for the permitted partitions.

    python scripts/build_features.py --data-root . --out data/features

Training rows to use: trip_label_ok == 1 & label_reliable == 1 (see FeatureConfig).

Writes, per partition:  {partition}_cands.parquet  (one row per observation x candidate,
FEATURE_COLS + label) and {partition}_obs.parquet (one row per GPS observation), plus a
candidate-generation summary in experiments/candidates/. The evaluator partition is
refused on purpose: its labels must never reach training.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from pathsnap.config import Config
from pathsnap.features import build_trip_frame
from pathsnap.io import discover_trips, read_trip
from pathsnap.network import build_network


def one_trip(prefix, trip_id, partition, cfg):
    t0 = time.perf_counter()
    trip = read_trip(prefix, trip_id, partition)
    net = build_network(trip, cfg.undirected, cfg.cand.sample_step, cfg.hmm.cache_limit)
    obs, cands = build_trip_frame(trip, net, cfg, with_labels=True)
    obs["partition"] = cands["partition"] = partition
    stats = {"trip_id": trip_id, "partition": partition, **trip.qc,
             "candidate_miss_rate": float(1 - obs.label_in_cands.mean()),
             "mean_cands": float(cands.groupby("obs_idx").size().mean()),
             "forced_frac": float(obs.obs_idx.isin(cands[cands.forced == 1].obs_idx).mean()),
             "label_dist_median": float(np.median(obs.label_dist)),
             "label_dist_p95": float(np.percentile(obs.label_dist, 95)),
             "label_reliable_frac": float(obs.label_reliable.mean()),
             "trip_label_ok": int(obs.trip_label_ok.iloc[0]),
             "seconds": time.perf_counter() - t0}
    return obs, cands, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=".")
    ap.add_argument("--splits", default=None)
    ap.add_argument("--out", default="data/features")
    ap.add_argument("--partitions", nargs="+", default=["train", "student_test"])
    ap.add_argument("--config", default=None, help="Config JSON (defaults if omitted)")
    ap.add_argument("--jobs", type=int, default=-1)
    args = ap.parse_args()
    if "evaluator" in args.partitions:
        sys.exit("refusing to build labelled features for the evaluator partition")

    cfg = Config.load(args.config) if args.config else Config()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    idx = discover_trips(args.data_root, args.splits)
    all_stats = []
    for part in args.partitions:
        rows = idx[idx.partition == part]
        res = Parallel(n_jobs=args.jobs, verbose=0)(
            delayed(one_trip)(r.prefix, r.trip_id, part, cfg) for r in rows.itertuples())
        obs = pd.concat([r[0] for r in res], ignore_index=True)
        cands = pd.concat([r[1] for r in res], ignore_index=True)
        obs.to_parquet(out / f"{part}_obs.parquet", index=False)
        cands.to_parquet(out / f"{part}_cands.parquet", index=False)
        all_stats += [r[2] for r in res]
        print(f"{part}: {len(rows)} trips, {len(obs)} observations, {len(cands)} candidate rows, "
              f"miss rate {1 - obs.label_in_cands.mean():.4f}, "
              f"positive rate {cands.label.mean():.3f}, "
              f"excluded trips {sorted(obs[obs.trip_label_ok == 0].trip_id.unique())}, "
              f"unreliable-label obs {1 - obs.label_reliable.mean():.4f}")

    exp = Path("experiments/candidates")
    exp.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_stats).to_csv(exp / "per_trip_summary.csv", index=False)
    cfg.save(out / "config.json")
    print(f"wrote {out}/ and {exp}/per_trip_summary.csv")


if __name__ == "__main__":
    main()
