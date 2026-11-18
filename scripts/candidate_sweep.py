"""Justify the candidate radius R and count k: miss rate vs. candidate-set size.

    python scripts/candidate_sweep.py --data-root .

Builds candidates once with a generous (R_max, k_max) on the TRAINING trips, then
evaluates every smaller (R, k) by filtering. Output: experiments/candidates/sweep.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from joblib import Parallel, delayed

from pathsnap.config import Config
from pathsnap.features import build_trip_frame
from pathsnap.io import discover_trips, read_trip
from pathsnap.network import build_network

RADII = [15, 20, 30, 40, 50, 75, 100]
KS = [2, 4, 6, 8, 12, 16]


def one(prefix, tid, cfg):
    trip = read_trip(prefix, tid)
    net = build_network(trip, cfg.undirected, cfg.cand.sample_step)
    obs, c = build_trip_frame(trip, net, cfg, with_labels=True)
    c = c[c.trip_label_ok == 1]   # drop trips with unusable reference routes (33, 54)
    return c[["trip_id", "obs_idx", "cand_rank", "dist", "label"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=".")
    ap.add_argument("--splits", default=None)
    args = ap.parse_args()
    cfg = Config()
    cfg.cand.radius, cfg.cand.k = max(RADII), max(KS)
    idx = discover_trips(args.data_root, args.splits)
    idx = idx[idx.partition == "train"]
    c = pd.concat(Parallel(n_jobs=-1)(delayed(one)(r.prefix, r.trip_id, cfg)
                                      for r in idx.itertuples()), ignore_index=True)
    n_obs = c.groupby(["trip_id", "obs_idx"]).ngroups
    rows = []
    for R in RADII:
        for k in KS:
            s = c[(c.dist <= R) & (c.cand_rank < k)]
            hit = s.groupby(["trip_id", "obs_idx"]).label.max().sum()
            rows.append({"radius": R, "k": k, "miss_rate": 1 - hit / n_obs,
                         "mean_cands": len(s) / n_obs})
    df = pd.DataFrame(rows)
    out = Path("experiments/candidates")
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "sweep.csv", index=False)
    print(df.pivot(index="radius", columns="k", values="miss_rate").round(4).to_string())


if __name__ == "__main__":
    main()
