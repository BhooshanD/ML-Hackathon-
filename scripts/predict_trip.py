"""Predict one trip from the command line (same code path the API uses).

    python scripts/predict_trip.py 00000000 [--out result.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pathsnap.pipeline import predict_trip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trip_id")
    ap.add_argument("--data-root", default=".")
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--splits", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    r = predict_trip(args.trip_id, args.data_root, args.artifacts, args.splits)
    print(json.dumps({k: r[k] for k in ("trip_id", "model", "n_obs", "metrics", "timings")}, indent=2))
    print(f"HMM breaks: {len(r['hmm_breaks'])}, route arcs: {len(r['route_arcs'])}")
    if args.out:
        Path(args.out).write_text(json.dumps(r))


if __name__ == "__main__":
    main()
