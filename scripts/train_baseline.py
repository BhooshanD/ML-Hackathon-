"""PLACEHOLDER model so the pipeline runs end-to-end before Person 2's models exist.

    python scripts/train_baseline.py

Fits StandardScaler + LogisticRegression on the training candidates and saves the bundle
in the agreed format to artifacts/model.joblib. Person 2 overwrites this file with the
selected, Bayesian-tuned model using the same format:
    {"model": estimator_with_predict_proba, "feature_cols": [...], "name": "..."}
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pathsnap.features import FEATURE_COLS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/features")
    ap.add_argument("--out", default="artifacts/model.joblib")
    args = ap.parse_args()
    tr = pd.read_parquet(Path(args.features) / "train_cands.parquet")
    tr = tr[(tr.trip_label_ok == 1) & (tr.label_reliable == 1)]  # drop bad reference labels
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    model.fit(tr[FEATURE_COLS].to_numpy(dtype=float), tr.label.to_numpy())
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_cols": FEATURE_COLS, "name": "baseline_logreg"}, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
