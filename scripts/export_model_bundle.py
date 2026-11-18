"""Train the selected classifier on the PIPELINE feature table and save the bundle the
decoder / API load.

    python scripts/export_model_bundle.py                 # LightGBM, interim config
    python scripts/export_model_bundle.py --balanced      # same, class_weight="balanced"

Why this exists: models.ipynb was trained on a separate candidates.parquet with 11 `f_*`
features whose generating code is not in the repo, so that model cannot be fed by
src/pathsnap at inference time. This script fits the same algorithm/configuration on
data/features/train_cands.parquet (pathsnap.features.FEATURE_COLS), reports GroupKFold
metrics, and writes {"model", "feature_cols", "name"} to artifacts/model.joblib.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import GroupKFold

from pathsnap.features import FEATURE_COLS

SEED = 42


def make_model(balanced: bool):
    return LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                          class_weight="balanced" if balanced else None,
                          random_state=SEED, verbose=-1, n_jobs=-1)


def top1(frame: pd.DataFrame, p: np.ndarray) -> float:
    f = frame[["trip_id", "obs_idx", "label"]].assign(p=p)
    return float(f.loc[f.groupby(["trip_id", "obs_idx"]).p.idxmax()].label.mean())


def scores(frame, p):
    y = frame.label.to_numpy()
    return {"macro_f1": f1_score(y, (p >= 0.5).astype(int), average="macro"),
            "log_loss": log_loss(y, np.clip(p, 1e-7, 1 - 1e-7)), "top1": top1(frame, p)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/features")
    ap.add_argument("--out", default="artifacts/model.joblib")
    ap.add_argument("--balanced", action="store_true")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    f = Path(args.features)
    tr = pd.read_parquet(f / "train_cands.parquet")
    tr = tr[(tr.trip_label_ok == 1) & (tr.label_reliable == 1)].reset_index(drop=True)
    te = pd.read_parquet(f / "student_test_cands.parquet")
    X, y, g = tr[FEATURE_COLS], tr.label.to_numpy(), tr.trip_id.to_numpy()

    folds = []
    for k, (a, b) in enumerate(GroupKFold(n_splits=args.folds).split(X, y, g), 1):
        m = make_model(args.balanced).fit(X.iloc[a], y[a])
        folds.append({"fold": k, **scores(tr.iloc[b], m.predict_proba(X.iloc[b])[:, 1])})
        print(f"fold {k}: " + " ".join(f"{n}={v:.4f}" for n, v in folds[-1].items() if n != "fold"))
    cv = pd.DataFrame(folds)

    t0 = time.perf_counter()
    model = make_model(args.balanced).fit(X, y)
    train_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    p = model.predict_proba(te[FEATURE_COLS])[:, 1]
    infer_s = time.perf_counter() - t0
    test = scores(te, p)

    name = "lightgbm_balanced" if args.balanced else "lightgbm"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_cols": FEATURE_COLS, "name": name}, args.out)
    summary = {"model": name, "features": FEATURE_COLS, "params": model.get_params(),
               "cv_mean": cv.drop(columns="fold").mean().to_dict(),
               "cv_std": cv.drop(columns="fold").std().to_dict(),
               "student_test": test, "train_s": train_s,
               "infer_ms_per_1k_rows": 1000 * infer_s / len(te) * 1000,
               "note": "interim - untuned; same configuration as models.ipynb LightGBM"}
    out = Path("experiments/models")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}_pipeline_features.json").write_text(json.dumps(summary, indent=2, default=str))
    cv.to_csv(out / f"{name}_cv_folds.csv", index=False)
    print(f"\nCV   : " + " ".join(f"{n}={v:.4f}" for n, v in summary["cv_mean"].items()))
    print(f"TEST : " + " ".join(f"{n}={v:.4f}" for n, v in test.items()))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
