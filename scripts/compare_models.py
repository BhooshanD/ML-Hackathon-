"""Train and compare all candidate classifiers on the pipeline feature table.

    python scripts/compare_models.py

Model line-up and configurations follow models.ipynb, but every model is fitted on
data/features/train_cands.parquet (pathsnap.features.FEATURE_COLS) so the numbers are
reproducible from this repo and comparable with the decoder results.

Protocol: 5-fold GroupKFold by trip on the training trips (reliable labels only), then a
refit on all training trips evaluated on the 10 student-test trips, classifier-only and
followed by HMM/Viterbi (decoder parameters from artifacts/hmm_config.json, i.e. tuned
once for the deployed model and held fixed here).

Outputs: experiments/models/model_comparison.csv, cv_folds_all.csv
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from eval_decoder import prepare, run
from pathsnap.config import Config, HMMConfig
from pathsnap.features import FEATURE_COLS
from pathsnap.io import discover_trips

SEED = 42
# name: (family, factory, fraction of training rows used, configuration text)
MODELS = {
    "LogReg": ("Linear", lambda: make_pipeline(StandardScaler(), LogisticRegression(
        class_weight="balanced", max_iter=1000)), 1.0, "C=1.0, balanced weights, scaled"),
    "GaussNB": ("Probabilistic", lambda: make_pipeline(StandardScaler(), GaussianNB()), 1.0,
                "default var_smoothing, scaled"),
    "KNN": ("Instance-based", lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(
        n_neighbors=25, weights="distance", n_jobs=-1)), 0.05,
        "k=25, distance-weighted, scaled, 5% row sample"),
    "RandomForest": ("Ensemble (bagging)", lambda: RandomForestClassifier(
        n_estimators=100, min_samples_leaf=10, max_features="sqrt", max_samples=0.2,
        class_weight="balanced", n_jobs=-1, random_state=SEED), 1.0,
        "100 trees, min_leaf=10, 20% bootstrap per tree, balanced"),
    "MLP": ("Neural network", lambda: make_pipeline(StandardScaler(), MLPClassifier(
        hidden_layer_sizes=(64, 32), early_stopping=True, max_iter=200, random_state=SEED)), 0.3,
        "64-32 ReLU, adam, early stopping, 30% row sample"),
    "LightGBM (balanced)": ("Boosting", lambda: LGBMClassifier(
        n_estimators=400, learning_rate=0.05, num_leaves=31, class_weight="balanced",
        random_state=SEED, verbose=-1, n_jobs=-1), 1.0, "400 trees, lr=0.05, 31 leaves, balanced"),
    "LightGBM": ("Boosting", lambda: LGBMClassifier(
        n_estimators=400, learning_rate=0.05, num_leaves=31,
        random_state=SEED, verbose=-1, n_jobs=-1), 1.0,
        "400 trees, lr=0.05, 31 leaves, unweighted"),
}


_best = Path(__file__).resolve().parents[1] / "experiments/tuning/best_params.json"
if _best.exists():   # add the Bayesian-tuned configuration as a final row
    _params = json.loads(_best.read_text())["best_params"]
    MODELS["LightGBM (tuned)"] = ("Boosting", lambda: LGBMClassifier(
        **_params, random_state=SEED, verbose=-1, n_jobs=-1), 1.0,
        "Optuna TPE, 40 trials: %d trees, lr=%.3f, %d leaves (deployed)" % (
            _params["n_estimators"], _params["learning_rate"], _params["num_leaves"]))


def sample(n, frac):
    if frac >= 1:
        return np.arange(n)
    return np.sort(np.random.default_rng(SEED).choice(n, int(n * frac), replace=False))


def top1(frame, p):
    f = frame[["trip_id", "obs_idx", "label"]].assign(p=p)
    return float(f.loc[f.groupby(["trip_id", "obs_idx"]).p.idxmax()].label.mean())


def scores(frame, p):
    y, pred = frame.label.to_numpy(), (p >= 0.5).astype(int)
    return {"macro_f1": f1_score(y, pred, average="macro"),
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred),
            "log_loss": log_loss(y, np.clip(p, 1e-7, 1 - 1e-7)), "top1": top1(frame, p)}


def main():
    f = Path("data/features")
    cfg = Config.load(f / "config.json")
    hmm = HMMConfig(**json.loads(Path("artifacts/hmm_config.json").read_text()))
    tr = pd.read_parquet(f / "train_cands.parquet")
    tr = tr[(tr.trip_label_ok == 1) & (tr.label_reliable == 1)].reset_index(drop=True)
    te = pd.read_parquet(f / "student_test_cands.parquet")
    te_obs = pd.read_parquet(f / "student_test_obs.parquet")
    X, y, g = tr[FEATURE_COLS].to_numpy(float), tr.label.to_numpy(), tr.trip_id.to_numpy()
    Xte = te[FEATURE_COLS].to_numpy(float)

    idx = discover_trips(".").set_index("trip_id")
    te_trips = Parallel(n_jobs=-1)(
        delayed(prepare)(idx.loc[tid, "prefix"], tid, "student_test",
                         te_obs[te_obs.trip_id == tid], cg.reset_index(drop=True), cfg)
        for tid, cg in te.groupby("trip_id", sort=False))
    rows_of = {tid: np.flatnonzero((te.trip_id == tid).to_numpy()) for tid in te.trip_id.unique()}

    out = Path("experiments/models")
    out.mkdir(parents=True, exist_ok=True)
    table, all_folds = [], []
    for name, (family, make, frac, text) in MODELS.items():
        t_all = time.perf_counter()
        folds = []
        for k, (a, b) in enumerate(GroupKFold(n_splits=5).split(X, y, g), 1):
            a = a[sample(len(a), frac)]
            m = make().fit(X[a], y[a])
            folds.append({"model": name, "fold": k, **scores(tr.iloc[b], m.predict_proba(X[b])[:, 1])})
        cv = pd.DataFrame(folds)
        all_folds.append(cv)

        i = sample(len(X), frac)
        t0 = time.perf_counter(); m = make().fit(X[i], y[i]); train_s = time.perf_counter() - t0
        t0 = time.perf_counter(); p = m.predict_proba(Xte)[:, 1]; infer_s = time.perf_counter() - t0
        s = scores(te, p)
        clf = [run(T, p[rows_of[T["trip_id"]]], hmm, "classifier", full=True) for T in te_trips]
        dec = [run(T, p[rows_of[T["trip_id"]]], hmm, "hmm_cls", full=True) for T in te_trips]
        mean = lambda rs, key: float(np.mean([r[key] for r in rs]))
        table.append({
            "model": name, "family": family, "config": text,
            "cv_macro_f1": cv.macro_f1.mean(), "cv_macro_f1_std": cv.macro_f1.std(),
            "cv_top1": cv.top1.mean(), "cv_top1_std": cv.top1.std(), "cv_log_loss": cv.log_loss.mean(),
            "test_macro_f1": s["macro_f1"], "test_precision": s["precision"],
            "test_recall": s["recall"], "test_log_loss": s["log_loss"],
            "test_top1_pooled": s["top1"],
            "test_top1": mean(clf, "top1_accuracy"), "test_continuity": mean(clf, "route_continuity"),
            "test_route_f1": mean(clf, "route_f1"),
            "hmm_top1": mean(dec, "top1_accuracy"), "hmm_continuity": mean(dec, "route_continuity"),
            "hmm_route_f1": mean(dec, "route_f1"),
            "train_s": train_s, "infer_ms_per_1k_rows": 1000 * infer_s / len(Xte) * 1000,
        })
        pd.DataFrame(table).to_csv(out / "model_comparison.csv", index=False)
        pd.concat(all_folds).to_csv(out / "cv_folds_all.csv", index=False)
        r = table[-1]
        print(f"{name:20s} cvF1={r['cv_macro_f1']:.4f} cvTop1={r['cv_top1']:.4f} "
              f"testF1={r['test_macro_f1']:.4f} top1={r['test_top1']:.4f} "
              f"+HMM={r['hmm_top1']:.4f} ({time.perf_counter() - t_all:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
