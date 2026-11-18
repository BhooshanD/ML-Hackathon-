"""Bayesian hyperparameter optimisation (Optuna TPE) for the selected model, LightGBM.

    python scripts/tune_lightgbm.py --trials 40

Objective : mean observation-level Top-1 accuracy over 5-fold GroupKFold (by trip) on the
            80 training trips (reliable labels only). Top-1 is the quantity the task is
            scored on and what the HMM consumes; CV macro F1 and log loss are recorded
            for every trial as well.
Leakage   : student-test is touched once, after the search, to compare tuned vs untuned.
            The evaluator partition is never read.

Outputs (experiments/tuning/): trials.csv (full search record), best_params.json,
tuned_vs_untuned.csv. The tuned bundle replaces artifacts/model.joblib only if it beats
the untuned model on CROSS-VALIDATION Top-1 (not on student-test).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import GroupKFold

from pathsnap.features import FEATURE_COLS

SEED = 42
UNTUNED = dict(n_estimators=400, learning_rate=0.05, num_leaves=31)
SEARCH_SPACE = {
    "n_estimators": "int 200..1000",
    "learning_rate": "log-uniform 0.01..0.2",
    "num_leaves": "int log 15..255",
    "min_child_samples": "int log 10..300",
    "subsample": "uniform 0.5..1.0 (subsample_freq=1)",
    "colsample_bytree": "uniform 0.5..1.0",
    "reg_lambda": "log-uniform 1e-3..10",
    "reg_alpha": "log-uniform 1e-3..10",
}


def make(params):
    return LGBMClassifier(**params, random_state=SEED, verbose=-1, n_jobs=-1)


def top1(frame, p):
    f = frame[["trip_id", "obs_idx", "label"]].assign(p=p)
    return float(f.loc[f.groupby(["trip_id", "obs_idx"]).p.idxmax()].label.mean())


def scores(frame, p):
    y = frame.label.to_numpy()
    return {"top1": top1(frame, p), "macro_f1": f1_score(y, (p >= 0.5).astype(int), average="macro"),
            "log_loss": log_loss(y, np.clip(p, 1e-7, 1 - 1e-7))}


def cv_scores(params, tr, X, y, g):
    rows = []
    for a, b in GroupKFold(n_splits=5).split(X, y, g):
        m = make(params).fit(X[a], y[a])
        rows.append(scores(tr.iloc[b], m.predict_proba(X[b])[:, 1]))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--features", default="data/features")
    args = ap.parse_args()

    f = Path(args.features)
    tr = pd.read_parquet(f / "train_cands.parquet")
    tr = tr[(tr.trip_label_ok == 1) & (tr.label_reliable == 1)].reset_index(drop=True)
    te = pd.read_parquet(f / "student_test_cands.parquet")
    X, y, g = tr[FEATURE_COLS].to_numpy(float), tr.label.to_numpy(), tr.trip_id.to_numpy()
    Xte = te[FEATURE_COLS].to_numpy(float)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 300, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
        }
        t0 = time.perf_counter()
        cv = cv_scores(params, tr, X, y, g)
        trial.set_user_attr("cv_macro_f1", float(cv.macro_f1.mean()))
        trial.set_user_attr("cv_log_loss", float(cv.log_loss.mean()))
        trial.set_user_attr("cv_top1_std", float(cv.top1.std()))
        trial.set_user_attr("seconds", time.perf_counter() - t0)
        return float(cv.top1.mean())

    out = Path("experiments/tuning")
    out.mkdir(parents=True, exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", study_name="lightgbm_top1",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.enqueue_trial({**UNTUNED, "min_child_samples": 20, "subsample": 1.0,
                         "colsample_bytree": 1.0, "reg_lambda": 1e-3, "reg_alpha": 1e-3})

    def log(study, trial):
        study.trials_dataframe().to_csv(out / "trials.csv", index=False)
        print(f"trial {trial.number:3d} top1={trial.value:.4f} best={study.best_value:.4f} "
              f"({trial.user_attrs['seconds']:.0f}s)", flush=True)

    t0 = time.perf_counter()
    study.optimize(objective, n_trials=args.trials, callbacks=[log])
    search_s = time.perf_counter() - t0
    best = {**study.best_params, "subsample_freq": 1}

    rows, models = [], {}
    for name, params in (("untuned", UNTUNED), ("tuned", best)):
        cv = cv_scores(params, tr, X, y, g)
        t1 = time.perf_counter(); m = make(params).fit(X, y); train_s = time.perf_counter() - t1
        t1 = time.perf_counter(); p = m.predict_proba(Xte)[:, 1]; infer_s = time.perf_counter() - t1
        s = scores(te, p)
        models[name] = m
        rows.append({"model": f"LightGBM {name}", "cv_macro_f1": cv.macro_f1.mean(),
                     "cv_macro_f1_std": cv.macro_f1.std(), "cv_top1": cv.top1.mean(),
                     "cv_top1_std": cv.top1.std(), "cv_log_loss": cv.log_loss.mean(),
                     "test_top1": s["top1"], "test_macro_f1": s["macro_f1"],
                     "test_log_loss": s["log_loss"], "train_s": train_s,
                     "infer_ms_per_1k_rows": 1000 * infer_s / len(Xte) * 1000})
    cmp = pd.DataFrame(rows)
    cmp.to_csv(out / "tuned_vs_untuned.csv", index=False)
    study.trials_dataframe().to_csv(out / "trials.csv", index=False)

    use_tuned = cmp.cv_top1[1] > cmp.cv_top1[0]
    (out / "best_params.json").write_text(json.dumps(
        {"objective": "mean 5-fold GroupKFold observation-level Top-1 accuracy",
         "sampler": "optuna TPESampler(seed=42)", "n_trials": len(study.trials),
         "search_seconds": search_s, "search_space": SEARCH_SPACE,
         "best_trial": study.best_trial.number, "best_cv_top1": study.best_value,
         "best_params": best, "untuned_params": UNTUNED,
         "deployed": "tuned" if use_tuned else "untuned"}, indent=2))
    name = "lightgbm_tuned" if use_tuned else "lightgbm"
    joblib.dump({"model": models["tuned" if use_tuned else "untuned"],
                 "feature_cols": FEATURE_COLS, "name": name}, "artifacts/model.joblib")
    print(cmp.round(4).to_string())
    print(f"best params: {best}\ndeployed: {name} ({search_s:.0f}s search)")


if __name__ == "__main__":
    main()
