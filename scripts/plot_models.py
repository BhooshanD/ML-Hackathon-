"""Figures for the model-comparison, tuning and decoder sections of the report.

    python scripts/plot_models.py

Reads only recorded experiment outputs (experiments/models, experiments/tuning,
experiments/decoder) and writes PNGs to artifacts/figures/. The one model refitted here is
the MLP, because it is the only model with an epoch-wise training history; its loss and
validation curves come from that actual fit (same configuration and seed as
scripts/compare_models.py). No curves are drawn for models without a training history.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIG = Path("artifacts/figures")
BLUE, ORANGE, GREEN, GREY = "#4C72B0", "#DD8452", "#55A868", "#8C8C8C"


def label(ax, bars, fmt="{:.3f}"):
    for b in bars:
        ax.annotate(fmt.format(b.get_height()), (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", fontsize=7, xytext=(0, 1), textcoords="offset points")


def comparison():
    c = pd.read_csv("experiments/models/model_comparison.csv")
    x, names = np.arange(len(c)), c.model.str.replace(" (", "\n(", regex=False)

    fig, ax = plt.subplots(figsize=(9, 4.2))
    b = ax.bar(x, c.cv_macro_f1, yerr=c.cv_macro_f1_std, capsize=3, color=BLUE)
    label(ax, b)
    ax.set(ylim=(0.8, 1.0), ylabel="macro F1", xticks=x,
           title="Cross-validation macro F1 (5-fold GroupKFold by trip, mean ± std)")
    ax.set_xticklabels(names, fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "model_cv_macro_f1.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4.4))
    w = 0.27
    for i, (col, name, colr) in enumerate([("test_top1", "Top-1, classifier only", BLUE),
                                           ("hmm_top1", "Top-1, classifier + HMM", GREEN),
                                           ("test_macro_f1", "macro F1 (candidate level)", ORANGE)]):
        label(ax, ax.bar(x + (i - 1) * w, c[col], w, label=name, color=colr))
    ax.set(ylim=(0.8, 1.0), xticks=x, ylabel="score",
           title="Student-test performance (10 trips; Top-1 = mean over trips)")
    ax.set_xticklabels(names, fontsize=8); ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "model_student_test.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for i, (col, name, colr) in enumerate([("cv_log_loss", "cross-validation", BLUE),
                                           ("test_log_loss", "student-test", ORANGE)]):
        label(ax[0], ax[0].bar(x + (i - 0.5) * 0.4, c[col], 0.4, label=name, color=colr))
    ax[0].set(xticks=x, ylabel="log loss (lower is better)", title="Log loss")
    ax[0].set_xticklabels(names, fontsize=7); ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, frameon=False, fontsize=8)
    for i, (col, name, colr) in enumerate([("test_continuity", "classifier only", BLUE),
                                           ("hmm_continuity", "classifier + HMM", GREEN)]):
        label(ax[1], ax[1].bar(x + (i - 0.5) * 0.4, c[col], 0.4, label=name, color=colr))
    ax[1].set(ylim=(0.8, 1.005), xticks=x, ylabel="route continuity", title="Route continuity (student-test)")
    ax[1].set_xticklabels(names, fontsize=7); ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "model_log_loss_continuity.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.scatter(c.infer_ms_per_1k_rows, c.hmm_top1, s=40, color=BLUE)
    for r in c.itertuples():
        ax.annotate(r.model, (r.infer_ms_per_1k_rows, r.hmm_top1), fontsize=7,
                    xytext=(4, 3), textcoords="offset points")
    ax.set(xscale="log", xlabel="inference time (ms per 1,000 candidate rows)",
           ylabel="Top-1, classifier + HMM", title="Accuracy vs inference cost")
    fig.tight_layout(); fig.savefig(FIG / "model_accuracy_vs_inference.png", dpi=200); plt.close(fig)


def mlp_curve():
    from compare_models import MODELS, SEED, sample
    from pathsnap.features import FEATURE_COLS
    tr = pd.read_parquet("data/features/train_cands.parquet")
    tr = tr[(tr.trip_label_ok == 1) & (tr.label_reliable == 1)].reset_index(drop=True)
    i = sample(len(tr), MODELS["MLP"][2])
    m = MODELS["MLP"][1]().fit(tr[FEATURE_COLS].to_numpy(float)[i], tr.label.to_numpy()[i])
    net = m[-1]
    pd.DataFrame({"epoch": np.arange(1, len(net.loss_curve_) + 1), "train_loss": net.loss_curve_,
                  "validation_accuracy": net.validation_scores_}).to_csv(
        "experiments/models/mlp_training_history.csv", index=False)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ep = np.arange(1, len(net.loss_curve_) + 1)
    ax.plot(ep, net.loss_curve_, color=BLUE, label="training loss (log loss)")
    ax.set(xlabel="epoch", ylabel="training loss")
    ax2 = ax.twinx()
    ax2.plot(ep, net.validation_scores_, color=ORANGE, label="validation accuracy (10% hold-out)")
    ax2.set_ylabel("validation accuracy")
    ax.set_title(f"MLP (64-32) training history: early stopping after {net.n_iter_} epochs")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="center right", fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "mlp_training_history.png", dpi=200); plt.close(fig)
    print(f"MLP stopped after {net.n_iter_} epochs (max_iter={net.max_iter})")


def tuning():
    p = Path("experiments/tuning/trials.csv")
    if not p.exists():
        return
    t = pd.read_csv(p)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ax[0].plot(t.number, t.value, "o", ms=4, color=GREY, label="trial")
    ax[0].plot(t.number, t.value.cummax(), color=BLUE, label="best so far")
    ax[0].axhline(t.value.iloc[0], ls="--", color=ORANGE, lw=1, label="untuned configuration (trial 0)")
    ax[0].set(xlabel="trial", ylabel="CV Top-1 (5-fold GroupKFold)", title="Optuna TPE optimisation history")
    ax[0].legend(fontsize=8)
    sc = ax[1].scatter(t.params_learning_rate, t.params_num_leaves, c=t.value, cmap="viridis", s=35)
    ax[1].set(xscale="log", yscale="log", xlabel="learning_rate", ylabel="num_leaves",
              title="Search coverage (colour = CV Top-1)")
    fig.colorbar(sc, ax=ax[1])
    fig.tight_layout(); fig.savefig(FIG / "tuning_history.png", dpi=200); plt.close(fig)

    c = Path("experiments/tuning/tuned_vs_untuned.csv")
    if c.exists():
        c = pd.read_csv(c)
        cols = ["cv_macro_f1", "cv_top1", "test_top1", "test_macro_f1"]
        fig, ax = plt.subplots(figsize=(8, 4.2))
        x = np.arange(len(cols))
        for i, r in enumerate(c.itertuples()):
            label(ax, ax.bar(x + (i - 0.5) * 0.4, [getattr(r, k) for k in cols], 0.4,
                             label=r.model, color=[GREY, GREEN][i]), "{:.4f}")
        ax.set(ylim=(0.88, 0.96), xticks=x, title="LightGBM: untuned vs Bayesian-tuned")
        ax.set_xticklabels(["CV macro F1", "CV Top-1", "student-test Top-1", "student-test macro F1"], fontsize=8)
        ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(FIG / "tuned_vs_untuned.png", dpi=200); plt.close(fig)


def decoder():
    d = pd.read_csv("experiments/decoder/per_trip_student_test.csv", dtype={"trip_id": str})
    methods = list(d.method.unique())
    clf, hmm = methods[1], methods[3]
    p = d.pivot(index="trip_id", columns="method", values="top1_accuracy")
    x = np.arange(len(p))
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.bar(x - 0.2, p[clf], 0.4, color=BLUE, label=clf)
    ax.bar(x + 0.2, p[hmm], 0.4, color=GREEN, label=hmm)
    for i, (a, b) in enumerate(zip(p[clf], p[hmm])):
        ax.annotate(f"{(b - a) * 100:+.1f}", (i, max(a, b)), ha="center", va="bottom", fontsize=8,
                    color="black" if b >= a else "#C44E52", xytext=(0, 2), textcoords="offset points")
    ax.set(ylim=(0.7, 1.02), xticks=x, ylabel="Top-1 accuracy",
           title="Classifier only vs classifier + HMM/Viterbi, per student-test trip (label = change in points)")
    ax.set_xticklabels(p.index, fontsize=8); ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "decoder_per_trip.png", dpi=200); plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    comparison(); decoder(); tuning(); mlp_curve()
    print("figures:", sorted(p.name for p in FIG.glob("*.png")))
