"""Additional dataset checks that complement experiments/eda/01_dataset_exploration.ipynb.

    python scripts/eda_checks.py

Writes to artifacts/eda/:
  sampling_interval_summary.csv  pooled sampling-interval statistics per partition
  route_validation.csv           route id validity + contiguity for EVERY trip
  arc_direction.csv              directed-graph structure (reverse twins, components)
  route_gps_agreement.csv        does the reference route follow the GPS track? (metres)
  trip_extent_metric.csv         trip size in kilometres (projected coordinates)
and figures to artifacts/figures/. Partitions used: train and student_test (labelled in
every table); the evaluator partition is not read.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from pathsnap.config import Config
from pathsnap.features import build_trip_frame
from pathsnap.io import discover_trips, read_trip
from pathsnap.network import build_network

EDA, FIG = Path("artifacts/eda"), Path("artifacts/figures")
C = {"train": "#4C72B0", "student_test": "#DD8452"}


def one_trip(prefix, tid, part, cfg):
    trip = read_trip(prefix, tid, part)
    net = build_network(trip)
    obs, cands = build_trip_frame(trip, net, cfg, with_labels=True)
    t = trip.track.t.to_numpy()
    dt = np.diff(t)
    xy = obs[["x", "y"]].to_numpy()
    step = np.hypot(*np.diff(xy, axis=0).T)
    r = trip.route
    ids_ok = (r >= 0) & (r < len(trip.arcs))
    disc = int(np.sum(trip.arcs[r[:-1], 1] != trip.arcs[r[1:], 0])) if ids_ok.all() else -1
    s = net.structure_report()
    nearest = cands.groupby("obs_idx").dist.min().to_numpy()
    return {
        "intervals": pd.DataFrame({"partition": part, "trip_id": tid, "dt": dt}),
        "route": {"partition": part, "trip_id": tid, "route_arcs": len(r),
                  "unique_route_arcs": int(len(np.unique(r))),
                  "invalid_arc_ids": int((~ids_ok).sum()), "disconnected_steps": disc},
        "direction": {"partition": part, "trip_id": tid, "n_nodes": s["n_nodes"],
                      "n_arcs": s["n_arcs"],
                      "frac_arcs_with_reverse_twin": s["frac_arcs_with_reverse_twin"],
                      "weak_components": s["n_weak_components"],
                      "strong_components": s["n_strong_components"],
                      "median_arc_length_m": s["arc_length_median_m"]},
        "agreement": {"partition": part, "trip_id": tid, "gps_points": len(obs),
                      "gps_to_nearest_arc_median_m": float(np.median(nearest)),
                      "gps_to_nearest_arc_p95_m": float(np.percentile(nearest, 95)),
                      "gps_to_reference_route_median_m": float(np.median(obs.label_dist)),
                      "gps_to_reference_route_p95_m": float(np.percentile(obs.label_dist, 95)),
                      "frac_points_over_50m_from_route": float(1 - obs.label_reliable.mean()),
                      "usable_for_training": int(obs.trip_label_ok.iloc[0])},
        "extent": {"partition": part, "trip_id": tid,
                   "utm_epsg": net.proj.crs.to_epsg(),
                   "bbox_width_km": float(np.ptp(xy[:, 0]) / 1000),
                   "bbox_height_km": float(np.ptp(xy[:, 1]) / 1000),
                   "track_length_km": float(step.sum() / 1000),
                   "duration_min": float((t[-1] - t[0]) / 60),
                   "median_speed_kmh": float(np.median(step / np.maximum(dt, 1e-9)) * 3.6),
                   "max_gap_s": float(dt.max())},
    }


def main():
    cfg = Config()
    idx = discover_trips(".")
    idx = idx[idx.partition.isin(["train", "student_test"])]
    res = Parallel(n_jobs=-1)(delayed(one_trip)(r.prefix, r.trip_id, r.partition, cfg)
                              for r in idx.itertuples())
    EDA.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    iv = pd.concat([r["intervals"] for r in res], ignore_index=True)
    tabs = {k: pd.DataFrame([r[k] for r in res]) for k in ("route", "direction", "agreement", "extent")}
    tabs["route"].to_csv(EDA / "route_validation.csv", index=False)
    tabs["direction"].to_csv(EDA / "arc_direction.csv", index=False)
    tabs["agreement"].to_csv(EDA / "route_gps_agreement.csv", index=False)
    tabs["extent"].to_csv(EDA / "trip_extent_metric.csv", index=False)

    summ = iv.groupby("partition").dt.agg(
        n_intervals="size", mean_s="mean", median_s="median", min_s="min", max_s="max",
        frac_exactly_1s=lambda d: np.isclose(d, 1.0).mean(),
        frac_over_1_5s=lambda d: (d > 1.5).mean(),
        n_over_60s=lambda d: (d > 60).sum(), n_over_1h=lambda d: (d > 3600).sum())
    per_trip_max = iv.groupby(["partition", "trip_id"]).dt.max().reset_index()
    summ["trips_with_gap_over_60s"] = per_trip_max[per_trip_max.dt > 60].groupby("partition").size()
    summ = summ.fillna(0)
    summ.to_csv(EDA / "sampling_interval_summary.csv")

    # ---- figures
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    bins = np.logspace(0, np.log10(iv.dt.max() * 1.1), 45)
    for p in ("train", "student_test"):
        ax[0].hist(iv.dt[iv.partition == p], bins=bins, alpha=0.75, label=p, color=C[p])
    ax[0].set(xscale="log", yscale="log", xlabel="interval between consecutive GPS points (s)",
              ylabel="count", title="Sampling intervals (log-log)")
    ax[0].legend()
    top = per_trip_max.sort_values("dt", ascending=False).head(15)
    ax[1].barh(top.trip_id, top.dt, color=[C[p] for p in top.partition])
    ax[1].set(xscale="log", xlabel="largest gap in trip (s)", title="15 trips with the largest gaps")
    ax[1].invert_yaxis()
    fig.tight_layout(); fig.savefig(FIG / "sampling_intervals.png", dpi=200); plt.close(fig)

    a = tabs["agreement"].sort_values("gps_to_reference_route_median_m", ascending=False)
    fig, ax = plt.subplots(figsize=(13, 4.2))
    col = ["#C44E52" if u == 0 else C[p] for u, p in zip(a.usable_for_training, a.partition)]
    ax.bar(range(len(a)), a.gps_to_reference_route_median_m, color=col)
    dots, = ax.plot(range(len(a)), a.gps_to_nearest_arc_median_m, "k.", ms=4)
    line = ax.axhline(30, ls="--", c="grey", lw=1)
    ax.set(yscale="log", ylabel="median distance (m)", xlabel="trips (sorted by distance to reference route)",
           title="Does the reference route follow the GPS track?")
    ax.set_xticks([])
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=C["train"], label="GPS -> reference route, train"),
                       Patch(color=C["student_test"], label="GPS -> reference route, student_test"),
                       Patch(color="#C44E52", label="GPS -> reference route, excluded from training"),
                       dots, line],
              labels=["GPS -> reference route, train", "GPS -> reference route, student_test",
                      "GPS -> reference route, excluded from training",
                      "GPS -> nearest road", "trip exclusion threshold (30 m)"])
    fig.tight_layout(); fig.savefig(FIG / "route_gps_agreement.png", dpi=200); plt.close(fig)

    d, e = tabs["direction"], tabs["extent"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for p in ("train", "student_test"):
        m = d.partition == p
        ax[0].hist(d.frac_arcs_with_reverse_twin[m], bins=np.linspace(0.3, 1, 29), alpha=0.75,
                   color=C[p], label=p)
        ax[1].scatter(d.n_nodes[m], d.n_arcs[m], s=18, color=C[p], label=p)
        ax[2].scatter(e.track_length_km[e.partition == p], e.duration_min[e.partition == p],
                      s=18, color=C[p], label=p)
    ax[0].set(xlabel="fraction of arcs that have a reverse twin", ylabel="trips",
              title="Arcs are directed (two-way road = 2 arcs)")
    ax[1].set(xscale="log", yscale="log", xlabel="nodes", ylabel="arcs", title="Road-network size per trip")
    ax[2].set(xscale="log", yscale="log", xlabel="track length (km)", ylabel="duration (min)",
              title="Trip length vs duration")
    for x in ax:
        x.legend()
    fig.tight_layout(); fig.savefig(FIG / "network_and_trip_structure.png", dpi=200); plt.close(fig)

    pd.set_option("display.width", 250)
    print(summ.round(4).T.to_string())
    r = tabs["route"]
    print("\nroutes: invalid ids", int(r.invalid_arc_ids.sum()), "| disconnected steps",
          int(r.disconnected_steps.sum()), "| trips revisiting an arc",
          int((r.unique_route_arcs < r.route_arcs).sum()))
    print("reverse-twin fraction: min %.3f median %.3f max %.3f" % (
        d.frac_arcs_with_reverse_twin.min(), d.frac_arcs_with_reverse_twin.median(),
        d.frac_arcs_with_reverse_twin.max()))
    print("components (mean) weak %.1f strong %.1f | median arc length %.1f m" % (
        d.weak_components.mean(), d.strong_components.mean(), d.median_arc_length_m.median()))
    print("\nworst route/GPS agreement:\n", a.head(6).round(2).to_string(index=False))
    print("\nextent by partition:\n", e.groupby("partition")[
        ["bbox_width_km", "bbox_height_km", "track_length_km", "duration_min", "median_speed_kmh"]
    ].agg(["median", "max"]).round(2).to_string())
    print("\nlargest gaps:\n", e.sort_values("max_gap_s", ascending=False).head(5)[
        ["partition", "trip_id", "max_gap_s", "duration_min", "track_length_km"]].to_string(index=False))
    print("UTM zones used:", e.utm_epsg.nunique())


if __name__ == "__main__":
    main()
