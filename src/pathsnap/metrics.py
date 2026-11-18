"""Evaluation metrics for map matching (all computed against server-side reference data)."""
from __future__ import annotations

import numpy as np

from .network import RoadNetwork


def top1_accuracy(pred_arc: np.ndarray, label_arc: np.ndarray) -> float:
    return float(np.mean(pred_arc == label_arc)) if len(pred_arc) else float("nan")


def candidate_miss_rate(label_in_cands: np.ndarray) -> float:
    """Fraction of observations whose reference arc is not in the candidate set (upper
    bound on achievable Top-1 accuracy = 1 - miss rate)."""
    return float(1.0 - np.mean(label_in_cands)) if len(label_in_cands) else float("nan")


def route_continuity(net: RoadNetwork, pred_arc: np.ndarray, gps_dist: np.ndarray | None = None,
                     slack: float = 30.0) -> float:
    """Fraction of consecutive arc changes that are physically drivable.

    A change a -> b counts as continuous if b directly follows a, or (when gps_dist is
    given) b is reachable from a within 2 * gps_dist + slack metres of driving (covers
    short arcs skipped between two 1 Hz samples).
    """
    a, b = pred_arc[:-1], pred_arc[1:]
    change = a != b
    if not change.any():
        return 1.0
    a, b = a[change], b[change]
    ok = net.arcs_adjacent(a, b)
    if gps_dist is not None:
        g = gps_dist[1:][change]
        for i in np.flatnonzero(~ok):
            lim = 2 * g[i] + slack
            d = net.sp_rows(np.array([net.arc_to[a[i]]]), lim)[0, net.arc_from[b[i]]]
            ok[i] = np.isfinite(d) and d <= lim
    return float(np.mean(ok))


def route_overlap(net: RoadNetwork, pred_arcs: np.ndarray, ref_route: np.ndarray) -> dict:
    """Length-weighted precision / recall / F1 of the set of traversed arcs."""
    p, r = set(map(int, pred_arcs)), set(map(int, ref_route))
    L = lambda s: float(sum(net.length[i] for i in s))
    inter = L(p & r)
    prec = inter / L(p) if p else 0.0
    rec = inter / L(r) if r else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"route_precision": prec, "route_recall": rec, "route_f1": f1}


def trip_metrics(net: RoadNetwork, pred_arc, label_arc, label_in_cands, ref_route,
                 gps_dist=None, filled_route=None) -> dict:
    m = {
        "top1_accuracy": top1_accuracy(pred_arc, label_arc),
        "candidate_miss_rate": candidate_miss_rate(label_in_cands),
        "route_continuity": route_continuity(net, pred_arc, gps_dist),
        "n_obs": int(len(pred_arc)),
    }
    m.update(route_overlap(net, filled_route if filled_route is not None else pred_arc, ref_route))
    return m
