"""Turn a reference route (sequence of arc ids) into one label arc per GPS observation.

The .route file is a sequence of arcs, not a per-point label, so we need an alignment.
We find the monotone assignment j(0) <= j(1) <= ... <= j(T-1) of observations to
route positions that minimises the total point-to-arc distance (dynamic programming,
O(T * L)). Monotonicity is what makes this correct for routes that revisit the same
road (e.g. out-and-back trips): a point can never jump back to an earlier route arc.
Skipping route arcs is allowed (short arcs traversed between two 1 Hz samples).

This uses ground truth, so it is ONLY for training labels and server-side scoring.
"""
from __future__ import annotations

import warnings

import numpy as np

from .network import RoadNetwork


def align_route(obs_xy: np.ndarray, route: np.ndarray, net: RoadNetwork,
                clip: float = 200.0, chunk: int = 2_000_000):
    """Returns (label_arc (T,), label_dist (T,), route_pos (T,))."""
    route = np.asarray(route, dtype=np.int64)
    ok = (route >= 0) & (route < net.n_arcs)
    ok[ok] &= net.valid[route[ok]]
    if not ok.all():
        warnings.warn(f"dropping {int((~ok).sum())} invalid arc ids from route")
        route = route[ok]
    T, L = len(obs_xy), len(route)
    if T == 0 or L == 0:
        return (np.full(T, -1, np.int64), np.full(T, np.nan), np.full(T, -1, np.int64))

    # cost[i, j] = distance from observation i to route arc j (clipped for robustness)
    cost = np.empty((T, L))
    rows_per_chunk = max(1, chunk // L)
    for s in range(0, T, rows_per_chunk):
        e = min(T, s + rows_per_chunk)
        P = np.repeat(obs_xy[s:e], L, axis=0)
        A = np.tile(route, e - s)
        d, _, _ = net.project(P, A)
        cost[s:e] = d.reshape(e - s, L)
    cost = np.minimum(cost, clip)

    acc = cost[0].copy()
    back = np.empty((T, L), dtype=np.int32)
    back[0] = np.arange(L)
    idx = np.arange(L)
    for i in range(1, T):
        # best predecessor j' <= j = argmin of acc[0..j] (running prefix argmin):
        # positions where acc hits its running minimum, carried forward.
        run_min = np.minimum.accumulate(acc)
        pm_idx = np.maximum.accumulate(np.where(acc <= run_min, idx, 0))
        back[i] = pm_idx
        acc = cost[i] + acc[pm_idx]

    pos = np.empty(T, dtype=np.int64)
    pos[-1] = int(np.argmin(acc))
    for i in range(T - 1, 0, -1):
        pos[i - 1] = back[i, pos[i]]
    label_arc = route[pos]
    label_dist = cost[np.arange(T), pos]
    return label_arc, label_dist, pos
