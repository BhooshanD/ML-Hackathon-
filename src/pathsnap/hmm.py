"""HMM + Viterbi route reconstruction.

Hidden state at time t : one candidate (arc_id, offset along arc) of GPS observation t.
Emission  log p(z_t | s) : w_cls * log P_classifier(s | z_t)  [+ w_dist * Gaussian(dist)]
                           (pure Gaussian distance if no classifier is supplied)
Transition log p(s'|s)   : Newson & Krumm (2009): |route_dist(s, s') - gps_dist(z_t, z_t+1)|
                           ~ Exponential(beta). route_dist comes from Dijkstra on the
                           directed road graph, so a transition onto a disconnected or
                           wrong-way segment gets a huge route distance -> ~0 probability.
Infeasible transitions   : unreachable within search limit, or implied speed > vmax -> -inf.
Breaks                   : if *no* transition into step t is feasible (e.g. a big GPS gap
                           or a missing road), the chain is cut and decoding restarts at
                           t from emissions only. Each segment is decoded independently.

Route distances do not depend on (beta, sigma, w_*), so they are precomputed once per trip
(`precompute_transitions`) and reused for every decoder-tuning trial.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import HMMConfig
from .network import RoadNetwork


@dataclass
class TripLattice:
    """Candidates grouped per observation (all arrays are lists over observations)."""
    arcs: list[np.ndarray]
    offsets: list[np.ndarray]
    dists: list[np.ndarray]
    row_index: list[np.ndarray]   # row positions in the candidate frame
    xy: np.ndarray                # (T, 2)
    t: np.ndarray                 # (T,)

    @classmethod
    def from_frames(cls, obs: pd.DataFrame, cands: pd.DataFrame) -> "TripLattice":
        cands = cands.reset_index(drop=True)
        order = np.lexsort((cands.cand_rank.to_numpy(), cands.obs_idx.to_numpy()))
        oi = cands.obs_idx.to_numpy()[order]
        bounds = np.searchsorted(oi, np.arange(len(obs) + 1))
        split = lambda col: [col[order[bounds[i]:bounds[i + 1]]] for i in range(len(obs))]
        return cls(split(cands.arc_id.to_numpy()), split(cands.offset.to_numpy()),
                   split(cands.dist.to_numpy()), [order[bounds[i]:bounds[i + 1]] for i in range(len(obs))],
                   obs[["x", "y"]].to_numpy(), obs.t.to_numpy(dtype=float))


def route_distances(net: RoadNetwork, sa, so, da, do, limit: float, back_tol: float,
                    back_penalty: float = 1.0) -> np.ndarray:
    """Driving distance from each source state (arc sa, offset so) to each dest state."""
    if net.undirected:
        # leave source through either end, enter destination through either end
        best = np.full((len(sa), len(da)), np.inf)
        for exit_node, exit_cost in ((net.arc_to[sa], net.length[sa] - so), (net.arc_from[sa], so)):
            rows = net.sp_rows(exit_node, limit)
            for entry_node, entry_cost in ((net.arc_from[da], do), (net.arc_to[da], net.length[da] - do)):
                best = np.minimum(best, exit_cost[:, None] + rows[:, entry_node] + entry_cost[None, :])
        same = sa[:, None] == da[None, :]
        return np.where(same, np.abs(do[None, :] - so[:, None]), best)

    rows = net.sp_rows(net.arc_to[sa], limit)                       # (k1, N)
    D = (net.length[sa] - so)[:, None] + rows[:, net.arc_from[da]] + do[None, :]
    fwd = do[None, :] - so[:, None]
    same_ok = (sa[:, None] == da[None, :]) & (fwd >= -back_tol)
    # forward on the same arc costs its length; small backward jitter is tolerated but
    # penalised, so the wrong-direction twin of a two-way road does not win for free
    same_cost = np.where(fwd >= 0, fwd, -fwd * (1.0 + back_penalty))
    return np.where(same_ok, same_cost, D)


def precompute_transitions(net: RoadNetwork, lat: TripLattice, cfg: HMMConfig):
    """Returns (D_list, gps_dist, dt): D_list[t] is (k_{t-1}, k_t) route distances."""
    T = len(lat.arcs)
    gps = np.r_[0.0, np.hypot(*np.diff(lat.xy, axis=0).T)]
    dt = np.r_[0.0, np.diff(lat.t)]
    D_list = [None] * T
    for t in range(1, T):
        limit = cfg.route_factor * gps[t] + cfg.route_slack
        limit = max(limit, cfg.vmax * max(dt[t], 1.0)) if dt[t] > 5 else limit
        D_list[t] = route_distances(net, lat.arcs[t - 1], lat.offsets[t - 1],
                                    lat.arcs[t], lat.offsets[t], limit, cfg.back_tol,
                                    cfg.back_penalty)
    return D_list, gps, dt


def emission_logprob(lat: TripLattice, proba: np.ndarray | None, cfg: HMMConfig) -> list[np.ndarray]:
    """`proba` is aligned to candidate-frame rows (P(candidate is correct))."""
    out = []
    for i, d in enumerate(lat.dists):
        le = np.zeros(len(d))
        if proba is not None:
            p = proba[lat.row_index[i]]
            p = p / max(p.sum(), cfg.eps)
            le += cfg.w_cls * np.log(p + cfg.eps)
            if cfg.w_dist > 0:
                le += -cfg.w_dist * 0.5 * (d / cfg.sigma) ** 2
        else:
            le += -0.5 * (d / cfg.sigma) ** 2
        out.append(le)
    return out


def viterbi(log_em: list[np.ndarray], D_list, gps: np.ndarray, dt: np.ndarray, cfg: HMMConfig):
    """Returns (state index per observation, list of break positions)."""
    T = len(log_em)
    deltas = [None] * T
    backs = [None] * T
    breaks = []
    delta = log_em[0] - log_em[0].max()
    deltas[0] = delta
    for t in range(1, T):
        D = D_list[t]
        beta = cfg.beta * (1.0 + cfg.gap_beta_scale * max(dt[t] - 1.0, 0.0))
        logtr = -np.abs(D - gps[t]) / beta - np.log(beta)
        bad = ~np.isfinite(D) | (D > cfg.vmax * max(dt[t], 1.0) + 2 * cfg.back_tol)
        logtr[bad] = -np.inf
        scores = delta[:, None] + logtr
        best_prev = np.argmax(scores, axis=0)
        new = scores[best_prev, np.arange(scores.shape[1])] + log_em[t]
        if not np.isfinite(new).any():          # chain broken: restart here
            breaks.append(t)
            backs[t] = None
            new = log_em[t].copy()
        else:
            backs[t] = best_prev
        delta = new - new[np.isfinite(new)].max()
        deltas[t] = delta

    path = np.empty(T, dtype=np.int64)
    path[-1] = int(np.argmax(deltas[-1]))
    for t in range(T - 1, 0, -1):
        path[t - 1] = backs[t][path[t]] if backs[t] is not None else int(np.argmax(deltas[t - 1]))
    return path, breaks


def decode(lat: TripLattice, log_em, D_list, gps, dt, cfg: HMMConfig):
    """Viterbi decode -> (predicted arc per observation, chosen row index, breaks)."""
    path, breaks = viterbi(log_em, D_list, gps, dt, cfg)
    arcs = np.array([lat.arcs[i][s] for i, s in enumerate(path)], dtype=np.int64)
    rows = np.array([lat.row_index[i][s] for i, s in enumerate(path)], dtype=np.int64)
    return arcs, rows, breaks


def classifier_only(lat: TripLattice, proba: np.ndarray | None) -> np.ndarray:
    """Independent per-observation argmax (nearest arc if no classifier)."""
    if proba is None:
        return np.array([a[np.argmin(d)] for a, d in zip(lat.arcs, lat.dists)], dtype=np.int64)
    return np.array([a[np.argmax(proba[r])] for a, r in zip(lat.arcs, lat.row_index)], dtype=np.int64)


def fill_route(net: RoadNetwork, arcs: np.ndarray, max_dist: float = 3000.0) -> list[int]:
    """Collapse per-point arcs to a connected arc sequence, inserting shortest-path arcs
    between consecutive distinct arcs that are not directly adjacent (for display)."""
    from scipy.sparse.csgraph import dijkstra
    seq = [int(a) for i, a in enumerate(arcs) if i == 0 or a != arcs[i - 1]]
    if not seq:
        return []
    out = [seq[0]]
    lookup = {(int(u), int(v)): i for i, (u, v) in enumerate(zip(net.arc_from, net.arc_to)) if net.valid[i]}
    for a, b in zip(seq[:-1], seq[1:]):
        if net.arc_to[a] != net.arc_from[b]:
            src, dst = int(net.arc_to[a]), int(net.arc_from[b])
            dist, pred = dijkstra(net.graph, directed=True, indices=src, limit=max_dist,
                                  return_predecessors=True)
            if np.isfinite(dist[dst]):
                nodes = [dst]
                while nodes[-1] != src:
                    nodes.append(int(pred[nodes[-1]]))
                nodes.reverse()
                out.extend(lookup[(u, v)] for u, v in zip(nodes[:-1], nodes[1:]) if (u, v) in lookup)
        out.append(b)
    return out
