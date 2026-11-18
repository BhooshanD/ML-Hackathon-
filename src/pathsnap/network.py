"""Road network: arc geometry, spatial candidate search and shortest paths."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from .geo import LocalProjector, bearing_deg


class RoadNetwork:
    """Arcs are straight segments from node `from` to node `to` (arc_id = row index)."""

    def __init__(self, node_lonlat: np.ndarray, arcs: np.ndarray, projector: LocalProjector,
                 undirected: bool = False, sample_step: float = 10.0,
                 cache_limit: float = 1500.0, cache_bytes: float = 100e6):
        self.proj = projector
        self.undirected = undirected
        self.node_xy = projector.to_xy(node_lonlat) if len(node_lonlat) else np.empty((0, 2))
        self.n_nodes = len(self.node_xy)
        self.n_arcs = len(arcs)
        self.arc_from = arcs[:, 0].astype(np.int64)
        self.arc_to = arcs[:, 1].astype(np.int64)

        n = self.n_nodes
        self.valid = ((self.arc_from >= 0) & (self.arc_from < n) &
                      (self.arc_to >= 0) & (self.arc_to < n))
        f = np.where(self.valid, self.arc_from, 0)
        t = np.where(self.valid, self.arc_to, 0)
        self.A = self.node_xy[f] if n else np.zeros((self.n_arcs, 2))
        self.B = self.node_xy[t] if n else np.zeros((self.n_arcs, 2))
        self.V = self.B - self.A
        self.length = np.hypot(self.V[:, 0], self.V[:, 1])
        self.bearing = bearing_deg(self.V[:, 0], self.V[:, 1])

        self._build_graph()
        self._build_tree(sample_step)
        self.cache_limit = cache_limit
        self._cache: dict[int, np.ndarray] = {}
        self._cache_max = max(64, int(cache_bytes / (4 * max(n, 1))))

    # ------------------------------------------------------------------ graph
    def _build_graph(self):
        ids = np.flatnonzero(self.valid)
        u, v = self.arc_from[ids], self.arc_to[ids]
        w = np.maximum(self.length[ids], 1e-3)  # zero weight == "no edge" in scipy sparse
        if self.undirected:
            u, v, w = np.concatenate([u, v]), np.concatenate([v, u]), np.concatenate([w, w])
        e = pd.DataFrame({"u": u, "v": v, "w": w}).groupby(["u", "v"], as_index=False).w.min()
        self.graph = csr_matrix((e.w, (e.u, e.v)), shape=(self.n_nodes, self.n_nodes))
        self.out_degree = np.bincount(u, minlength=self.n_nodes)
        self.in_degree = np.bincount(v, minlength=self.n_nodes)

    def sp_rows(self, sources: np.ndarray, limit: float) -> np.ndarray:
        """Shortest-path distances from each source node to all nodes, shape (len, N)."""
        sources = np.asarray(sources, dtype=np.int64)
        if limit > self.cache_limit:
            return dijkstra(self.graph, directed=True, indices=sources, limit=limit)
        missing = [s for s in np.unique(sources) if s not in self._cache]
        if missing:
            if len(self._cache) + len(missing) > self._cache_max:
                self._cache.clear()   # evict everything, then recompute all requested rows
                missing = list(np.unique(sources))
            rows = dijkstra(self.graph, directed=True, indices=missing, limit=self.cache_limit)
            for s, row in zip(missing, np.atleast_2d(rows)):
                self._cache[int(s)] = row.astype(np.float32)
        return np.stack([self._cache[int(s)] for s in sources])

    # --------------------------------------------------------------- geometry
    def project(self, P: np.ndarray, arc_ids: np.ndarray):
        """Project points P (n,2) onto arcs (n,). Returns dist, offset along arc, foot point."""
        A, V = self.A[arc_ids], self.V[arc_ids]
        L2 = np.einsum("ij,ij->i", V, V)
        with np.errstate(invalid="ignore", divide="ignore"):
            s = np.where(L2 > 0, np.einsum("ij,ij->i", P - A, V) / L2, 0.0)
        s = np.clip(s, 0.0, 1.0)
        foot = A + s[:, None] * V
        dist = np.hypot(*(P - foot).T)
        return dist, s * self.length[arc_ids], foot

    def _build_tree(self, step: float):
        self.sample_step = step
        ids = np.flatnonzero(self.valid)
        n_samples = np.maximum(1, np.ceil(self.length[ids] / step).astype(int)) + 1
        arc_of = np.repeat(ids, n_samples)
        frac = np.concatenate([np.linspace(0, 1, m) for m in n_samples]) if len(ids) else np.empty(0)
        pts = self.A[arc_of] + frac[:, None] * self.V[arc_of]
        self._sample_arc = arc_of
        self._tree = cKDTree(pts) if len(pts) else None

    def candidates(self, P: np.ndarray, radius: float, k: int, fallback_k: int = 3):
        """Candidate arcs per point, nearest first.

        Correctness of the prefilter: every point on an arc lies within step/2 of some
        sample, so an arc at true distance <= radius always has a sample within
        radius + step/2.
        Returns lists (arc_ids, dists, offsets) and a bool array `forced` (fallback used).
        """
        if self._tree is None:
            raise ValueError("road network has no valid arcs")
        hits = self._tree.query_ball_point(P, r=radius + self.sample_step / 2)
        out_arcs, out_dist, out_off = [], [], []
        forced = np.zeros(len(P), dtype=bool)
        for i, h in enumerate(hits):
            arcs = np.unique(self._sample_arc[h]) if h else np.empty(0, dtype=np.int64)
            if len(arcs):
                d, off, _ = self.project(np.repeat(P[i:i + 1], len(arcs), 0), arcs)
                keep = d <= radius
                arcs, d, off = arcs[keep], d[keep], off[keep]
            if not len(arcs):  # nothing in radius: take nearest arcs regardless
                forced[i] = True
                _, nn = self._tree.query(P[i], k=min(len(self._sample_arc), 50 * fallback_k))
                arcs = np.unique(self._sample_arc[np.atleast_1d(nn)])
                d, off, _ = self.project(np.repeat(P[i:i + 1], len(arcs), 0), arcs)
                order = np.argsort(d, kind="stable")[:fallback_k]
                arcs, d, off = arcs[order], d[order], off[order]
            order = np.argsort(d, kind="stable")[:k]
            out_arcs.append(arcs[order])
            out_dist.append(d[order])
            out_off.append(off[order])
        return out_arcs, out_dist, out_off, forced

    # ------------------------------------------------------------- utilities
    def arcs_adjacent(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """True where arc b can directly follow arc a (or is the same arc)."""
        same = a == b
        if self.undirected:
            ends_a = np.stack([self.arc_from[a], self.arc_to[a]], 1)
            ends_b = np.stack([self.arc_from[b], self.arc_to[b]], 1)
            share = (ends_a[:, :, None] == ends_b[:, None, :]).any(axis=(1, 2))
            return same | share
        return same | (self.arc_to[a] == self.arc_from[b])

    def arc_lonlat(self, arc_id: int) -> list[list[float]]:
        ll = self.proj.to_lonlat(np.stack([self.A[arc_id], self.B[arc_id]]))
        return ll.tolist()

    def structure_report(self) -> dict:
        """Quick facts used to decide `undirected` and to sanity-check the network."""
        from scipy.sparse.csgraph import connected_components
        pairs = set(zip(self.arc_from[self.valid].tolist(), self.arc_to[self.valid].tolist()))
        twin = np.mean([(v, u) in pairs for u, v in pairs]) if pairs else float("nan")
        n_weak, _ = connected_components(self.graph, directed=True, connection="weak")
        n_strong, _ = connected_components(self.graph, directed=True, connection="strong")
        isolated = int(np.sum((self.out_degree + self.in_degree) == 0))
        return {"n_nodes": self.n_nodes, "n_arcs": self.n_arcs,
                "n_invalid_arcs": int((~self.valid).sum()),
                "frac_arcs_with_reverse_twin": float(twin),
                "n_weak_components": int(n_weak), "n_strong_components": int(n_strong),
                "n_isolated_nodes": isolated,
                "arc_length_median_m": float(np.median(self.length[self.valid]))
                if self.valid.any() else float("nan")}


def build_network(trip, undirected: bool = False, sample_step: float = 10.0,
                  cache_limit: float = 1500.0) -> RoadNetwork:
    lonlat = trip.track[["lon", "lat"]].to_numpy()
    proj = LocalProjector.for_points(lonlat if len(lonlat) else trip.nodes)
    return RoadNetwork(trip.nodes, trip.arcs, proj, undirected, sample_step, cache_limit)
