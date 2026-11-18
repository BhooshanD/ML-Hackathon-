"""Tests on a tiny synthetic road network with known ground truth.

    .venv/bin/python -m pytest tests -q
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pathsnap.config import Config
from pathsnap.features import build_trip_frame
from pathsnap.geo import LocalProjector
from pathsnap.hmm import TripLattice, decode, emission_logprob, precompute_transitions
from pathsnap.io import Trip
from pathsnap.labels import align_route
from pathsnap.network import RoadNetwork

LON0, LAT0 = 8.6, 49.85
PROJ = LocalProjector(LON0, LAT0)


def grid_trip(seed=0, noise=6.0, spacing=100.0, n=6):
    """n x n grid of two-way streets; vehicle drives an L-shaped route."""
    rng = np.random.default_rng(seed)
    origin = PROJ.to_xy(np.array([[LON0, LAT0]]))[0]
    xy = np.array([[origin[0] + i * spacing, origin[1] + j * spacing]
                   for j in range(n) for i in range(n)])
    nid = lambda i, j: j * n + i
    arcs = []
    for j in range(n):
        for i in range(n):
            if i + 1 < n:
                arcs += [(nid(i, j), nid(i + 1, j)), (nid(i + 1, j), nid(i, j))]
            if j + 1 < n:
                arcs += [(nid(i, j), nid(i, j + 1)), (nid(i, j + 1), nid(i, j))]
    arcs = np.array(arcs)
    lookup = {tuple(a): k for k, a in enumerate(arcs)}
    path_nodes = [nid(i, 1) for i in range(5)] + [nid(4, j) for j in range(2, 6)]
    route = np.array([lookup[(a, b)] for a, b in zip(path_nodes[:-1], path_nodes[1:])])

    pts, true_arc = [], []
    for r in route:
        A, B = xy[arcs[r, 0]], xy[arcs[r, 1]]
        for s in np.arange(0, 1, 10.0 / spacing):   # 10 m/s at 1 Hz
            pts.append(A + s * (B - A))
            true_arc.append(r)
    pts = np.array(pts) + rng.normal(0, noise, (len(pts), 2))
    ll = PROJ.to_lonlat(pts)
    track = pd.DataFrame({"lon": ll[:, 0], "lat": ll[:, 1], "t": np.arange(len(pts), dtype=float),
                          "raw_idx": np.arange(len(pts))})
    trip = Trip("synthetic", track, PROJ.to_lonlat(xy), arcs, route)
    return trip, np.array(true_arc)


def test_projection_distance_is_metric():
    a = PROJ.to_xy(np.array([[LON0, LAT0], [LON0, LAT0 + 0.001]]))
    assert abs(np.hypot(*(a[1] - a[0])) - 111.2) < 1.0   # 0.001 deg lat ~ 111 m


def test_label_alignment_recovers_truth():
    trip, true_arc = grid_trip(noise=3.0)
    net = RoadNetwork(trip.nodes, trip.arcs, PROJ)
    xy = PROJ.to_xy(trip.track[["lon", "lat"]].to_numpy())
    label, dist, pos = align_route(xy, trip.route, net)
    assert np.all(np.diff(pos) >= 0)                  # monotone along the route
    assert np.mean(label == true_arc) > 0.95


def test_candidates_contain_truth_and_hmm_beats_nearest():
    trip, true_arc = grid_trip(noise=8.0, seed=1)
    cfg = Config()
    net = RoadNetwork(trip.nodes, trip.arcs, PROJ)
    obs, cands = build_trip_frame(trip, net, cfg, with_labels=True)
    assert obs.label_in_cands.mean() > 0.99
    lat = TripLattice.from_frames(obs, cands)
    D, gps, dt = precompute_transitions(net, lat, cfg.hmm)
    arcs, _, breaks = decode(lat, emission_logprob(lat, None, cfg.hmm), D, gps, dt, cfg.hmm)
    nearest = np.array([a[np.argmin(d)] for a, d in zip(lat.arcs, lat.dists)])
    hmm_acc, near_acc = np.mean(arcs == true_arc), np.mean(nearest == true_arc)
    assert hmm_acc > 0.9 and hmm_acc > near_acc        # twin arcs make nearest ~50 %
    assert not breaks


def test_prediction_does_not_read_route():
    from pathsnap.pipeline import PathSnapPipeline
    trip, _ = grid_trip()
    trip.route = None                                  # predict must work without GT
    out = PathSnapPipeline().predict(trip)
    assert len(out["pred_arcs"]) == len(trip.track)
    assert out["route_geojson"]["geometry"]["coordinates"]
