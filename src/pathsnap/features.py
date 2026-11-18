"""Candidate generation + feature engineering.

Output is a long table with one row per (GPS observation, candidate arc). The ML task
is binary: is this candidate the arc the vehicle was really on? At inference the
per-observation probabilities become HMM emission scores.

Features deliberately avoid absolute coordinates / arc ids so the model generalises to
unseen trips (and cannot memorise a location).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config
from .geo import angle_diff, bearing_deg
from .labels import align_route
from .network import RoadNetwork

FEATURE_COLS = [
    "dist", "dist_rank", "dist_minus_min", "dist_ratio", "n_cands", "forced",
    "heading_diff", "heading_diff_undirected", "heading_reliable", "speed",
    "arc_length", "offset_frac", "dist_to_arc_end",
    "ctx_dist_mean", "ctx_dist_max", "ctx_dist_minus_min",
    "dt_prev", "dt_next", "out_deg_to", "in_deg_from",
    "heading_diff_long", "heading_long_reliable", "progress_short", "progress_long",
]


def observation_frame(trip, net: RoadNetwork, cfg: Config) -> pd.DataFrame:
    """Per-observation kinematics computed from the GPS track only."""
    tr = trip.track
    xy = net.proj.to_xy(tr[["lon", "lat"]].to_numpy())
    t = tr.t.to_numpy(dtype=float)
    T = len(tr)
    w = cfg.feat.heading_window
    i0 = np.clip(np.arange(T) - w, 0, T - 1)
    i1 = np.clip(np.arange(T) + w, 0, T - 1)
    d = xy[i1] - xy[i0]
    disp = np.hypot(d[:, 0], d[:, 1])
    span = np.maximum(t[i1] - t[i0], 1e-6)
    dt = np.diff(t)
    return pd.DataFrame({
        "obs_idx": np.arange(T), "raw_idx": tr.raw_idx.to_numpy(), "t": t,
        "lon": tr.lon.to_numpy(), "lat": tr.lat.to_numpy(), "x": xy[:, 0], "y": xy[:, 1],
        "heading": bearing_deg(d[:, 0], d[:, 1]),
        "heading_reliable": (disp >= cfg.feat.min_heading_disp).astype(np.int8),
        "speed": disp / span,
        "dt_prev": np.r_[0.0, dt], "dt_next": np.r_[dt, 0.0],
    })


def candidate_frame(obs: pd.DataFrame, net: RoadNetwork, cfg: Config) -> pd.DataFrame:
    P = obs[["x", "y"]].to_numpy()
    arcs, dists, offs, forced = net.candidates(P, cfg.cand.radius, cfg.cand.k, cfg.cand.fallback_k)
    n = np.array([len(a) for a in arcs])
    oi = np.repeat(obs.obs_idx.to_numpy(), n)
    c = pd.DataFrame({
        "obs_idx": oi,
        "cand_rank": np.concatenate([np.arange(m) for m in n]),
        "arc_id": np.concatenate(arcs), "dist": np.concatenate(dists),
        "offset": np.concatenate(offs),
    })
    c["n_cands"] = np.repeat(n, n)
    c["forced"] = np.repeat(forced, n).astype(np.int8)
    c["dist_rank"] = c.cand_rank
    dmin = c.groupby("obs_idx").dist.transform("min")
    c["dist_minus_min"] = c.dist - dmin
    c["dist_ratio"] = c.dist / (dmin + 1.0)

    a = c.arc_id.to_numpy()
    o = obs.set_index("obs_idx")
    hd = angle_diff(o.heading.to_numpy()[oi], net.bearing[a])
    c["heading_diff"] = hd
    c["heading_diff_undirected"] = np.minimum(hd, 180.0 - hd)
    for col in ("heading_reliable", "speed", "dt_prev", "dt_next"):
        c[col] = o[col].to_numpy()[oi]
    L = net.length[a]
    c["arc_length"] = L
    c["offset_frac"] = np.where(L > 0, c.offset / np.maximum(L, 1e-9), 0.0)
    c["dist_to_arc_end"] = np.minimum(c.offset, L - c.offset)
    c["out_deg_to"] = net.out_degree[net.arc_to[a]]
    c["in_deg_from"] = net.in_degree[net.arc_from[a]]

    # context: how close are the neighbouring GPS points to this same arc?
    T, w = len(obs), cfg.feat.context_window
    ctx = np.stack([net.project(P[np.clip(oi + k, 0, T - 1)], a)[0] for k in range(-w, w + 1)], 1)
    c["ctx_dist_mean"] = ctx.mean(1)
    c["ctx_dist_max"] = ctx.max(1)
    c["ctx_dist_minus_min"] = c.ctx_dist_mean - c.groupby("obs_idx").ctx_dist_mean.transform("min")

    # direction over a longer baseline. In slow / jittery stretches the short-window GPS
    # heading is noise, and the two directions of a two-way road (twin arcs with identical
    # geometry) become indistinguishable. Two remedies:
    #  * heading_diff_long: GPS heading from point i-W to i+W (W = long_window)
    #  * progress_*: signed distance travelled ALONG this candidate arc between an earlier
    #    and a later point (offset of the later minus offset of the earlier projection).
    #    Positive = moving with the arc's direction, negative = against it.
    W = cfg.feat.long_window
    lo, hi = np.clip(np.arange(T) - W, 0, T - 1), np.clip(np.arange(T) + W, 0, T - 1)
    dl = P[hi] - P[lo]
    disp_long = np.hypot(dl[:, 0], dl[:, 1])
    hd_long = angle_diff(bearing_deg(dl[:, 0], dl[:, 1])[oi], net.bearing[a])
    c["heading_diff_long"] = hd_long
    c["heading_long_reliable"] = (disp_long >= 2 * cfg.feat.min_heading_disp).astype(np.int8)[oi]
    for name, win in (("progress_short", w), ("progress_long", W)):
        fwd = net.project(P[np.clip(oi + win, 0, T - 1)], a)[1]
        back = net.project(P[np.clip(oi - win, 0, T - 1)], a)[1]
        c[name] = fwd - back
    return c


def build_trip_frame(trip, net: RoadNetwork, cfg: Config, with_labels: bool = False):
    """Returns (obs, cands). With labels, adds label columns using the reference route."""
    obs = observation_frame(trip, net, cfg)
    cands = candidate_frame(obs, net, cfg)
    obs.insert(0, "trip_id", trip.trip_id)
    cands.insert(0, "trip_id", trip.trip_id)
    if with_labels:
        if trip.route is None:
            raise ValueError(f"trip {trip.trip_id} has no route; cannot label")
        la, ld, pos = align_route(obs[["x", "y"]].to_numpy(), trip.route, net)
        obs["label_arc"], obs["label_dist"], obs["route_pos"] = la, ld, pos
        cands["label"] = (cands.arc_id.to_numpy() == la[cands.obs_idx.to_numpy()]).astype(np.int8)
        hit = cands.groupby("obs_idx").label.max()
        obs["label_in_cands"] = hit.reindex(obs.obs_idx).fillna(0).astype(np.int8).to_numpy()
        # data quality: GPS far from its reference arc => reference route (or GPS) is wrong there
        obs["label_reliable"] = (ld <= cfg.feat.label_max_dist).astype(np.int8)
        trip_ok = bool(np.median(ld) <= cfg.feat.trip_max_median_label_dist)
        obs["trip_label_ok"] = cands["trip_label_ok"] = np.int8(trip_ok)
        cands["label_reliable"] = obs.label_reliable.to_numpy()[cands.obs_idx.to_numpy()]
    return obs, cands
