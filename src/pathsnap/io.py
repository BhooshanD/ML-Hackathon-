"""Reading raw PathSnap trips (.track / .nodes / .arcs / .route) and partition discovery."""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

PARTITIONS = ("train", "student_test", "evaluator")


@dataclass
class Trip:
    trip_id: str
    track: pd.DataFrame            # cleaned: columns lon, lat, t, raw_idx (sorted by t)
    nodes: np.ndarray              # (N, 2) lon, lat
    arcs: np.ndarray               # (M, 2) from_node, to_node
    route: np.ndarray | None       # ground-truth arc ids (None if not supplied)
    partition: str | None = None
    qc: dict = field(default_factory=dict)


def _read_table(path: Path, ncols: int, dtype) -> np.ndarray:
    if not path.exists() or path.stat().st_size == 0:
        return np.empty((0, ncols), dtype=dtype)
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    if df.shape[1] < ncols:
        raise ValueError(f"{path}: expected >= {ncols} columns, got {df.shape[1]}")
    return df.iloc[:, :ncols].to_numpy(dtype=dtype)


def clean_track(raw: np.ndarray) -> tuple[pd.DataFrame, dict]:
    """Drop invalid rows, sort by time, remove duplicate timestamps. Returns (track, qc)."""
    df = pd.DataFrame(raw, columns=["lon", "lat", "t"])
    df["raw_idx"] = np.arange(len(df))
    qc = {"n_raw": len(df)}

    finite = np.isfinite(df[["lon", "lat", "t"]]).all(axis=1)
    in_range = df.lon.between(-180, 180) & df.lat.between(-90, 90)
    null_island = (df.lon.abs() < 1e-9) & (df.lat.abs() < 1e-9)
    ok = finite & in_range & ~null_island
    qc["n_invalid"] = int((~ok).sum())
    df = df[ok]

    qc["n_out_of_order"] = int((np.diff(df.t.to_numpy()) < 0).sum())
    df = df.sort_values("t", kind="stable")
    dup = df.t.duplicated(keep="first")
    qc["n_duplicate_t"] = int(dup.sum())
    df = df[~dup].reset_index(drop=True)

    dt = np.diff(df.t.to_numpy())
    qc["n_clean"] = len(df)
    qc["dt_median"] = float(np.median(dt)) if len(dt) else float("nan")
    qc["dt_max"] = float(dt.max()) if len(dt) else float("nan")
    qc["frac_dt_1s"] = float(np.mean(np.isclose(dt, 1.0))) if len(dt) else float("nan")
    return df, qc


def read_trip(prefix: str | Path, trip_id: str | None = None, partition: str | None = None,
              load_route: bool = True) -> Trip:
    """`prefix` is the path without extension, e.g. data/train/00000001/00000001."""
    prefix = Path(prefix)
    trip_id = trip_id or prefix.name
    track, qc = clean_track(_read_table(prefix.with_suffix(".track"), 3, float))
    nodes = _read_table(prefix.with_suffix(".nodes"), 2, float)
    arcs = _read_table(prefix.with_suffix(".arcs"), 2, np.int64)
    route = None
    route_path = prefix.with_suffix(".route")
    if load_route and route_path.exists():
        route = _read_table(route_path, 1, np.int64).ravel()
    qc.update(n_nodes=len(nodes), n_arcs=len(arcs),
              n_route=None if route is None else len(route))
    return Trip(trip_id, track, nodes, arcs, route, partition, qc)


def _infer_partition(path: Path, root: Path) -> str | None:
    for part in reversed(path.relative_to(root).parts[:-1]):
        p = part.lower()
        if "eval" in p:
            return "evaluator"
        if "student" in p or p in ("test", "student_test", "studenttest"):
            return "student_test"
        if "train" in p:
            return "train"
    return None


def discover_trips(root: str | Path, splits_file: str | Path | None = None) -> pd.DataFrame:
    """Find every *.track under `root`. Returns DataFrame[trip_id, prefix, partition].

    Partition comes from `splits_file` (JSON {"train": [...], "student_test": [...],
    "evaluator": [...]}) if given, otherwise from folder names (train / student / eval).
    """
    root = Path(root)
    rows = []
    for track in sorted(root.rglob("*.track")):
        rows.append({"trip_id": track.stem, "prefix": str(track.with_suffix("")),
                     "partition": _infer_partition(track, root)})
    df = pd.DataFrame(rows, columns=["trip_id", "prefix", "partition"])
    if df.trip_id.duplicated().any():
        raise ValueError("duplicate trip ids found under data root")
    if splits_file:
        splits = json.loads(Path(splits_file).read_text())
        lookup = {str(t): p for p, ids in splits.items() for t in ids}
        df["partition"] = df.trip_id.map(lookup)
    if df.partition.isna().any():
        warnings.warn(f"{int(df.partition.isna().sum())} trips have no partition; "
                      "pass a splits file")
    return df


def load_partition(root: str | Path, partition: str, splits_file=None,
                   load_route: bool = True) -> list[Trip]:
    idx = discover_trips(root, splits_file)
    idx = idx[idx.partition == partition]
    return [read_trip(r.prefix, r.trip_id, partition, load_route) for r in idx.itertuples()]
