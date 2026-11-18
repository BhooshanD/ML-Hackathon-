"""Coordinate handling: lon/lat (degrees) -> local metric UTM coordinates.

Euclidean distance in degrees is wrong (1 deg lon shrinks with cos(lat)), so every
trip is projected into the UTM zone of its own centroid. Within a city-sized area UTM
distortion is < 0.1 %, far below GPS noise.
"""
from __future__ import annotations

import numpy as np
from pyproj import CRS, Transformer


class LocalProjector:
    def __init__(self, lon0: float, lat0: float):
        zone = int((lon0 + 180.0) // 6.0) % 60 + 1
        epsg = (32700 if lat0 < 0 else 32600) + zone
        self.crs = CRS.from_epsg(epsg)
        self._fwd = Transformer.from_crs("EPSG:4326", self.crs, always_xy=True)
        self._inv = Transformer.from_crs(self.crs, "EPSG:4326", always_xy=True)

    @classmethod
    def for_points(cls, lonlat: np.ndarray) -> "LocalProjector":
        lon0, lat0 = np.nanmedian(lonlat, axis=0)
        return cls(float(lon0), float(lat0))

    def to_xy(self, lonlat: np.ndarray) -> np.ndarray:
        x, y = self._fwd.transform(lonlat[:, 0], lonlat[:, 1])
        return np.column_stack([x, y])

    def to_lonlat(self, xy: np.ndarray) -> np.ndarray:
        lon, lat = self._inv.transform(xy[:, 0], xy[:, 1])
        return np.column_stack([lon, lat])


def bearing_deg(dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """Compass bearing (0 = north, clockwise) of a displacement in projected metres."""
    return np.degrees(np.arctan2(dx, dy)) % 360.0


def angle_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Absolute difference between two bearings, in [0, 180]."""
    d = np.abs(a - b) % 360.0
    return np.minimum(d, 360.0 - d)
