"""Folium trip maps: raw GPS, reference route, sampled road network, predicted route.

`create_trip_map` is the function developed in experiments/eda/01_dataset_exploration.ipynb
(same layers, colours and controls), moved here so maps can be regenerated from the
command line whenever the model changes (scripts/make_trip_map.py).

Maps that draw the reference route are for the report only. The evaluator UI never
receives reference data.
"""
from __future__ import annotations

from pathlib import Path

import folium
import numpy as np
import pandas as pd

ESRI = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}"
COLORS = {"gps": "blue", "reference": "orange", "predicted": "purple", "road": "green"}


def _read(path: Path, names):
    return pd.read_csv(path, sep=r"\s+", header=None, names=names)


def create_trip_map(trip_dir: str | Path, predicted_lines=None, max_road_segments: int = 5000,
                    title: str | None = None, legend: bool = True) -> folium.Map:
    """`predicted_lines`: list of polylines, each a list of (lat, lon) pairs."""
    trip_dir = Path(trip_dir)
    tid = trip_dir.name
    track = _read(trip_dir / f"{tid}.track", ["longitude", "latitude", "timestamp"])
    nodes = _read(trip_dir / f"{tid}.nodes", ["longitude", "latitude"])
    arcs = _read(trip_dir / f"{tid}.arcs", ["source", "target"])
    lat, lon = nodes.latitude.to_numpy(), nodes.longitude.to_numpy()
    gps = list(zip(track.latitude, track.longitude))

    m = folium.Map(location=[track.latitude.mean(), track.longitude.mean()], zoom_start=11,
                   tiles=None, control_scale=True)
    folium.TileLayer(tiles=ESRI, attr="Tiles © Esri", name="Esri Street Map").add_to(m)

    # road network: sample of the arcs inside the GPS bounding box (hidden by default)
    road = folium.FeatureGroup(name="Road Network (sampled)", show=False)
    mg = 0.01
    inside = (np.abs(lat - (track.latitude.min() + track.latitude.max()) / 2)
              <= np.ptp(track.latitude) / 2 + mg) & \
             (np.abs(lon - (track.longitude.min() + track.longitude.max()) / 2)
              <= np.ptp(track.longitude) / 2 + mg)
    near = arcs[inside[arcs.source.to_numpy()] & inside[arcs.target.to_numpy()]]
    near = near.sample(n=min(max_road_segments, len(near)), random_state=42)
    u, v = near.source.to_numpy(), near.target.to_numpy()
    folium.PolyLine([[[lat[a], lon[a]], [lat[b], lon[b]]] for a, b in zip(u, v)],
                    color=COLORS["road"], weight=1, opacity=0.4).add_to(road)
    road.add_to(m)

    g = folium.FeatureGroup(name="Raw GPS Trajectory")
    folium.PolyLine(gps, color=COLORS["gps"], weight=3, opacity=0.8,
                    tooltip="Raw GPS").add_to(g)
    g.add_to(m)

    route_file = trip_dir / f"{tid}.route"
    if route_file.exists():
        r = _read(route_file, ["arc_id"]).arc_id.to_numpy(dtype=int)
        r = r[(r >= 0) & (r < len(arcs))]
        src, dst = arcs.source.to_numpy()[r], arcs.target.to_numpy()[r]
        # one polyline per contiguous run (a few routes have disconnected steps)
        lines, cur = [], [[lat[src[0]], lon[src[0]]]]
        for i in range(len(r)):
            if i and src[i] != dst[i - 1]:
                lines.append(cur)
                cur = [[lat[src[i]], lon[src[i]]]]
            cur.append([lat[dst[i]], lon[dst[i]]])
        lines.append(cur)
        ref = folium.FeatureGroup(name="Reference Route")
        folium.PolyLine(lines, color=COLORS["reference"], weight=7, opacity=0.85,
                        tooltip="Reference Route").add_to(ref)
        ref.add_to(m)

    if predicted_lines is not None:
        pred = folium.FeatureGroup(name="Predicted Route")
        folium.PolyLine(predicted_lines, color=COLORS["predicted"], weight=3, opacity=0.95,
                        tooltip="Predicted Route").add_to(pred)
        pred.add_to(m)

    folium.Marker(gps[0], tooltip="Trip Start", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker(gps[-1], tooltip="Trip End", icon=folium.Icon(color="red")).add_to(m)

    box = ("position: fixed; z-index: 9999; background: white; color: #222; padding: 10px 14px; "
           "border: 1px solid grey; border-radius: 5px;")
    if title:
        m.get_root().html.add_child(folium.Element(
            f'<div style="{box} top: 10px; left: 50px; font-size: 16px; font-weight: bold;">{title}</div>'))
    if legend:
        rows = [("gps", "Raw GPS Trajectory"), ("reference", "Reference Route")]
        if predicted_lines is not None:
            rows.append(("predicted", "Predicted Route"))
        rows.append(("road", "Road Network (sampled)"))
        body = "<br>".join(f'<span style="color:{COLORS[k]};">━━</span> {name}' for k, name in rows)
        m.get_root().html.add_child(folium.Element(
            f'<div style="{box} bottom: 35px; left: 35px; font-size: 13px;"><b>Map Legend</b><br>{body}</div>'))
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[track.latitude.min(), track.longitude.min()],
                  [track.latitude.max(), track.longitude.max()]])
    return m
