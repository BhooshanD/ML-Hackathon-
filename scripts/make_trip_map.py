"""Regenerate a report map (GPS + reference route + predicted route) with the CURRENT model.

    python scripts/make_trip_map.py 00000000            # one student-test trip
    python scripts/make_trip_map.py 00000000 00000065   # several

Writes artifacts/maps/report_ready_trip_<id>.html, artifacts/prediction_<id>.json
(prediction + metrics, no reference route) and artifacts/predicted_coordinates_<id>.json
(the [[lat, lon], [lat, lon]] segments the EDA notebook reads). Run again after the model
or decoder changes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pathsnap.io import discover_trips
from pathsnap.pipeline import PathSnapPipeline, predict_trip
from pathsnap.viz import create_trip_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trip_ids", nargs="+")
    ap.add_argument("--data-root", default=".")
    ap.add_argument("--artifacts", default="artifacts")
    args = ap.parse_args()

    pipe = PathSnapPipeline.from_artifacts(args.artifacts)
    idx = discover_trips(args.data_root).set_index("trip_id")
    art = Path(args.artifacts)
    maps = art / "maps"
    maps.mkdir(parents=True, exist_ok=True)
    for tid in args.trip_ids:
        r = predict_trip(tid, args.data_root, pipeline=pipe)
        lines = [[[lat, lon] for lon, lat in seg]
                 for seg in r["route_geojson"]["geometry"]["coordinates"]]
        acc = r["metrics"]["hmm"]["top1_accuracy"]
        m = create_trip_map(
            Path(idx.loc[tid, "prefix"]).parent, predicted_lines=lines,
            title=f"PathSnap — Predicted vs Reference Route (Trip {tid}, {idx.loc[tid, 'partition']}, "
                  f"{r['model']} + HMM, Top-1 {acc:.1%})")
        m.save(str(maps / f"report_ready_trip_{tid}.html"))
        (art / f"prediction_{tid}.json").write_text(json.dumps(r))
        (art / f"predicted_coordinates_{tid}.json").write_text(json.dumps(lines))
        print(f"{tid}: model={r['model']} top1={acc:.4f} -> {maps / f'report_ready_trip_{tid}.html'}")


if __name__ == "__main__":
    main()
