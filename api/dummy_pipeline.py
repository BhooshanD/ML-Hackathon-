# TEMPORARY PLACEHOLDER - replace with the real predict_trip before evaluation.
# It returns synthetic data only, flagged with "placeholder": True so the UI
# shows a warning banner. Never use these numbers or screenshots in the report.
import math
import random


def list_trip_ids():
    return [f"trip_{i}" for i in range(1, 11)]


def predict_trip(trip_id: str):
    rng = random.Random(trip_id)          # same trip -> same fake path
    lat0, lon0 = 12.9716, 77.5946
    phase = rng.uniform(0, 3)
    n = 90

    route, gps = [], []
    for i in range(n):
        t = i / (n - 1)
        lat = lat0 + 0.012 * t + 0.002 * math.sin(t * 7 + phase)
        lon = lon0 + 0.020 * t + 0.003 * math.cos(t * 5 + phase)
        route.append([lon, lat])                                   # GeoJSON order
        gps.append([lat + rng.gauss(0, 0.00025),                   # noisy GPS
                    lon + rng.gauss(0, 0.00025)])

    return {
        "trip_id": trip_id,
        "gps": gps,
        "predicted_route": {"type": "LineString", "coordinates": route},
        "accuracy": None,        # N/A until the real pipeline computes it
        "placeholder": True,
    }
