import logging, time
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from real_pipeline import list_trip_ids, partition, predict_trip

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("pathsnap")

app = FastAPI(title="PathSnap API")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    log.info("%s %s -> %s (%.0f ms) from %s", request.method, request.url.path,
             response.status_code, (time.time() - start) * 1000, request.client.host)
    return response

@app.on_event("startup")
def warm_up():
    ids = list_trip_ids()          # fails fast if data or model are missing
    log.info("serving %d '%s' trips: %s", len(ids), partition(), ", ".join(ids))

@app.get("/trips")
def trips():
    return {"trips": list_trip_ids(), "partition": partition()}

@app.get("/predict/{trip_id}")
def predict(trip_id: str):
    if trip_id not in list_trip_ids():
        raise HTTPException(404, "Unknown trip")
    result = predict_trip(trip_id)
    log.info("trip %s: %d GPS points, top-1 %.4f, %.2f s", trip_id, len(result["gps"]),
             result["accuracy"] if result["accuracy"] is not None else float("nan"),
             result["metrics"]["inference_seconds"])
    return result
