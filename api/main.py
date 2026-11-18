import logging, time
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from dummy_pipeline import list_trip_ids, predict_trip   # swap import later

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

@app.get("/trips")
def trips():
    return {"trips": list_trip_ids()}

@app.get("/predict/{trip_id}")
def predict(trip_id: str):
    if trip_id not in list_trip_ids():
        raise HTTPException(404, "Unknown trip")
    return predict_trip(trip_id)