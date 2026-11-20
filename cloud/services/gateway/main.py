"""gateway: REST API + live dashboard for the cloud side (FastAPI, port 8080).

    GET  /                         dashboard
    GET  /api/health
    GET  /api/latest               newest message of each kind (climate, soil, irrigation, diagnosis, ...)
    GET  /api/history?kind=climate&minutes=120
    GET  /api/diagnoses?limit=20   edge + cloud diagnoses, newest first, with the photo URL
    GET  /api/alerts?limit=20
    GET  /api/storage              where data is kept + row counts (for the "How it works" card)
    POST /api/images               edge upload: form fields image (file), meta (JSON string, needs image_id)
    GET  /images/<image_id>.jpg
    GET  /api/stream               Server-Sent Events: every plant/# message, live
"""
import asyncio
import json
import logging
import re
import os
import threading
import urllib.request
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from common import db, logs
from common.mqtt import Bus

logs.setup()
log = logging.getLogger("gateway")
IMAGES = Path("/data/images")
PI_URL = os.getenv("PI_URL", "http://10.93.8.209:8000").rstrip("/")   # edge web app (live preview)
STATIC = Path(__file__).parent / "static"
app = FastAPI(title="Plant farm gateway")
bus = Bus("gateway")
listeners = set()            # asyncio queues of connected SSE clients
loop = None
lock = threading.Lock()


def on_any(topic, data):
    if loop is None:
        return
    msg = json.dumps({"topic": topic, "data": data}, default=str)
    with lock:
        for q in list(listeners):
            loop.call_soon_threadsafe(q.put_nowait, msg)


@app.on_event("startup")
async def startup():
    global loop
    loop = asyncio.get_running_loop()
    IMAGES.mkdir(parents=True, exist_ok=True)
    await asyncio.to_thread(db.pool)
    bus.subscribe("plant/#", on_any, qos=0)
    await asyncio.to_thread(bus.start)


@app.get("/api/health")
def health():
    db.query("SELECT 1")
    return {"ok": True, "mqtt": bus.client.is_connected()}


@app.get("/api/latest")
def latest():
    rows = db.query("""SELECT DISTINCT ON (kind) kind, ts, topic, device, data FROM telemetry
                       ORDER BY kind, ts DESC""")
    return {r["kind"]: {"ts": r["ts"], "topic": r["topic"], "device": r["device"], **r["data"]} for r in rows}


@app.get("/api/history")
def history(kind: str = "climate", minutes: int = 120, limit: int = 2000):
    rows = db.query("""SELECT ts, device, data FROM telemetry WHERE kind = %s AND ts > now() - make_interval(mins => %s)
                       ORDER BY ts LIMIT %s""", (kind, minutes, limit))
    return [{"ts": r["ts"], "device": r["device"], **r["data"]} for r in rows]


@app.get("/api/diagnoses")
def diagnoses(limit: int = 20):
    rows = db.query("""SELECT d.ts, d.source, d.device, d.image_id, d.class_id, d.crop, d.status, d.confidence, d.data,
                              (i.image_id IS NOT NULL) AS has_image
                       FROM diagnoses d LEFT JOIN images i USING (image_id)
                       ORDER BY d.ts DESC LIMIT %s""", (limit,))
    for r in rows:
        r["image_url"] = f"/images/{r['image_id']}.jpg" if r.pop("has_image") else None
    return rows


@app.get("/api/alerts")
def alerts(limit: int = 20):
    return db.query("SELECT ts, level, kind, message, data FROM alerts ORDER BY ts DESC LIMIT %s", (limit,))


@app.get("/api/storage")
def storage():
    """Row counts and size of what the cloud has stored, for the dashboard's "How it works" card."""
    tables = ["telemetry", "diagnoses", "advice", "alerts", "irrigation_events", "images"]
    counts = {t: db.query(f"SELECT count(*) AS n FROM {t}")[0]["n"] for t in tables}   # fixed names, no user input
    by_kind = {r["kind"]: r["n"] for r in db.query("SELECT kind, count(*) AS n FROM telemetry GROUP BY kind")}
    size = db.query("SELECT pg_size_pretty(pg_database_size(current_database())) AS s")[0]["s"]
    photos = list(IMAGES.glob("*.jpg"))
    return {"postgres": {"tables": counts, "telemetry_by_kind": by_kind, "size": size},
            "photos": {"count": len(photos), "mb": round(sum(f.stat().st_size for f in photos) / 1e6, 1)}}


@app.post("/api/images")
async def upload(image: UploadFile = File(...), meta: str = Form("{}")):
    try:
        m = json.loads(meta)
    except json.JSONDecodeError:
        raise HTTPException(400, "meta must be JSON")
    image_id = str(m.get("image_id", ""))
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", image_id) or ".." in image_id:
        raise HTTPException(400, "meta.image_id missing or invalid")
    body = await image.read()
    if len(body) > 8_000_000:
        raise HTTPException(413, "image too large")
    path = IMAGES / f"{image_id}.jpg"
    path.write_bytes(body)
    db.execute("""INSERT INTO images (image_id, device, path, meta) VALUES (%s, %s, %s, %s)
                  ON CONFLICT (image_id) DO UPDATE SET meta = EXCLUDED.meta""",
               (image_id, m.get("device"), str(path), db.jsonb(m)))
    # tells cloud-inference (and the dashboard) that a new photo is ready
    bus.publish("plant/image", {"image_id": image_id, "url": f"/images/{image_id}.jpg", "device": m.get("device"),
                                "edge": m.get("edge")})
    return {"ok": True, "image_id": image_id}


@app.get("/api/camera/preview.jpg")
def camera_preview():
    """Live view: one frame from the Pi camera (proxied from the Pi app, not saved, no prediction)."""
    try:
        with urllib.request.urlopen(f"{PI_URL}/api/preview.jpg", timeout=10) as r:
            return Response(r.read(), media_type="image/jpeg", headers={"Cache-Control": "no-store"})
    except Exception as e:
        raise HTTPException(503, f"Pi camera not reachable: {e}")


@app.post("/api/camera/capture")
def camera_capture():
    """Take photo: MQTT plant/capture -> bridge -> Pi takes a photo and runs the model.
    The result comes back as plant/diagnosis + the photo upload (live on the dashboard)."""
    bus.publish("plant/capture", {"device": "cloud-dashboard", "time": __import__("time").time()})
    return {"ok": True, "sent": "plant/capture"}


@app.post("/api/water")
def water(amount_pct: float = 35):
    """Dashboard "I watered it" button (and later a real pump relay): tells soil-sim / the pump node."""
    amount_pct = max(5.0, min(60.0, amount_pct))
    bus.publish("plant/irrigation/cmd", {"action": "water", "amount_pct": amount_pct, "by": "dashboard"})
    return {"ok": True, "amount_pct": amount_pct}


@app.get("/images/{name}")
def get_image(name: str):
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}\.jpg", name) or ".." in name:
        raise HTTPException(404)
    path = IMAGES / name
    if not path.exists():
        raise HTTPException(404)
    return FileResponse(path, media_type="image/jpeg")


@app.get("/api/stream")
async def stream(request: Request):
    q = asyncio.Queue(maxsize=500)
    with lock:
        listeners.add(q)

    async def gen():
        try:
            yield "retry: 3000\n\n"
            while not await request.is_disconnected():
                try:
                    msg = await asyncio.wait_for(q.get(), timeout=15)
                    yield f"data: {msg}\n\n"
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
        finally:
            with lock:
                listeners.discard(q)

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
