"""Offline plant-disease demo app (runs on the Raspberry Pi; also runs on a laptop for testing).

    python app/server.py                         # uses config.yaml
    python app/server.py --model models/x.tflite --no-mqtt --camera none

Open http://<pi-ip>:8000 on any phone/laptop on the same Wi-Fi (or on the Pi's own screen).
Photo sources: Pi camera / webcam button, file upload, or the NodeMCU button over MQTT.
A DHT11/DHT22 on the Pi (optional, config.yaml -> sensor.dht) adds temperature and humidity.
Every result is logged to captures/log.csv with the image, and published to MQTT.
"""
import argparse
import csv
import io
import sys
import threading
import time
from collections import deque
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pde.config import load_config, p  # noqa: E402
from pde.predictor import Predictor  # noqa: E402
from pde import quality  # noqa: E402

app = Flask(__name__, template_folder="templates", static_folder="static")
STATE = {"predictor": None, "camera": None, "mqtt": None, "climate": None, "quality": None, "cloud": None,
         "device": "pi-node-01", "history": deque(maxlen=20), "advice": {},
         "captures": ROOT / "captures", "lock": threading.Lock()}


def run_prediction(img, source):
    """img: PIL image or RGB array. Saves the photo, predicts, logs, publishes."""
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    img = img.convert("RGB")
    ts = time.strftime("%Y%m%d_%H%M%S") + f"_{int(time.time() * 1000) % 1000:03d}"
    fname = f"{ts}.jpg"
    caps = STATE["captures"]
    caps.mkdir(exist_ok=True)
    small = img.copy()
    small.thumbnail((800, 800))
    small.save(caps / fname, quality=88)

    with STATE["lock"]:
        pred = STATE["predictor"].predict(img)
    result = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "source": source, "image": fname,
              "image_id": f"{STATE['device']}_{ts}", "device": STATE["device"], **pred.to_dict()}
    # Photo check: the model has no "not a leaf" class, so blurry / leafless photos must not
    # get a disease name. The model's guess stays in top_label/top3 for transparency.
    if STATE["quality"] is not None:
        q = quality.check(img, **STATE["quality"])
        result["quality"] = q
        if not q["ok"]:
            result.update(label="not_sure", is_sure=False, advice=q["message"], disease="")
    # one simple field for LEDs / dashboards: healthy | disease | not_sure
    result["status"] = ("not_sure" if not result["is_sure"]
                        else "healthy" if result["disease"] == "healthy" else "disease")
    clim = STATE["climate"].snapshot() if STATE["climate"] else {}
    if clim.get("temp_c") is not None:
        result.update(temp_c=clim["temp_c"], humidity=clim["humidity"], fungal_risk=clim["fungal_risk"])

    header = ["time", "source", "image", "label", "top_label", "crop", "status", "confidence", "latency_ms",
              "temp_c", "humidity", "photo_problem", "sharpness", "leaf_fraction"]
    log = caps / "log.csv"
    if log.exists():
        with open(log) as f:
            if f.readline().strip() != ",".join(header):  # old format -> keep it aside
                log.rename(caps / f"log_old_{ts}.csv")
    new = not log.exists()
    with open(log, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        q = result.get("quality", {})
        w.writerow([result["time"], source, fname, result["label"], pred.top_label, pred.crop, result["status"], pred.confidence, pred.latency_ms,
                    result.get("temp_c", ""), result.get("humidity", ""),
                    q.get("problem") or "", q.get("sharpness", ""), q.get("leaf_fraction", "")])

    STATE["history"].appendleft(result)
    if STATE["cloud"]:   # photo + edge result to the cloud gateway (queued on disk if offline)
        STATE["cloud"].enqueue(fname, {"image_id": result["image_id"], "device": STATE["device"],
                                       "edge": {k: v for k, v in result.items() if k != "image"}})
    if STATE["mqtt"]:
        STATE["mqtt"].publish_result({k: v for k, v in result.items() if k != "image"})
    qmsg = f" [photo: {result['quality']['problem']}]" if not result.get("quality", {}).get("ok", True) else ""
    print(f"[app] {source}: {result['label']} ({pred.confidence:.2f}) {pred.latency_ms:.0f} ms{qmsg}")
    return result


# ------------------------------------------------------------------ routes
@app.get("/")
def index():
    return render_template("index.html", camera=STATE["camera"].available if STATE["camera"] else False,
                           threshold=STATE["predictor"].threshold)


@app.post("/api/predict")
def api_predict():
    f = request.files.get("image")
    if not f:
        return jsonify(error="no image"), 400
    try:
        img = Image.open(io.BytesIO(f.read()))
        img.load()
    except Exception:
        return jsonify(error="not an image"), 400
    return jsonify(run_prediction(img, "upload"))


@app.post("/api/capture")
def api_capture():
    cam = STATE["camera"]
    if not cam or not cam.available:
        return jsonify(error="no camera on this device"), 503
    try:
        img = cam.capture()
    except Exception as e:
        print(f"[app] camera error: {e}")
        return jsonify(error=f"camera error: {e}"), 503
    return jsonify(run_prediction(img, "camera"))


@app.get("/api/preview.jpg")
def api_preview():
    """Live preview frame (no prediction, not saved) for aiming the camera."""
    cam = STATE["camera"]
    if not cam or not cam.available:
        return jsonify(error="no camera on this device"), 503
    try:
        img = Image.fromarray(cam.capture())
    except Exception as e:
        return jsonify(error=f"camera error: {e}"), 503
    img.thumbnail((640, 640))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=75)
    return buf.getvalue(), 200, {"Content-Type": "image/jpeg", "Cache-Control": "no-store"}


@app.get("/api/history")
def api_history():
    return jsonify(list(STATE["history"]))


@app.get("/api/status")
def api_status():
    m = STATE["mqtt"]
    return jsonify(camera=STATE["camera"].kind if STATE["camera"] else None,
                   mqtt=None if m is None else m.connected, threshold=STATE["predictor"].threshold,
                   climate=STATE["climate"].snapshot() if STATE["climate"] else {"sensor": None},
                   cloud=STATE["cloud"].status() if STATE["cloud"] else None)


@app.get("/api/advice/<image_id>")
def api_advice(image_id):
    """Cloud advisor measures for one photo (via MQTT plant/advice). 404 until they arrive / when offline."""
    a = STATE["advice"].get(image_id)
    return (jsonify(a), 200) if a else (jsonify(pending=True), 404)


@app.get("/api/climate")
def api_climate():
    return jsonify(STATE["climate"].snapshot() if STATE["climate"] else {"sensor": None})


@app.get("/captures/<path:name>")
def captures(name):
    return send_from_directory(STATE["captures"], name)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--threshold", type=float)
    ap.add_argument("--camera", default="auto", choices=["auto", "picamera2", "opencv", "none"])
    ap.add_argument("--no-mqtt", action="store_true")
    ap.add_argument("--no-dht", action="store_true", help="run without the DHT temperature/humidity sensor")
    ap.add_argument("--no-photo-check", action="store_true", help="skip the blur / leaf-present check")
    ap.add_argument("--no-cloud", action="store_true", help="do not upload photos to the cloud gateway")
    ap.add_argument("--broker")
    ap.add_argument("--port", type=int)
    ap.add_argument("--config")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ac, mc = cfg["app"], cfg["mqtt"]
    STATE["device"] = mc.get("device_id", STATE["device"])

    model = p(args.model or ac["model_path"])
    if not model.exists():
        sys.exit(f"Model not found: {model}. Copy the .tflite from training into models/.")
    STATE["predictor"] = Predictor(model, threshold=args.threshold or ac["confidence_threshold"])
    print(f"[app] model {model.name} (input {STATE['predictor'].inp['dtype'].__name__})")
    qc = ac.get("photo_check", {"enabled": True})
    if qc.get("enabled", True) and not args.no_photo_check:
        STATE["quality"] = {k: qc.get(k, v) for k, v in quality.DEFAULTS.items()}
        print(f"[app] photo check on: {STATE['quality']}")

    if args.camera != "none":
        from camera import Camera
        STATE["camera"] = Camera(prefer=args.camera)

    if mc.get("enabled") and not args.no_mqtt:
        from mqtt_bridge import MqttBridge
        if args.broker:
            mc["broker"] = args.broker

        def capture_and_predict():
            try:
                run_prediction(STATE["camera"].capture(), "mqtt-button")
            except Exception as e:
                print(f"[app] MQTT capture failed: {e}")

        def on_capture(_req):
            cam = STATE["camera"]
            if cam and cam.available:
                # capture in a thread: never block (or crash) the MQTT network loop
                threading.Thread(target=capture_and_predict, daemon=True).start()
            else:
                print("[app] capture requested over MQTT but there is no camera")

        def on_advice(a):  # measures from the cloud advisor, kept for the last 50 photos
            if a.get("image_id"):
                adv = STATE["advice"]
                adv[a["image_id"]] = a
                while len(adv) > 50:
                    adv.pop(next(iter(adv)))
        STATE["mqtt"] = MqttBridge(mc, on_capture_request=on_capture, on_advice=on_advice)
        STATE["mqtt"].start()

    cc = cfg.get("cloud", {})
    if cc.get("enabled") and not args.no_cloud:
        from cloud_uploader import CloudUploader
        STATE["cloud"] = CloudUploader(cc["gateway_url"], STATE["captures"], timeout=cc.get("timeout_s", 10)).start()
        print(f"[app] cloud upload on: {cc['gateway_url']}")

    dc = cfg.get("sensor", {}).get("dht", {})
    if dc.get("enabled") and not args.no_dht:
        from climate import Climate
        last_pub = [0.0]

        def on_reading(snap):  # publish to MQTT at most every publish_every_s
            m = STATE["mqtt"]
            if m and time.time() - last_pub[0] >= dc.get("publish_every_s", 30):
                last_pub[0] = time.time()
                m.publish_climate(snap)

        STATE["climate"] = Climate(dc.get("type", "DHT11"), dc.get("gpio", 4),
                                   interval_s=dc.get("interval_s", 5), on_reading=on_reading).start()

    try:
        app.run(host=ac["host"], port=args.port or ac["port"], threaded=True)
    finally:
        if STATE["mqtt"]:
            STATE["mqtt"].stop()
        if STATE["camera"]:
            STATE["camera"].close()
        if STATE["climate"]:
            STATE["climate"].stop()
        if STATE["cloud"]:
            STATE["cloud"].stop()


if __name__ == "__main__":
    main()
