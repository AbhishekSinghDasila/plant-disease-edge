"""End-to-end check of the whole system (docs/PLAN_iot_cloud.md, "Verification"). Prints PASS / FAIL per check.

    .venv/bin/python scripts/e2e_check.py                 # all checks except the offline test (~2-3 min)
    .venv/bin/python scripts/e2e_check.py --offline-test  # also stops the cloud broker+gateway to test store-and-forward
    .venv/bin/python scripts/e2e_check.py --no-tests --no-crops   # quick health check only (~40 s)

Note: the crop check uploads one real photo per crop (14) through the Pi, so they appear in the dashboard history
(source "upload") and may raise disease alerts. That is the real pipeline, not fake data.
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import paho.mqtt.client as mqtt

ROOT = Path(__file__).resolve().parent.parent
PI = os.getenv("PI_URL", "http://10.93.8.209:8000")
GW = os.getenv("GATEWAY_URL", "http://localhost:8080")
EXPECTED_CONTAINERS = ["mosquitto", "postgres", "ingest", "gateway", "soil-sim", "irrigation", "advisor", "notifier",
                       "cloud-inference"]
# one photo per crop (disease where the dataset has one) -> expected class id
CROPS = {
    "Apple___Apple_scab": "apple.apple_scab", "Blueberry___healthy": "blueberry.healthy",
    "Cherry_(including_sour)___Powdery_mildew": "cherry.powdery_mildew",
    "Corn_(maize)___Northern_Leaf_Blight": "corn.northern_leaf_blight", "Grape___Black_rot": "grape.black_rot",
    "Orange___Haunglongbing_(Citrus_greening)": "orange.citrus_greening", "Peach___Bacterial_spot": "peach.bacterial_spot",
    "Pepper,_bell___Bacterial_spot": "pepper.bacterial_spot", "Potato___Early_blight": "potato.early_blight",
    "Raspberry___healthy": "raspberry.healthy", "Soybean___healthy": "soybean.healthy",
    "Squash___Powdery_mildew": "squash.powdery_mildew", "Strawberry___Leaf_scorch": "strawberry.leaf_scorch",
    "Tomato___Late_blight": "tomato.late_blight",             # tomato photos are in samples/
}
results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""), flush=True)


def get(url, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def upload(path):
    """POST a photo to the Pi app (multipart, no extra libraries)."""
    boundary = "e2e" + str(time.time_ns())
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"leaf.jpg\"\r\n"
            f"Content-Type: image/jpeg\r\n\r\n").encode() + Path(path).read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(f"{PI}/api/predict", body, {"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


class Listener:
    """Collects plant/# messages from the laptop broker."""
    def __init__(self):
        self.msgs, self.lock = [], threading.Lock()
        self.c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"e2e-{os.getpid()}")
        self.c.on_message = self._on
        self.c.on_connect = lambda c, *a: c.subscribe("plant/#")
        self.c.connect("localhost", 1883)
        self.c.loop_start()

    def _on(self, c, u, m):
        try:
            d = json.loads(m.payload or b"null")
        except ValueError:
            d = None
        with self.lock:
            self.msgs.append((time.time(), m.topic, d if isinstance(d, dict) else {}, m.retain))

    def find(self, topic, pred=lambda d: True, since=0, live=True):
        with self.lock:
            return [d for t, tp, d, r in self.msgs if tp == topic and t >= since and pred(d) and not (live and r)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-tests", action="store_true")
    ap.add_argument("--no-crops", action="store_true")
    ap.add_argument("--offline-test", action="store_true")
    a = ap.parse_args()
    py = str(ROOT / ".venv/bin/python")

    print("== 1. unit tests")
    if not a.no_tests:
        r = subprocess.run([py, "-m", "pytest", "-q", "tests"], cwd=ROOT, capture_output=True, text=True)
        check("project tests", r.returncode == 0, r.stdout.strip().splitlines()[-1] if r.stdout else r.stderr[-200:])
        r = subprocess.run([py, "-m", "pytest", "-q", "tests"], cwd=ROOT / "cloud", capture_output=True, text=True,
                           env={**os.environ, "PYTHONPATH": "."})
        check("cloud tests", r.returncode == 0, r.stdout.strip().splitlines()[-1] if r.stdout else r.stderr[-200:])

    print("== 2. cloud containers")
    out = subprocess.run(["docker", "ps", "--filter", "name=cloud_", "--format", "{{.Names}}"],
                         capture_output=True, text=True).stdout.split()
    missing = [s for s in EXPECTED_CONTAINERS if f"cloud_{s}_1" not in out]
    check(f"{len(EXPECTED_CONTAINERS)} containers running", not missing, f"missing: {missing}" if missing else "")
    try:
        h = get(f"{GW}/api/health")
        check("gateway + database + MQTT", h.get("ok") and h.get("mqtt"), str(h))
    except Exception as e:
        check("gateway + database + MQTT", False, str(e))

    print("== 3. live data on the laptop broker (waits up to 40 s)")
    L = Listener()
    t0 = time.time()
    while time.time() - t0 < 40 and not (L.find("plant/soil", since=t0) and L.find("plant/irrigation", since=0, live=False)):
        time.sleep(1)
    soil = L.find("plant/soil", since=t0)
    check("soil readings arrive", bool(soil), f"{soil[-1].get('moisture_pct')} %, "
          f"{'simulated' if soil and soil[-1].get('simulated') else 'real probe'}" if soil else "none in 40 s")
    check("irrigation decision present", bool(L.find("plant/irrigation", live=False)))

    print("== 4. Pi (edge)")
    try:
        s = get(f"{PI}/api/status")
        check("Pi app reachable", True, f"camera {s.get('camera')}, threshold {s.get('threshold')}")
        check("Pi camera available", bool(s.get("camera")),
              "" if s.get("camera") else "upload-only mode: another program may hold /dev/video0 (sudo fuser -v /dev/video0)")
        check("Pi app connected to MQTT", bool(s.get("mqtt")))
        check("Pi photo upload queue empty", (s.get("cloud") or {}).get("pending", 1) == 0, str(s.get("cloud")))
    except Exception as e:
        check("Pi app reachable", False, str(e))
    bridge = L.find("plant/bridge/pi-node-01", live=False)
    raw = [m for m in L.msgs if m[1] == "plant/bridge/pi-node-01"]
    check("Pi → laptop MQTT bridge connected", bool(raw), "state message seen" if raw else "no bridge state message")
    status = L.find("plant/status", lambda d: d.get("device") == "pi-node-01", live=False)
    check("Pi status online", bool(status) and status[-1].get("state") == "online")

    if not a.no_crops:
        print("== 5. one photo per crop: edge → cloud check → advice (14 crops)")
        ok_edge = ok_full = 0
        for folder, want in CROPS.items():
            d = ROOT / "data/pv_allcrops" / folder
            photo = sorted(d.iterdir())[0] if d.exists() else ROOT / "samples" / f"{folder}.jpg"
            t = time.time()
            r = upload(photo)
            iid = r.get("image_id")
            while time.time() - t < 25 and not L.find("plant/advice", lambda d: d.get("image_id") == iid
                                                         and not d.get("pending_cloud"), since=t):
                time.sleep(0.3)
            cloud = L.find("plant/cloud_diagnosis", lambda d: d.get("image_id") == iid, since=t)
            adv = L.find("plant/advice", lambda d: d.get("image_id") == iid and not d.get("pending_cloud"), since=t)
            e_ok = r.get("class_id") == want and r.get("status") != "not_sure"
            full = bool(cloud) and bool(adv)
            ok_edge += e_ok
            ok_full += full
            print(f"      {want:28} edge {r.get('class_id', '?'):28} {100 * r.get('confidence', 0):3.0f}%  "
                  f"cloud {cloud[-1]['class_id'] if cloud else '—':28} → {adv[-1]['verdict'] if adv else 'no advice'}")
        check("edge model names the right crop + disease", ok_edge >= 12, f"{ok_edge}/{len(CROPS)}")
        check("cloud check + final advice for every photo", ok_full == len(CROPS), f"{ok_full}/{len(CROPS)}")
        check("disease alerts raised", bool(L.find("plant/alert", lambda d: d.get("kind") == "disease", since=t0)),
              "(none is OK if the same diseases alerted in the last 10 min)")

    if a.offline_test:
        print("== 6. store-and-forward: cloud broker + gateway stopped")
        cdir = ROOT / "cloud"
        subprocess.run(["docker-compose", "stop", "gateway", "mosquitto"], cwd=cdir, capture_output=True)
        r = upload(ROOT / "samples/Tomato___Early_blight.jpg")
        iid = r.get("image_id")
        check("Pi still answers while the cloud is down", r.get("status") in ("disease", "healthy", "not_sure"), iid)
        subprocess.run(["docker-compose", "start", "mosquitto", "gateway"], cwd=cdir, capture_output=True)
        t = time.time()
        found = False
        while time.time() - t < 120 and not found:
            time.sleep(5)
            try:
                found = any(d.get("image_id") == iid for d in get(f"{GW}/api/diagnoses?limit=30"))
            except Exception:
                pass
        check("queued diagnosis reached the cloud after restart", found, f"{time.time() - t:.0f} s")

    L.c.loop_stop()
    n_ok = sum(results)
    print(f"\n{n_ok}/{len(results)} checks passed")
    sys.exit(0 if n_ok == len(results) else 1)


if __name__ == "__main__":
    main()
