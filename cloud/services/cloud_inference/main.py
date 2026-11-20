"""cloud-inference: second opinion on every uploaded photo, with a bigger / more careful model.

Input:   plant/image {image_id, device, edge}: the gateway publishes it after the Pi uploaded the photo
         (file in the shared `images` volume, /data/images/<image_id>.jpg).
Output:  plant/cloud_diagnosis {image_id, class_id, status, confidence, agreement, top3, model, ...}.
Model:   MODEL_PATH / LABELS_PATH (default: the 38-class FP32 export in /models, mounted from ../models),
         run with 6-view test-time augmentation (tta.py). Phase 5 v2: point MODEL_PATH at a new model.
"""
import logging
import os
import queue
import threading
import time
from pathlib import Path

from common import logs
from common.mqtt import Bus
from service.tta import CloudModel

log = logging.getLogger("cloud-inference")
IMAGES = Path(os.getenv("IMAGES_DIR", "/data/images"))
bus = Bus("cloud-inference")
jobs = queue.Queue(maxsize=200)


def on_image(topic, d):
    if d.get("image_id"):
        try:
            jobs.put_nowait(d)
        except queue.Full:
            log.warning("queue full, skipping %s", d["image_id"])


def worker(model):
    while True:
        d = jobs.get()
        path = IMAGES / f"{d['image_id']}.jpg"
        for _ in range(10):                   # the file is written before plant/image, but be safe
            if path.exists():
                break
            time.sleep(0.5)
        else:
            log.warning("photo %s not found", path)
            continue
        try:
            r = model.predict(path)
        except Exception:
            log.exception("prediction failed for %s", d["image_id"])
            continue
        crop, dis = r["class_id"].split(".", 1)
        display = dis.replace("_", " ").capitalize() if crop == "other" else f"{crop.capitalize()}: {dis.replace('_', ' ')}"
        r.update(image_id=d["image_id"], device=d.get("device"), source="cloud",
                 display=display, time=time.strftime("%Y-%m-%d %H:%M:%S"))
        bus.publish("plant/cloud_diagnosis", r)
        edge = (d.get("edge") or {}).get("class_id")
        log.info("%s -> %s %s %.0f %% (views agree %.0f %%, %.0f ms)%s", d["image_id"], r["status"], r["class_id"],
                 100 * r["confidence"], 100 * r["agreement"], r["latency_ms"],
                 "" if not edge else (" = edge" if edge == r["class_id"] else f" != edge {edge}"))


def main():
    logs.setup()
    model = CloudModel(os.getenv("MODEL_PATH", "/models/pv38_fp32.tflite"),
                       os.getenv("LABELS_PATH", "/models/pv38.labels.json"),
                       threshold=float(os.getenv("THRESHOLD", "0.8")),
                       min_agreement=float(os.getenv("MIN_AGREEMENT", "0.8")),
                       num_threads=int(os.getenv("THREADS", "2")))
    log.info("model %s loaded (%d classes, %d views)", model.name, len(model.ids), len(model.views))
    threading.Thread(target=worker, args=(model,), daemon=True).start()
    bus.subscribe("plant/image", on_image, skip_retained=True)
    bus.start(background=False)


if __name__ == "__main__":
    main()
