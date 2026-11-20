"""Send each captured photo + its edge result to the cloud gateway (POST /api/images).

Store-and-forward: every upload is first written to captures/outbox/<image_id>.json; a background
thread sends the outbox oldest-first and deletes an entry only after the gateway accepted it.
If the cloud (laptop) is off, entries wait on the SD card and go out when it is back.
The MQTT diagnosis itself does not depend on this: it travels over the Mosquitto bridge.
"""
import json
import threading
import time
from pathlib import Path

import requests


class CloudUploader:
    def __init__(self, gateway_url, captures_dir, timeout=10, max_outbox=500):
        self.url = gateway_url.rstrip("/") + "/api/images"
        self.captures = Path(captures_dir)
        self.outbox = self.captures / "outbox"
        self.outbox.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.max_outbox = max_outbox
        self.wake = threading.Event()
        self.stop_flag = False
        self.last_error = None
        self.sent = 0

    def enqueue(self, image_file, meta):
        entries = sorted(self.outbox.glob("*.json"))
        for old in entries[: max(0, len(entries) - self.max_outbox + 1)]:  # cap the queue: drop the oldest
            old.unlink(missing_ok=True)
        (self.outbox / f"{meta['image_id']}.json").write_text(json.dumps({"image": image_file, "meta": meta}, default=str))
        self.wake.set()

    def pending(self):
        return len(list(self.outbox.glob("*.json")))

    def _send(self, entry):
        job = json.loads(entry.read_text())
        img = self.captures / job["image"]
        if not img.exists():          # photo deleted: nothing to send
            entry.unlink(missing_ok=True)
            return True
        with open(img, "rb") as f:
            r = requests.post(self.url, files={"image": (img.name, f, "image/jpeg")},
                              data={"meta": json.dumps(job["meta"], default=str)}, timeout=self.timeout)
        if r.status_code == 400:      # bad entry, retrying will not help
            print(f"[cloud] rejected {entry.name}: {r.text[:200]}")
            entry.unlink(missing_ok=True)
            return True
        r.raise_for_status()
        entry.unlink(missing_ok=True)
        self.sent += 1
        return True

    def _loop(self):
        backoff = 5
        while not self.stop_flag:
            entries = sorted(self.outbox.glob("*.json"))
            try:
                for e in entries:
                    self._send(e)
                if entries and self.last_error:
                    print(f"[cloud] gateway reachable again, outbox sent")
                self.last_error = None
                backoff = 5
                self.wake.wait(60)    # new photo or periodic recheck
            except Exception as ex:   # cloud down / network error -> keep the entry, retry later
                if self.last_error is None:
                    print(f"[cloud] upload failed ({ex}); {len(entries)} waiting in outbox, will retry")
                self.last_error = str(ex)
                self.wake.wait(backoff)
                backoff = min(backoff * 2, 120)
            self.wake.clear()

    def start(self):
        threading.Thread(target=self._loop, daemon=True, name="cloud-uploader").start()
        return self

    def stop(self):
        self.stop_flag = True
        self.wake.set()

    def status(self):
        return {"url": self.url, "pending": self.pending(), "sent": self.sent, "last_error": self.last_error}
