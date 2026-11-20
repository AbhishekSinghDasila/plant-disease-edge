"""disease-advisor: turns every diagnosis into a verdict + measures.

Inputs:  plant/diagnosis (edge, Pi), plant/cloud_diagnosis (second opinion, Phase 5), plant/climate.
Output:  plant/advice (retained) = verdict, title, summary, severity, immediate/cultural/chemical/prevention.
Edge-only advice is published at once with pending_cloud=true (the notifier waits). When the cloud diagnosis for
the same image_id arrives, the advice is re-published with both opinions; if it does not come within CLOUD_WAIT_S
(cloud model down), the edge-only advice is re-published as final.
"""
import logging
import os
import threading
import time
from collections import OrderedDict

from common import logs
from common.mqtt import Bus
from service.logic import advise

log = logging.getLogger("advisor")
bus = Bus("advisor")
state = {"climate": {}, "edge": OrderedDict(), "cloud": OrderedDict()}   # image_id -> diagnosis (last 50)
CLOUD_WAIT_S = float(os.getenv("CLOUD_WAIT_S", "20"))
lock = threading.Lock()


def remember(store, d):
    if d.get("image_id"):
        store[d["image_id"]] = d
        while len(store) > 50:
            store.popitem(last=False)


def climate_now():
    c = state["climate"]
    return c if time.time() - c.get("ts", 0) < 900 else {}


def publish(edge, cloud=None, pending=False):
    a = advise(edge, cloud, climate_now())
    a["pending_cloud"] = pending
    a["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    bus.publish("plant/advice", a, retain=True)
    log.info("%s -> %s (%s, severity %s)", a["image_id"], a["verdict"], a["title"], a["severity"])


def on_climate(topic, d):
    if d.get("temp_c") is not None:
        state["climate"] = {"temp_c": d["temp_c"], "humidity": d.get("humidity"),
                            "fungal_risk": d.get("fungal_risk"), "ts": time.time()}


def on_edge(topic, d):
    with lock:
        remember(state["edge"], d)
        cloud = state["cloud"].get(d.get("image_id"))
        # a cloud check only matters when the edge gave an answer; uncertain stays uncertain
        wait = cloud is None and d.get("status") != "not_sure" and CLOUD_WAIT_S > 0
        publish(d, cloud, pending=wait)
    if wait:
        threading.Timer(CLOUD_WAIT_S, cloud_timeout, args=(d.get("image_id"),)).start()


def cloud_timeout(image_id):
    with lock:
        if image_id in state["edge"] and image_id not in state["cloud"]:
            log.info("%s: no cloud answer in %.0f s, edge-only advice is final", image_id, CLOUD_WAIT_S)
            publish(state["edge"][image_id])


def on_cloud(topic, d):
    with lock:
        remember(state["cloud"], d)
        edge = state["edge"].get(d.get("image_id"))
        if edge:
            publish(edge, d)


def main():
    logs.setup()
    bus.subscribe("plant/climate", on_climate)
    bus.subscribe("plant/diagnosis", on_edge)
    bus.subscribe("plant/cloud_diagnosis", on_cloud)
    log.info("advisor running")
    bus.start(background=False)


if __name__ == "__main__":
    main()
