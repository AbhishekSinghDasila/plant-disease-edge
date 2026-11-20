"""ingest: every plant/# MQTT message -> postgres.

All messages go to `telemetry` (raw JSON). Diagnoses, advice, irrigation decisions and alerts are
also written to their own tables so the dashboard and other services can query them easily.
"""
import logging

from common import db, logs
from common.mqtt import Bus

log = logging.getLogger("ingest")


def device_of(d):
    return d.get("device") or d.get("device_id")


def store(topic, d):
    kind = topic.split("/")[-1]
    if topic.startswith("plant/image"):     # the gateway writes images itself
        return
    db.execute("INSERT INTO telemetry (topic, kind, device, data) VALUES (%s, %s, %s, %s)",
               (topic, kind, device_of(d), db.jsonb(d)))
    if kind in ("diagnosis", "cloud_diagnosis"):
        db.execute("""INSERT INTO diagnoses (source, device, image_id, class_id, crop, status, confidence, data)
                      VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                   ("edge" if kind == "diagnosis" else "cloud", device_of(d), d.get("image_id"),
                    d.get("class_id") or d.get("top_label"), d.get("crop"), d.get("status"),
                    d.get("confidence"), db.jsonb(d)))
    elif kind == "advice":
        db.execute("INSERT INTO advice (image_id, class_id, verdict, data) VALUES (%s, %s, %s, %s)",
                   (d.get("image_id"), d.get("class_id"), d.get("verdict"), db.jsonb(d)))
    elif kind == "irrigation":
        db.execute("INSERT INTO irrigation_events (need_water, urgency, data) VALUES (%s, %s, %s)",
                   (d.get("need_water"), d.get("urgency"), db.jsonb(d)))
    elif kind == "alert":
        db.execute("INSERT INTO alerts (level, kind, message, data) VALUES (%s, %s, %s, %s)",
                   (d.get("level"), d.get("kind"), d.get("message"), db.jsonb(d)))


def main():
    logs.setup()
    db.pool()
    bus = Bus("ingest")
    bus.subscribe("plant/#", store)
    log.info("ingest running")
    bus.start(background=False)


if __name__ == "__main__":
    main()
