"""irrigation: "does the plant need water?"

Inputs:  plant/soil (moisture_pct), plant/climate (temp_c, humidity), plant/diagnosis (crop of the plant).
Output:  plant/irrigation (retained) = need_water, urgency, reason, hours_to_dry, vpd_kpa, crop, last_watered.
         plant/irrigation/watered (retained) = watering detected from a sudden rise in soil moisture.
Crop: CROP env var, or the crop of the latest confident edge diagnosis when CROP=auto (default).
"""
import logging
import os
import time
from pathlib import Path

import yaml

from common import db, logs
from common.mqtt import Bus
from service.logic import decide, detect_watering, trend_pct_per_hour

log = logging.getLogger("irrigation")
BANDS = yaml.safe_load((Path(__file__).parent / "crops.yaml").read_text())
CROP = os.getenv("CROP", "auto")
MIN_RISE = float(os.getenv("WATERING_MIN_RISE_PCT", "8"))
state = {"crop": None if CROP == "auto" else CROP, "climate": {}, "was_needed": False, "last": None, "last_pub": 0,
         "prev_m": {}, "watered": None}
bus = Bus("irrigation")


def on_climate(topic, d):
    if d.get("temp_c") is not None:
        state["climate"] = {"temp_c": d["temp_c"], "humidity": d.get("humidity"), "ts": time.time()}


def on_diagnosis(topic, d):
    if CROP == "auto" and d.get("crop") and d.get("status") in ("healthy", "disease"):
        state["crop"] = d["crop"]


def soil_points(device, hours=2):
    rows = db.query("""SELECT extract(epoch FROM ts) AS t, (data->>'moisture_pct')::float AS m FROM telemetry
                       WHERE kind = 'soil' AND device IS NOT DISTINCT FROM %s AND ts > now() - make_interval(hours => %s)
                       ORDER BY ts""", (device, hours))
    return [(float(r["t"]), r["m"]) for r in rows if r["m"] is not None]


def on_soil(topic, d):
    m = d.get("moisture_pct")
    if m is None:
        return
    dev = d.get("device")
    if dev not in state["prev_m"]:          # after a restart: compare with the last stored reading
        try:
            rows = db.query("""SELECT (data->>'moisture_pct')::float AS m FROM telemetry WHERE kind = 'soil'
                               AND device IS NOT DISTINCT FROM %s AND ts < now() - interval '3 seconds'
                               ORDER BY ts DESC LIMIT 1""", (dev,))
            state["prev_m"][dev] = rows[0]["m"] if rows else None
        except Exception:
            log.exception("last soil reading query failed")
    w = detect_watering(state["prev_m"].get(dev), m, MIN_RISE)
    state["prev_m"][dev] = m
    if w:
        w.update(device=dev, how="detected: soil moisture jumped up", simulated=bool(d.get("simulated")),
                 time=time.strftime("%Y-%m-%d %H:%M:%S"), ts=time.time())
        state["watered"] = w
        bus.publish("plant/irrigation/watered", w, retain=True)
        log.info("watering detected on %s: %.0f %% -> %.0f %%", dev, w["from_pct"], w["to_pct"])
    crop = state["crop"] or "tomato"
    band = BANDS.get(crop, BANDS["default"])
    clim = state["climate"] if time.time() - state["climate"].get("ts", 0) < 600 else {}
    try:
        slope = trend_pct_per_hour(soil_points(d.get("device")) + [(time.time(), m)])
    except Exception:
        log.exception("trend query failed")
        slope = None
    r = decide(m, band, clim.get("temp_c"), clim.get("humidity"), slope, state["was_needed"])
    state["was_needed"] = r["need_water"]
    r.update(crop=crop, device=d.get("device"), trend_pct_per_h=None if slope is None else round(slope, 2),
             climate_source="sensor" if clim else "none",
             last_watered=state["watered"], time=time.strftime("%Y-%m-%d %H:%M:%S"))
    key = (r["need_water"], r["urgency"])
    # publish on every change, otherwise at most once a minute
    if w or key != state["last"] or time.time() - state["last_pub"] > 60:
        bus.publish("plant/irrigation", r, retain=True)
        if key != state["last"]:
            log.info("%s: need_water=%s urgency=%s (%s)", crop, *key, r["reason"])
        state["last"], state["last_pub"] = key, time.time()


def main():
    logs.setup()
    db.pool()
    bus.subscribe("plant/climate", on_climate)
    bus.subscribe("plant/diagnosis", on_diagnosis)
    bus.subscribe("plant/soil", on_soil)
    log.info("irrigation running (crop=%s)", CROP)
    bus.start(background=False)


if __name__ == "__main__":
    main()
