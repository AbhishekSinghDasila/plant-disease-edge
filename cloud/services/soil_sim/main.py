"""soil-sim: simulated soil-moisture sensor until the real NodeMCU probe exists.

Publishes plant/soil {device, moisture_pct, raw, simulated: true} every INTERVAL_S seconds.
Drying speed follows the latest plant/climate (hotter / drier air -> faster), with fallback values
when the DHT11 has no reading. Watering: plant/irrigation/cmd {"action": "water", "amount_pct": 35}
(the dashboard button sends this). SIM_SPEED makes simulated time run faster for demos
(60 = one simulated hour per real minute). AUTO_WATER_BELOW waters by itself so the demo keeps cycling
(0 = off). It pauses by itself while a real probe (plant/soil with simulated=false, e.g. the NodeMCU) is sending,
and starts again REAL_TIMEOUT_S after the last real reading. ENABLED=0 turns it off completely.
"""
import logging
import os
import random
import time

from common import logs
from common.agro import vpd_kpa
from common.mqtt import Bus

log = logging.getLogger("soil-sim")
ENABLED = os.getenv("ENABLED", "1") == "1"
INTERVAL = float(os.getenv("INTERVAL_S", "30"))
SPEED = float(os.getenv("SIM_SPEED", "60"))
BASE_DRY = float(os.getenv("BASE_DRY_PCT_PER_H", "1.2"))   # % per hour at 20 °C, VPD ~1 kPa
AUTO_WATER_BELOW = float(os.getenv("AUTO_WATER_BELOW", "15"))
DEVICE = os.getenv("DEVICE", "soil-sim-01")
REAL_TIMEOUT = float(os.getenv("REAL_TIMEOUT_S", "600"))
state = {"m": float(os.getenv("START_PCT", "62")), "temp": 27.0, "rh": 60.0, "clim_ts": 0, "real_ts": 0}
bus = Bus("soil-sim")


def on_climate(topic, d):
    if d.get("temp_c") is not None and d.get("humidity") is not None:
        state.update(temp=d["temp_c"], rh=d["humidity"], clim_ts=time.time())


def on_soil(topic, d):
    if d.get("simulated") is False and d.get("device") != DEVICE:
        if time.time() - state["real_ts"] > REAL_TIMEOUT:
            log.info("real soil probe %s is sending: simulator paused", d.get("device"))
        state["real_ts"] = time.time()


def real_probe_active():
    return time.time() - state["real_ts"] < REAL_TIMEOUT


def on_cmd(topic, d):
    if d.get("action") == "water" and not real_probe_active():
        add = float(d.get("amount_pct", 35))
        state["m"] = min(95.0, state["m"] + add)
        log.info("watered +%.0f %% -> %.1f %% (by %s)", add, state["m"], d.get("by", "?"))
        publish()


def rate_per_h():
    vpd = vpd_kpa(state["temp"], state["rh"]) or 1.0
    heat = 1 + max(0.0, state["temp"] - 20) / 15
    dryness = 0.6 + 0.6 * vpd        # more evaporation in dry air
    wet = 0.5 + state["m"] / 100     # wet soil loses water faster than nearly dry soil
    return BASE_DRY * heat * dryness * wet


def publish():
    m = state["m"]
    raw = int(820 - m / 100 * (820 - 380))   # what a capacitive probe on the NodeMCU A0 would read (dry 820, wet 380)
    bus.publish("plant/soil", {"device": DEVICE, "moisture_pct": round(m, 1), "raw": raw, "simulated": True,
                               "sim_speed": SPEED, "time": time.strftime("%Y-%m-%d %H:%M:%S")})


def main():
    logs.setup()
    if not ENABLED:
        log.info("disabled (ENABLED=0): real sensor in use")
        while True:
            time.sleep(3600)
    bus.subscribe("plant/climate", on_climate)
    bus.subscribe("plant/irrigation/cmd", on_cmd)
    bus.subscribe("plant/soil", on_soil, skip_retained=True)
    bus.start()
    log.info("soil-sim running: %.0f %% start, speed x%.0f, every %.0f s", state["m"], SPEED, INTERVAL)
    while True:
        if real_probe_active():
            time.sleep(INTERVAL)
            continue
        dt_h = INTERVAL * SPEED / 3600
        state["m"] = max(3.0, state["m"] - rate_per_h() * dt_h + random.gauss(0, 0.15))
        if AUTO_WATER_BELOW and state["m"] < AUTO_WATER_BELOW:
            on_cmd("", {"action": "water", "amount_pct": 50, "by": "auto (simulated farmer)"})
        publish()
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
