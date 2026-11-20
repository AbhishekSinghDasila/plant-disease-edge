"""Alert rules: MQTT messages in -> alerts out. Pure (time is passed in), unit-tested in cloud/tests.

Each method returns a list of alerts {level: info|warning|critical, kind, message, key}.
Cooldowns stop the same problem from alerting again and again.
"""

SEV_LEVEL = {"medium": "warning", "high": "warning", "critical": "critical"}


class Rules:
    def __init__(self, disease_cooldown_s=600, water_cooldown_s=1800, silent_after_s=300):
        self.disease_cd, self.water_cd, self.silent_after = disease_cooldown_s, water_cooldown_s, silent_after_s
        self.last_sent = {}       # key -> time
        self.last_seen = {}       # sensor kind -> time of last message
        self.silent = set()       # sensor kinds currently reported as silent
        self.status = {}          # device -> online|offline

    def _once(self, key, now, cooldown):
        if now - self.last_sent.get(key, -1e18) < cooldown:
            return False
        self.last_sent[key] = now
        return True

    def on_advice(self, a, now):
        level = SEV_LEVEL.get(a.get("severity"))
        if a.get("verdict") != "disease" or level is None or a.get("pending_cloud"):
            return []
        key = f"disease:{a.get('class_id')}"
        if not self._once(key, now, self.disease_cd):
            return []
        first = (a.get("immediate") or [""])[0]
        conf = a.get("edge_confidence")
        pct = f" ({100 * conf:.0f} %)" if conf is not None else ""
        return [{"level": level, "kind": "disease", "key": key, "image_id": a.get("image_id"),
                 "message": f"{a.get('title')}{pct} on {a.get('device') or 'the plant'}. {first}".strip()}]

    def on_irrigation(self, d, now):
        if not d.get("need_water"):
            self.last_sent.pop("water", None)          # next dry spell alerts again at once
            return []
        if d.get("urgency") != "high" or not self._once("water", now, self.water_cd):
            return []
        return [{"level": "warning", "kind": "water", "key": "water",
                 "message": f"Water the plant now: {d.get('reason', '')}"}]

    def on_watered(self, d, now):
        if not self._once("watered", now, 60):
            return []
        return [{"level": "info", "kind": "water", "key": "watered",
                 "message": f"Watering detected: soil moisture {d.get('from_pct')} % → {d.get('to_pct')} %."}]

    def on_status(self, d, now):
        dev, st = d.get("device", "?"), d.get("state")
        prev = self.status.get(dev)
        self.status[dev] = st
        if st == "offline" and prev != "offline":
            return [{"level": "critical", "kind": "offline", "key": f"offline:{dev}",
                     "message": f"{dev} went offline (no camera / edge model until it is back)."}]
        if st == "online" and prev == "offline":
            return [{"level": "info", "kind": "offline", "key": f"online:{dev}", "message": f"{dev} is back online."}]
        return []

    def seen(self, kind, now):
        """A sensor message arrived. Returns a 'back' alert if that sensor was reported silent."""
        self.last_seen[kind] = now
        if kind in self.silent:
            self.silent.discard(kind)
            return [{"level": "info", "kind": "sensor", "key": f"back:{kind}", "message": f"{kind} sensor is sending again."}]
        return []

    def watchdog(self, now):
        """Sensors that sent before but are quiet for silent_after seconds -> one warning each."""
        out = []
        for kind, t in self.last_seen.items():
            if kind not in self.silent and now - t > self.silent_after:
                self.silent.add(kind)
                out.append({"level": "warning", "kind": "sensor", "key": f"silent:{kind}",
                            "message": f"No {kind} reading for {int((now - t) // 60)} min: check the sensor / node."})
        return out
