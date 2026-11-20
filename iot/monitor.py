"""Laptop-side monitor: prints every plant/# message and logs diagnoses to iot/diagnoses.csv.

Run:  python iot/monitor.py                 broker = localhost
      python iot/monitor.py 172.29.1.135    broker = the Pi
Stop: Ctrl+C

Simulate the NodeMCU button without hardware:
      mosquitto_pub -h <pi-ip> -t plant/capture -m '{"device":"laptop-test"}'
"""
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import paho.mqtt.client as mqtt

BROKER = sys.argv[1] if len(sys.argv) > 1 else "localhost"
LOG = Path(__file__).resolve().parent / "diagnoses.csv"
FIELDS = ["received", "device", "time", "source", "label", "top_label", "confidence", "latency_ms"]


def on_connect(client, _u, _f, reason_code, _p):
    if reason_code == 0:
        print(f"Connected to {BROKER}, listening on 'plant/#' ...")
        client.subscribe("plant/#")
    else:
        print(f"Connect failed: {reason_code}")


def on_message(_c, _u, msg):
    now = datetime.now().strftime("%H:%M:%S")
    text = msg.payload.decode(errors="replace")
    if msg.topic == "plant/diagnosis":
        try:
            d = json.loads(text)
        except ValueError:
            print(f"{now}  {msg.topic}  (bad JSON) {text}")
            return
        tag = "NOT SURE" if not d.get("is_sure") else d["label"].upper()
        print(f"{now}  [{d.get('device')}] {tag:24s} conf={d.get('confidence'):.2f}  "
              f"{d.get('latency_ms')} ms  via {d.get('source')}")
        if d.get("is_sure"):
            print(f"          advice: {d.get('advice')}")
        new = not LOG.exists()
        with open(LOG, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow({"received": datetime.now().isoformat(timespec="seconds"), **d})
    else:
        print(f"{now}  {msg.topic}  {text}")


client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.on_connect = on_connect
client.on_message = on_message
client.connect(BROKER, 1883, keepalive=60)
try:
    client.loop_forever()
except KeyboardInterrupt:
    print("\nStopped.")
