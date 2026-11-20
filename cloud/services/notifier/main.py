"""notifier: watches the system and raises alerts.

Inputs:  plant/advice, plant/irrigation, plant/irrigation/watered, plant/status, plant/soil, plant/climate.
Output:  plant/alert (ingest stores it in the alerts table; the dashboard shows it live).
Optional push to Telegram for warning/critical alerts: set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID (off by default).
"""
import json
import logging
import os
import threading
import time
import urllib.parse
import urllib.request

from common import logs
from common.mqtt import Bus
from service.rules import Rules

log = logging.getLogger("notifier")
bus = Bus("notifier")
rules = Rules(silent_after_s=float(os.getenv("SENSOR_SILENT_S", "300")))
lock = threading.Lock()
TG_TOKEN, TG_CHAT = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")


def push(alert):
    if not (TG_TOKEN and TG_CHAT) or alert["level"] == "info":
        return
    try:
        body = urllib.parse.urlencode({"chat_id": TG_CHAT, "text": f"🌱 {alert['level'].upper()}: {alert['message']}"}).encode()
        urllib.request.urlopen(f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage", body, timeout=10)
    except Exception as e:
        log.warning("telegram push failed: %s", e)


def emit(alerts):
    for a in alerts:
        a["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        bus.publish("plant/alert", a)
        log.info("[%s] %s", a["level"], a["message"])
        push(a)


def handler(fn):
    def h(topic, d):
        with lock:
            emit(fn(d, time.time()))
    return h


def sensor(kind):
    return handler(lambda d, now: rules.seen(kind, now))


def main():
    logs.setup()
    bus.subscribe("plant/advice", handler(rules.on_advice), skip_retained=True)
    bus.subscribe("plant/irrigation", handler(rules.on_irrigation), skip_retained=True)
    bus.subscribe("plant/irrigation/watered", handler(rules.on_watered), skip_retained=True)
    bus.subscribe("plant/status", handler(rules.on_status))
    bus.subscribe("plant/soil", sensor("soil"), skip_retained=True)
    bus.subscribe("plant/climate", sensor("climate"), skip_retained=True)
    bus.start()
    log.info("notifier running (telegram %s)", "on" if TG_TOKEN and TG_CHAT else "off")
    while True:
        with lock:
            emit(rules.watchdog(time.time()))
        time.sleep(30)


if __name__ == "__main__":
    main()
