"""MQTT client with auto-reconnect and JSON payloads.

    bus = Bus("ingest"); bus.subscribe("plant/#", handler); bus.start()   # handler(topic, dict)
    bus.publish("plant/advice", {...}, retain=False)
"""
import json
import logging
import time

import paho.mqtt.client as mqtt

from common import settings

log = logging.getLogger("mqtt")


class Bus:
    def __init__(self, name, host=None, port=None):
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"cloud-{name}", clean_session=False)
        self.host, self.port = host or settings.MQTT_HOST, port or settings.MQTT_PORT
        self.subs = []
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.reconnect_delay_set(1, 30)

    def subscribe(self, pattern, handler, qos=1, skip_retained=False):
        """skip_retained=True: ignore old retained messages the broker replays on (re)connect."""
        self.subs.append((pattern, handler, qos, skip_retained))

    def _on_connect(self, client, userdata, flags, rc, props=None):
        log.info("connected to %s:%s (rc=%s)", self.host, self.port, rc)
        for pattern, _, qos, _ in self.subs:
            client.subscribe(pattern, qos)

    def _on_message(self, client, userdata, msg):
        try:
            data = json.loads(msg.payload.decode() or "null")
        except (UnicodeDecodeError, json.JSONDecodeError):
            data = {"raw": msg.payload.decode(errors="replace")}
        if not isinstance(data, dict):
            data = {"value": data}
        for pattern, handler, _, skip_retained in self.subs:
            if skip_retained and msg.retain:
                continue
            if mqtt.topic_matches_sub(pattern, msg.topic):
                try:
                    handler(msg.topic, data)
                except Exception:  # one bad message must not kill the service
                    log.exception("handler failed for %s", msg.topic)

    def publish(self, topic, data, retain=False, qos=1):
        self.client.publish(topic, json.dumps(data, default=str), qos=qos, retain=retain)

    def start(self, background=True):
        while True:  # broker may still be starting
            try:
                self.client.connect(self.host, self.port, keepalive=30)
                break
            except OSError as e:
                log.warning("broker not reachable (%s), retrying in 3 s", e)
                time.sleep(3)
        if background:
            self.client.loop_start()
        else:
            self.client.loop_forever()
