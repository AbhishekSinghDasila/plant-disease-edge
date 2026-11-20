"""MQTT link between the Pi app and the rest of the IoT setup.

Pi app --publish--> plant/diagnosis   (every prediction, JSON)
Pi app --publish--> plant/status      (retained "online"; broker sends "offline" if the Pi drops)
Pi app --publish--> plant/climate     (DHT temperature/humidity, retained, every ~30 s)
NodeMCU --publish--> plant/capture    (button pressed -> Pi takes a photo and predicts)
cloud   --publish--> plant/advice     (measures from the cloud disease-advisor, shown on the Pi page)

The app works fully offline: if the broker is down, predictions still run and MQTT reconnects
in the background.
"""
import json
import time

import paho.mqtt.client as mqtt


class MqttBridge:
    def __init__(self, cfg, on_capture_request=None, on_advice=None):
        self.cfg = cfg
        self.on_capture_request = on_capture_request
        self.on_advice = on_advice
        self.connected = False
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=cfg["device_id"])
        self.client.will_set(cfg["topic_status"], self._status("offline"), qos=1, retain=True)
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message
        self.client.reconnect_delay_set(1, 30)

    def _status(self, state):
        return json.dumps({"device": self.cfg["device_id"], "state": state, "time": int(time.time())})

    def start(self):
        self.client.connect_async(self.cfg["broker"], self.cfg["port"], keepalive=30)
        self.client.loop_start()

    def stop(self):
        if self.connected:
            self.client.publish(self.cfg["topic_status"], self._status("offline"), qos=1, retain=True)
        self.client.loop_stop()
        self.client.disconnect()

    def _on_connect(self, client, _u, _f, reason_code, _p):
        if reason_code == 0:
            self.connected = True
            print(f"[mqtt] connected to {self.cfg['broker']}")
            client.publish(self.cfg["topic_status"], self._status("online"), qos=1, retain=True)
            client.subscribe(self.cfg["topic_capture"], qos=1)
            client.subscribe(self.cfg.get("topic_advice", "plant/advice"), qos=1)
        else:
            print(f"[mqtt] connect failed: {reason_code}")

    def _on_disconnect(self, *_a):
        self.connected = False

    def _on_message(self, _c, _u, msg):
        if msg.topic == self.cfg.get("topic_advice", "plant/advice") and self.on_advice:
            try:
                self.on_advice(json.loads(msg.payload.decode() or "{}"))
            except Exception as e:
                print(f"[mqtt] advice handler error: {e}")
            return
        if msg.topic == self.cfg["topic_capture"] and self.on_capture_request:
            try:
                req = json.loads(msg.payload.decode() or "{}")
            except ValueError:
                req = {"raw": msg.payload.decode(errors="replace")}
            print(f"[mqtt] capture requested by {req.get('device', '?')}")
            try:
                self.on_capture_request(req)
            except Exception as e:  # an exception here would kill paho's network thread
                print(f"[mqtt] capture handler error: {e}")

    def publish_result(self, result):
        payload = json.dumps({"device": self.cfg["device_id"], **result})
        self.client.publish(self.cfg["topic_result"], payload, qos=1)

    def publish_climate(self, snap):
        payload = json.dumps({"device": self.cfg["device_id"], "time": int(time.time()), **snap})
        self.client.publish(self.cfg.get("topic_climate", "plant/climate"), payload, qos=0, retain=True)
