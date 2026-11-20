"""DHT11 / DHT22 temperature + humidity sensor on the Raspberry Pi.

A background thread reads the sensor every few seconds and keeps the last good reading.
DHT sensors often fail a single read (timing-sensitive), so failures are normal and only
counted; a reading older than `stale_after_s` is reported as stale.

Backends (first one that imports wins):
  1. Adafruit_DHT                 (legacy library, already on the lab Pi, Bullseye)
  2. adafruit_dht (CircuitPython)  pip install adafruit-circuitpython-dht  (Bookworm)
No backend or no sensor -> the app runs normally without climate data.

Wiring (DHT11 module): VCC -> pin 1 (3.3 V), DATA -> pin 7 (GPIO4), GND -> pin 6.

    python app/climate.py            # test: prints a reading every 3 s
"""
import threading
import time


def _open_backend(kind, gpio):
    """Return a read() function giving (temp_c, humidity) or (None, None)."""
    try:
        import Adafruit_DHT
        sensor = Adafruit_DHT.DHT22 if kind.upper() in ("DHT22", "AM2302") else Adafruit_DHT.DHT11

        def read():
            h, t = Adafruit_DHT.read(sensor, gpio)  # single attempt; the thread retries
            return t, h
        return "Adafruit_DHT", read
    except ImportError:
        pass
    try:
        import adafruit_dht
        import board
        pin = getattr(board, f"D{gpio}")
        dev = (adafruit_dht.DHT22 if kind.upper() in ("DHT22", "AM2302") else adafruit_dht.DHT11)(pin)

        def read():
            try:
                return dev.temperature, dev.humidity
            except RuntimeError:  # checksum / timing error: normal for DHT sensors
                return None, None
        return "adafruit_dht", read
    except (ImportError, AttributeError, NotImplementedError, RuntimeError):
        pass
    return None, None


def fungal_risk(temp_c, humidity):
    """Rough hint: most tomato leaf fungi (late blight, leaf mold, early blight, septoria)
    spread fastest in long periods of high humidity at mild temperatures. General guidance only."""
    if temp_c is None or humidity is None:
        return None
    if humidity >= 85 and 15 <= temp_c <= 30:
        return "high"
    if humidity >= 70 and 10 <= temp_c <= 32:
        return "medium"
    return "low"


class Climate:
    def __init__(self, kind="DHT11", gpio=4, interval_s=5, stale_after_s=60, on_reading=None):
        self.kind, self.gpio = kind, gpio
        self.interval_s, self.stale_after_s = interval_s, stale_after_s
        self.on_reading = on_reading
        self.backend, self._read = _open_backend(kind, gpio)
        self.temp_c = self.humidity = None
        self.last_ok = 0.0
        self.reads = self.fails = 0
        self._stop = threading.Event()
        self._thread = None
        if self.backend:
            print(f"[climate] {kind} on GPIO{gpio} via {self.backend}")
        else:
            print("[climate] no DHT library found - running without temperature/humidity")

    @property
    def available(self):
        return self.backend is not None

    def start(self):
        if self.available and not self._thread:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def stop(self):
        self._stop.set()

    def read_once(self):
        t, h = self._read()
        self.reads += 1
        # DHT11 range is 0-50 C / 20-90 %; reject obvious garbage
        if t is None or h is None or not (-40 <= t <= 80) or not (0 <= h <= 100):
            self.fails += 1
            return False
        self.temp_c, self.humidity, self.last_ok = round(float(t), 1), round(float(h), 1), time.time()
        if self.on_reading:
            self.on_reading(self.snapshot())
        return True

    def _loop(self):
        while not self._stop.is_set():
            try:
                ok = self.read_once()
            except Exception as e:  # never let the sensor kill the app
                self.fails += 1
                ok = False
                print(f"[climate] read error: {e}")
            # a failed read is retried sooner (DHT11 needs >= 1-2 s between reads)
            self._stop.wait(self.interval_s if ok else 2.5)

    def snapshot(self):
        """Dict for the web page, MQTT and the CSV log."""
        if not self.available:
            return {"sensor": None}
        age = time.time() - self.last_ok if self.last_ok else None
        fresh = age is not None and age <= self.stale_after_s
        return {
            "sensor": self.kind,
            "temp_c": self.temp_c if fresh else None,
            "humidity": self.humidity if fresh else None,
            "fungal_risk": fungal_risk(self.temp_c, self.humidity) if fresh else None,
            "age_s": None if age is None else round(age, 1),
            "reads": self.reads,
            "fails": self.fails,
        }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--type", default="DHT11")
    ap.add_argument("--gpio", type=int, default=4)
    a = ap.parse_args()
    c = Climate(a.type, a.gpio)
    if not c.available:
        raise SystemExit("No DHT library (install Adafruit_DHT or adafruit-circuitpython-dht).")
    print("Ctrl+C to stop")
    while True:
        ok = c.read_once()
        s = c.snapshot()
        print(f"{time.strftime('%H:%M:%S')}  " + (
            f"{s['temp_c']} C  {s['humidity']} %RH  fungal risk: {s['fungal_risk']}" if ok
            else f"read failed ({c.fails}/{c.reads}) - normal now and then; always failing = check wiring"))
        time.sleep(3)
