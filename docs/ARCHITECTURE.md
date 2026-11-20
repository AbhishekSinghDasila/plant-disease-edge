# Architecture: plant disease edge + cloud system

A leaf photo is diagnosed **on the Raspberry Pi** (edge) in about 66 ms, even with no network. Every result then
flows to a **cloud tier** (Docker on the laptop) that adds a second opinion from a bigger model, treatment measures,
watering decisions, alerts, storage and a live dashboard.

```
 FIELD / POT                         EDGE: Raspberry Pi 4 (pi-node-01)            CLOUD: laptop, docker-compose (cloud/)
 ───────────                         ─────────────────────────────────            ──────────────────────────────────────
 Pi Camera ──────────────────────▶  app/server.py  (Flask :8000, systemd)
                                     ├ photo check (blur / leaf)       HTTP POST   gateway (FastAPI :8080)
                                     ├ pv38_int8.tflite (38 classes)  ─────────▶   ├ /api/images → volume `images`
 DHT11 (GPIO4, not wired yet) ─────▶ ├ climate reader                   photos     ├ dashboard + live SSE stream
                                     └ captures/ + outbox/ (queue)                 └ /api/camera/* → proxied to the Pi
 NodeMCU leaf node                                                                  │
  ├ button ─── plant/capture ──┐    Mosquitto (Pi) ═══ MQTT bridge plant/# ═══▶  Mosquitto (laptop :1883)
  ├ LEDs  ◀── plant/advice ────┤    (store-and-forward, 10 000 msgs)               │
  └ soil probe A0 ─ plant/soil ┘                                                    ├ ingest ─────────▶ PostgreSQL 16 (volume pgdata)
     (sketch ready, not uploaded)                                                   ├ cloud-inference (FP32 + 6-view TTA)
                                                                                    ├ advisor (KB, 38 classes)
                                                                                    ├ irrigation (needs water? watering seen?)
                                                                                    ├ notifier (alerts, optional Telegram)
                                                                                    └ soil-sim (until a real probe sends)
```

## One photo, step by step
1. **Capture**: Pi page, dashboard camera card (`POST /api/camera/capture` → MQTT `plant/capture`), or NodeMCU button.
2. **Edge** (Pi, ~66 ms when the Pi is idle): photo check (blurry / no leaf → "not sure"), INT8 model → top 3, confidence;
   below 0.80 → "not sure". Saved in `captures/` + `log.csv`, published on `plant/diagnosis`, photo uploaded to the gateway
   (queued in `captures/outbox/` while the cloud is unreachable).
3. **Cloud second opinion** (`cloud-inference`, ~35 ms): the gateway announces the photo on `plant/image`; the same
   38-class network at FP32 sees 6 versions of the photo (mirror, flip, zooms), averages them → `plant/cloud_diagnosis`.
4. **Verdict + measures** (`advisor`): a disease is reported only if edge and cloud are **both sure and agree**.
   Otherwise "not sure / models disagree / cloud check failed" with retake tips and no treatment. Until the cloud answers,
   the advice is marked `pending_cloud`; after 20 s without an answer the edge result alone becomes final.
   → `plant/advice` (dashboard "What to do", Pi page, NodeMCU LEDs).
5. **Alert** (`notifier`): disease of medium+ severity (10 min cooldown per disease), urgent water, Pi offline, silent
   sensor, watering detected → `plant/alert` (+ Telegram if configured).
6. **Storage** (`ingest`): every MQTT message → `telemetry`; typed copies in `diagnoses`, `advice`, `alerts`,
   `irrigation_events`; photos in the `images` volume, indexed in `images`.

## Watering
* **Soil moisture** comes from `plant/soil`. Today the source is `soil-sim` (a model of a drying pot, 60× real time).
  The NodeMCU sketch reads a capacitive probe on A0 (dry ≈ 820, wet ≈ 380 → 0-100 %) and publishes the same message with
  `simulated: false`; `soil-sim` pauses by itself while real readings arrive.
* **Needs water?** (`irrigation`): per-crop band from `crops.yaml` (dry / target / wilting), crop = latest confident
  diagnosis, hysteresis between dry and target, urgency raised by heat (≥ 32 °C) or dry air (VPD ≥ 1.6 kPa), 2 h trend →
  "dry in ~N h". → retained `plant/irrigation`.
* **Watering detected**: soil rises ≥ 8 points between two readings → `plant/irrigation/watered` (works for hand
  watering, rain or a pump). The dashboard's 💧 button only tells the simulator that water was added.

## MQTT topics
| Topic | From → to | Notes |
|---|---|---|
| `plant/capture` | dashboard / NodeMCU → Pi | take a photo now |
| `plant/diagnosis` | Pi → all | edge result (class_id, status, confidence, top3, quality) |
| `plant/status` | Pi → all | retained online/offline (last will) |
| `plant/climate` | Pi → all | DHT11 temp / humidity / VPD / fungal risk |
| `plant/image` | gateway → cloud-inference | a new photo is stored |
| `plant/cloud_diagnosis` | cloud-inference → advisor, dashboard | second opinion (+ view agreement) |
| `plant/advice` | advisor → dashboard, Pi, NodeMCU | retained final verdict + measures |
| `plant/alert` | notifier → dashboard | info / warning / critical |
| `plant/soil` | soil-sim or NodeMCU → irrigation | moisture_pct, raw, simulated |
| `plant/irrigation` | irrigation → dashboard | retained need_water, urgency, reason, last_watered |
| `plant/irrigation/watered` | irrigation → notifier, dashboard | retained detected watering |
| `plant/irrigation/cmd` | dashboard → soil-sim (later: pump) | `{"action":"water"}` |
| `plant/bridge/pi-node-01` | Pi broker → all | bridge connected 1 / 0 |

## Design decisions
* **Two tiers, edge first.** The farmer gets an answer from the Pi even with no internet. The cloud only adds
  things that need more compute or history (second model, KB, trends, alerts, storage). MQTT QoS 1 + the bridge's
  persistent queue + the photo outbox give store-and-forward (tested: queued result arrived 5 s after the cloud came back).
* **The Pi is not dockerised.** Raspbian Bullseye 32-bit (armhf) with the legacy camera stack and GPIO access:
  containers would need privileged device access and armhf images for little gain. It runs as one systemd service.
* **One Dockerfile for all cloud services** (build arg `SERVICE`) + `common/` (MQTT with reconnect, DB pool). Each
  service is small, single-purpose and restarts on its own (`restart: unless-stopped`).
* **Cloud model on LiteRT, not TensorFlow**: same FP32 network, 256 MB image instead of ~1 GB.
* **"Both models must agree"** was chosen from measurements (`scripts/eval_cloud_tta.py`, `results/cloud_tta.csv`):
  on 663 real field photos it cuts confident-wrong answers from 56 % (edge alone) to 30 %; on lab photos from 1.3 % to
  0.2 %, while 93 % of lab photos still get an answer.
* **Pure logic, thin services.** Rules live in pure modules (`irrigation/logic.py`, `advisor/logic.py`,
  `notifier/rules.py`, `cloud_inference/tta.py`) with unit tests; `main.py` files only wire MQTT/DB.

## Limits (be honest about these)
* **Field accuracy is ~20 %** for the current models (lab ~98 %). The agree rule makes the system say "not sure" more
  often instead of being confidently wrong, but it cannot make it right. Fix: Phase 5 v2
  (`notebooks/cloud_v2/train_cloud_v2_colab.ipynb`: field photos + "not a leaf" + "unknown plant" classes).
* Only 14 crops; any other plant gets a (wrong) crop label from the edge model until v2 is installed.
* Soil moisture is **simulated** until the NodeMCU probe is wired; the DHT11 is not wired yet.
* No authentication or TLS on MQTT / HTTP: LAN demo only.
* The laptop IP comes from DHCP; if it changes, rerun `pi/install_bridge.sh <ip>` and update `cloud.gateway_url`.
* docker-compose 1.29: after a rebuild run `docker-compose rm -fs <service>` before `up -d` ('ContainerConfig' bug).

## Checking it
`.venv/bin/python scripts/e2e_check.py` runs the whole verification (unit tests, containers, live data, Pi, one photo
per crop through edge → cloud → advice; `--offline-test` adds store-and-forward). Details per phase: `HARDWARE_LOG.md`.
