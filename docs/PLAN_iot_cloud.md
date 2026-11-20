# Plan: Smart-farm IoT system (edge + cloud microservices) for plant_disease_edge

## Context
The single-app demo works end to end (Pi camera → INT8 model → web page → MQTT), but live tests showed
the limits: the model knows only 10 tomato classes, has no "unknown" output, and gives confident wrong
answers on unfamiliar photos (fixed partly by the photo check in `pde/quality.py`). The user now wants:
- a stronger pretrained model trained on more crops and data (chosen: **two-tier** edge + cloud model, **all 14 PlantVillage crops**)
- separate **Docker microservices** for climate/soil monitoring, "does the plant need water?", disease
  detected or not, and **what measures to take**
- a complete **IoT → edge → cloud** system, with everything recorded so work can resume next session.

User decisions: soil sensor **not available yet → simulated for now** (real NodeMCU A0 sensor later);
cloud = **docker-compose on the laptop**; crops = **all PlantVillage crops**.

## Architecture
```
FIELD (NodeMCU)           EDGE (Raspberry Pi 4, systemd: needs camera + GPIO)        CLOUD (laptop, docker-compose)
 button/LEDs ──MQTT──►  Mosquitto (Pi) ◄── plant-app: camera, photo check,   ══bridge══►  mosquitto
 soil probe A0 (later)     │               38-class INT8 model, DHT11                      │
                           │               └─HTTP image upload (async, retry)──►  gateway/dashboard (FastAPI :8080)
                           └ store-and-forward: the bridge queues while the cloud is offline     │
   ingest ─► postgres   irrigation   disease-advisor   cloud-inference   notifier   soil-sim
```
Edge keeps working with the cloud down: it still diagnoses locally and the Pi broker queues messages for the bridge.
Docker is cloud-only. The Pi stays on systemd because Docker on 32-bit Bullseye with camera + GPIO passthrough
adds risk for no gain. This choice is written down in the docs.

### MQTT topics (all JSON, prefix `plant/`)
| Topic | From → To | Purpose |
|---|---|---|
| `plant/diagnosis` | Pi → all | edge result; now also `crop`, `disease`, `quality`, `image_id` |
| `plant/climate` | Pi DHT11 | temp_c, humidity, VPD (retained) |
| `plant/soil` | soil-sim, later NodeMCU | moisture_pct, raw |
| `plant/cloud_diagnosis` | cloud-inference | second opinion for an `image_id` |
| `plant/advice` | disease-advisor | final verdict + measures |
| `plant/irrigation` | irrigation | need_water, urgency, reason, hours_to_dry (retained) |
| `plant/alert` | notifier | human-readable alerts (disease, water, device offline) |
| `plant/status`, `plant/capture` | unchanged | |

## Phase 1: Edge model upgrade to all 14 crops (38 classes)
- `scripts/import_pretrained.py`: add `--all-crops`. It exports the full 38-class model (no tomato slicing) to
  `models/pv38_fp32.tflite` and `models/pv38_int8.tflite` (INT8 calibration uses tomato lab images, which is fine
  for activation ranges), plus `models/pv38.labels.json`. Reuse the existing `inference_model()`/`check_same()` minus the `take`.
- New `pde/catalog.py`: 38 PlantVillage class ids → `{crop, disease, display, healthy}`. `Predictor`
  (`pde/predictor.py`) loads `<model>.labels.json` when present; otherwise it keeps today's tomato `SHORT_CLASSES`,
  so the research pipeline, evaluation and tests stay unchanged.
- `Prediction` gains `crop`, `disease`. The page shows "Tomato: Late blight" / "Apple: healthy".
  Advice for non-tomato classes comes from the advisor KB (`pde/catalog.py` holds short fallback lines for offline use).
- Benchmark `pv38_int8` on the Pi with the existing `scripts/benchmark_pi.py`. `config.yaml` app.model_path → pv38_int8.
- Keep the photo check (`pde/quality.py`).

## Phase 2: Cloud stack skeleton (`cloud/` folder, `docker-compose.yml`, compose file v3.8 for docker-compose 1.29)
- **mosquitto** (eclipse-mosquitto:2): listener 1883, persistence.
- **postgres** (postgres:16-alpine, volume `pgdata`). Tables: `telemetry(ts, device, kind, data jsonb)`,
  `diagnoses(...)`, `advice`, `irrigation_events`, `alerts`, `images`. Init SQL in `cloud/db/init.sql`.
- **ingest**: subscribes `plant/#` and writes every message to postgres.
- **gateway** (FastAPI + static dashboard, :8080): REST (`/api/latest`, `/api/history?kind=`, `/api/diagnoses`,
  `/api/advice/{class}`), `POST /api/images` (edge upload → volume `images`, publishes for cloud-inference),
  SSE `/api/stream` for live updates. Dashboard (vanilla JS + Chart.js from cdnjs): live temp/humidity/soil
  charts, "needs water?" card, latest diagnosis with photo, edge vs cloud verdict, measures, alerts list.
- Shared code `cloud/common/` (MQTT client with reconnect, DB helper, pydantic schemas, settings from env)
  built into one base image `plantfarm-base`; each service is a small `main.py` with its own Dockerfile `FROM` base.
- Pi Mosquitto bridge: `pi/mosquitto-bridge.conf.template` → `/etc/mosquitto/conf.d/bridge.conf`
  (`connection cloud`, `address <laptop-ip>:1883`, `topic plant/# both 1`, `cleansession false`,
  `queue_qos0_messages true`), installed by a new `pi/install_bridge.sh <laptop-ip>`.
- Pi app: after each diagnosis, a background uploader thread POSTs the image + metadata to the gateway, with a
  retry queue on disk (`captures/outbox/`). Configured in `config.yaml` → `cloud.gateway_url`, `cloud.enabled`.

## Phase 3: Irrigation ("does the plant need water?")
- **soil-sim**: realistic drying curve (faster when hot/dry, using the latest `plant/climate`), watering events
  reset it. Publishes `plant/soil` every 30 s. Disabled by an env flag when the real sensor arrives.
- **irrigation**: per-crop moisture thresholds (`cloud/irrigation/crops.yaml`, a dry/ok band for the 14 crops).
  Rule: need water if moisture < dry threshold; urgency raised by high VPD (computed from temp + RH) and heat.
  Hysteresis to avoid flapping. `hours_to_dry` by linear trend over the last 2 h of soil readings from postgres.
  Publishes `plant/irrigation` (retained). Optional `plant/irrigation/cmd` for a future relay/pump (documented, off).
- `iot/nodemcu_soil_node/` sketch: capacitive sensor on A0 → `plant/soil` (dry/wet calibration constants).
  Ready for when the sensor is bought; wiring goes in HARDWARE_LOG.

## Phase 4: Disease detection verdict + measures
- **disease-advisor**: knowledge base `cloud/advisor/kb.yaml` covering all 38 classes, each with: crop, pathogen
  type (fungal/bacterial/viral/pest/none), symptoms, immediate actions, organic/cultural measures, chemical options
  (generic active ingredients only, "follow the local label/extension advice"), prevention, spread risk, severity.
  Combines edge + cloud results into a final verdict (agree → confident; disagree or photo-check fail → "uncertain,
  retake / check by eye") and adds climate risk (e.g. high RH raises fungal urgency). Publishes `plant/advice`,
  serves `GET /advice/{class}`.
- **notifier**: rules → `plant/alert` + DB: disease found (severity ≥ medium), needs water (urgent),
  device offline (no `plant/status`/telemetry for N min), sensor failures. Dashboard shows them; email/Telegram
  hooks stubbed behind env vars (off).
- Pi page: shows the advisor's measures when the cloud is reachable, else the offline fallback line.

## Phase 5: Cloud inference (bigger model, second opinion)
- **cloud-inference** (Python + TF CPU, ~1 GB image): subscribes to new images; runs the configured model; publishes
  `plant/cloud_diagnosis`. v1 model = the 38-class FP32 Keras model (`models/pretrained_mnv2_38class.keras`) at full
  precision with test-time augmentation (flip + centre crop averaging).
- v2 (needs the user's Colab GPU): new notebook `notebooks/train_cloud_model_colab.ipynb` fine-tunes EfficientNet-B1
  (Keras) on full PlantVillage (38 classes + `Background_without_leaves` from the augmented Mendeley set → "no leaf"
  class) + PlantDoc field photos mapped to the same classes, with strong augmentation (reuses `pde/data.py` augment
  functions). Exports `cloud_effb1.keras` + labels. Drop it into `cloud/models/`, set the env var, restart the service.
  The evaluation script reports lab vs field accuracy, like the existing results.

## Phase 6: Docs, state, tests
- `cloud/README.md`: how to run (`docker-compose up -d --build`), services table, ports, topics, env vars.
- `docs/ARCHITECTURE.md`: diagram, data flow, design decisions (why the Pi is not dockerised, why two tiers, limits).
- `HARDWARE_LOG.md`: keep the per-step log; the status table gets cloud rows. New top section **"Resume here"**:
  current phase, what is running where, next action.
- Memory file updated with the cloud layout pointer.
- Tests: pytest unit tests per service (irrigation rules, VPD, advisor verdict merge, KB covers all 38 classes,
  catalog/labels consistency); keep the existing `tests/test_pipeline.py` passing.

## Verification (end to end)
1. `pytest -q tests/ cloud/tests/` on the laptop.
2. `docker-compose -f cloud/docker-compose.yml up -d --build`, then `docker-compose ps` shows all services healthy.
3. `mosquitto_sub -h localhost -t 'plant/#' -v` shows soil-sim data and an `irrigation` decision within 1 min.
4. On the Pi: `bash pi/install_bridge.sh 10.93.8.158`. Climate/status messages appear on the laptop broker;
   stopping the laptop stack, then restarting it, shows queued messages arriving (store-and-forward).
5. Website on the Pi: take a photo of a sample leaf on a phone screen → edge result on the Pi page; the cloud dashboard
   `http://localhost:8080` shows the photo, edge + cloud verdicts, measures, and an alert for a disease.
6. Upload samples of other crops (apple, potato...) → correct crop + disease names.
7. Pi benchmark for pv38_int8 recorded in `results/pi_benchmark.csv` and the log.

## Status (2026-11-19)
Phases 1-4, 5 v1 and 6 done; 5 v2 notebook ready and smoke-tested, waiting for a Colab GPU run.
See `HARDWARE_LOG.md` for the per-step log and `docs/ARCHITECTURE.md` for the final design.

## Order of work
Phase 1 → 2 → 3 → 4 → 5 (v1) → 6, logging each step in HARDWARE_LOG.md as it is done.
v2 cloud model training waits for the user to run the Colab notebook.
