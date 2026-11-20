# Cloud side (laptop, docker-compose)

```bash
cd cloud
docker-compose up -d --build        # build + start (containers restart by themselves after a reboot)
docker-compose ps                   # all Up?
docker-compose logs -f gateway      # logs of one service
docker-compose down                 # stop (data stays in volumes; `down -v` deletes it)
```
Dashboard: **http://localhost:8080** (from other devices: http://<laptop-ip>:8080).

| Service | What it does | Port |
|---|---|---|
| mosquitto | cloud MQTT broker; the Pi's broker bridges `plant/#` to it | 1883 |
| postgres | storage (`db/init.sql`: telemetry, images, diagnoses, advice, irrigation_events, alerts) | 127.0.0.1:5432 |
| ingest | every `plant/#` message → postgres | – |
| gateway | REST API + dashboard + photo upload + live SSE stream (`services/gateway/main.py`) | 8080 |
| soil-sim | **simulated** soil probe → `plant/soil` every 30 s (60× speed); `ENABLED=0` once the real NodeMCU probe exists | – |
| irrigation | "needs water?" per crop band + trend + VPD → `plant/irrigation`; detects watering (moisture jump ≥ 8 points) → `plant/irrigation/watered` | – |
| advisor | diagnosis (+ cloud 2nd opinion + climate) → verdict + measures from `services/advisor/kb.yaml` (38 classes) → `plant/advice` | – |
| cloud-inference | second opinion on every uploaded photo: 38-class FP32 model (`../models`, mounted) + 6-view test-time augmentation (`services/cloud_inference/tta.py`) → `plant/cloud_diagnosis`. Swap the model with `MODEL_PATH` | – |
| notifier | alerts → `plant/alert`: disease (medium+), urgent water, Pi offline, silent sensor, watering detected. Telegram if `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID` are set | – |

All Python services share one `Dockerfile` (build arg `SERVICE`) and the `common/` package
(MQTT client with reconnect, DB pool, settings from env vars).

Pi side: `bash pi/install_bridge.sh <laptop-ip>` (MQTT bridge) and `config.yaml` → `cloud.gateway_url` (photo upload,
queued in `captures/outbox/` while the cloud is unreachable).

Quick checks:
```bash
docker exec cloud_mosquitto_1 mosquitto_sub -t 'plant/#' -v                      # watch all messages
docker exec cloud_postgres_1 psql -U plant -c "select ts, source, class_id, status from diagnoses order by ts desc limit 5"
curl localhost:8080/api/latest
```

## Checking everything
`.venv/bin/python scripts/e2e_check.py` (from the project root): unit tests, containers, live data, Pi, one photo per
crop through edge → cloud → advice. `--offline-test` adds the store-and-forward test, `--no-tests --no-crops` is a
40-second health check. Architecture and design decisions: `docs/ARCHITECTURE.md`.

## Phase 5 v2: better cloud model (Colab GPU)
1. Open `notebooks/cloud_v2/train_cloud_v2_colab.ipynb` in Google Colab (File → Upload notebook).
2. Runtime → Change runtime type → T4 GPU; Runtime → Run all (~1.5-2 h). It downloads `cloud_v2.tflite` and
   `cloud_v2.labels.json`.
3. `bash scripts/install_cloud_v2.sh ~/Downloads`: evaluates v1 vs v2 on the held-out lab + field photos and switches
   `cloud-inference` to v2 only if it is better (`--force` to switch anyway, `--rollback` to go back).
The notebook source is `notebooks/cloud_v2/train_cloud_v2.py`; rebuild the .ipynb with `python scripts/make_colab_notebook.py`.

