# Plan: edge-first (the Pi does all computing, the cloud only gets a few updates)

Goal set by the user on 2026-11-19: the Raspberry Pi runs every prediction and decision itself with a
quantized / fine-tuned model; only a few images and small summaries are sent to the cloud.

## Today vs target
| Job | Today | Target |
|---|---|---|
| Disease prediction | Pi (INT8 MobileNetV2, 38 classes) | Pi, **fine-tuned on field photos** + "not a leaf" / "unknown plant", INT8 |
| "Is it sure?" second check | laptop (cloud-inference, FP32 + 6 views) | **Pi** (same INT8 model, several views of the photo) |
| Treatment measures (KB) | laptop (advisor) | **Pi** (same code + kb.yaml) |
| Needs water? / watering detected | laptop (irrigation) | **Pi** |
| Alerts | laptop (notifier) | **Pi** (page banner, NodeMCU LEDs) |
| Photos to the cloud | every photo | **only some** (disease, not sure, 1 in N healthy) |
| MQTT to the cloud | everything (`plant/#`) | **results + summaries only**, climate/soil thinned out |
| Cloud | logic + storage + dashboard | **storage + dashboard** (+ optional audit of uploaded photos) |

## Steps
| # | Work | Estimate |
|---|---|---|
| E1 | On-device check: INT8 model on several views of the photo; measure gain + Pi latency, pick the number of views | 30 min |
| E2 | Advisor on the Pi: reuse `cloud/services/advisor/logic.py` + `kb.yaml` (one source of truth); full measures on the Pi page offline; Pi publishes `plant/advice` | 45 min |
| E3 | Irrigation + alerts on the Pi: reuse `irrigation/logic.py`, `notifier/rules.py`; soil from the NodeMCU over the Pi's own broker | 45 min |
| E4 | Upload policy: config `cloud.upload` (which photos, how often climate/soil); MQTT bridge only forwards result topics; cloud advisor/irrigation/notifier/inference moved to an optional compose profile | 45 min |
| E5 | Better edge model: fine-tune pv38 MobileNetV2 on all PlantDoc field photos (test photos held out by checksum) + "not a leaf" / "unknown plant", INT8 quantize, evaluate lab/field, benchmark on the Pi, install only if better | 1.5-2.5 h (mostly training; laptop CPU with the charger in, or Colab) |
| E6 | Docs, tests, e2e check updated for edge-first | 30 min |

Rule kept from Phase 5: a disease is reported only when the result is sure (now: sure on the Pi's own multi-view check).

## Learning loop: Pi photos → cloud → retrain → new model back to the Pi
Added by the user on 2026-11-19: captured images should be used to fine-tune the model, and the improved model
should go back to the Pi.

```
 Pi: capture → predict (INT8) → upload policy picks useful photos (not sure, disease, 1 in N healthy)
        │  HTTP upload (queued offline)                                   ▲ download + verify + self-test + swap
        ▼                                                                 │  (keeps the old model for rollback)
 Cloud: photo store → REVIEW QUEUE on the dashboard → person confirms / corrects the label
        → labelled pool (postgres `labels`) → retrain job (fine-tune + INT8 + evaluate on held-out tests)
        → only if better: model registry v2, v3, ... → MQTT retained plant/model/available {version, sha256, url}
```

* **People label, the model never trains on its own guesses.** Photos with the model's own label would teach it its
  own mistakes (confirmation bias). The dashboard shows each uploaded photo with the Pi's top 3; one click confirms,
  or picks the right class, "not a leaf", "unknown plant" or "skip / unclear".
* **Uncertain photos first** (active learning): "not sure" and low-confidence photos are at the top of the queue,
  because they teach the model the most.
* **Retrain** (laptop CPU or Colab): the E5 fine-tune script, run on base data + all confirmed field photos, then INT8.
  Triggered by a dashboard button, or automatically after N new labels (default 50).
* **Guard:** the new model goes to the Pi only if it beats the current one on the fixed held-out lab + field test
  sets (same rule as `install_cloud_v2.sh`): fewer confident-wrong answers, accuracy not lower. Labelled photos are
  split, so a share of them also becomes new test data.
* **Over-the-air update:** the gateway serves `/api/models/<version>.tflite`; `plant/model/available` is retained, so a
  Pi that was offline gets it when it reconnects. The Pi checks the SHA-256, runs a self-test on sample photos, swaps
  the model without stopping the app, and reports `plant/model/status`. If the self-test fails it keeps the old
  model; the dashboard has a "roll back" button.

| # | Work | Estimate |
|---|---|---|
| E7 | Review queue + labels table + dashboard labelling UI | 1 h |
| E8 | Retrain job + model registry + guard (reuses E5) | 1 h + training time |
| E9 | OTA update on the Pi (download, verify, self-test, hot swap, rollback, status) | 1 h |
