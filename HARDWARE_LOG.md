# Hardware Log: plant_disease_edge

This is a running log of the hardware work: what is connected where, what was installed, what
worked and what failed. **Add to it after every step.** Anyone (or Claude in a new session) can
read this file to pick up where we left off.

---

## 0. RESUME HERE (read first after a restart / new session)

**Last updated:** 2026-11-19 08:00.
**Done:** Phases 1, 2, 3 of `docs/PLAN_iot_cloud.md` + live camera card on the cloud dashboard + quick fixes (threshold 0.80, top 3 + low-confidence badge) + **Phase 4** (advisor, notifier, watering detection, "How it works" card) + **Phase 5 v1** (cloud second opinion) + **Phase 6** (docs, tests, e2e check). **Phase 5 v2** notebook ready, needs a Colab GPU run.
**Next, in order:**
1. ~~Quick fixes~~ ✅ 2026-11-19.
2. ~~Phase 4~~ ✅ 2026-11-19.
3. ~~Phase 5 v1~~ ✅, ~~Phase 6~~ ✅ 2026-11-19.
4. **User:** test everything (`scripts/e2e_check.py` + the websites). **Pi camera is blocked** by another program
   (`/home/pi/plant-edge/pi/live.py`), see the 08:00 entry.
5. **User:** Phase 5 v2: run `notebooks/cloud_v2/train_cloud_v2_colab.ipynb` on Colab (T4, ~1.5-2 h), then
   `bash scripts/install_cloud_v2.sh ~/Downloads`.
6. **User (hands-on):** wire DHT11 (section 3) and the soil probe (sketch header), put the Wi-Fi password in the
   NodeMCU sketch and upload it.

What is running, and how to bring it back after a laptop restart:
0. **Quick check of everything:** `.venv/bin/python scripts/e2e_check.py` (~3 min; `--no-tests --no-crops` = 40 s).
1. Laptop cloud stack: `cd ~/Documents/plant_disease_edge/cloud && docker-compose up -d` (it normally restarts by itself),
   then `docker ps --filter name=cloud_` should show 9 containers. Dashboard: http://localhost:8080
2. Pi: `ssh pi@raspberrypi.local 'systemctl is-active plant-app mosquitto'` (both start on boot). Pi page: http://raspberrypi.local:8000
3. Laptop IP still 10.93.8.158? (`ip -4 addr show wlo1`). If not: on the Pi run `bash pi/install_bridge.sh <new-ip>` and change
   `cloud.gateway_url` in `config.yaml`, rsync it to the Pi, then `sudo systemctl restart plant-app`.
4. Keep the **charger plugged in** before training or builds (the laptop powers off under heavy load on battery).

Remaining plan, with rough times:
| Phase | Work | Estimate |
|---|---|---|
| user | wire DHT11 (section 3) + soil probe, Wi-Fi password in the NodeMCU sketch, upload it, test with real leaves | ~30-45 min hands-on |
| 5 v2 | run the ready notebook on Colab (EfficientNet-B1, field + "not a leaf" + "unknown plant"), then the install script | ~1.5–2 h GPU + 10 min |

### How to check each phase on the real websites
Open two tabs: **Pi page http://10.93.8.209:8000** (camera, edge model) and **cloud dashboard http://localhost:8080**.
- **Camera is on both sites:** dashboard card "Live camera" (👁 Live preview, 📷 Take photo & detect, via MQTT) or the Pi page.
- **Phase 1 (14 crops):** Pi page (or dashboard camera card) → 👁 Live preview → hold a leaf (or a leaf photo on a phone) 40–60 cm away → 📷 Take photo.
  The result reads "Crop: Disease" with a confidence %. Upload works too (⬆ Upload photo, e.g. `samples/` or `data/pv_allcrops/`).
- **Phase 2 (cloud):** within ~5 s the same photo + result appears in the dashboard's "Latest diagnosis" and history table.
  Offline test: `cd cloud && docker-compose stop gateway mosquitto`, take a photo (the Pi still answers), `docker-compose start mosquitto gateway`;
  the photo arrives within ~1 min.
- **Phase 3 (water):** the dashboard's "Soil moisture" and "Does the plant need water?" update every 30 s, and the green line on the chart
  falls (simulated, 1 h per minute). Press 💧 I watered it → moisture jumps up.
- **Phase 4:** after Take photo, the "What to do" card shows verdict, severity and measures (Pi page too, under the result);
  a disease of medium+ severity shows under "Alerts". 💧 I watered it → "Last watering … detected" + an info alert.
  The "How it works" card shows where data is measured / predicted / stored, with live row counts.
- **Phase 5:** the dashboard shows "☁ Cloud model (FP32 + 6-view check): … · n/6 views agree" under the edge result; "What to do"
  shows ⏳ until the cloud check is in, then ☁✓ (agree) or "Models disagree / cloud check failed" (no measures).

---

## 1. Current status (update this first)

| Item | Status |
|---|---|
| Laptop ↔ Pi network | ✅ `raspberrypi.local` (10.93.8.209), user `pi` |
| Passwordless SSH (laptop → Pi) | ✅ laptop key in Pi `~/.ssh/authorized_keys`; `sudo` works without password |
| Project on Pi (`/home/pi/plant_disease_edge`) | ✅ same code as laptop (copy changes with rsync, section 4) |
| Pi software (Mosquitto, tflite-runtime 2.13, Flask, paho, OpenCV, Adafruit_DHT) | ✅ |
| Pretrained model on Pi (`pretrained_mnv2_int8.tflite`) | ✅ 66 ms/photo, 15 FPS; same answers as laptop |
| Pi Camera | ✅ works after power-cycle (20:27). Fixed focus: blurry closer than ~40 cm |
| Photo check (blur / leaf present) | ✅ `pde/quality.py`, on by default (`config.yaml` → `app.photo_check`) |
| DHT11 sensor | ⏳ code ready (`app/climate.py`), not wired yet (wiring in section 3) |
| Web app `http://raspberrypi.local:8000` | ✅ runs as systemd service `plant-app` (starts on boot) |
| MQTT (Mosquitto on Pi) | ✅ `plant/status` online, `plant/capture` received |
| NodeMCU Node 1 (button + LEDs) | ✅ board tested, ⏳ leaf-node sketch not uploaded |
| Edge model: 38 classes / 14 crops (`pv38_int8.tflite`) | ✅ Phase 1 done, 66 ms on Pi |
| Cloud stack (laptop docker-compose, `cloud/`) | ✅ Phases 2–5 v1: mosquitto, postgres, ingest, gateway :8080, soil-sim, irrigation, advisor, notifier, cloud-inference; Pi bridge + photo upload. ⏳ Phase 6 next |
| Laptop power-offs | ⚠ heavy CPU on battery; keep the charger in (see 20:58 entry) |

**Next step:** see the last entry in section 5.

---

## 2. Hardware inventory

| Part | Details |
|---|---|
| Laptop | Ubuntu, Wi-Fi IP 10.93.8.158 (DHCP, may change), project in `~/Documents/plant_disease_edge` |
| Raspberry Pi | **Pi 4 Model B 4 GB**, Raspbian 11 Bullseye **32-bit (armhf)**, 64-bit kernel, Python 3.9.2. Hostname `raspberrypi`, user `pi`, IP 10.93.8.209 (DHCP), monitor attached |
| Pi Camera | ribbon cable to CAMERA port. Pi is in **legacy camera mode** (`start_x=1` in `/boot/config.txt`), so picamera2/libcamera can't see it. The app uses OpenCV on `/dev/video0` (bcm2835-v4l2) with **MJPG** format. `raspistill` also works. Do NOT switch off legacy mode: other lab scripts in `~` use it |
| DHT11 | temperature + humidity sensor (blue). Range 0–50 °C ±2 °C, 20–90 % RH ±5 % |
| NodeMCU Node 1 | ESP8285 (ESP8266 family), MAC `5C:CF:7F:A3:FF:C2`, flash 4 MB, board setting "NodeMCU 1.0 (ESP-12E)" |
| USB cable | first cable was faulty (error -71, drop-outs); use the working one |

### Model used on the Pi
`models/pretrained_mnv2_int8.tflite` (pretrained MobileNetV2, quantized to INT8).
Laptop evaluation (`results/summary.csv`): lab photos 98.8 % accuracy, field photos 29.1 %.
Expect good results on clean single-leaf photos and weak results on messy field shots.

---

## 3. Wiring

### DHT11 → Raspberry Pi
Use **3.3 V, not 5 V.** The Pi's GPIO pins are 3.3 V only, and 5 V on the data line can damage them.

**3-pin module (sensor on a small board, pins marked `+ out -` or `VCC DATA GND`):**

| DHT11 module pin | Pi physical pin | Pi name |
|---|---|---|
| `+` / VCC | **pin 1** | 3.3 V |
| `out` / DATA | **pin 7** | GPIO4 |
| `-` / GND | **pin 6** | GND |

The module already has a pull-up resistor, so no extra resistor is needed.

**Bare 4-pin sensor (no small board):** looking at the grille, pins from left to right are
1 VCC → Pi pin 1, 2 DATA → Pi pin 7, 3 not connected, 4 GND → Pi pin 6.
Add a **10 kΩ resistor between DATA and VCC**.

Pi header orientation: pin 1 is the corner pin nearest the SD card slot, on the inner row.
Odd pins (1, 3, 5, 7…) are on the inner row, even pins (2, 4, 6…) on the outer edge row.

```
          3.3V  (1) (2)  5V
          GPIO2 (3) (4)  5V
          GPIO3 (5) (6)  GND   ◄── DHT GND
 DHT DATA ► GPIO4 (7) (8)  GPIO14
            ...
   DHT VCC ► pin 1 (3.3V)
```

**Turn the Pi off before wiring:** run `sudo poweroff`, wait for the green LED to stop, then unplug power.

### Pi Camera
Ribbon cable in the CAMERA port (not DISPLAY). On a Pi 4 the blue side faces the USB/Ethernet
ports. Check with `rpicam-hello --list-cameras` (older OS: `libcamera-hello --list-cameras`).

### NodeMCU
See the README section "NodeMCU (ESP8266) button + LED node" (D5 green, D6 red, D7 yellow, D3 button).

---

## 4. Useful commands

```bash
# from the laptop
ssh pi@raspberrypi.local
rsync -av --exclude .venv --exclude data --exclude __pycache__ \
  ~/Documents/plant_disease_edge/ pi@raspberrypi.local:~/plant_disease_edge/

# laptop → Pi: copy code changes, then restart the app
rsync -a ~/Documents/plant_disease_edge/{app,pde,config.yaml,HARDWARE_LOG.md} pi@raspberrypi.local:~/plant_disease_edge/
ssh pi@raspberrypi.local 'sudo systemctl restart plant-app'

# on the Pi: the app runs as a service (installed with: bash pi/install_service.sh)
sudo systemctl status plant-app        # running?
sudo systemctl restart plant-app       # after changing code/config
sudo systemctl stop plant-app          # before running server.py by hand
journalctl -u plant-app -f             # live logs
cd ~/plant_disease_edge && .venv/bin/python -u app/climate.py    # DHT11 test (stop the service first)
raspistill -n -t 1000 -o /tmp/t.jpg    # camera test (stop the service first)
dmesg | grep -i camera                 # "Failed enabling camera" = power-cycle the Pi

# web API (from any device on the Wi-Fi)
curl http://raspberrypi.local:8000/api/status          # camera / mqtt / climate
curl -X POST http://raspberrypi.local:8000/api/capture # take photo + predict
mosquitto_pub -h raspberrypi.local -t plant/capture -m '{"device":"test"}'   # same, via MQTT
mosquitto_sub -h raspberrypi.local -t 'plant/#' -v   # watch all MQTT messages (from laptop too)
```

---

## 5. Step-by-step log (newest at the bottom)

### 2026-11-17, ESP8266 Node 1 board test
- The first USB cable caused `error -71`. A working cable fixed it.
- Uploaded `Documents/setup/esp8266_board_test` at 9600 baud. Chip ID, MAC and Wi-Fi scan
  (9 networks) were printed and the LED blinks. **Passed.**
- The serial port needs `sudo chmod a+rw /dev/ttyUSB0` after every replug, until
  `sudo usermod -aG dialout $USER` is run and you log out and back in.

### 2026-11-17, Pi found on the network
- The Pi answers at `raspberrypi.local` → 10.93.8.209, and SSH (port 22) is open.
- SSH asks for a password. **Next:** run `ssh-copy-id pi@raspberrypi.local` on the laptop
  (one-time) so Claude can install things without a password.
- DHT11 is not wired yet. Wiring plan is in section 3 (pin 1, pin 7 = GPIO4, pin 6).

### 2026-11-17 20:12, SSH access and Pi inspection
- `ssh-copy-id` typed with `!` in Claude can't ask for a password. Instead the laptop's public key
  was pasted into `~/.ssh/authorized_keys` on the Pi from a logged-in terminal. ✅
- Found the Pi already set up from earlier today: project in `~/plant_disease_edge`, `.venv` (with
  system site-packages), both pretrained models, and `results/pi_benchmark.csv`:
  INT8 2.8 MB, **65.8 ms, 15.2 FPS, 36 MB RAM**; FP32 9.1 MB, 133.6 ms, 7.5 FPS, 58 MB.
- Old lab folder `~/dht11_project` already read a DHT11 on GPIO4 with `Adafruit_DHT` (installed in `~/.local`).

### 2026-11-17 20:13, camera
- `rpicam-hello` not installed, and `libcamera-hello` says "No cameras available". Reason: legacy camera mode.
  `vcgencmd get_camera` → `supported=1 detected=1`, and `raspistill` takes photos. ✅
- The app's OpenCV fallback got a flat `(1, 3686400)` buffer (default BGR3 format).
  **Fix in `app/camera.py`:** open with `CAP_V4L2` and set `MJPG`, which gives a proper 960×1280×3 image. ✅
- Pi and laptop give identical predictions on `samples/` (9/10 right; the Early_blight sample is a
  genuine model mistake: it predicts yellow leaf curl virus).
- Known limit: the model has no "not a leaf" class. A photo of random objects still gets a disease label
  (an ultrasonic sensor came out as "mosaic virus 67 %").

### 2026-11-17 20:15, DHT11 support added (code)
- New `app/climate.py`: background thread, read every 5 s, keeps the last good value, counts failures,
  rough "fungal disease weather risk" (high = RH ≥ 85 % at 15–30 °C).
- `config.yaml` → `sensor.dht` (type DHT11, gpio 4). `--no-dht` turns it off.
- Web page: temperature / humidity / risk card. Every diagnosis gets `temp_c`, `humidity`
  (page, `captures/log.csv`, MQTT). New MQTT topic **`plant/climate`** (retained, every 30 s).
- Without wiring: library loads, every read fails, app carries on normally. ✅

### 2026-11-17 20:20, app as a service, and the camera got stuck
- Installed the service with `bash pi/install_service.sh`. `/api/status` →
  `camera: opencv, mqtt: true, climate: DHT11 (no reading)`. ✅
- **Lesson:** `ssh pi@… 'pkill -f app/server.py; … app/server.py'` kills its own shell, because the
  pattern is in its own command line. Stop and start in separate commands, or use systemctl.
- First capture through the service → `500`. `dmesg`: `bcm2835_v4l2: Failed enabling camera, ret -2`,
  and `raspistill` fails too ("mmal: Failed to run camera app"). The camera firmware lost the sensor,
  probably after test processes were killed mid-capture (or a loose ribbon).
  **Fix = full power-cycle** (not just a reboot, if it persists: reseat the ribbon cable).
- That camera error also **killed the MQTT thread** (an exception in paho's `on_message`). Fixed:
  - `mqtt_bridge.py`: the capture handler is wrapped in try/except
  - `server.py`: MQTT captures run in a thread with error handling; `/api/capture` returns `503 {"error": ...}`
    instead of 500
  - `camera.py`: the camera is **opened per capture and released** (no stream left running all day),
    with one automatic retry
- Added a **👁 Live preview** button on the page (`/api/preview.jpg`, refreshes every ~2 s, not saved,
  no prediction) to aim the camera before **📷 Take photo**.
- Laptop tests: `pytest -q tests/` → 4 passed. Code copied to the Pi and service restarted.

**NEXT (user):** `sudo poweroff` on the Pi → unplug power → wire DHT11 (section 3) → check the camera
ribbon is firmly seated → power on. The app starts by itself.
**THEN:** open `http://raspberrypi.local:8000` (or `http://10.93.8.209:8000`) → Live preview →
hold a tomato leaf (or a leaf photo on a phone screen) in front of the camera → Take photo.

### 2026-11-17 20:27, first live test from the website ✅ (with wrong answers → photo check added)
- Pi power-cycled → camera OK. The user opened `http://10.93.8.209:8000` on the laptop, used Live preview + Take photo
  (screenshot: `~/Pictures/Screenshots/Screenshot from 2026-11-17 20-29-17.png`).
- Results were **wrong**:
  - 20:27:54 ultrasonic sensor board → "yellow leaf curl virus 83 %"
  - 20:28:48 healthy green leaf (very blurry, probably not tomato) → "**late blight 99.6 %**"
- Why:
  1. The model knows only 10 tomato classes and has **no "not a leaf / unknown" class**. It always picks one, often with
     very high confidence, so the 0.60 confidence threshold does not catch it.
  2. The photo was far out of focus (fixed-focus Pi camera, leaf too close). Training photos are sharp.
  3. Checked: the full 38-class model (`models/pretrained_mnv2_38class.keras`) makes the **same** mistakes
     (leaf → 99 % tomato late blight, sensor → 97 % tomato mosaic), so the tomato-only renormalising is NOT the cause, and
     switching to 38 classes would not help.
- **Fix: photo check before trusting the model** (`pde/quality.py`). Thresholds picked from data:

  | | sharpness (p1) | leaf-colour fraction (p1) |
  |---|---|---|
  | PlantVillage lab val | 45 | 0.13 |
  | PlantDoc field test | 139 | 0.12 |
  | blurry Pi leaf | **14** | 0.51 |
  | sensor-board photos | 27 / 107 | **0.06 / 0.09** |

  Reject if sharpness < 35 ("Retake: too blurry, hold it 40–60 cm away") or leaf fraction < 0.10 ("Retake: no leaf found").
  Both bad captures are now rejected; all 10 `samples/` photos pass and are diagnosed as before.
  The page shows the reason; the model's ignored guess stays in `top_label` / `top3`; `captures/log.csv` gets
  `photo_problem, sharpness, leaf_fraction`. NodeMCU: the result is `not_sure`, so the yellow LED lights.
- Laptop tests 4 passed. Deployed to the Pi and checked via `/api/predict` with the two bad captures and a sample. ✅
- **Still a limit:** a *sharp* photo of a healthy leaf from another plant will pass the check and get a tomato label.
  Only tomato leaves give meaningful results. A real fix needs an "unknown" detector (feature-similarity to the training
  set) or our own model trained with a "not tomato" class (see "Ideas" below).

### Ideas / backlog
- "Unfamiliar image" detector: compare the model's internal features with the training-set class centres; far away →
  "this doesn't look like a tomato leaf". Needs a 2-output TFLite export plus calibration on lab/field photos.
- Focus: Pi Camera v2 lenses can be turned (with the focus tool / tweezers) to focus at ~25 cm.
- Finish our own models (mobilenetv3small strong + field fine-tuning) and compare them with the pretrained one in the app.

### 2026-11-17 20:45, NEW GOAL from the user: full IoT-edge-cloud system with microservices
User request (paraphrased): use a stronger pretrained plant-disease model trained on much more data; run
separate **Docker microservices** for: temperature/moisture monitoring, "does the plant need water?",
disease detected or not, and **what measures to take** if a disease is found. A complete IoT + edge + cloud
project. Keep recording everything here so work can resume from the same place.

Facts gathered:
- Laptop: Ryzen 5 7530U (12 threads), 14 GB RAM (microstack/OpenStack snap also running), no NVIDIA GPU,
  Docker 29.1.3, user in `docker` group, **`docker-compose` v1.29.2 only** (no `docker compose` plugin).
- Pi: no Docker; `/dev/i2c-1` enabled; 43 GB free.
- Model research (2026-11-17):
  - PlantVillage-only models (like ours) fail on field photos; the literature says the same (lightweight CNNs have the
    lowest F1 on PlantDoc). Mixed PlantVillage + PlantDoc + PlantWild training reported ~94.7 % (EfficientNet-B1).
  - `wambugu71/crop_leaf_diseases_vit` (ViT-tiny, 5.5 M params) has an "Invalid" class but covers corn/potato/rice/wheat only (no tomato).
  - Kindwise crop.health cloud API: ~300 issues across 23 crops; needs an API key (demo 10/month) → optional cloud second opinion.
  - The augmented PlantVillage set (Mendeley, 39 classes) has a **Background_without_leaves** class → gives a "not a leaf" output.
- Status: planning (architecture + questions to the user), nothing built yet for this goal.

### 2026-11-17 20:55, plan approved: IoT-edge-cloud with microservices
User answers: soil sensor **none yet → simulate**; model **two-tier** (edge 38-class + cloud bigger model);
cloud = **laptop docker-compose**; crops = **all 14 PlantVillage crops**.
Full plan: **`docs/PLAN_iot_cloud.md`** (phases 1–6, MQTT topics, services, verification).
Laptop ports free for the stack: 1883 (MQTT), 5432 (postgres), 8080 (gateway). Port 5000 is taken.
Laptop's `.local` name resolves only to IPv6 → the Pi bridge uses the laptop IPv4 (10.93.8.158, DHCP: recheck if it changes).

### 2026-11-17 20:58, laptop shut down twice by itself (diagnosis)
- Crashes at **19:55** and **20:47**. The system log just stops: no shutdown, no kernel panic, no out-of-memory
  message, no thermal message. That pattern = power cut at the hardware level.
- Battery was **not empty**: about 49 % at the first crash and about 25 % at the second. The charger was **unplugged**
  (`ACAD online=0`), and upower recorded **31–38 W** drawn from the battery just before the 19:55 crash
  (model training on all 12 CPU threads). The second crash came during the dataset download and model export.
- Likely cause: heavy CPU load **on battery**. The 36 Wh battery hits its protection cutoff (voltage sag), or the
  CPU overheats and the firmware trips. Idle temperatures are fine now (CPU 41 °C).
- Extra load: the **microstack (OpenStack) snap** is broken (its DB IP 192.168.0.191 isn't on this network).
  `mysqld` restarts every ~3 s (857 restarts in 47 min), and nova/cinder keep retrying. That wastes CPU and heat all the time.
- **What to do:** (1) keep the charger plugged in for training/exports; (2) if it still powers off on AC, clean the
  fans/vents and use a cooling pad (thermal cause); (3) stop the broken microstack when not needed:
  `sudo snap disable microstack` (bring it back with `sudo snap enable microstack`); (4) heavy jobs from now on run with
  fewer threads (`taskset -c 0-5 nice -n 10 ...`) and save checkpoints every epoch.
- Pi checked at 20:56: up 52 min, 49.6 °C, `throttled=0x0`, `plant-app` + `mosquitto` active, 43 GB free. OK.

### 2026-11-17 21:05, Phase 1 done: edge model now covers all 14 crops (38 classes) ✅
- `pde/catalog.py`: 38 classes → crop, disease, display name, stable id `crop.disease` (e.g. `apple.black_rot`), offline advice.
- `scripts/import_pretrained.py --all-crops` → `models/pv38_fp32.tflite` (9.1 MB), `models/pv38_int8.tflite` (2.8 MB),
  `models/pv38.labels.json`. INT8 calibrated on 982 images (150 tomato + all of `data/pv_allcrops`).
  `data/pv_allcrops` = about 30 images per non-tomato class from the PlantVillage GitHub repo (`urls.tsv`).
- `pde/predictor.py`: if `<model>.labels.json` exists, labels are catalog ids. `Prediction` gains `crop`, `disease`,
  `display`, `class_id`. The old tomato models work unchanged.
- `scripts/check_pv38.py` (per-crop accuracy, 1132 images): **INT8 98.6 %, FP32 99.2 %, crop right 99.6 %**.
  Caveat: these are PlantVillage photos, probably seen by the pretrained model in training → a sanity check, not a field score.
- App: page shows "Apple: Black rot"; every result gets `status` = healthy | disease | not_sure (also in `captures/log.csv`
  and MQTT). NodeMCU sketch updated to use `status` / `*.healthy`.
- `config.yaml` app.model_path → `models/pv38_int8.tflite`. Tests: `tests/test_catalog.py` added, **9 passed**.
- Deployed to the Pi, service restarted. Live check via `/api/predict`: apple black rot ✅, potato late blight ✅ (99.6 %, ~67 ms).
- Pi benchmark (`results/pi_benchmark.csv`): **pv38_int8 66.0 ms, 15.2 FPS, 36 MB RAM**; pv38_fp32 133.8 ms, 7.5 FPS, 58 MB.
  Same speed as the tomato model (only the last layer differs).

**NEXT:** Phase 2 = cloud stack skeleton in `cloud/` (docker-compose: mosquitto, postgres, ingest, gateway/dashboard),
then the Pi→cloud Mosquitto bridge. See `docs/PLAN_iot_cloud.md`.

### 2026-11-17 21:12, Phase 2 done: cloud stack + Pi→cloud link ✅
- Charger plugged in at 21:05, so heavy work is safe again.
- `cloud/` (see `cloud/README.md`): docker-compose with **mosquitto** (:1883), **postgres 16** (127.0.0.1:5432,
  schema `cloud/db/init.sql`), **ingest** (all `plant/#` → DB), **gateway** (FastAPI :8080: REST, photo upload,
  live SSE, dashboard at http://localhost:8080). One shared `cloud/Dockerfile` + `cloud/common/`.
  Start: `cd cloud && docker-compose up -d --build`. `restart: unless-stopped`, so it comes back after a reboot.
- Pi MQTT bridge: `pi/mosquitto-bridge.conf.template` → `/etc/mosquitto/conf.d/bridge.conf` via
  `bash pi/install_bridge.sh 10.93.8.158`. `plant/#` both ways, QoS 1, persistent session, queue up to 10 000 messages.
  Bridge state is on topic `plant/bridge/pi-node-01` (1 = connected).
- Pi app: every result now has `image_id` (`pi-node-01_<time>`) + `device`. New `app/cloud_uploader.py` posts photo + edge
  result to the gateway, using `captures/outbox/` as a disk queue. `config.yaml` → `cloud:` section. `/api/status` shows `cloud`.
  `--no-cloud` turns it off.
- Tests: Pi diagnoses (grape, corn) appear in cloud postgres with the photo. **Store-and-forward test:** stopped the cloud broker +
  gateway, then diagnosed a peach leaf on the Pi (worked offline, 1 in the outbox). Restarted the cloud: within 40 s the photo was
  uploaded (outbox 0) and the queued MQTT diagnosis arrived through the bridge. ✅
- If the laptop IP changes: rerun `bash pi/install_bridge.sh <new-ip>` on the Pi and edit `cloud.gateway_url` in `config.yaml`.

**NEXT:** Phase 3 = soil-sim + irrigation services ("does the plant need water?").

### 2026-11-17 21:15, Phase 3 done: soil simulator + irrigation ("needs water?") ✅
- `cloud/services/soil_sim/`: simulated probe → `plant/soil` every 30 s, **SIM_SPEED=60** (1 simulated hour per real minute).
  It dries faster when hot or dry (uses `plant/climate`, else 27 °C / 60 %). It waters itself below 15 % (demo), and
  `plant/irrigation/cmd {"action":"water"}` adds water. Set `ENABLED=0` once the real NodeMCU probe exists.
- `cloud/services/irrigation/`: per-crop bands in `crops.yaml` (dry / target / wilting for all 14 crops), crop = latest
  confident diagnosis (`CROP=auto`, default tomato). The rules (`logic.py`): below dry → need water; heat ≥ 32 °C or VPD ≥ 1.6 kPa → high
  urgency; below wilting → high; hysteresis until target; `hours_to_dry` from a 2 h trend in postgres. Publishes retained `plant/irrigation`.
- `cloud/common/agro.py` (VPD). Gateway `POST /api/water`; dashboard has a "💧 I watered it" button.
- Tests: `cd cloud && PYTHONPATH=. ../.venv/bin/python -m pytest -q tests` → 3 passed; project tests 9 passed.
- **docker-compose 1.29 bug:** after a rebuild, recreating containers fails with `ERROR: 'ContainerConfig'`.
  Fix: `docker-compose rm -fs <service>` (and `docker rm -f` any `<hash>_cloud_<service>_1` leftovers), then `docker-compose up -d`.
- All 6 containers Up at 21:15.

**NEXT:** Phase 4 = `disease-advisor` (KB `cloud/advisor/kb.yaml` for 38 classes: symptoms, actions, organic/chemical,
prevention; merges edge + cloud verdicts + climate risk → `plant/advice`) and `notifier` (`plant/alert`).

### 2026-11-17 21:25, live camera on the cloud dashboard + fixes (from the user's screenshots)
- The user expected the camera on the dashboard (:8080); until now it was only on the Pi page (:8000).
  Added a **"Live camera" card** to the dashboard: 👁 Live preview (gateway proxies `GET /api/camera/preview.jpg` from the Pi
  app, env `PI_URL=http://10.93.8.209:8000`) and **📷 Take photo & detect** (`POST /api/camera/capture` → MQTT `plant/capture`
  → bridge → Pi camera + model → `plant/diagnosis` + photo upload → dashboard). Tested: result back in **~4 s**. ✅
- Deleted my fake test rows (device `test`: 24.5 °C / 88 %, apple photo) and the retained test climate message. Climate now
  shows "–" until the DHT11 is wired.
- Fixed "dry in ~1.3 h" being shown twice.
- Live capture at 21:23: the camera points at the ceiling/wall (cables). Model 57 % → **not sure** (correct). But the photo check passed it
  (leaf_fraction 0.19: the yellowish wall counts as leaf colour). Add to the "unknown image" backlog: a stricter leaf check or a "no leaf" class (Phase 5 v2).
- **User action:** aim the Pi camera at a plant (40–60 cm), then use the dashboard's camera card.

### 2026-11-17 21:27, user test from the dashboard camera: leaf photo on a phone screen
- 21:24:57: the user held a phone showing a web image (Wikipedia, appears mirrored) of a **yellow leaf with round black spots**
  (looks like rose black spot / a leaf-spot disease, probably **not one of the 14 crops**). Result: **Corn: Gray leaf spot 67 %**
  (2nd potato healthy 15 %, 3rd tomato mosaic 9 %). Photo check passed (sharpness 86, leaf 0.24).
- The pipeline worked (dashboard → MQTT → Pi camera → model → cloud). The **answer is wrong** because the crop is unknown to the model, the photo
  shows a screen (blurry, glare, hand and phone frame in view), and 67 % is barely over the 0.60 cutoff.
- Ideas: raise `app.confidence_threshold` to ~0.80; show the top 3 + a "low confidence" badge on the dashboard; prioritise the
  "not a leaf / unknown crop" work (Phase 5 v2 model with a background class, or the feature-distance check).

### 2026-11-17 21:30, session paused (continue tomorrow)
- State left running: the laptop cloud stack (6 containers, restarts with Docker) and the Pi `plant-app` + Mosquitto bridge (start on boot).
- Nothing half-done: all code changes are deployed and tests pass (project 9, cloud 3).
- Tomorrow: start with "0. RESUME HERE" → step 1 (quick fixes), then Phase 4.

### 2026-11-19 06:50, session resumed: quick fixes done ✅
- Start-up check: laptop IP still 10.93.8.158, charger in, 6 cloud containers Up (restarts=0; `docker ps` showed "Up Less than a second"
  only because the system clock jumped ~5 h after boot), Pi `plant-app` + `mosquitto` active.
- `config.yaml` → `app.confidence_threshold` **0.60 → 0.80**, rsynced to the Pi, `plant-app` restarted (`/api/status` shows threshold 0.8).
- Dashboard "Latest diagnosis": now shows the **top 3 with bars**, a **⚠ low confidence** badge (not sure, or confidence < 90 %),
  the **best guess** when not sure, and the photo-check reason. Gateway rebuilt (`docker-compose rm -fs gateway` workaround).
- Live test from the dashboard camera at 06:44: tomato early blight 73 % (late blight 27 %), so now **not sure** (with 0.60 it would have been
  shown as a diagnosis); the photo check also flagged it as too blurry.
- `pytest.ini`: added `testpaths = tests` (a bare `pytest` from the root tried to collect `cloud/tests` and failed on imports).
  Tests: project 9 passed, cloud 3 passed.

**NEXT:** Phase 4 = disease-advisor (KB for 38 classes) + notifier (alerts).

### 2026-11-19 07:00, user questions answered on the website + Phase 4 done ✅
- **The user asked:** how is soil moisture captured, how is watering detected, where is data stored, where does prediction run.
  Honest answer: soil moisture is **simulated** (`soil-sim` container), no probe yet; watering was not detected before, only
  "told" by the 💧 button. Now on the dashboard:
  - Soil card: **SIMULATED** badge + "How is this measured?" (simulator now; planned capacitive probe → NodeMCU A0, raw ≈820 dry / ≈380 wet).
  - Water card: "How is watering decided and detected?" + **"Last watering: hh:mm · a % → b % (detected)"**.
  - New **"How it works"** card: Sensing → Prediction & decisions → Transport → Storage, with live counts from new `GET /api/storage`.
- **Watering detection** (`irrigation`): moisture rise ≥ 8 points between readings (`WATERING_MIN_RISE_PCT`) → retained
  `plant/irrigation/watered`, also in `plant/irrigation.last_watered`. After a restart it compares with the last reading in postgres
  (the first test missed a watering because of that; fixed). Test: 59 % → 89 % detected + info alert. ✅
- **Phase 4 advisor** (`cloud/services/advisor/`): `kb.yaml` = 26 diseases + shared healthy entry (cause, type, severity, spread,
  symptoms, now / organic / chemical (active ingredients only) / prevention). Rules (`logic.py`): edge not sure → "retake" (no measures,
  shows best guess + its symptoms); cloud disagrees → "models disagree, check by eye"; humid air (≥ 85 % RH) raises severity of
  humidity-driven diseases; hot dry air does the same for spider mites. Publishes retained `plant/advice`.
- **Phase 4 notifier** (`cloud/services/notifier/`): `plant/alert` for disease (medium → warning, critical → critical, 10 min cooldown
  per class), urgent water (30 min cooldown), Pi offline/online, sensor silent > 5 min, watering detected. Ignores retained replays
  (new `skip_retained` option in `common/mqtt.py`). Telegram push stubbed: set `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID`.
- **Pi page**: subscribes to `plant/advice` (bridge), new `GET /api/advice/<image_id>`; under each result it shows "☁ What to do"
  or "Cloud advisor not reachable" after ~15 s. Deployed with rsync + `systemctl restart plant-app`.
- Live tests: tomato late blight sample → advisor critical + critical alert ✅; potato early blight → medium measures shown on
  the Pi page API ✅; dashboard camera capture (no leaf in view) → "Not sure: retake the photo" with best guess ✅.
- Tests: cloud 11 passed (new `tests/test_phase4.py`: KB covers all 38 model ids, verdicts, cooldowns, watchdog), project 9 passed.
- 8 containers up.

**NEXT:** Phase 5 v1 = cloud-inference container (38-class FP32 Keras + test-time augmentation → `plant/cloud_diagnosis`;
the advisor already merges it). Consider 5 v2 ("not a leaf" class) early.

### 2026-11-19 07:25, Phase 5 v1 done: cloud second opinion ✅
- **Change from the plan:** instead of the Keras model in a TensorFlow image (~1 GB), the cloud runs the same 38-class network as
  the FP32 export `models/pv38_fp32.tflite` on LiteRT (`ai-edge-litert`). Image 256 MB, ~30-40 ms for 6 views.
- `cloud/services/cloud_inference/`: `tta.py` (6 views: full, mirror, flip, centre 85 %, 85 % + mirror, centre 70 %; probabilities
  averaged; sure = mean conf ≥ 0.8 and ≥ 80 % of views agree) + `main.py` (on `plant/image` → reads the photo from the shared
  `images` volume → `plant/cloud_diagnosis`). Model mounted from `../models` (`MODEL_PATH` env, for the v2 model later).
  Dockerfile: per-service `requirements.txt` is now installed if present.
- **Evaluation** (`scripts/eval_cloud_tta.py` → `results/cloud_tta.csv`, threshold 0.8; lab = 600 PlantVillage tomato test photos,
  field = 663 real tomato field photos):

  | set | setup | accuracy | answered | acc. on answered | confident-wrong |
  |---|---|---|---|---|---|
  | lab | edge INT8 | 98.0 % | 98.5 % | 98.7 % | 1.3 % |
  | lab | cloud FP32 + TTA | 98.3 % | 93.0 % | 99.8 % | 0.2 % |
  | lab | edge + cloud must agree | 98.0 % | 92.7 % | 99.8 % | 0.2 % |
  | field | edge INT8 | 20.1 % | 72.3 % | 22.1 % | **56.3 %** |
  | field | cloud FP32 (1 view) | 20.1 % | 73.5 % | 20.7 % | 58.2 % |
  | field | cloud FP32 + TTA | 20.5 % | 47.8 % | 28.4 % | 34.2 % |
  | field | edge + cloud must agree | 20.1 % | 42.2 % | 28.6 % | **30.2 %** |

  FP32 alone gives nothing; TTA makes the model more honest, not more accurate. Field accuracy stays ~20 % (the 38-class model
  confuses field tomato photos with other crops), so **v2 (field-trained model with a "no leaf" class) is still needed**.
  The view-agreement rule added nothing beyond the averaged confidence (same numbers); kept as information.
- **Advisor policy** (from the numbers): a diagnosis only when edge and cloud are both sure and agree; edge sure + cloud not sure →
  "Not sure: the cloud check failed"; different classes → "Models disagree". Edge-only advice is published with
  `pending_cloud=true` (notifier waits); if the cloud does not answer in 20 s (`CLOUD_WAIT_S`) it becomes final.
- Dashboard: cloud line shows "n/6 views agree", ⏳ while waiting; "How it works" card updated. Pi page waits for the final advice and
  shows the "models disagree" header.
- Live tests (photos uploaded to the Pi): potato late blight lab photo → edge 99.6 % = cloud 100 % (6/6) → "both models agree" +
  critical alert ✅. Real field photo of tomato early blight → edge said **cherry healthy 88 %** (confidently wrong), cloud not sure
  (strawberry leaf scorch 67 %, 4/6) → "Models disagree: check the leaf by eye" ✅. Cloud stopped → grape black rot edge-only advice
  final after 20 s + alert ✅.
- Tests: cloud 14 passed, project 9 passed. 9 containers up.

**NEXT:** Phase 6 (docs/ARCHITECTURE.md, more tests, end-to-end check, final log), then the user's hands-on work (DHT11, NodeMCU
sketch, real leaves). Phase 5 v2 (Colab GPU, field + "no leaf" model) is the real fix for field accuracy.

### 2026-11-19 08:00, Phase 6 done + Phase 5 v2 prepared ✅
- **Phase 5 v2, ready for Colab** (`notebooks/cloud_v2/`): `train_cloud_v2.py` (source, "# %%" cells) →
  `train_cloud_v2_colab.ipynb` via `scripts/make_colab_notebook.py`. EfficientNet-B1 @240 px, 40 classes = the 38 ids
  + `other.not_a_leaf` (DTD textures + Imagenette) + `other.unknown_plant` (TFDS plant_leaves without lemon).
  Data: PlantVillage (≤1500/class) + **all PlantDoc field photos ×4**, strong augmentation (crop/zoom, flips, 90° turns,
  colour, blur, JPEG). **The laptop's lab + field test photos are excluded by MD5** (3396 checksums embedded; all 662
  unique field test files and 2734 lab test files matched their originals), so the comparison stays fair.
  Head 2 epochs, then fine-tune 8 (cosine LR, early stopping), mixed precision; FP32 TFLite export checked against Keras.
  - **Smoke-tested on the laptop** (`SMOKE=1`, tiny data, 64 px): data → train → evaluate → export OK, TFLite = Keras
    (diff 4e-9). First export attempt (`from_concrete_functions`) crashed with Keras 3 → now SavedModel ExportArchive.
    Unreadable web images are skipped (PIL check + `ignore_errors`).
  - Cloud side ready: `tta.py` turns a sure `other.*` answer into not-sure with a reason; the advisor shows
    "No leaf found" / "Unknown plant: not one of the 14 crops". Unit tests with the smoke model's 40-class export.
  - `scripts/install_cloud_v2.sh <folder>`: copies the files, evaluates v1 vs v2 (`eval_cloud_tta.py --cloud-model`),
    switches only if v2 has fewer field confident-wrong answers, ≥ v1 field accuracy and ≥ 95 % lab accuracy.
    Tested with the smoke model: correctly **kept v1**; `--rollback` and the compose edit tested.
- **NodeMCU sketch** (`iot/nodemcu_leaf_node`): broker now the Pi **10.93.8.209** (was 172.29.1.135), SSID "Abhishek"
  (**password still to fill in**), capacitive soil probe on A0 → `plant/soil` every 30 s (10-sample average, calibration
  constants, "unplugged" check), LEDs follow the final `plant/advice` verdict, MQTT buffer 4096. **Compiles**
  (arduino-cli, esp8266:nodemcuv2, 24 % flash); not uploaded.
- `soil-sim` pauses by itself while a real probe sends (`simulated:false`), restarts 10 min after the last real reading.
  Tested with one probe-style message (then deleted from the DB).
- **Docs:** `docs/ARCHITECTURE.md` (diagram, photo flow, watering, topics, design decisions, limits), `cloud/README.md`
  (services, e2e, v2 steps), plan status.
- **Tests:** project 12 (new `tests/test_consistency.py`: KB / water bands / Colab class map vs catalog), cloud 15 + 1 skip.
  ⚠ I had overwritten the existing `tests/test_catalog.py` (5 tests) by mistake; restored byte-identical from the Pi copy.
- **`scripts/e2e_check.py`** (verification list of the plan): ran with `--offline-test` → all passed; final run
  **14/14 crops correct on edge and cloud, every photo got final advice**, store-and-forward back in 5 s.
- **Pi benchmark** (`results/pi_benchmark.csv`): pv38_int8 **167 ms** mean / 214 ms p95, 6 FPS, 36 MB RAM, **measured while
  another program used 65-80 % CPU** (see below); the idle figure from 2026-11-17 is 66 ms.
- ⚠ **Pi camera blocked:** `/home/pi/plant-edge/pi/live.py --model models/current.tflite --sensor dht11` (PID 2670,
  started ~06:35 UTC, not part of this project) holds `/dev/video0`. Since the `plant-app` restarts today it runs in
  **upload-only mode** (camera None), so "Take photo" on the dashboard / Pi page does not work. Not stopped by me
  (another project). Fix when it is free: stop that program, then `sudo systemctl restart plant-app`.

