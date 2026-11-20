# Offline Plant-Disease Detection on a Raspberry Pi (IoT edition)

End-to-end implementation of *Plant_Disease_Edge_Proposal.docx*: a tomato leaf-disease classifier
trained on PlantVillage (lab photos), tested on PlantDoc (field photos), quantized to INT8 and
run **offline** on a Raspberry Pi with a camera app. An MQTT layer joins it to the existing lab
IoT setup (NodeMCU → Pi Mosquitto gateway → laptop).

**Research question:** how much accuracy is lost going from lab images to field images, and from
an FP32 model to an INT8 model on the Pi? Can light fine-tuning on a few field images recover it?

## System

```
             TRAINING (Colab / Kaggle GPU)                         EDGE (Raspberry Pi, offline)
 PlantVillage ─┐                                          ┌──────────────────────────────────────┐
 (lab, tomato) ├─ prepare_data ─ train ─ finetune_field ─ │ quantize ─► *_int8.tflite            │
 PlantDoc ─────┘                  │          │            │   Pi Camera ─► Predictor (LiteRT) ─┐ │
 (field)                       evaluate   evaluate        │   Flask page :8000 ◄───────────────┤ │
                                                          │   Mosquitto broker ◄─ plant/diagnosis│
                                                          └────────▲────────────────┬───────────┘
                                                     plant/capture │                │ plant/diagnosis
                                                   ┌───────────────┴──┐     ┌───────▼──────────┐
                                                   │ NodeMCU ESP8266  │     │ Laptop           │
                                                   │ button + 3 LEDs  │     │ iot/monitor.py   │
                                                   └──────────────────┘     └──────────────────┘
```

MQTT topics (broker = Mosquitto on the Pi, port 1883):

| Topic | Publisher | Payload |
|---|---|---|
| `plant/capture` | NodeMCU button (or `mosquitto_pub`) | `{"device": "leaf-node-xxxx"}` → Pi takes a photo |
| `plant/diagnosis` | Pi app, after every prediction | `{"device", "time", "source", "label", "top_label", "confidence", "is_sure", "advice", "top3", "latency_ms"}` |
| `plant/status` | Pi app (retained, plus last will) | `{"device", "state": "online" \| "offline"}` |

`label` is a short class name (`early_blight`, `healthy`, …) or `not_sure` when confidence is below
the threshold (default 0.60, in `config.yaml`).

## Folder layout

| Path | What |
|---|---|
| `config.yaml` | all settings (paths, epochs, threshold, MQTT) |
| `pde/` | shared code: classes and advice, data and augmentation, models, TFLite predictor |
| `scripts/prepare_data.py` | download PlantVillage + PlantDoc, keep tomato, split |
| `scripts/train.py` | transfer learning: MobileNetV3-Small / EfficientNet-Lite0, standard or strong augmentation |
| `scripts/finetune_field.py` | fine-tune on a few field images (mixed with lab images) |
| `scripts/quantize.py` | SavedModel → FP32 and INT8 `.tflite` |
| `scripts/evaluate.py` | accuracy, macro-F1, confusion matrices, "not sure" rate (lab and field separately) |
| `scripts/benchmark_pi.py` | on the Pi: latency, FPS, peak RAM, size (and accuracy) |
| `notebooks/train_colab.ipynb` | runs steps 1–4 on Colab in order |
| `app/` | Pi demo app: Flask page, camera, MQTT bridge |
| `iot/nodemcu_leaf_node/` | ESP8266 sketch: button → capture, LEDs show result |
| `iot/monitor.py` | laptop subscriber that logs diagnoses to CSV |
| `pi/` | Pi setup script and systemd service |
| `tests/` | unit tests + an end-to-end smoke test on synthetic images |

## Classes

The 10 PlantVillage tomato classes: bacterial spot, early blight, late blight, leaf mold, septoria leaf
spot, spider mites, target spot, yellow leaf curl virus, mosaic virus, healthy.
PlantDoc has no Target Spot, so the field test set has 9 classes. Macro-F1 is computed over the
classes present in each test set.

## Phase 1 — software (no hardware needed)

### 1. Local setup and smoke test (laptop)
```bash
cd plant_disease_edge
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q tests/          # ~3 min: builds fake data and runs every stage end to end
```

### 2. Train on Colab (or Kaggle)
1. Copy this folder (without `.venv`) to Google Drive → `My Drive/plant_disease_edge`.
2. Open `notebooks/train_colab.ipynb` in Colab with a GPU runtime and run all cells. You need a
   Kaggle API token (`kaggle.json`) for PlantVillage.
3. Outputs land in `My Drive/plant_disease_edge/models` and `.../results`.

Or run the same steps by hand:
```bash
python scripts/prepare_data.py --download
python scripts/train.py --model mobilenetv3small --augment standard     # baseline
python scripts/train.py --model efficientnet_lite0 --augment standard
python scripts/evaluate.py models/mobilenetv3small_standard models/efficientnet_lite0_standard   # domain gap
python scripts/train.py --model mobilenetv3small --augment strong
python scripts/finetune_field.py --base models/mobilenetv3small_strong
python scripts/quantize.py models/mobilenetv3small_field
python scripts/evaluate.py models/*.tflite                              # FP32 vs INT8
```

### 3. Try the app on the laptop (no Pi, no camera)
```bash
python app/server.py --no-mqtt --camera none --model models/mobilenetv3small_field_int8.tflite
# open http://localhost:8000 and upload a leaf photo
```
With MQTT: install Mosquitto (`sudo apt install mosquitto mosquitto-clients`), run
`python app/server.py --camera none`, and in another terminal `python iot/monitor.py`.

### Team field photos (optional)
Put labelled photos in `data/own_photos/<class>/` (folder name like `early_blight` or
`Tomato___Early_blight`) before running `prepare_data.py`. They are split into fine-tune and test
sets like the PlantDoc images.

## Phase 2 — hardware (last)

### Raspberry Pi 4/5 + Camera Module 3
```bash
# on the Pi (64-bit Raspberry Pi OS Bookworm), project copied to /home/pi/plant_disease_edge
bash pi/setup_pi.sh                       # Mosquitto, Picamera2, LiteRT, Flask, paho-mqtt
cp /path/to/*.tflite models/
.venv/bin/python scripts/benchmark_pi.py models/*.tflite            # latency / FPS / RAM / size
tar xzf test_sets.tgz && .venv/bin/python scripts/benchmark_pi.py models/*.tflite --accuracy
.venv/bin/python app/server.py            # http://<pi-ip>:8000
sudo cp pi/plant-app.service /etc/systemd/system/ && sudo systemctl enable --now plant-app   # start on boot
```
Test the MQTT trigger without the NodeMCU: `mosquitto_pub -h <pi-ip> -t plant/capture -m '{"device":"test"}'`.

### NodeMCU (ESP8266) button + LED node
Open `iot/nodemcu_leaf_node/nodemcu_leaf_node.ino` in the Arduino IDE (the board setup is the same as
`setup/ESP8266_setup_guide.pdf`, library *PubSubClient*). Set `WIFI_SSID`, `WIFI_PASS` and
`BROKER` (the Pi's IP), then upload.

| NodeMCU pin | Part |
|---|---|
| D5 (GPIO14) | green LED + 220 Ω → GND (healthy) |
| D6 (GPIO12) | red LED + 220 Ω → GND (disease) |
| D7 (GPIO13) | yellow LED + 220 Ω → GND (not sure, retake / blinking = waiting) |
| D3 (GPIO0)  | on-board FLASH button, or a push button to GND |

Press the button: the Pi takes a photo, predicts, the LED lights for 10 s, the Pi web page updates
and the laptop monitor logs the result.

## Evaluation outputs (proposal section 7)

| Metric | Where |
|---|---|
| Accuracy and macro-F1, lab and field separately | `results/summary.csv`, `results/<model>__<set>.json` |
| Confusion matrices | `results/<model>__<set>_cm.png` |
| FP32 → INT8 accuracy change | compare `*_fp32` and `*_int8` rows in `summary.csv` |
| Latency (ms), FPS, peak RAM, model size on the Pi | `results/pi_benchmark.csv` |
| Share rejected as "not sure" and accuracy on the rest | `rejected_share`, `accuracy_on_accepted`, `threshold_sweep` in the JSON |

## Notes
- EfficientNet-Lite0 is only published on TF-Hub in the old TF1 format, which cannot be fine-tuned.
  `pde/efficientnet_lite.py` rebuilds it as normal Keras layers and copies the ImageNet weights in
  from that checkpoint. Its output matches the TF-Hub module exactly (`check_against_hub()`). If the
  download fails, `efficientnet_lite0` falls back to Keras EfficientNetB0 and prints a warning.
- Keras 2 (`tf_keras`) is used throughout because INT8 TFLite conversion is most reliable with it.
- The INT8 model takes the raw uint8 photo as input; scaling happens inside the model.
- Advice lines in `pde/classes.py` are general guidance, not agronomic advice.
