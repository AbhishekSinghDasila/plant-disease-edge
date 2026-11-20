"""Import a ready-made PlantVillage model and turn it into tomato-only TFLite models for the Pi.

Source: huggingface.co/rarfileexe/Plant-Disease-Detector (MIT licence), a Keras MobileNetV2
fine-tuned on PlantVillage (38 classes, all crops). Pixel scaling is built into the model, so it
takes raw [0, 255] RGB 224x224 images, the same as our own models.

We keep only its 10 tomato outputs (indices 28-37, same order as pde.classes.CLASSES) and
renormalise them, so the result is a drop-in replacement for our own trained model.

    python scripts/import_pretrained.py            # -> models/pretrained_mnv2_fp32.tflite, _int8.tflite
    python scripts/import_pretrained.py --all-crops   # -> models/pv38_fp32.tflite, pv38_int8.tflite, pv38.labels.json
                                                     #    (all 38 classes / 14 crops, see pde/catalog.py)
INT8 calibration uses data/lab/train if present (run prepare_data.py first), else it is skipped.

Note: this model is for the hardware demo only. The report's lab-vs-field and fine-tuning results
need our own trained models (scripts/train.py).
"""
import argparse
import json
import os
import random
import sys
import urllib.request
from pathlib import Path

os.environ["TF_USE_LEGACY_KERAS"] = "0"   # this .keras file needs Keras 3
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pde.catalog import PV38, PV38_IDS  # noqa: E402
from pde.classes import CLASSES  # noqa: E402
from pde.predictor import load_rgb  # noqa: E402

URL = "https://huggingface.co/rarfileexe/Plant-Disease-Detector/resolve/main/model_4_mobilenet_finetuned.keras"
TOMATO_IDX = list(range(28, 38))  # their CLASS_NAMES[28:38] = Tomato___Bacterial_spot ... Tomato___Healthy


def inference_model(model, take=TOMATO_IDX):
    """Rebuild without the RandomFlip/Rotation/Zoom training layers (their random state cannot
    be exported): image [0,255] -> /127.5 - 1 -> MobileNetV2 -> pool -> dense(38) -> tomato 10."""
    import keras
    inp = keras.Input((224, 224, 3), name="image")
    x = inp / 127.5 - 1.0
    x = model.get_layer("mobilenetv2_1.00_224")(x)
    if isinstance(x, (list, tuple)):  # this saved sub-model returns its output as a list
        x = x[0]
    x = model.get_layer("global_average_pooling2d")(x)
    x = model.get_layer("dense")(x)
    if take is None:   # all 38 classes, unchanged
        return keras.Model(inp, x, name="pretrained_mnv2_pv38")
    p = keras.ops.take(x, take, axis=-1)
    p = p / keras.ops.sum(p, axis=-1, keepdims=True)
    return keras.Model(inp, p, name="pretrained_mnv2_tomato")


def check_same(original, rebuilt, n=6, take=TOMATO_IDX):
    """Rebuilt model must give the same probabilities as the original (inference mode)."""
    x = np.random.default_rng(0).integers(0, 256, (n, 224, 224, 3)).astype(np.float32)
    a = original.predict(x, verbose=0)
    if take is not None:
        a = a[:, take]
        a = a / a.sum(-1, keepdims=True)
    b = rebuilt.predict(x, verbose=0)
    diff = float(np.abs(a - b).max())
    print(f"rebuilt vs original: max diff {diff:.2e}")
    assert diff < 1e-4, "rebuilt model differs from original"


def main():
    import keras
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-crops", action="store_true", help="keep all 38 classes (14 crops)")
    args = ap.parse_args()
    take = None if args.all_crops else TOMATO_IDX
    prefix = "pv38" if args.all_crops else "pretrained_mnv2"
    models = ROOT / "models"
    models.mkdir(exist_ok=True)
    src = models / "pretrained_mnv2_38class.keras"
    if not src.exists():
        print("Downloading pretrained model ...")
        urllib.request.urlretrieve(URL, src)
    model = keras.models.load_model(src, compile=False)
    assert model.output_shape[-1] == 38, model.output_shape
    print(f"Loaded {src.name}: {model.count_params():,} params")

    rebuilt = inference_model(model, take)
    check_same(model, rebuilt, take=take)
    sm = models / f"{prefix}_savedmodel"
    rebuilt.export(str(sm))
    if args.all_crops:
        labels = models / "pv38.labels.json"
        labels.write_text(json.dumps({"ids": PV38_IDS, "folders": PV38}, indent=1))
        print(f"{labels.name}: {len(PV38_IDS)} classes")

    conv = tf.lite.TFLiteConverter.from_saved_model(str(sm))
    fp32 = models / f"{prefix}_fp32.tflite"
    fp32.write_bytes(conv.convert())
    print(f"{fp32.name}: {fp32.stat().st_size / 1e6:.2f} MB")

    calib_dir = ROOT / "data" / "lab" / "train"
    paths = sorted(str(f) for c in CLASSES for f in (calib_dir / c).glob("*") if (calib_dir / c).exists())
    if args.all_crops:   # calibrate on every crop, not just tomato: tomato subset + all of data/pv_allcrops
        random.Random(0).shuffle(paths)
        paths = paths[:150] + sorted(str(f) for f in (ROOT / "data" / "pv_allcrops").glob("*/*")
                                     if f.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not paths:
        print("No data/lab/train images: skipping INT8 (run prepare_data.py, then this script again).")
        return
    random.Random(0).shuffle(paths)
    paths = paths[:300] if not args.all_crops else paths[:1000]

    def representative():
        for f in paths:
            yield [load_rgb(f, 224)[None].astype(np.float32)]

    conv = tf.lite.TFLiteConverter.from_saved_model(str(sm))
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = representative
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.uint8
    conv.inference_output_type = tf.uint8
    int8 = models / f"{prefix}_int8.tflite"
    int8.write_bytes(conv.convert())
    print(f"{int8.name}: {int8.stat().st_size / 1e6:.2f} MB (calibrated on {len(paths)} images)")


if __name__ == "__main__":
    main()
