"""Convert a trained SavedModel to TensorFlow Lite: FP32 and full-integer INT8.

INT8 calibration uses a few hundred lab-train + field-finetune images (never test images).
INT8 model input is uint8 [0,255] (the raw photo), output uint8 probabilities.

    python scripts/quantize.py models/mobilenetv3small_field
-> models/mobilenetv3small_field_fp32.tflite, models/mobilenetv3small_field_int8.tflite
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pde.config import load_config, p  # noqa: E402
from pde.data import list_files  # noqa: E402
from pde.predictor import load_rgb  # noqa: E402
from pde.tfcompat import tf  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("saved_model")
    ap.add_argument("--calib", type=int, default=300, help="number of calibration images")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    size = cfg["train"]["img_size"]
    src = Path(args.saved_model)

    # FP32
    conv = tf.lite.TFLiteConverter.from_saved_model(str(src))
    fp32 = conv.convert()
    out32 = src.parent / f"{src.name}_fp32.tflite"
    out32.write_bytes(fp32)

    # INT8 (full integer)
    paths = list_files(p(cfg["data"]["lab_dir"]) / "train")[0] + \
        list_files(p(cfg["data"]["field_dir"]) / "finetune")[0]
    if not paths:
        sys.exit("No calibration images found (need data/lab/train).")
    random.Random(0).shuffle(paths)
    paths = paths[:args.calib]

    def representative():
        for f in paths:
            yield [load_rgb(f, size)[None].astype(np.float32)]

    conv = tf.lite.TFLiteConverter.from_saved_model(str(src))
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = representative
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.uint8
    conv.inference_output_type = tf.uint8
    int8 = conv.convert()
    out8 = src.parent / f"{src.name}_int8.tflite"
    out8.write_bytes(int8)

    print(f"{out32.name}: {len(fp32) / 1e6:.2f} MB")
    print(f"{out8.name}: {len(int8) / 1e6:.2f} MB  (calibrated on {len(paths)} images)")


if __name__ == "__main__":
    main()
