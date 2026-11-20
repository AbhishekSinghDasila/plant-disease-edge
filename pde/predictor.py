"""TFLite inference with a confidence threshold. Used by the Pi app, the benchmark and evaluation.

No TensorFlow needed on the Pi: uses ai-edge-litert (new name of tflite-runtime), then
tflite-runtime, then full TensorFlow - whichever is installed.
"""
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from pde import catalog
from pde.classes import ADVICE, NOT_SURE_MSG, SHORT_CLASSES


def make_interpreter(model_path, num_threads=4):
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            import tensorflow as tf
            Interpreter = tf.lite.Interpreter
    return Interpreter(model_path=str(model_path), num_threads=num_threads)


def load_rgb(image, size):
    """Path / PIL image / HxWx3 uint8 array -> (size, size, 3) uint8, same as training (plain resize)."""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    elif not isinstance(image, Image.Image):
        image = Image.open(image)
    image = ImageOps.exif_transpose(image).convert("RGB")
    return np.asarray(image.resize((size, size), Image.BILINEAR), dtype=np.uint8)


@dataclass
class Prediction:
    label: str           # short class name, or "not_sure"
    top_label: str       # best class even when not sure
    confidence: float
    is_sure: bool
    advice: str
    top3: list
    latency_ms: float
    crop: str = "tomato"         # crop id, e.g. "apple"
    disease: str = ""            # disease id, e.g. "black_rot" / "healthy"; "" when not sure
    display: str = ""            # "Apple: Black rot" (best guess, shown on the page)
    class_id: str = ""           # "apple.black_rot" (best guess; stable id for MQTT / DB / advisor)

    def to_dict(self):
        return asdict(self)


class Predictor:
    def __init__(self, model_path, threshold=0.6, num_threads=4, classes=SHORT_CLASSES):
        self.interp = make_interpreter(model_path, num_threads)
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()[0]
        self.out = self.interp.get_output_details()[0]
        self.size = int(self.inp["shape"][1])
        self.threshold = threshold
        # Multi-crop models ship <name>.labels.json next to them (from import_pretrained.py --all-crops).
        # Labels are then catalog ids "crop.disease"; otherwise the old tomato short names are used.
        labels = Path(str(model_path).rsplit(".", 1)[0].removesuffix("_int8").removesuffix("_fp32") + ".labels.json")
        self.multi_crop = labels.exists() and classes is SHORT_CLASSES
        self.classes = json.loads(labels.read_text())["ids"] if self.multi_crop else classes
        n_out = int(self.out["shape"][-1])
        assert n_out == len(self.classes), f"model has {n_out} outputs but {len(self.classes)} labels"

    def _describe(self, label):
        """label -> (crop, disease, display, class_id, advice)."""
        if self.multi_crop:
            i = catalog.BY_ID[label]
            return i["crop"], i["disease"], i["display"], label, i["advice"]
        return ("tomato", label, f"Tomato: {label.replace('_', ' ')}", f"tomato.{label}",
                ADVICE.get(label, ""))

    # ---- low level
    def _quant_input(self, x):
        dtype = self.inp["dtype"]
        if dtype == np.float32:
            return x.astype(np.float32)
        scale, zp = self.inp["quantization"]
        if scale == 0:
            return x.astype(dtype)
        q = np.round(x.astype(np.float32) / scale + zp)
        info = np.iinfo(dtype)
        return np.clip(q, info.min, info.max).astype(dtype)

    def _dequant_output(self, y):
        if self.out["dtype"] == np.float32:
            return y
        scale, zp = self.out["quantization"]
        return (y.astype(np.float32) - zp) * scale

    def probs(self, rgb_uint8):
        """(size,size,3) uint8 -> probability vector."""
        x = self._quant_input(rgb_uint8[None].astype(np.float32))
        self.interp.set_tensor(self.inp["index"], x)
        self.interp.invoke()
        return self._dequant_output(self.interp.get_tensor(self.out["index"])[0])

    # ---- high level
    def predict(self, image):
        rgb = load_rgb(image, self.size)
        t0 = time.perf_counter()
        p = self.probs(rgb)
        ms = (time.perf_counter() - t0) * 1000
        order = np.argsort(p)[::-1]
        best = self.classes[order[0]]
        conf = float(p[order[0]])
        sure = conf >= self.threshold
        crop, disease, display, cid, advice = self._describe(best)
        return Prediction(
            label=best if sure else "not_sure", top_label=best, confidence=round(conf, 4),
            is_sure=sure, advice=advice if sure else NOT_SURE_MSG,
            top3=[(self.classes[i], round(float(p[i]), 4)) for i in order[:3]],
            latency_ms=round(ms, 2), crop=crop, disease=disease if sure else "",
            display=display, class_id=cid)
