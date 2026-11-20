"""Cloud model: full-precision (FP32) 38-class network + test-time augmentation (TTA).

The same photo is shown to the model several times (as is, mirrored, flipped, zoomed in). The class
probabilities are averaged, and the share of views that agree with the final answer is an extra
"how sure" signal: a real, clear leaf gives the same answer from every view, while unknown things
(other plants, screens, walls) often flip between classes.

No TensorFlow needed: runs on LiteRT (ai-edge-litert), falls back to tflite-runtime / tensorflow.
"""
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


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


def crop_center(img, frac):
    w, h = img.size
    cw, ch = int(w * frac), int(h * frac)
    left, top = (w - cw) // 2, (h - ch) // 2
    return img.crop((left, top, left + cw, top + ch))


# name -> PIL image transform. Leaves have no fixed "up", so flips do not change the disease.
VIEWS = {
    "full": lambda im: im,
    "hflip": ImageOps.mirror,
    "vflip": ImageOps.flip,
    "crop85": lambda im: crop_center(im, 0.85),
    "crop85_hflip": lambda im: ImageOps.mirror(crop_center(im, 0.85)),
    "crop70": lambda im: crop_center(im, 0.70),
}


OTHER_REASON = {"other.not_a_leaf": "no plant leaf in the photo",
                "other.unknown_plant": "a plant the model does not know (not one of the 14 crops)"}


class CloudModel:
    def __init__(self, model_path, labels_path, threshold=0.8, min_agreement=0.8, views=tuple(VIEWS), num_threads=4):
        self.interp = make_interpreter(model_path, num_threads)
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()[0]
        self.out = self.interp.get_output_details()[0]
        assert self.inp["dtype"] == np.float32, "cloud model must be the FP32 export"
        self.size = int(self.inp["shape"][1])
        self.ids = json.loads(Path(labels_path).read_text())["ids"]
        self.threshold, self.min_agreement, self.views = threshold, min_agreement, list(views)
        self.name = Path(model_path).stem

    def probs(self, img):
        x = np.asarray(img.convert("RGB").resize((self.size, self.size), Image.BILINEAR), dtype=np.float32)[None]
        self.interp.set_tensor(self.inp["index"], x)
        self.interp.invoke()
        return self.interp.get_tensor(self.out["index"])[0].astype(np.float64)

    def predict(self, image):
        """image: path or PIL image -> dict with class_id, status, confidence, top3, agreement, per-view answers."""
        img = image if isinstance(image, Image.Image) else Image.open(image)
        img = ImageOps.exif_transpose(img).convert("RGB")
        t0 = time.perf_counter()
        per_view = {v: self.probs(VIEWS[v](img)) for v in self.views}
        ms = (time.perf_counter() - t0) * 1000
        p = np.mean(list(per_view.values()), axis=0)
        order = np.argsort(p)[::-1]
        best = int(order[0])
        votes = {v: self.ids[int(np.argmax(q))] for v, q in per_view.items()}
        agreement = sum(1 for c in votes.values() if c == self.ids[best]) / len(votes)
        conf = float(p[best])
        sure = conf >= self.threshold and agreement >= self.min_agreement
        cid = self.ids[best]
        crop, disease = cid.split(".", 1)
        why = None if sure else ("views disagree" if agreement < self.min_agreement else "low confidence")
        if crop == "other" and sure:             # v2 model: "not a leaf" / "unknown plant" are answers, not diseases
            sure, why = False, OTHER_REASON.get(cid, "not one of the 14 crops")
        return {"class_id": cid, "crop": crop, "disease": disease if sure else "",
                "status": "not_sure" if not sure else ("healthy" if disease == "healthy" else "disease"),
                "is_sure": sure, "confidence": round(conf, 4), "agreement": round(agreement, 3),
                "not_sure_reason": why, "top3": [(self.ids[int(i)], round(float(p[i]), 4)) for i in order[:3]],
                "views": votes, "latency_ms": round(ms, 1), "model": f"{self.name}+tta{len(self.views)}"}
