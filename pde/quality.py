"""Photo check before the disease model runs: is the photo sharp, and is there a leaf in it?

The disease model only knows 10 tomato classes and has no "not a leaf" class, so it gives a
confident disease name for anything (a sensor board came out as "mosaic virus 97 %", a blurry
healthy leaf as "late blight 99.6 %"). This check runs first and turns such photos into
"retake the photo", with the reason.

Thresholds were picked from data (2026-11-17, 400 random photos per set, image shrunk to 320 px):
                         sharpness p1 / p5      leaf fraction p1 / p5
  PlantVillage lab val      45 /  86              0.13 / 0.20
  PlantDoc field test      139 / 226              0.12 / 0.23
  blurry Pi-camera leaf     14                    0.51           -> rejected (blur)
  sensor board photos      27 and 107             0.06 and 0.09  -> rejected (no leaf)
min_sharpness 35 and min_leaf_fraction 0.10 reject both bad captures and keep > 99 % of real photos.
"""
import numpy as np
from PIL import Image

DEFAULTS = {"min_sharpness": 35.0, "min_leaf_fraction": 0.10}


def measure(image):
    """PIL image / HxWx3 uint8 array -> (sharpness, leaf_fraction)."""
    im = Image.fromarray(image) if isinstance(image, np.ndarray) else image
    im = im.convert("RGB")
    im.thumbnail((320, 320))  # same scale for every camera resolution
    g = np.asarray(im.convert("L"), np.float32)
    # variance of the Laplacian: low = blurry
    lap = 4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    sharpness = float(lap.var())
    hsv = np.asarray(im.convert("HSV"), np.float32)
    h, s, v = hsv[..., 0] * (360 / 255), hsv[..., 1] / 255, hsv[..., 2] / 255
    # leaf-coloured pixels: yellow -> green -> teal hues with some saturation, not too dark
    # (brown/dark lesions sit inside a leaf that is mostly in this range)
    leaf = (h >= 20) & (h <= 170) & (s >= 0.18) & (v >= 0.12)
    return sharpness, float(leaf.mean())


def check(image, min_sharpness=DEFAULTS["min_sharpness"], min_leaf_fraction=DEFAULTS["min_leaf_fraction"]):
    """Return a dict: ok, problem (None / "blurry" / "no_leaf"), message, sharpness, leaf_fraction."""
    sharp, leaf = measure(image)
    problem, msg = None, None
    if leaf < min_leaf_fraction:
        problem = "no_leaf"
        msg = ("No leaf found in the photo. Hold one leaf in front of the camera so it fills "
               "most of the picture, on a plain background.")
    elif sharp < min_sharpness:
        problem = "blurry"
        msg = ("Photo is too blurry. The Pi camera has a fixed focus: move the leaf further "
               "away (about 40-60 cm), keep it still, and use good light.")
    return {"ok": problem is None, "problem": problem, "message": msg,
            "sharpness": round(sharp, 1), "leaf_fraction": round(leaf, 3)}
