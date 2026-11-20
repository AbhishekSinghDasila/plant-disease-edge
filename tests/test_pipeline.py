"""End-to-end smoke test on tiny synthetic data (no downloads, no GPU, ~2-4 min on a laptop).

Checks every stage runs and hands the right files to the next:
fake PlantVillage/PlantDoc folders -> prepare_data -> train -> finetune_field -> quantize
-> evaluate -> benchmark_pi -> Predictor -> Flask app routes.

    pytest -q tests/                # full smoke test
    pytest -q tests/ -k "not slow"  # only fast unit tests
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pde.classes import CLASSES, PLANTDOC_TO_CLASS, SHORT  # noqa: E402

# PlantVillage folder names as they appear in two popular Kaggle copies
PV_NAMES_ALT = {
    "Tomato___Bacterial_spot": "Tomato_Bacterial_spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Tomato_Spider_mites_Two_spotted_spider_mite",
}


def fake_leaf(cls_idx, rng, size=96, field=False):
    """Class = hue of a blob, so a tiny model can learn it. Field images get a cluttered background."""
    bg = rng.integers(0, 255, (size, size, 3)).astype(np.uint8) if field else np.full((size, size, 3), 200, np.uint8)
    yy, xx = np.mgrid[:size, :size]
    cy, cx, r = rng.integers(35, 60), rng.integers(35, 60), rng.integers(22, 32)
    mask = (yy - cy) ** 2 + (xx - cx) ** 2 < r ** 2
    color = np.array(Image.new("HSV", (1, 1), (int(cls_idx * 25), 220, 200)).convert("RGB")).reshape(3)
    img = bg.copy()
    img[mask] = np.clip(color + rng.normal(0, 10, (mask.sum(), 3)), 0, 255).astype(np.uint8)
    return Image.fromarray(img)


# ------------------------------------------------------------------ fast unit tests
def test_name_normalisation():
    sys.path.insert(0, str(ROOT / "scripts"))
    import prepare_data as pd
    for canon, alt in PV_NAMES_ALT.items():
        assert pd.CANON_BY_NORM[pd.norm(alt)] == canon
    for pdname, canon in PLANTDOC_TO_CLASS.items():
        assert pd.PLANTDOC_BY_NORM[pd.norm(pdname)] == canon
    assert set(SHORT) == set(CLASSES)


def test_advice_for_every_class():
    from pde.classes import ADVICE, SHORT_CLASSES
    assert set(ADVICE) == set(SHORT_CLASSES)


def test_lite0_matches_checkpoint_layout():
    """Our Keras EfficientNet-Lite0 has exactly the 245 tensors of the TF-Hub checkpoint."""
    from pde.efficientnet_lite import build_lite0
    m = build_lite0()
    keys = set()
    for layer in m.layers:
        for w in layer.weights:
            keys.add(f"{layer.name.replace('__', '/')}/{w.name.split('/')[-1].split(':')[0]}")
    assert len(keys) == 245
    assert "blocks_15/conv2d_1/kernel" in keys and "head/tpu_batch_normalization/moving_variance" in keys
    assert m.output_shape == (None, 1280)


# ------------------------------------------------------------------ full pipeline
@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    ws = tmp_path_factory.mktemp("pde")
    rng = np.random.default_rng(0)
    pv = ws / "raw" / "plantvillage" / "color"
    for i, c in enumerate(CLASSES):
        d = pv / PV_NAMES_ALT.get(c, c)
        d.mkdir(parents=True)
        for k in range(30):
            fake_leaf(i, rng).save(d / f"{k}.jpg")
    (pv / "Potato___healthy").mkdir()  # non-tomato class must be ignored
    pdoc = ws / "raw" / "PlantDoc-Dataset"
    for split, n in [("train", 14), ("test", 4)]:
        for name, c in PLANTDOC_TO_CLASS.items():
            d = pdoc / split / name
            d.mkdir(parents=True)
            for k in range(n):
                fake_leaf(CLASSES.index(c), rng, field=True).save(d / f"{k}.jpg")

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    cfg["output"] = {"models_dir": str(ws / "models"), "results_dir": str(ws / "results")}
    for k in ("raw_dir", "lab_dir", "field_dir", "own_photos_dir"):
        cfg["data"][k] = str(ws / k.replace("_dir", ""))
    cfg["data"]["field_finetune_per_class"] = 5
    cfg["train"].update(img_size=96, batch_size=16, epochs_head=3, epochs_finetune=1)
    cfg["field_finetune"]["epochs"] = 1
    cfg["mqtt"]["enabled"] = False
    (ws / "config.yaml").write_text(yaml.safe_dump(cfg))
    return ws


def run(*args):
    r = subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, f"{args}\nSTDOUT:\n{r.stdout[-3000:]}\nSTDERR:\n{r.stderr[-3000:]}"
    return r.stdout


@pytest.mark.slow
def test_end_to_end(workspace):
    ws, cfg = workspace, workspace / "config.yaml"
    S = ROOT / "scripts"

    out = run(S / "prepare_data.py", "--config", cfg, "--plantvillage", ws / "raw/plantvillage",
              "--plantdoc", ws / "raw/PlantDoc-Dataset")
    assert "no field test images for: target_spot" in out
    assert len(list((ws / "lab/train").rglob("*.jpg"))) == 10 * 21
    # 18 PlantDoc images per class -> 5 finetune / 13 test, for 9 classes
    assert len(list((ws / "field/finetune").rglob("*.jpg"))) == 45
    assert not (ws / "lab/train/Potato___healthy").exists()

    run(S / "train.py", "--config", cfg, "--model", "mobilenetv3small", "--augment", "strong", "--weights", "none")
    base = ws / "models/mobilenetv3small_strong"
    assert (base / "saved_model.pb").exists()
    assert "head" in json.loads((ws / "models/mobilenetv3small_strong_history.json").read_text())

    run(S / "finetune_field.py", "--config", cfg, "--base", base)
    field = ws / "models/mobilenetv3small_field"
    assert (field / "saved_model.pb").exists()

    run(S / "quantize.py", "--config", cfg, field, "--calib", "40")
    fp32, int8 = ws / "models/mobilenetv3small_field_fp32.tflite", ws / "models/mobilenetv3small_field_int8.tflite"
    assert int8.stat().st_size < fp32.stat().st_size

    out = run(S / "evaluate.py", "--config", cfg, field, fp32, int8)
    assert (ws / "results/mobilenetv3small_field_int8__field.json").exists()
    assert (ws / "results/mobilenetv3small_field_int8__lab_cm.png").exists()
    r = json.loads((ws / "results/mobilenetv3small_field_int8__lab.json").read_text())
    assert 0 <= r["accuracy"] <= 1 and len(r["threshold_sweep"]) == 8
    # FP32 TFLite must match the Keras model closely
    k = json.loads((ws / "results/mobilenetv3small_field_keras__lab.json").read_text())
    f = json.loads((ws / "results/mobilenetv3small_field_fp32__lab.json").read_text())
    assert abs(k["accuracy"] - f["accuracy"]) < 0.02

    out = run(S / "benchmark_pi.py", fp32, int8, "--runs", "10", "--out", ws / "results/pi_benchmark.csv")
    assert "FPS" in out

    # Predictor + app
    from pde.predictor import Predictor
    pred = Predictor(int8, threshold=0.0)
    res = pred.predict(next((ws / "lab/test").rglob("*.jpg")))
    assert res.is_sure and res.label in SHORT.values() and len(res.top3) == 3
    assert Predictor(int8, threshold=1.01).predict(fake_leaf(0, np.random.default_rng(1))).label == "not_sure"

    sys.path.insert(0, str(ROOT / "app"))
    import server
    server.STATE["predictor"] = pred
    server.STATE["captures"] = ws / "captures"
    c = server.app.test_client()
    assert c.get("/").status_code == 200
    img = next((ws / "field/test").rglob("*.jpg"))
    with open(img, "rb") as fh:
        r = c.post("/api/predict", data={"image": (fh, "leaf.jpg")}, content_type="multipart/form-data")
    assert r.status_code == 200 and "label" in r.get_json()
    assert c.post("/api/capture").status_code == 503  # no camera in tests
    assert len(c.get("/api/history").get_json()) == 1
    assert (ws / "captures/log.csv").exists()
