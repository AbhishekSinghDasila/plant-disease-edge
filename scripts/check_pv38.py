"""Quick accuracy check of the 38-class models on all 14 crops.

Images: data/pv_allcrops/<folder>/* (about 30 per non-tomato class, also used for INT8 calibration)
and data/lab/test/<tomato folder>/* (first 30 per class). Prints per-crop accuracy and appends to
results/summary.csv.

    python scripts/check_pv38.py models/pv38_int8.tflite [models/pv38_fp32.tflite ...]
"""
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pde import catalog  # noqa: E402
from pde.predictor import Predictor  # noqa: E402


def images():
    for folder in catalog.PV38:
        d = ROOT / "data" / "pv_allcrops" / folder
        if not d.exists():
            d = ROOT / "data" / "lab" / "test" / folder
        files = sorted(f for f in d.glob("*") if f.suffix.lower() in (".jpg", ".jpeg", ".png"))[:30]
        for f in files:
            yield catalog.class_id(folder), f


def main():
    data = list(images())
    for model in sys.argv[1:] or ["models/pv38_int8.tflite"]:
        pred = Predictor(ROOT / model, threshold=0.6)
        per_crop = defaultdict(lambda: [0, 0])
        ok = crop_ok = 0
        for cid, f in data:
            p = pred.predict(f)
            hit = p.top_label == cid
            ok += hit
            crop_ok += p.crop == catalog.BY_ID[cid]["crop"]
            per_crop[catalog.BY_ID[cid]["crop"]][0] += hit
            per_crop[catalog.BY_ID[cid]["crop"]][1] += 1
        n = len(data)
        print(f"\n{model}: {n} images, class accuracy {ok / n:.4f}, crop accuracy {crop_ok / n:.4f}")
        for crop, (a, b) in sorted(per_crop.items()):
            print(f"  {crop:12s} {a:3d}/{b:3d}  {a / b:.2f}")
        with open(ROOT / "results" / "summary.csv", "a") as fh:
            fh.write(f"{time.strftime('%Y-%m-%d %H:%M')},{Path(model).stem},pv38_allcrops,{n},{ok / n:.4f},,,\n")


if __name__ == "__main__":
    main()
