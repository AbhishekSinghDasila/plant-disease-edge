"""Phase 5: does the cloud model (FP32 + test-time augmentation) beat the edge model (INT8, one view)?

    python scripts/eval_cloud_tta.py [--lab-n 600]
    python scripts/eval_cloud_tta.py --cloud-model models/cloud_v2.tflite --out results/cloud_v2_eval.csv   # Phase 5 v2

Sets: lab = PlantVillage tomato test photos (sample), field = real tomato field photos (data/field/test, all).
Per setup: accuracy, share accepted (not "not sure"), accuracy on accepted, and confident-wrong rate
(wrong answer given without "not sure": the failure that matters for a farmer).
Writes results/cloud_tta.csv and prints a table.
"""
import argparse
import csv
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "cloud/services/cloud_inference"))

from pde import catalog                       # noqa: E402
from pde.predictor import Predictor           # noqa: E402
from tta import CloudModel                    # noqa: E402

IMG = {".jpg", ".jpeg", ".png"}


def samples(folder, n=None, seed=0):
    items = [(p, catalog.class_id(d.name)) for d in sorted(folder.iterdir()) if d.is_dir()
             for p in sorted(d.iterdir()) if p.suffix.lower() in IMG]
    if n and len(items) > n:
        items = random.Random(seed).sample(items, n)
    return items


def score(rows):
    n = len(rows)
    acc = sum(r["pred"] == r["true"] for r in rows) / n
    accepted = [r for r in rows if r["sure"]]
    acc_acc = sum(r["pred"] == r["true"] for r in accepted) / max(1, len(accepted))
    conf_wrong = sum(r["sure"] and r["pred"] != r["true"] for r in rows) / n
    return {"n": n, "accuracy": round(acc, 4), "accepted": round(len(accepted) / n, 4),
            "acc_on_accepted": round(acc_acc, 4), "confident_wrong": round(conf_wrong, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lab-n", type=int, default=600)
    ap.add_argument("--threshold", type=float, default=0.8)
    ap.add_argument("--cloud-model", default="models/pv38_fp32.tflite")
    ap.add_argument("--cloud-labels", help="default: <model>.labels.json, or models/pv38.labels.json")
    ap.add_argument("--out", default="results/cloud_tta.csv")
    a = ap.parse_args()
    edge = Predictor(ROOT / "models/pv38_int8.tflite", threshold=a.threshold)
    cm = ROOT / a.cloud_model
    labels = Path(a.cloud_labels) if a.cloud_labels else cm.with_suffix(".labels.json")
    if not labels.exists():
        labels = ROOT / "models/pv38.labels.json"
    one = CloudModel(cm, labels, threshold=a.threshold, min_agreement=0, views=["full"])
    tta = CloudModel(cm, labels, threshold=a.threshold, min_agreement=0.8)
    print(f"cloud model {cm.name} ({len(tta.ids)} classes, {tta.size} px)")
    sets = {"lab": samples(ROOT / "data/lab/test", a.lab_n), "field": samples(ROOT / "data/field/test")}
    out = []
    for set_name, items in sets.items():
        res = {"edge_int8": [], "cloud_fp32": [], "cloud_fp32_tta6": [], "tta6_conf_only": [], "edge+cloud_agree": []}
        t0 = time.time()
        for path, true in items:
            e = edge.predict(path)
            res["edge_int8"].append({"true": true, "pred": e.top_label, "sure": e.is_sure})
            o = one.predict(path)
            res["cloud_fp32"].append({"true": true, "pred": o["class_id"], "sure": o["is_sure"]})
            t = tta.predict(path)
            res["cloud_fp32_tta6"].append({"true": true, "pred": t["class_id"], "sure": t["is_sure"]})
            res["tta6_conf_only"].append({"true": true, "pred": t["class_id"], "sure": t["confidence"] >= a.threshold})
            # advisor policy: a diagnosis only when both are sure and name the same class
            res["edge+cloud_agree"].append({"true": true, "pred": e.top_label,
                                            "sure": e.is_sure and t["is_sure"] and t["class_id"] == e.top_label})
        print(f"{set_name}: {len(items)} photos in {time.time() - t0:.0f} s")
        for name, rows in res.items():
            out.append({"set": set_name, "setup": name, **score(rows)})
    stamp = time.strftime("%Y-%m-%d %H:%M")
    path = ROOT / a.out
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["time", "model", "set", "setup", "n", "accuracy", "accepted", "acc_on_accepted", "confident_wrong"])
        w.writeheader()
        for r in out:
            w.writerow({"time": stamp, "model": cm.name, **r})
    print(f"\n{'set':6} {'setup':17} {'n':>5} {'acc':>6} {'accepted':>9} {'acc_acc':>8} {'conf_wrong':>10}")
    for r in out:
        print(f"{r['set']:6} {r['setup']:17} {r['n']:>5} {r['accuracy']:>6.3f} {r['accepted']:>9.3f} "
              f"{r['acc_on_accepted']:>8.3f} {r['confident_wrong']:>10.3f}")
    print(f"-> {path}")


if __name__ == "__main__":
    main()
