"""Evaluate Keras (SavedModel) or TFLite models on the lab and field test sets.

Metrics (proposal section 7): accuracy and macro-F1 on lab and field separately, plus the share
of images rejected as "not sure" at the confidence threshold and accuracy on the rest.

    python scripts/evaluate.py models/mobilenetv3small_standard models/mobilenetv3small_field_int8.tflite
    python scripts/evaluate.py --sets field models/*.tflite

Writes results/<model>__<set>.json, results/<model>__<set>_cm.png and appends to results/summary.csv.
Both model types use the same PIL image loading as the Pi app, so numbers are comparable.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pde.classes import CLASSES, SHORT_CLASSES  # noqa: E402
from pde.config import load_config, p  # noqa: E402
from pde.data import list_files  # noqa: E402
from pde.predictor import Predictor, load_rgb  # noqa: E402

SETS = {"lab": ("lab_dir", "test"), "field": ("field_dir", "test")}


def predict_all(model_path, paths, img_size, batch=64):
    model_path = Path(model_path)
    if model_path.suffix == ".tflite":
        pred = Predictor(model_path)
        return np.stack([pred.probs(load_rgb(f, pred.size)) for f in paths])
    from pde.models import load_model
    model = load_model(model_path)
    out = []
    for i in range(0, len(paths), batch):
        x = np.stack([load_rgb(f, img_size) for f in paths[i:i + batch]]).astype(np.float32)
        out.append(model.predict(x, verbose=0))
    return np.concatenate(out)


def metrics(y, probs, threshold):
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
    yhat = probs.argmax(1)
    conf = probs.max(1)
    present = sorted(set(y.tolist()))  # field set has no Target Spot -> F1 over present classes
    res = {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, yhat)),
        "macro_f1": float(f1_score(y, yhat, labels=present, average="macro", zero_division=0)),
        "per_class_f1": {SHORT_CLASSES[c]: float(v) for c, v in zip(
            present, f1_score(y, yhat, labels=present, average=None, zero_division=0))},
        "confusion_matrix": confusion_matrix(y, yhat, labels=range(len(CLASSES))).tolist(),
        "threshold": threshold,
    }
    sweep = []
    for t in [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        keep = conf >= t
        sweep.append({"threshold": t, "rejected_share": float(1 - keep.mean()),
                      "accuracy_on_accepted": float((yhat[keep] == y[keep]).mean()) if keep.any() else None})
    res["threshold_sweep"] = sweep
    keep = conf >= threshold
    res["rejected_share"] = float(1 - keep.mean())
    res["accuracy_on_accepted"] = float((yhat[keep] == y[keep]).mean()) if keep.any() else None
    return res


def plot_cm(cm, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = np.array(cm, dtype=float)
    rows = cm.sum(1, keepdims=True)
    norm = np.divide(cm, rows, out=np.zeros_like(cm), where=rows > 0)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(SHORT_CLASSES)), SHORT_CLASSES, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(SHORT_CLASSES)), SHORT_CLASSES, fontsize=8)
    for i in range(len(cm)):
        for j in range(len(cm)):
            if cm[i, j]:
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=7,
                        color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def model_name(path):
    path = Path(path)
    return path.stem if path.suffix == ".tflite" else f"{path.name}_keras"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--sets", nargs="+", default=["lab", "field"], choices=list(SETS))
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    thr = cfg["app"]["confidence_threshold"]
    out_dir = p(cfg["output"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = out_dir / "summary.csv"
    new = not summary.exists()

    with open(summary, "a", newline="") as fsum:
        w = csv.writer(fsum)
        if new:
            w.writerow(["time", "model", "set", "n", "accuracy", "macro_f1",
                        f"rejected_share@{thr}", f"acc_on_accepted@{thr}"])
        for m in args.models:
            for s in args.sets:
                key, split = SETS[s]
                paths, y = list_files(p(cfg["data"][key]) / split)
                if not paths:
                    print(f"skip {s}: no images")
                    continue
                probs = predict_all(m, paths, cfg["train"]["img_size"])
                r = metrics(y, probs, thr)
                name = model_name(m)
                r["model"], r["set"] = name, s
                (out_dir / f"{name}__{s}.json").write_text(json.dumps(r, indent=1))
                plot_cm(r["confusion_matrix"], f"{name} on {s} test (acc {r['accuracy']:.3f})",
                        out_dir / f"{name}__{s}_cm.png")
                acc_ok = r["accuracy_on_accepted"]
                w.writerow([time.strftime("%Y-%m-%d %H:%M"), name, s, r["n"], f"{r['accuracy']:.4f}",
                            f"{r['macro_f1']:.4f}", f"{r['rejected_share']:.4f}",
                            "" if acc_ok is None else f"{acc_ok:.4f}"])
                print(f"{name:40s} {s:5s} n={r['n']:5d} acc={r['accuracy']:.3f} "
                      f"macroF1={r['macro_f1']:.3f} rejected={r['rejected_share']:.2%}")
    print(f"Results in {out_dir}")


if __name__ == "__main__":
    main()
