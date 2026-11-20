"""Benchmark TFLite models on the Raspberry Pi (proposal section 7).

Per model: file size, mean / p95 latency per image, FPS, peak RAM (each model runs in its own
process so RAM is measured separately), and optionally accuracy on the test sets.
No TensorFlow needed - only ai-edge-litert (or tflite-runtime), numpy, pillow.

    python scripts/benchmark_pi.py models/*.tflite
    python scripts/benchmark_pi.py models/*.tflite --accuracy     # also needs data/lab/test, data/field/test

Writes results/pi_benchmark.csv
"""
import argparse
import csv
import json
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pde.predictor import Predictor, load_rgb  # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png"}


def test_images(split_dir):
    from pde.classes import CLASSES
    paths, labels = [], []
    for i, c in enumerate(CLASSES):
        d = Path(split_dir) / c
        if d.exists():
            for f in sorted(d.iterdir()):
                if f.suffix.lower() in IMG_EXT:
                    paths.append(f)
                    labels.append(i)
    return paths, np.array(labels)


def run_single(model, runs, threads, accuracy, threshold):
    """Runs inside a child process; prints one JSON line."""
    pred = Predictor(model, threshold=threshold, num_threads=threads)
    rng = np.random.default_rng(0)
    x = rng.integers(0, 256, (pred.size, pred.size, 3), dtype=np.uint8)
    for _ in range(10):
        pred.probs(x)  # warm-up
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        pred.probs(x)
        times.append((time.perf_counter() - t0) * 1000)
    t = np.array(times)
    res = {"model": Path(model).name, "size_mb": round(Path(model).stat().st_size / 1e6, 3),
           "input_dtype": str(pred.inp["dtype"].__name__), "threads": threads,
           "latency_mean_ms": round(float(t.mean()), 2), "latency_p95_ms": round(float(np.percentile(t, 95)), 2),
           "fps": round(1000 / float(t.mean()), 1)}
    if accuracy:
        for name, d in [("lab", ROOT / "data/lab/test"), ("field", ROOT / "data/field/test")]:
            paths, y = test_images(d)
            if not len(paths):
                continue
            probs = np.stack([pred.probs(load_rgb(f, pred.size)) for f in paths])
            yhat, conf = probs.argmax(1), probs.max(1)
            keep = conf >= threshold
            res[f"{name}_acc"] = round(float((yhat == y).mean()), 4)
            res[f"{name}_rejected"] = round(float(1 - keep.mean()), 4)
            res[f"{name}_acc_accepted"] = round(float((yhat[keep] == y[keep]).mean()), 4) if keep.any() else None
    # ru_maxrss: KiB on Linux
    res["peak_ram_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    print("RESULT " + json.dumps(res))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--accuracy", action="store_true")
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--out", default=str(ROOT / "results" / "pi_benchmark.csv"))
    ap.add_argument("--single", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.single:
        run_single(args.models[0], args.runs, args.threads, args.accuracy, args.threshold)
        return

    rows = []
    for m in args.models:
        cmd = [sys.executable, __file__, m, "--single", "--runs", str(args.runs),
               "--threads", str(args.threads), "--threshold", str(args.threshold)]
        if args.accuracy:
            cmd.append("--accuracy")
        out = subprocess.run(cmd, capture_output=True, text=True)
        line = next((ln for ln in out.stdout.splitlines() if ln.startswith("RESULT ")), None)
        if not line:
            print(f"{m}: failed\n{out.stderr[-2000:]}")
            continue
        r = json.loads(line[7:])
        rows.append(r)
        print(f"{r['model']:42s} {r['size_mb']:6.2f} MB  {r['latency_mean_ms']:7.1f} ms  "
              f"{r['fps']:5.1f} FPS  RAM {r['peak_ram_mb']:6.1f} MB"
              + (f"  lab {r.get('lab_acc')}  field {r.get('field_acc')}" if args.accuracy else ""))

    if rows:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["device"] + keys)
            w.writeheader()
            for r in rows:
                w.writerow({"device": platform.node() + " " + platform.machine(), **r})
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
