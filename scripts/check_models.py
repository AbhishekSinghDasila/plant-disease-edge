"""Sanity check on the Pi: run every model on samples/*.jpg (file name = true class) and flag
models whose output is broken (all classes ~equal, e.g. INT8 on an old TFLite runtime).

    .venv/bin/python scripts/check_models.py            # runtime inside the venv
    python3 scripts/check_models.py                      # whatever runtime system Python has
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pde.classes import SHORT  # noqa: E402
from pde.predictor import Predictor, load_rgb  # noqa: E402


def runtime_name():
    for mod, name in [("ai_edge_litert", "ai-edge-litert"), ("tflite_runtime", "tflite-runtime"),
                      ("tensorflow", "tensorflow")]:
        try:
            m = __import__(mod)
            return f"{name} {getattr(m, '__version__', '?')}"
        except ImportError:
            continue
    return "none"


def main():
    models = sorted((ROOT / "models").glob(sys.argv[1] if len(sys.argv) > 1 else "*.tflite"))
    samples = sorted((ROOT / "samples").glob("*.jpg"))
    print(f"Runtime: {runtime_name()}  |  {len(samples)} sample photos\n")
    for m in models:
        pred = Predictor(m, threshold=0.0)
        correct, spreads = 0, []
        for f in samples:
            p = pred.probs(load_rgb(f, pred.size))
            spreads.append(float(p.max() - p.min()))
            correct += pred.classes[int(p.argmax())] == SHORT.get(f.stem, f.stem)
        broken = max(spreads) < 0.05
        status = "BROKEN (all classes equal - don't use)" if broken else "OK"
        print(f"{m.name:40s} {correct}/{len(samples)} correct   {status}")


if __name__ == "__main__":
    main()
