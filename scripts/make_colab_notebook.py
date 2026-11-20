"""Build notebooks/cloud_v2/train_cloud_v2_colab.ipynb from train_cloud_v2.py ("# %%" cells).

Embeds classes.json and the held-out test checksums, so the notebook needs no extra uploads.
    python scripts/make_colab_notebook.py
"""
import base64
import gzip
import json
from pathlib import Path

D = Path(__file__).resolve().parent.parent / "notebooks/cloud_v2"
src = (D / "train_cloud_v2.py").read_text()
classes = json.dumps(json.loads((D / "classes.json").read_text()))
holdout = base64.b64encode(gzip.compress((D / "holdout_md5.txt").read_bytes(), 9)).decode()
src = src.replace("CLASSES_JSON = None", f"CLASSES_JSON = {classes!r}", 1)
src = src.replace("HOLDOUT_B64 = None", f"HOLDOUT_B64 = {holdout!r}", 1)
assert "CLASSES_JSON = None" not in src and "HOLDOUT_B64 = None" not in src

cells = []
for chunk in src.split("# %%")[1:]:
    md = chunk.startswith(" [markdown]")
    body = chunk.split("\n", 1)[1].strip("\n")
    if md:
        text = "\n".join(line[2:] if line.startswith("# ") else line.lstrip("#") for line in body.splitlines())
        cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)})
    else:
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                      "source": body.splitlines(keepends=True)})
nb = {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                   "kernelspec": {"name": "python3", "display_name": "Python 3"}}}
out = D / "train_cloud_v2_colab.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(f"{out} ({len(cells)} cells, {out.stat().st_size // 1024} KB)")
