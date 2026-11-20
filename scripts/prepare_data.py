"""Download PlantVillage + PlantDoc, keep the tomato classes, and split them.

Output layout (one sub-folder per class in each split):
    data/lab/{train,val,test}/<class>/      PlantVillage (lab photos)
    data/field/{finetune,test}/<class>/     PlantDoc (+ team photos) (field photos)

Usage:
    python scripts/prepare_data.py --download          # needs ~/.kaggle/kaggle.json and git
    python scripts/prepare_data.py --plantvillage /path/to/plantvillage/color --plantdoc /path/to/PlantDoc-Dataset

PlantVillage source: Kaggle "abdallahalidev/plantvillage-dataset" (we use the 'color' images).
PlantDoc source:     https://github.com/pratikkayal/PlantDoc-Dataset
Team photos (optional): data/own_photos/<class short name or PlantVillage name>/*.jpg
"""
import argparse
import hashlib
import random
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pde.classes import CLASSES, PLANTDOC_TO_CLASS, SHORT  # noqa: E402
from pde.config import load_config, p  # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def norm(name):
    """'Tomato__Tomato_YellowLeaf__Curl_Virus' and 'Tomato___Tomato_Yellow_Leaf_Curl_Virus' -> same key."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


CANON_BY_NORM = {norm(c): c for c in CLASSES}
CANON_BY_NORM.update({norm(s): c for c, s in SHORT.items()})
PLANTDOC_BY_NORM = {norm(k): v for k, v in PLANTDOC_TO_CLASS.items()}


def images_in(folder):
    return sorted(f for f in Path(folder).rglob("*") if f.suffix.lower() in IMG_EXT)


def md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------- download
def download(raw):
    raw.mkdir(parents=True, exist_ok=True)
    pv = raw / "plantvillage"
    if not pv.exists():
        print("Downloading PlantVillage from Kaggle (~2 GB) ...")
        subprocess.run(["kaggle", "datasets", "download", "-d", "abdallahalidev/plantvillage-dataset",
                        "-p", str(pv), "--unzip"], check=True)
    pd = raw / "PlantDoc-Dataset"
    if not pd.exists():
        print("Cloning PlantDoc from GitHub ...")
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/pratikkayal/PlantDoc-Dataset.git", str(pd)], check=True)
    return find_plantvillage_color(pv), pd


def find_plantvillage_color(root):
    """Find the folder that directly contains the PlantVillage class folders (prefer 'color')."""
    candidates = []
    for d in Path(root).rglob("*"):
        if d.is_dir() and norm(d.name) == norm("Tomato___healthy"):
            candidates.append(d.parent)
    if not candidates:
        sys.exit(f"Could not find PlantVillage tomato folders under {root}")
    candidates.sort(key=lambda d: (0 if "color" in str(d).lower() else 1, len(str(d))))
    return candidates[0]


# ---------------------------------------------------------------- collect
def collect_plantvillage(src):
    out = {}
    for d in Path(src).iterdir():
        if d.is_dir() and norm(d.name) in CANON_BY_NORM and d.name.lower().startswith("tomato"):
            out[CANON_BY_NORM[norm(d.name)]] = images_in(d)
    missing = set(CLASSES) - set(out)
    if missing:
        sys.exit(f"PlantVillage is missing classes: {missing}")
    return out


def collect_plantdoc(root):
    out = {c: [] for c in CLASSES}
    for split in ("train", "test"):
        base = Path(root) / split
        if not base.exists():
            continue
        for d in base.iterdir():
            if d.is_dir() and norm(d.name) in PLANTDOC_BY_NORM:
                out[PLANTDOC_BY_NORM[norm(d.name)]] += images_in(d)
    return out


def collect_own(folder):
    out = {c: [] for c in CLASSES}
    folder = Path(folder)
    if not folder.exists():
        return out
    for d in folder.iterdir():
        if d.is_dir() and norm(d.name) in CANON_BY_NORM:
            out[CANON_BY_NORM[norm(d.name)]] += images_in(d)
        elif d.is_dir():
            print(f"  ! ignoring team-photo folder '{d.name}' (not a known class name)")
    return out


def dedupe(files):
    seen, keep = set(), []
    for f in files:
        h = md5(f)
        if h not in seen:
            seen.add(h)
            keep.append(f)
    return keep


# ---------------------------------------------------------------- write
def copy_split(files_by_split, dest_root, prefix=""):
    for split, by_class in files_by_split.items():
        for cls, files in by_class.items():
            d = dest_root / split / cls
            d.mkdir(parents=True, exist_ok=True)
            for i, f in enumerate(files):
                shutil.copy2(f, d / f"{prefix}{i:05d}{f.suffix.lower()}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--download", action="store_true", help="download both datasets into data/raw")
    ap.add_argument("--plantvillage", help="folder containing PlantVillage class folders (e.g. .../color)")
    ap.add_argument("--plantdoc", help="PlantDoc-Dataset folder (has train/ and test/)")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)["data"]
    rng = random.Random(cfg["seed"])

    if args.download:
        pv_src, pd_src = download(p(cfg["raw_dir"]))
    else:
        if not (args.plantvillage and args.plantdoc):
            ap.error("give --download, or both --plantvillage and --plantdoc")
        pv_src, pd_src = find_plantvillage_color(args.plantvillage), Path(args.plantdoc)

    lab_dir, field_dir = p(cfg["lab_dir"]), p(cfg["field_dir"])
    for d in (lab_dir, field_dir):
        if d.exists():
            shutil.rmtree(d)

    # --- lab: PlantVillage train/val/test, stratified per class
    print(f"PlantVillage source: {pv_src}")
    pv = collect_plantvillage(pv_src)
    s = cfg["split"]
    lab = {"train": {}, "val": {}, "test": {}}
    for cls, files in pv.items():
        files = files[:]
        rng.shuffle(files)
        n_tr, n_va = int(len(files) * s["train"]), int(len(files) * s["val"])
        lab["train"][cls], lab["val"][cls], lab["test"][cls] = (
            files[:n_tr], files[n_tr:n_tr + n_va], files[n_tr + n_va:])
    copy_split(lab, lab_dir, "pv_")

    # --- field: PlantDoc (+ team photos) -> finetune (k per class) / test (rest)
    print(f"PlantDoc source: {pd_src}")
    field_sources = [("pd_", collect_plantdoc(pd_src)), ("own_", collect_own(p(cfg["own_photos_dir"])))]
    k = cfg["field_finetune_per_class"]
    for prefix, by_class in field_sources:
        field = {"finetune": {}, "test": {}}
        for cls, files in by_class.items():
            files = dedupe(files)
            rng.shuffle(files)
            # keep at least half for testing when a class is small
            n_ft = min(k, len(files) // 2)
            field["finetune"][cls], field["test"][cls] = files[:n_ft], files[n_ft:]
        copy_split(field, field_dir, prefix)

    # --- summary
    print("\nImages per class:")
    print(f"{'class':28s} {'lab tr':>7s} {'lab va':>7s} {'lab te':>7s} {'fld ft':>7s} {'fld te':>7s}")
    for cls in CLASSES:
        row = [len(images_in(d)) if d.exists() else 0 for d in (
            lab_dir / "train" / cls, lab_dir / "val" / cls, lab_dir / "test" / cls,
            field_dir / "finetune" / cls, field_dir / "test" / cls)]
        print(f"{SHORT[cls]:28s} " + " ".join(f"{n:7d}" for n in row))
    empty = [SHORT[c] for c in CLASSES if not (field_dir / "test" / c).exists()
             or not images_in(field_dir / "test" / c)]
    if empty:
        print(f"\nNote: no field test images for: {', '.join(empty)} (PlantDoc has no Target Spot).")


if __name__ == "__main__":
    main()
