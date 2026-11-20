"""Close the lab->field gap: fine-tune a lab-trained model on a few field images.

Field images (data/field/finetune) are mixed with lab images so the model does not forget the
lab data. Strong augmentation is used. Evaluate afterwards on data/field/test (never seen).

    python scripts/finetune_field.py --base models/mobilenetv3small_strong
Saves models/<model>_field/
"""
import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pde.config import load_config, p  # noqa: E402
from pde.data import class_weights, list_files, make_dataset  # noqa: E402
from pde.models import compile_model, load_model, save_model, set_backbone_trainable  # noqa: E402
from pde.tfcompat import seed_everything  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="SavedModel folder from train.py")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    tc, dc, fc = cfg["train"], cfg["data"], cfg["field_finetune"]
    seed_everything(dc["seed"])

    f_paths, f_labels = list_files(p(dc["field_dir"]) / "finetune")
    if not f_paths:
        sys.exit("No field fine-tune images. Run scripts/prepare_data.py first.")
    l_paths, l_labels = list_files(p(dc["lab_dir"]) / "train")
    rng = random.Random(dc["seed"])
    idx = rng.sample(range(len(l_paths)), min(len(l_paths), fc["mix_lab_ratio"] * len(f_paths)))
    paths = f_paths + [l_paths[i] for i in idx]
    labels = list(f_labels) + [int(l_labels[i]) for i in idx]
    print(f"Fine-tuning on {len(f_paths)} field + {len(idx)} lab images")

    import numpy as np
    labels = np.array(labels, dtype=np.int32)
    ds = make_dataset(paths, labels, tc["img_size"], tc["batch_size"], training=True, augment="strong")

    model = load_model(args.base)
    set_backbone_trainable(model, tc["unfreeze_fraction"])
    compile_model(model, fc["lr"])
    model.fit(ds, epochs=args.epochs or fc["epochs"], class_weight=class_weights(labels), verbose=2)

    name = Path(args.base).name.rsplit("_", 1)[0]  # mobilenetv3small_strong -> mobilenetv3small
    out = p(cfg["output"]["models_dir"]) / f"{name}_field"
    save_model(model, out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
