"""Train a backbone on PlantVillage tomato (lab images) with transfer learning.

Two stages: (1) frozen backbone, train the head; (2) unfreeze the top of the backbone.

    python scripts/train.py --model mobilenetv3small --augment standard   # baseline
    python scripts/train.py --model mobilenetv3small --augment strong     # field-style augmentation
    python scripts/train.py --model efficientnet_lite0 --augment standard

Saves models/<model>_<augment>/ (SavedModel) and models/<model>_<augment>_history.json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pde.config import load_config, p  # noqa: E402
from pde.data import class_weights, split_dataset  # noqa: E402
from pde.models import build_model, compile_model, count_params, save_model, set_backbone_trainable  # noqa: E402
from pde.tfcompat import keras, seed_everything  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["mobilenetv3small", "efficientnet_lite0"])
    ap.add_argument("--augment", default="standard", choices=["none", "standard", "strong"])
    ap.add_argument("--epochs-head", type=int)
    ap.add_argument("--epochs-finetune", type=int)
    ap.add_argument("--weights", default="imagenet", help="'imagenet' or 'none' (for offline smoke tests)")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    tc, dc = cfg["train"], cfg["data"]
    seed_everything(dc["seed"])
    size, bs = tc["img_size"], tc["batch_size"]
    lab = p(dc["lab_dir"])

    train_ds, train_labels = split_dataset(lab / "train", size, bs, training=True, augment=args.augment)
    val_ds, _ = split_dataset(lab / "val", size, bs)
    cw = class_weights(train_labels)

    weights = None if args.weights == "none" else args.weights
    model = build_model(args.model, size, weights=weights)
    print(f"{args.model}: {count_params(model):,} parameters, {len(train_labels)} training images")

    out = p(cfg["output"]["models_dir"]) / f"{args.model}_{args.augment}"
    out.parent.mkdir(parents=True, exist_ok=True)
    best_w = str(out) + "_best.weights.h5"
    cbs = [keras.callbacks.ModelCheckpoint(best_w, monitor="val_accuracy", save_best_only=True,
                                           save_weights_only=True),
           keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=False)]

    history = {}
    t0 = time.time()
    # stage 1: head only
    set_backbone_trainable(model, 0)
    compile_model(model, tc["lr_head"])
    e1 = args.epochs_head if args.epochs_head is not None else tc["epochs_head"]
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=e1, class_weight=cw, callbacks=cbs, verbose=2)
    history["head"] = h1.history

    # stage 2: unfreeze top of backbone
    e2 = args.epochs_finetune if args.epochs_finetune is not None else tc["epochs_finetune"]
    if e2 > 0:
        model.load_weights(best_w)
        set_backbone_trainable(model, tc["unfreeze_fraction"])
        compile_model(model, tc["lr_finetune"])
        cbs[0].best = max(h1.history["val_accuracy"])  # only overwrite if stage 2 beats stage 1
        h2 = model.fit(train_ds, validation_data=val_ds, epochs=e2, class_weight=cw, callbacks=cbs, verbose=2)
        history["finetune"] = h2.history

    model.load_weights(best_w)
    save_model(model, out)
    Path(best_w).unlink(missing_ok=True)
    history["train_minutes"] = round((time.time() - t0) / 60, 1)
    Path(str(out) + "_history.json").write_text(json.dumps(history, indent=1, default=float))
    print(f"Saved {out}  ({history['train_minutes']} min)")


if __name__ == "__main__":
    main()
