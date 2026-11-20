# %% [markdown]
# # Phase 5 v2: cloud model for the plant project (Colab GPU)
#
# Trains **EfficientNet-B1** on 40 classes:
# * the 38 PlantVillage crop/disease classes (same ids as the edge model),
# * **`other.not_a_leaf`**: walls, screens, hands, textures, objects (DTD textures + Imagenette),
# * **`other.unknown_plant`**: leaves of plants the model does not know (TFDS `cassava` field photos).
#
# Training data: PlantVillage (lab photos) + **PlantDoc field photos** (repeated ×4) with strong augmentation.
# The project's own lab/field **test photos are excluded by checksum** (`HOLDOUT`), so the results on the laptop stay fair.
#
# **How to run:** Runtime → Change runtime type → **T4 GPU** → Save. Then Runtime → **Run all**.
# Takes about 1.5–2 h. At the end the browser downloads **`cloud_v2.tflite`** and **`cloud_v2.labels.json`**
# (also saved to Google Drive `MyDrive/plant_cloud_v2/` if you allow Drive access).
# On the laptop: `bash scripts/install_cloud_v2.sh ~/Downloads` (evaluates and switches the cloud model).

# %%
import base64
import gzip
import hashlib
import json
import os
import random
import shutil
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path

SMOKE = os.environ.get("SMOKE") == "1"          # tiny local test run (no GPU, no downloads)
WORK = Path(os.environ.get("WORK", "/content/work"))
IMG = 64 if SMOKE else 240                       # EfficientNet-B1 native size
BATCH = 8 if SMOKE else 48
EPOCHS_HEAD, EPOCHS_FT = (1, 1) if SMOKE else (2, 8)
CAP_PER_CLASS = 30 if SMOKE else 1500            # PlantVillage images per class (balance + time)
FIELD_REPEAT = 4                                 # PlantDoc field photos are rare and the most valuable
CAP_OTHER = 40 if SMOKE else 3000                # per "other" class
SEED = 42
random.seed(SEED)
WORK.mkdir(parents=True, exist_ok=True)
print("SMOKE" if SMOKE else "FULL", "run, work dir", WORK)

# --- filled in by scripts/make_colab_notebook.py (class map + holdout checksums from the laptop)
CLASSES_JSON = None
HOLDOUT_B64 = None
if CLASSES_JSON is None:                         # running the .py directly
    here = Path(__file__).resolve().parent
    CLASSES = json.loads((here / "classes.json").read_text())
    HOLDOUT = set((here / "holdout_md5.txt").read_text().split())
else:
    CLASSES = json.loads(CLASSES_JSON)
    HOLDOUT = set(gzip.decompress(base64.b64decode(HOLDOUT_B64)).decode().split())
OTHER = ["other.not_a_leaf", "other.unknown_plant"]
IDS = CLASSES["ids"] + OTHER
IDX = {c: i for i, c in enumerate(IDS)}
print(len(IDS), "classes,", len(HOLDOUT), "held-out test photos")


def norm(s):
    return " ".join(s.lower().replace("_", " ").split())


# PlantDoc folder -> class id (PlantDoc has no target spot, esca, leaf blight, citrus greening, ...)
PLANTDOC = {norm(k): v for k, v in {
    "Apple Scab Leaf": "apple.apple_scab", "Apple leaf": "apple.healthy", "Apple rust leaf": "apple.cedar_apple_rust",
    "Bell_pepper leaf": "pepper.healthy", "Bell_pepper leaf spot": "pepper.bacterial_spot",
    "Blueberry leaf": "blueberry.healthy", "Cherry leaf": "cherry.healthy",
    "Corn Gray leaf spot": "corn.gray_leaf_spot", "Corn leaf blight": "corn.northern_leaf_blight",
    "Corn rust leaf": "corn.common_rust", "Peach leaf": "peach.healthy",
    "Potato leaf": "potato.healthy", "Potato leaf early blight": "potato.early_blight",
    "Potato leaf late blight": "potato.late_blight", "Raspberry leaf": "raspberry.healthy",
    "Soyabean leaf": "soybean.healthy", "Soybean leaf": "soybean.healthy",
    "Squash Powdery mildew leaf": "squash.powdery_mildew", "Strawberry leaf": "strawberry.healthy",
    "Tomato Early blight leaf": "tomato.early_blight", "Tomato Septoria leaf spot": "tomato.septoria_leaf_spot",
    "Tomato leaf": "tomato.healthy", "Tomato leaf bacterial spot": "tomato.bacterial_spot",
    "Tomato leaf late blight": "tomato.late_blight", "Tomato leaf mosaic virus": "tomato.mosaic_virus",
    "Tomato leaf yellow virus": "tomato.yellow_leaf_curl_virus", "Tomato mold leaf": "tomato.leaf_mold",
    "Tomato two spotted spider mites leaf": "tomato.spider_mites",
    "grape leaf": "grape.healthy", "grape leaf black rot": "grape.black_rot"}.items()}
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def images(folder):
    return sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in IMG_EXT)


def readable(p):
    """Web-scraped photos can be broken or in odd formats: keep only files TensorFlow can decode."""
    from PIL import Image as _I
    try:
        with _I.open(p) as im:
            return im.format in ("JPEG", "PNG", "BMP", "GIF")
    except Exception:
        return False


def split_of(path, val=0.1, test=0.05):
    """Deterministic split from the file name hash (same split on every run)."""
    h = int(hashlib.sha1(str(path).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "test" if h < test else "val" if h < test + val else "train"


# %% [markdown]
# ## 1. Download the data
# PlantVillage and PlantDoc from GitHub (same sources as the laptop's `scripts/prepare_data.py`).

# %%
def git_clone(url, dest, sparse=None):
    if dest.exists():
        return
    cmd = ["git", "clone", "--depth", "1"] + (["--filter=blob:none", "--sparse"] if sparse else []) + [url, str(dest)]
    subprocess.run(cmd, check=True)
    if sparse:
        subprocess.run(["git", "-C", str(dest), "sparse-checkout", "set", sparse], check=True)


if SMOKE:
    PV_COLOR = Path(os.environ["PV_DIR"])        # e.g. data/pv_allcrops (folders = PlantVillage names)
    PD_ROOT = Path(os.environ["PD_DIR"])
else:
    git_clone("https://github.com/spMohanty/PlantVillage-Dataset.git", WORK / "PlantVillage", sparse="raw/color")
    git_clone("https://github.com/pratikkayal/PlantDoc-Dataset.git", WORK / "PlantDoc")
    PV_COLOR, PD_ROOT = WORK / "PlantVillage/raw/color", WORK / "PlantDoc"

items = defaultdict(list)                        # split -> [(path, class_id, source)]
skipped = Counter()
for d in sorted(PV_COLOR.iterdir()):
    cid = CLASSES["pv_folder_to_id"].get(d.name)
    if not cid:
        continue
    files = images(d)
    random.Random(SEED).shuffle(files)
    kept = 0
    for p in files:
        if md5(p) in HOLDOUT:
            skipped["lab holdout"] += 1
            continue
        s = split_of(p)
        if s == "train" and kept >= CAP_PER_CLASS:
            continue
        kept += s == "train"
        items[s].append((p, cid, "lab"))
for split_dir in ("train", "test"):
    for d in sorted((PD_ROOT / split_dir).iterdir()) if (PD_ROOT / split_dir).exists() else []:
        cid = PLANTDOC.get(norm(d.name))
        if not cid:
            print("PlantDoc folder not mapped:", d.name)
            continue
        for p in images(d):
            if md5(p) in HOLDOUT:
                skipped["field holdout"] += 1
                continue
            if not readable(p):
                skipped["unreadable"] += 1
                continue
            items[split_of(p, val=0.1, test=0.1)].append((p, cid, "field"))
print("held out (not used):", dict(skipped))

# %% [markdown]
# ## 2. "Not a leaf" and "unknown plant" images
# From TensorFlow Datasets. Each source is optional: if a download fails, training continues without it.

# %%
import numpy as np                               # noqa: E402
from PIL import Image                            # noqa: E402

OTHER_DIR = WORK / "other"


def save_tfds(name, split, cid, cap, skip_label=None, size=320):
    """Saves up to `cap` images of a TFDS dataset as JPEGs into OTHER_DIR/<cid>/. Returns the count."""
    try:                                         # Colab's Python 3.13 image lacks it and tfds.load then fails
        import importlib_resources  # noqa: F401
    except ImportError:
        subprocess.run(["pip", "-q", "install", "importlib_resources"], check=False)
    import tensorflow_datasets as tfds
    out = OTHER_DIR / cid
    out.mkdir(parents=True, exist_ok=True)
    try:
        ds, info = tfds.load(name, split=split, with_info=True, shuffle_files=True)
    except Exception as e:
        print(f"!! {name} not available ({e}); continuing without it")
        return 0
    names = info.features["label"].names if "label" in info.features else None
    n = 0
    for ex in ds:
        if n >= cap:
            break
        if skip_label and names and skip_label in names[int(ex["label"])].lower():
            continue
        im = Image.fromarray(ex["image"].numpy()).convert("RGB")
        im.thumbnail((size, size))
        im.save(out / f"{name.split('/')[0]}_{n:05d}.jpg", quality=90)
        n += 1
    print(f"{name}: {n} images -> {cid}")
    return n


if SMOKE:                                        # synthetic stand-ins: flat colours and noise
    for cid in OTHER:
        (OTHER_DIR / cid).mkdir(parents=True, exist_ok=True)
        for i in range(CAP_OTHER):
            a = np.random.default_rng(i).integers(0, 255, (96, 96, 3), dtype=np.uint8)
            if cid == "other.not_a_leaf" and i % 2:
                a[:] = a[0, 0]
            Image.fromarray(a).save(OTHER_DIR / cid / f"s{i:03d}.jpg")
elif not any(OTHER_DIR.glob("*/*.jpg")):
    save_tfds("dtd", "train+validation+test", "other.not_a_leaf", CAP_OTHER // 2)
    save_tfds("imagenette/320px-v2", "train", "other.not_a_leaf", CAP_OTHER // 2)
    # plant_leaves (Mendeley) and beans returned HTTP 403 on 2026-11-19; cassava (field photos of a crop the model
    # does not know) downloads fine. Add more unknown-plant sources here when they work again.
    save_tfds("cassava", "train+validation+test", "other.unknown_plant", CAP_OTHER)

for cid in OTHER:
    for p in images(OTHER_DIR / cid):
        items[split_of(p, val=0.1, test=0.1)].append((p, cid, "other"))

for s in ("train", "val", "test"):
    c = Counter(src for _, _, src in items[s])
    print(f"{s:5}: {len(items[s]):6} images {dict(c)}")
missing = [c for c in IDS if not any(cid == c for _, cid, _ in items["train"])]
print("classes without training images:", missing or "none")

# %% [markdown]
# ## 3. Input pipeline with strong augmentation
# Field photos differ from lab photos in background, light, blur and angle, so training makes lab photos look more
# like field photos: random crops/zoom, flips and 90° turns, colour/brightness changes, blur and JPEG noise.

# %%
import tensorflow as tf                          # noqa: E402

AUTOTUNE = tf.data.AUTOTUNE


def to_lists(rows, repeat_field=1):
    paths, labels = [], []
    for p, cid, src in rows:
        for _ in range(repeat_field if src == "field" else 1):
            paths.append(str(p))
            labels.append(IDX[cid])
    return paths, labels


def load(path, label):
    img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    return tf.cast(img, tf.float32), label


def augment(img, label):
    shape = tf.shape(img)
    side = tf.cast(tf.cast(tf.minimum(shape[0], shape[1]), tf.float32) * tf.random.uniform([], 0.6, 1.0), tf.int32)
    img = tf.image.random_crop(img, [side, side, 3])
    img = tf.image.resize(img, [IMG, IMG])
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.rot90(img, tf.random.uniform([], 0, 4, tf.int32))
    img = tf.image.random_brightness(img, 40.0)
    img = tf.image.random_contrast(img, 0.7, 1.3)
    img = tf.image.random_saturation(img, 0.7, 1.3)
    img = tf.image.random_hue(img, 0.04)
    if tf.random.uniform([]) < 0.25:             # out-of-focus look (fixed-focus Pi camera)
        k = tf.ones([5, 5, 1, 1]) / 25.0
        img = tf.concat([tf.nn.conv2d(img[None, ..., i:i + 1], k, 1, "SAME")[0] for i in range(3)], -1)
    img = tf.clip_by_value(img, 0, 255)
    if tf.random.uniform([]) < 0.3:              # phone / webcam compression
        img = tf.cast(tf.image.random_jpeg_quality(tf.cast(img, tf.uint8), 30, 90), tf.float32)
    return img, label


def plain(img, label):
    return tf.image.resize(img, [IMG, IMG]), label


def dataset(rows, train):
    paths, labels = to_lists(rows, FIELD_REPEAT if train else 1)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if train:
        ds = ds.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(load, num_parallel_calls=AUTOTUNE).ignore_errors()     # one bad file must not stop a 1.5 h run
    ds = ds.map(augment if train else plain, num_parallel_calls=AUTOTUNE)
    return ds.batch(BATCH).prefetch(AUTOTUNE)


train_ds, val_ds = dataset(items["train"], True), dataset(items["val"], False)

# %% [markdown]
# ## 4. Train: head first, then fine-tune the whole network

# %%
if not SMOKE and tf.config.list_physical_devices("GPU"):
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
print("GPU:", tf.config.list_physical_devices("GPU") or "none (slow!)")


def build(weights="imagenet"):
    base = tf.keras.applications.EfficientNetB1(include_top=False, weights=weights, input_shape=(IMG, IMG, 3),
                                                pooling="avg")      # takes raw 0-255 pixels (rescaling built in)
    x = tf.keras.layers.Dropout(0.3)(base.output)
    out = tf.keras.layers.Dense(len(IDS), activation="softmax", dtype="float32")(x)
    return tf.keras.Model(base.input, out), base


model, base = build(None if SMOKE else "imagenet")
paths_tr, labels_tr = to_lists(items["train"], FIELD_REPEAT)
counts = Counter(labels_tr)
class_weight = {i: min(5.0, len(labels_tr) / (len(IDS) * counts[i])) for i in counts}   # rare classes count more
ckpt = str(WORK / "best.weights.h5")
cbs = [tf.keras.callbacks.ModelCheckpoint(ckpt, monitor="val_accuracy", save_best_only=True, save_weights_only=True),
       tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True)]

base.trainable = False
model.compile(tf.keras.optimizers.Adam(1e-3), "sparse_categorical_crossentropy", metrics=["accuracy"])
t0 = time.time()
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_HEAD, class_weight=class_weight,
          steps_per_epoch=3 if SMOKE else None, validation_steps=2 if SMOKE else None)

base.trainable = True
for layer in base.layers:                        # keep BatchNorm statistics from ImageNet
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False
# len(train_ds) is unknown because of ignore_errors(), so count batches from the file list (bug in the first Colab run)
steps = (3 if SMOKE else -(-len(paths_tr) // BATCH)) * EPOCHS_FT
model.compile(tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.CosineDecay(1e-4, steps)),
              "sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FT, class_weight=class_weight, callbacks=cbs,
          steps_per_epoch=3 if SMOKE else None, validation_steps=2 if SMOKE else None)
print(f"training took {(time.time() - t0) / 60:.0f} min")

# %% [markdown]
# ## 5. Test on the held-back split (lab, field and "other" separately)

# %%
def evaluate(rows, name):
    if not rows:
        return None
    ds = dataset(rows, False)
    p = model.predict(ds.take(2) if SMOKE else ds, verbose=0)
    y = np.array([IDX[c] for _, c, _ in rows][:len(p)])
    pred, conf = p.argmax(1), p.max(1)
    sure = conf >= 0.8
    r = {"n": int(len(y)), "accuracy": round(float((pred == y).mean()), 4),
         "answered": round(float(sure.mean()), 4),
         "confident_wrong": round(float((sure & (pred != y)).mean()), 4)}
    print(f"{name:7} {r}")
    return r


metrics = {src: evaluate([r for r in items["test"] if r[2] == src], src) for src in ("lab", "field", "other")}

# %% [markdown]
# ## 6. Export to TFLite (FP32, input = raw 0-255 RGB, batch 1) and check it matches Keras

# %%
tf.keras.mixed_precision.set_global_policy("float32")
m32, _ = build(None)
m32.set_weights(model.get_weights())


import keras                                    # noqa: E402

# SavedModel with a fixed batch-1 signature, then TFLite (from_concrete_functions crashes with Keras 3)
saved = WORK / "saved_v2"
shutil.rmtree(saved, ignore_errors=True)
archive = keras.export.ExportArchive()
archive.track(m32)
archive.add_endpoint(name="serve", fn=lambda x: m32(x, training=False),
                     input_signature=[tf.TensorSpec([1, IMG, IMG, 3], tf.float32)])
archive.write_out(str(saved))
conv = tf.lite.TFLiteConverter.from_saved_model(str(saved))
OUT = WORK / "export"
OUT.mkdir(exist_ok=True)
(OUT / "cloud_v2.tflite").write_bytes(conv.convert())

interp = tf.lite.Interpreter(model_path=str(OUT / "cloud_v2.tflite"))
interp.allocate_tensors()
i_in, i_out = interp.get_input_details()[0]["index"], interp.get_output_details()[0]["index"]
diffs = []
for p, _, _ in items["test"][:20]:
    x = plain(*load(str(p), 0))[0][None].numpy()
    interp.set_tensor(i_in, x)
    interp.invoke()
    diffs.append(float(np.abs(interp.get_tensor(i_out) - m32(x).numpy()).max()))
print("TFLite vs Keras max difference:", max(diffs))
assert max(diffs) < 1e-3, "TFLite export does not match Keras"

(OUT / "cloud_v2.labels.json").write_text(json.dumps({
    "ids": IDS, "image_size": IMG, "model": "efficientnet_b1_v2", "trained": time.strftime("%Y-%m-%d %H:%M"),
    "test_metrics": metrics, "train_images": len(paths_tr)}, indent=1))
print("exported:", sorted(p.name for p in OUT.iterdir()), f"{(OUT / 'cloud_v2.tflite').stat().st_size / 1e6:.1f} MB")

# %% [markdown]
# ## 7. Download the two files (and copy them to Google Drive)

# %%
if not SMOKE:
    try:
        from google.colab import drive, files
        try:
            drive.mount("/content/drive")
            dst = Path("/content/drive/MyDrive/plant_cloud_v2")
            dst.mkdir(parents=True, exist_ok=True)
            for f in OUT.iterdir():
                shutil.copy(f, dst / f.name)
            print("copied to", dst)
        except Exception as e:
            print("Drive copy skipped:", e)
        files.download(str(OUT / "cloud_v2.tflite"))
        files.download(str(OUT / "cloud_v2.labels.json"))
    except ImportError:
        print("not on Colab: files are in", OUT)
