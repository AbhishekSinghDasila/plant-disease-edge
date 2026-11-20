"""tf.data pipelines and augmentation.

Images enter the model as float32 in [0, 255], shape (H, W, 3). Each model does its own
scaling inside the graph, so the Pi app only has to resize the photo.

Augmentation levels:
  'standard' - flips, rotation, zoom, mild brightness/contrast (typical PlantVillage training)
  'strong'   - standard + field-style changes: background clutter (leaf pasted smaller onto
               another photo), hue/saturation shifts, blur, noise, random crops
"""
from pathlib import Path

import numpy as np

from pde.classes import CLASSES
from pde.tfcompat import tf

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
AUTOTUNE = tf.data.AUTOTUNE


def list_files(split_dir, classes=CLASSES):
    """Return (paths, labels) for split_dir/<class>/*. Missing class folders are allowed."""
    paths, labels = [], []
    for i, cls in enumerate(classes):
        d = Path(split_dir) / cls
        if not d.exists():
            continue
        for f in sorted(d.iterdir()):
            if f.suffix.lower() in IMG_EXT:
                paths.append(str(f))
                labels.append(i)
    return paths, np.array(labels, dtype=np.int32)


def load_image(path, img_size):
    raw = tf.io.read_file(path)
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, (img_size, img_size), antialias=True)
    return tf.cast(img, tf.float32)


# ------------------------------------------------------------------ augmentation
def _standard(img):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    k = tf.random.uniform([], 0, 4, dtype=tf.int32)
    img = tf.image.rot90(img, k)
    # random zoom-in crop (80-100%)
    size = tf.shape(img)[0]
    scale = tf.random.uniform([], 0.8, 1.0)
    crop = tf.cast(tf.cast(size, tf.float32) * scale, tf.int32)
    img = tf.image.random_crop(img, [crop, crop, 3])
    img = tf.image.resize(img, (size, size))
    img = tf.image.random_brightness(img, 0.1 * 255)
    img = tf.image.random_contrast(img, 0.85, 1.15)
    return tf.clip_by_value(img, 0, 255)


def _blur(img):
    k = tf.constant([1., 4., 6., 4., 1.]) / 16.
    k2 = tf.tensordot(k, k, axes=0)[:, :, None, None]
    k2 = tf.tile(k2, [1, 1, 3, 1])
    return tf.nn.depthwise_conv2d(img[None], k2, [1, 1, 1, 1], "SAME")[0]


def _strong_single(img):
    img = _standard(img)
    img = tf.image.random_hue(img / 255., 0.04) * 255.
    img = tf.image.random_saturation(img, 0.6, 1.4)
    img = tf.image.random_brightness(img, 0.2 * 255)
    img = tf.cond(tf.random.uniform([]) < 0.3, lambda: _blur(img), lambda: img)
    noise = tf.random.normal(tf.shape(img), stddev=tf.random.uniform([], 0, 8))
    img = tf.cond(tf.random.uniform([]) < 0.3, lambda: img + noise, lambda: img)
    return tf.clip_by_value(img, 0, 255)


def background_clutter(images, prob=0.5):
    """Batch-level: shrink each leaf image (55-90%) and paste it at a random spot on top of
    another image of the batch, so the model sees leaves on busy, non-uniform backgrounds."""
    b = tf.shape(images)[0]
    size = tf.shape(images)[1]
    bg = tf.random.shuffle(images)
    bg = tf.image.flip_left_right(bg)
    bg = tf.image.adjust_brightness(bg / 255., -0.1) * 255.

    def paste(args):
        fg, back = args
        scale = tf.random.uniform([], 0.55, 0.9)
        s = tf.cast(tf.cast(size, tf.float32) * scale, tf.int32)
        small = tf.image.resize(fg, (s, s))
        oy = tf.random.uniform([], 0, size - s + 1, dtype=tf.int32)
        ox = tf.random.uniform([], 0, size - s + 1, dtype=tf.int32)
        padded = tf.image.pad_to_bounding_box(small, oy, ox, size, size)
        mask = tf.image.pad_to_bounding_box(tf.ones_like(small), oy, ox, size, size)
        return padded * mask + back * (1 - mask)

    pasted = tf.map_fn(paste, (images, bg), fn_output_signature=tf.float32)
    use = tf.cast(tf.random.uniform([b, 1, 1, 1]) < prob, tf.float32)
    return use * pasted + (1 - use) * images


# ------------------------------------------------------------------ datasets
def make_dataset(paths, labels, img_size, batch_size, training=False, augment="standard",
                 num_classes=len(CLASSES), seed=42):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(lambda p, y: (load_image(p, img_size), y), num_parallel_calls=AUTOTUNE)
    if training and augment == "standard":
        ds = ds.map(lambda x, y: (_standard(x), y), num_parallel_calls=AUTOTUNE)
    elif training and augment == "strong":
        ds = ds.map(lambda x, y: (_strong_single(x), y), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=training and len(paths) >= batch_size)
    if training and augment == "strong":
        ds = ds.map(lambda x, y: (background_clutter(x), y), num_parallel_calls=AUTOTUNE)
    ds = ds.map(lambda x, y: (x, tf.one_hot(y, num_classes)), num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)


def split_dataset(split_dir, img_size, batch_size, **kw):
    paths, labels = list_files(split_dir)
    if not paths:
        raise FileNotFoundError(f"No images in {split_dir}. Run scripts/prepare_data.py first.")
    return make_dataset(paths, labels, img_size, batch_size, **kw), labels


def class_weights(labels, num_classes=len(CLASSES)):
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    counts[counts == 0] = 1
    w = counts.sum() / (num_classes * counts)
    return {i: float(w[i]) for i in range(num_classes)}
