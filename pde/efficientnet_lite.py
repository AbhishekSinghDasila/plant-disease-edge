"""EfficientNet-Lite0 in plain Keras, with ImageNet weights ported from the official TF-Hub module.

Why: the TF-Hub Lite0 is only published in the TF1 Hub format, which cannot be fine-tuned
(hub.KerasLayer(trainable=True) is refused). Rebuilding it as normal Keras layers makes it fully
trainable and lets TFLite quantize it like any other model.

Lite0 = EfficientNet-B0 without squeeze-and-excitation, with ReLU6 instead of swish
(Tan & Le, "EfficientNet-Lite", TensorFlow blog 2020). Input: float RGB in [0, 255].
"""
import os

import numpy as np

from pde.tfcompat import keras, tf

HUB_URL = "https://tfhub.dev/tensorflow/efficientnet/lite0/feature-vector/2"

# (kernel, repeats, in_ch, out_ch, expand_ratio, stride) - same as EfficientNet-B0
BLOCKS = [
    (3, 1, 32, 16, 1, 1),
    (3, 2, 16, 24, 6, 2),
    (5, 2, 24, 40, 6, 2),
    (3, 3, 40, 80, 6, 2),
    (5, 3, 80, 112, 6, 1),
    (5, 4, 112, 192, 6, 2),
    (3, 1, 192, 320, 6, 1),
]
BN = dict(momentum=0.99, epsilon=1e-3)


def _bn(name):
    return keras.layers.BatchNormalization(name=name, **BN)


def _relu6(name):
    return keras.layers.ReLU(6.0, name=name)


def build_lite0(input_shape=(224, 224, 3), dropout_connect=0.0):
    """Returns a Keras Model: image [0,255] -> 1280-d pooled features. Layer names follow the
    TF-Hub checkpoint ('blocks_3__conv2d' = 'blocks_3/conv2d') so weights can be copied by name."""
    inp = keras.Input(input_shape)
    x = keras.layers.Rescaling(1 / 128.0, offset=-127.0 / 128.0, name="normalize")(inp)
    x = keras.layers.Conv2D(32, 3, 2, "same", use_bias=False, name="stem__conv2d")(x)
    x = _bn("stem__tpu_batch_normalization")(x)
    x = _relu6("stem__relu6")(x)

    b = 0
    for k, reps, cin, cout, e, s in BLOCKS:
        for r in range(reps):
            stride = s if r == 0 else 1
            c_in = cin if r == 0 else cout
            p = f"blocks_{b}__"
            h = x
            bn_i = 0
            conv_i = 0
            if e != 1:
                h = keras.layers.Conv2D(c_in * e, 1, 1, "same", use_bias=False, name=p + "conv2d")(h)
                h = _bn(p + "tpu_batch_normalization")(h)
                h = _relu6(p + "expand_relu6")(h)
                bn_i, conv_i = 1, 1
            h = keras.layers.DepthwiseConv2D(k, stride, "same", use_bias=False, name=p + "depthwise_conv2d")(h)
            h = _bn(p + "tpu_batch_normalization" + (f"_{bn_i}" if bn_i else ""))(h)
            h = _relu6(p + "dw_relu6")(h)
            h = keras.layers.Conv2D(cout, 1, 1, "same", use_bias=False,
                                    name=p + "conv2d" + (f"_{conv_i}" if conv_i else ""))(h)
            h = _bn(p + f"tpu_batch_normalization_{bn_i + 1}")(h)
            if stride == 1 and c_in == cout:
                if dropout_connect:
                    h = keras.layers.Dropout(dropout_connect, noise_shape=(None, 1, 1, 1), name=p + "drop")(h)
                h = keras.layers.Add(name=p + "add")([h, x])
            x = h
            b += 1

    x = keras.layers.Conv2D(1280, 1, 1, "same", use_bias=False, name="head__conv2d")(x)
    x = _bn("head__tpu_batch_normalization")(x)
    x = _relu6("head__relu6")(x)
    x = keras.layers.GlobalAveragePooling2D(name="avg_pool")(x)
    return keras.Model(inp, x, name="backbone")


def _checkpoint_path():
    import tensorflow_hub as hub
    return os.path.join(hub.resolve(HUB_URL), "variables", "variables")


def load_imagenet_weights(model, ckpt=None):
    """Copy ImageNet weights from the TF-Hub checkpoint into the Keras model, by name."""
    ckpt = ckpt or _checkpoint_path()
    reader = tf.train.load_checkpoint(ckpt)
    names = set(reader.get_variable_to_shape_map())
    copied = 0
    for layer in model.layers:
        if not layer.weights:
            continue
        values = []
        for w in layer.weights:
            # layer "blocks_0__conv2d", weight "kernel" -> "efficientnet-lite0/blocks_0/conv2d/kernel"
            short = w.name.split("/")[-1].split(":")[0]
            key = f"efficientnet-lite0/{layer.name.replace('__', '/')}/{short}"
            if key not in names:
                raise KeyError(f"{key} not in checkpoint")
            v = reader.get_tensor(key)
            if v.shape != tuple(w.shape):
                raise ValueError(f"shape mismatch for {key}: {v.shape} vs {tuple(w.shape)}")
            values.append(v)
        layer.set_weights(values)
        copied += len(values)
    expected = len([n for n in names if n.startswith("efficientnet-lite0/") and "ExponentialMovingAverage" not in n])
    if copied != expected:
        raise RuntimeError(f"copied {copied} of {expected} checkpoint tensors")
    return copied


def lite0_backbone(input_shape=(224, 224, 3), weights="imagenet"):
    model = build_lite0(input_shape)
    if weights == "imagenet":
        load_imagenet_weights(model)
    return model


def check_against_hub(n=4, seed=0):
    """Max abs difference between this model and the official TF-Hub module (should be ~1e-4)."""
    import tensorflow_hub as hub
    ours = lite0_backbone()
    ref = hub.KerasLayer(HUB_URL, trainable=False)
    x = np.random.default_rng(seed).integers(0, 256, (n, 224, 224, 3)).astype(np.float32)
    a = ours.predict(x, verbose=0)
    b = ref(x / 255.0).numpy()
    return float(np.abs(a - b).max()), float(np.abs(b).mean())
