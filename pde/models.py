"""Backbones: MobileNetV3-Small (keras.applications) and EfficientNet-Lite0 (pde/efficientnet_lite.py,
ImageNet weights from TF-Hub).

All models take float32 images in [0, 255] and output softmax probabilities over CLASSES.
"""
from pde.classes import CLASSES
from pde.tfcompat import keras, tf


def build_model(name, img_size=224, num_classes=len(CLASSES), weights="imagenet", dropout=0.2):
    inputs = keras.Input((img_size, img_size, 3), name="image")
    if name == "mobilenetv3small":
        backbone = keras.applications.MobileNetV3Small(
            input_shape=(img_size, img_size, 3), include_top=False, weights=weights,
            pooling="avg", include_preprocessing=True)  # rescales [0,255] internally
        backbone._name = "backbone"
        x = backbone(inputs, training=False)
    elif name == "efficientnet_lite0":
        try:
            # Keras re-implementation with ImageNet weights ported from TF-Hub (fully trainable)
            from pde.efficientnet_lite import lite0_backbone
            backbone = lite0_backbone((img_size, img_size, 3), weights=weights)
            x = backbone(inputs, training=False)
        except Exception as e:  # no internet / hub problem -> closest keras.applications model
            print(f"[models] EfficientNet-Lite0 weights failed ({e}); using EfficientNetB0 instead.")
            backbone = keras.applications.EfficientNetB0(
                input_shape=(img_size, img_size, 3), include_top=False, weights=weights, pooling="avg")
            backbone._name = "backbone"
            x = backbone(inputs, training=False)
    else:
        raise ValueError(f"unknown model '{name}'")
    x = keras.layers.Dropout(dropout, name="dropout")(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="probs")(x)
    return keras.Model(inputs, outputs, name=name)


def set_backbone_trainable(model, fraction):
    """fraction=0 -> backbone frozen; else unfreeze the top `fraction` of its layers
    (BatchNorm layers stay frozen, the usual recipe for small-data fine-tuning)."""
    bb = model.get_layer("backbone")
    bb.trainable = fraction > 0
    if fraction <= 0:
        return
    n = len(bb.layers)
    cut = int(n * (1 - fraction))
    for i, layer in enumerate(bb.layers):
        layer.trainable = i >= cut and not isinstance(layer, keras.layers.BatchNormalization)


def compile_model(model, lr):
    model.compile(optimizer=keras.optimizers.Adam(lr),
                  loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
                  metrics=["accuracy"])


def save_model(model, path):
    model.save(str(path), save_format="tf")  # SavedModel dir


def load_model(path):
    return keras.models.load_model(str(path), compile=False)


def count_params(model):
    return int(sum(tf.size(w).numpy() for w in model.weights))
