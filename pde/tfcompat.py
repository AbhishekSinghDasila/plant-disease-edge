"""Import TensorFlow with Keras 2 (tf_keras).

Keras 2 is used because the TF-Hub EfficientNet-Lite0 layer and TFLite INT8 conversion
are most reliable with it. Always import keras from here:  from pde.tfcompat import tf, keras
"""
import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf  # noqa: E402
import tf_keras as keras  # noqa: E402

__all__ = ["tf", "keras", "seed_everything"]


def seed_everything(seed):
    """Like keras.utils.set_random_seed, which breaks tf_keras initializers on Python 3.12
    (it makes them call random.randint(1, 1e9) with a float)."""
    import random

    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
