import tensorflow as tf
import numpy as np

#Random seed initialization
seed = 1234
np.random.seed(seed)
tf.random.set_seed(seed)

class Dictionary:
    def __init__(self):
        # Initializers as objects (avoid string ambiguity)
        self.Dict_initializers = {
            "GlorotUniform": tf.keras.initializers.GlorotUniform(),
            "GlorotNormal": tf.keras.initializers.GlorotNormal(),
            "HeUniform": tf.keras.initializers.HeUniform(),
            "HeNormal": tf.keras.initializers.HeNormal(),
            "LecunNormal": tf.keras.initializers.LecunNormal(),
        }
        # Activations (callables)
        self.Dict_activations = {
            "tanh": tf.nn.tanh,
            "sigmoid": tf.nn.sigmoid,
            "selu": tf.nn.selu,
            "relu": tf.nn.relu,
            "sin": tf.math.sin,
            "linear": tf.keras.activations.linear,
        }
        # Optimizers (instances; LR may be overridden by schedule in training)
        self.Dict_optimizers = {
            "Adam": tf.keras.optimizers.Adam(learning_rate=1e-3),
            "Adamax": tf.keras.optimizers.Adamax(learning_rate=1e-3),
            "Nadam": tf.keras.optimizers.Nadam(learning_rate=1e-3),
        }
        self.Dict = {
            "initializer": self.Dict_initializers,
            "activation": self.Dict_activations,
            "optimizer": self.Dict_optimizers,
        }


