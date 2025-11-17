# models/global_bias.py
import tensorflow as tf
from tensorflow.keras import layers

class GlobalBias(layers.Layer):
    def __init__(self, initial_value=0.0, **kwargs):
        super().__init__(**kwargs)
        self.initial_value = float(initial_value)

    def build(self, input_shape):
        self.bias = self.add_weight(
            name="global_bias",
            shape=(),
            initializer=tf.keras.initializers.Constant(self.initial_value),
            trainable=True,
        )

    def call(self, inputs):
        return inputs + self.bias

    # following 2 methods make things serializable.  Only get_config is necessary.
    def get_config(self):
        cfg = super().get_config()
        cfg.update({"initial_value": self.initial_value})
        return cfg

    @classmethod
    def from_config(cls, config):
        # superfluous in most cases, but explicit:
        return cls(**config)
