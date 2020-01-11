"""
Models
"""

from tensorflow.keras import models, layers, activations


def build_model():
    """
    Create a TF model.
    """
    m = models.Sequential()
    m.add(layers.Input(shape=(28, 28)))
    m.add(layers.Flatten())
    m.add(layers.Dense(256, activation=activations.relu))
    m.add(layers.Dense(128, activation=activations.relu))
    m.add(layers.Dense(64, activation=activations.relu))
    m.add(layers.Dense(10, activation=activations.softmax))
    return m

