import collections
from typing import Any, cast
#import torch.nn.functional as F
#from torch import nn
import tensorflow as tf

class Model(tf.keras.Model):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(6, 5, padding="same", activation="relu")
        self.pool1 = tf.keras.layers.MaxPool2D(2, 2)
        self.conv2 = tf.keras.layers.Conv2D(16, 5, activation="relu")
        self.pool2 = tf.keras.layers.MaxPool2D(2, 2)
        self.conv3 = tf.keras.layers.Conv2D(120, 5, activation="relu")
        self.flatten = tf.keras.layers.Flatten()
        self.fc4 = tf.keras.layers.Dense(84, activation="relu")
        self.fc5 = tf.keras.layers.Dense(num_classes)
        self.build((None, 28, 28, 1))

    def call(self, inputs, training= None, mask = None):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.fc4(x)
        x = self.fc5(x)
        return x
