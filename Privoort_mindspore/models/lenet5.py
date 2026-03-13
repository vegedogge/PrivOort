import collections
from typing import Any, cast

# import torch.nn.functional as F
# from torch import nn
# import tensorflow as tf
# import paddle
# import paddle.nn as nn
# import paddle.nn.functional as F

import mindspore
from mindspore import nn, ops



class Model(nn.Cell):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, pad_mode="pad", padding=2)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, pad_mode="valid")
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5, pad_mode="valid")
        self.flatten = nn.Flatten()
        self.fc4 = nn.Dense(120, 84)  # type: ignore
        self.fc5 = nn.Dense(84, num_classes)  # type: ignore

    def construct(self, x):
        x = ops.relu(self.conv1(x))
        x = self.pool1(x)
        x = ops.relu(self.conv2(x))
        x = self.pool2(x)
        x = ops.relu(self.conv3(x))
        x = self.flatten(x)
        x = ops.relu(self.fc4(x))
        logits = self.fc5(x)
        return logits  # logits，loss 用 from_logits=True
