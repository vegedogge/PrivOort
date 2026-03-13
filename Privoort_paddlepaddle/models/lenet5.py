import collections
from typing import Any, cast

# import torch.nn.functional as F
# from torch import nn
# import tensorflow as tf
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class Model(nn.Layer):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2D(1, 6, 5, padding=2)
        self.pool1 = nn.MaxPool2D(2, 2)
        self.conv2 = nn.Conv2D(6, 16, 5)
        self.pool2 = nn.MaxPool2D(2, 2)
        self.conv3 = nn.Conv2D(16, 120, 5)
        self.fc4 = nn.Linear(120 , 84)  # 28x28 -> 4x4 after conv/pool
        self.fc5 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = paddle.flatten(x, start_axis=1)
        x = F.relu(self.fc4(x))
        logits = self.fc5(x)
        return logits  # logits，loss 用 from_logits=True
