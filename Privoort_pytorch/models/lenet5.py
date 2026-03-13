import collections
from typing import Any, cast
import torch.nn.functional as F
from torch import nn

class Model(nn.Module):
    def __init__(self, num_classes: int = 10, cut_layer=None):
        super().__init__()
        self.cut_layer = cast(Any, cut_layer)
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2, bias=True)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0, bias=True)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5, bias=True)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc5 = nn.Linear(84, num_classes)

        self.layers = []
        self.layerdict = collections.OrderedDict()
        for name, layer in [
            ("conv1", self.conv1),
            ("relu1", self.relu1),
            ("pool1", self.pool1),
            ("conv2", self.conv2),
            ("relu2", self.relu2),
            ("pool2", self.pool2),
            ("conv3", self.conv3),
            ("relu3", self.relu3),
            ("flatten", self.flatten),
            ("fc4", self.fc4),
            ("relu4", self.relu4),
            ("fc5", self.fc5),
        ]:
            self.layers.append(name)
            self.layerdict[name] = layer

    def flatten(self, x):
        return x.view(x.size(0), -1)

    def forward(self, x):
        if self.cut_layer is not None and self.training:
            layer_index = self.layers.index(self.cut_layer)
            for i in range(layer_index + 1, len(self.layers)):
                x = self.layerdict[self.layers[i]](x)
        else:
            x = self.conv1(x)
            x = self.relu1(x)
            x = self.pool1(x)
            x = self.conv2(x)
            x = self.relu2(x)
            x = self.pool2(x)
            x = self.conv3(x)
            x = self.relu3(x)
            x = self.flatten(x)
            x = self.fc4(x)
            x = self.relu4(x)
            x = self.fc5(x)
        return F.log_softmax(x, dim=1)