import torch
import torch.nn as nn

import math
import torch.nn.functional as F

from core.dynamic_conv import Dynamic_conv2d


class Dense(nn.Module):
    """A fully connected layer that reshapes outputs to feature maps."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.dense(x)[..., None, None]


class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class SpatialModel(nn.Module):
    def __init__(self, spectral_num=4, inner_num=16):
        super().__init__()
        self.spectral_num = spectral_num
        self.conv1 = Dynamic_conv2d(self.spectral_num, inner_num, kernel_size=3, stride=2, padding=1)
        self.conv2 = Dynamic_conv2d(inner_num, inner_num, kernel_size=3, stride=1, padding=1)
        self.conv3 = Dynamic_conv2d(inner_num, inner_num, kernel_size=3, stride=1, padding=1)
        self.conv4 = Dynamic_conv2d(inner_num, self.spectral_num, kernel_size=3, stride=2, padding=1)
        self.act = SiLU()

    def forward(self, x):
        h1 = self.act(self.conv1(x))
        h1 = self.act(self.conv2(h1))
        h1 = self.act(self.conv3(h1))
        h1 = self.act(self.conv4(h1))
        return h1


# MS2P NET
class SpectralModel(nn.Module):
    def __init__(self, spectral_num=4, inner_num=16):
        super().__init__()
        self.spectral_num = spectral_num
        self.conv1 = Dynamic_conv2d(self.spectral_num, inner_num, kernel_size=3, stride=1, padding=1)
        self.conv2 = Dynamic_conv2d(inner_num, inner_num, kernel_size=3, stride=1, padding=1)
        self.conv3 = Dynamic_conv2d(inner_num, 1, kernel_size=1, stride=1, padding=0)
        self.act = SiLU()

    def forward(self, x):
        h1 = self.act(self.conv1(x))
        h1 = self.act(self.conv2(h1))
        h1 = self.act(self.conv3(h1))
        return h1


if __name__ == "__main__":
    x = torch.randn((1, 4, 256, 256))
    model = SpatialModel()
    print(model(x).shape)
