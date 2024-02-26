import torch
import torch.nn as nn
from torch import Tensor

from typing import List

__all__ = ["FC"]


class FC(nn.Module):
    def __init__(self, feats: List[int]) -> None:
        super(FC, self).__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feats[i], feats[i+1]),
                nn.ReLU()
            ) for i in range(len(feats) - 2)
        ])
        self.out = nn.Linear(feats[-2], feats[-1])

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers: x = layer(x)
        return self.out(x)
