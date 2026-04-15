from typing import Callable, Optional

import torch.nn.functional as F
from torch import nn


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, act_fn: Optional[Callable] = None):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.act_fn = act_fn or F.silu

    def forward(self, x):
        return self.w2(self.act_fn(self.w1(x)) * self.w3(x))
