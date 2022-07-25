"""Policies used by various agents."""

import torch
import numpy as np
from torch import nn, Tensor
from typing import Union


class DeterministicPolicy(nn.Module):
    """Returns deterministic action."""

    def __init__(self, a_network: nn.Module) -> None:
        super(DeterministicPolicy, self).__init__()
        self._a_network = a_network

    def forward(self, observation: Union[np.ndarray, Tensor]) -> Tensor:
        with torch.no_grad():
            action = self._a_network(observation)
        return action
