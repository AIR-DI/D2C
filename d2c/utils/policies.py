"""Policies used by various agents."""

import torch
import numpy as np
from torch import nn, Tensor
from typing import Union
from d2c.utils.utils import to_array_as


class DeterministicPolicy(nn.Module):
    """Returns deterministic action."""

    def __init__(self, a_network: nn.Module) -> None:
        super(DeterministicPolicy, self).__init__()
        self._a_network = a_network

    def forward(self, observation: Union[np.ndarray, Tensor]) -> np.ndarray:
        with torch.no_grad():
            action = self._a_network(observation)
        return to_array_as(action, observation)


class DeterministicSoftPolicy(nn.Module):
    """Returns mode of policy distribution."""

    def __init__(self, a_network: nn.Module) -> None:
        super(DeterministicSoftPolicy, self).__init__()
        self._a_network = a_network

    def forward(self, observation: Union[np.ndarray, Tensor]) -> np.ndarray:
        with torch.no_grad():
            action = self._a_network(observation)[0]
        return to_array_as(action, observation)
