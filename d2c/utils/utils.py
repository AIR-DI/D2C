"""A collection of some little utils."""

import numpy as np
import torch
from typing import Dict


def get_summary_str(step: int = None, info: Dict = None, prefix: str = '') -> str:
    summary_str = prefix
    if step is not None:
        summary_str += 'Step %d; ' % step
    for key, val in info.items():
        if isinstance(val, (int, np.int32, np.int64)):
            summary_str += '%s %d; ' % (key, val)
        elif isinstance(val, (float, np.float32, np.float64)):
            summary_str += '%s %.4g; ' % (key, val)
    return summary_str


def write_summary(*args, **kwargs):
    raise NotImplementedError


def get_optimizer(name):
    """Get an optimizer generator that returns an optimizer according to lr."""
    if name == 'adam':
        def adam_opt_(parameters, lr, weight_decay=0.0):
            return torch.optim.Adam(params=parameters, lr=lr, weight_decay=weight_decay)

        return adam_opt_
    else:
        raise ValueError('Unknown optimizer %s.' % name)


class Flags(object):

    def __init__(self, **kwargs) -> None:
        for key, val in kwargs.items():
            setattr(self, key, val)

