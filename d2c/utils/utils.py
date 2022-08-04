"""A collection of some little utils."""

import os
import random
import numpy as np
import torch
from typing import Dict, Generator, List


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


def chain_gene(*args: List[Generator]) -> Generator:
    """Connect several Generator objects into one Generator object."""
    for x in args:
        yield from x


def maybe_makedirs(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)


def set_seed(seed):
    seed %= 4294967294
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    print("Using seed {}".format(seed))


def abs_file_path(file, relative_path):
    return os.path.abspath(os.path.join(os.path.split(os.path.abspath(file))[0], relative_path))