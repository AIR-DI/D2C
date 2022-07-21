"""A collection of some little utils."""

import numpy as np


def get_summary_str(step=None, info=None, prefix=''):
    summary_str = prefix
    if step is not None:
        summary_str += 'Step %d; ' % (step)
    for key, val in info.items():
        if isinstance(val, (int, np.int32, np.int64)):
            summary_str += '%s %d; ' % (key, val)
        elif isinstance(val, (float, np.float32, np.float64)):
            summary_str += '%s %.4g; ' % (key, val)
    return summary_str


def write_summary(*args, **kwargs):
    raise NotImplementedError

