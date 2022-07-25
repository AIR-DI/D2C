"""Tools for logging the training information."""

import wandb
import numpy as np
from typing import Dict
from torch.utils.tensorboard import SummaryWriter


def write_summary_tensorboard(writer: SummaryWriter, step: int, info: Dict) -> None:
    for key, val in info.items():
        if isinstance(
                val, (int, float, np.int32, np.int64, np.float32, np.float64)):
            writer.add_scalar(key, val, step)


def write_summary_wandb(info: Dict) -> None:
    wandb.log(info)
