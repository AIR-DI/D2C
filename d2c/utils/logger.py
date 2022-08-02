"""Tools for logging the training information."""

import os
import wandb
import numpy as np
from typing import Dict, Optional
from torch.utils.tensorboard import SummaryWriter


def write_summary_tensorboard(writer: SummaryWriter, step: int, info: Dict) -> None:
    for key, val in info.items():
        if isinstance(
                val, (int, float, np.int32, np.int64, np.float32, np.float64)):
            writer.add_scalar(key, val, step)


class WandbLogger:
    """Weights and Biases logger that sends data to https://wandb.ai/.

    :param str project: W&B project name.
    :param str entity: W&B team/organization name. Default to None.
    :param str name: W&B run name. Default to None. If None, random name is assigned.
    :param str run_id: run id of W&B run to be resumed. Default to None.
    :param str dir_: An absolute path to a directory where metadata will be stored.
    """

    def __init__(
            self,
            project: Optional[str] = None,
            entity: Optional[str] = None,
            name: Optional[str] = None,
            run_id: Optional[str] = None,
            config: Optional[dict] = None,
            dir_: Optional[str] = None,
    ) -> None:
        if project is None:
            project = os.getenv("WANDB_PROJECT", "d2c")
        self.wandb_run = wandb.init(
            project=project,
            name=name,
            id=run_id,
            entity=entity,
            config=config,  # type: ignore
            dir=dir_,
        )

    def finish(self) -> None:
        self.wandb_run.finish()

    @staticmethod
    def write_summary(info: Dict) -> None:
        wandb.log(info)
