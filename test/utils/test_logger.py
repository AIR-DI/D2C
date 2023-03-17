import os
import sys
sys.path.append('../../')
import pytest
from torch.utils.tensorboard import SummaryWriter
from d2c.utils.logger import write_summary_tensorboard, WandbLogger


class TestLogger:

    tmp_dir = 'temp'
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)

    def test_tb_logger(self):
        writer = SummaryWriter(self.tmp_dir+'/tb')
        for step in range(10):
            info = {'a': step, 'b': step+1}
            write_summary_tensorboard(writer, step, info)

    def test_wandb_logger(self):  # TODO Bug need to be fixed.
        project = 'test_project'
        entity = 'd2c'
        name = 'test_zack'
        dir_ = self.tmp_dir + '/wandb'
        config = {
            "learning_rate": 0.001,
            "epochs": 100,
            "batch_size": 128
        }

        logger = WandbLogger(
            project=project,
            entity=entity,
            name=name,
            config=config,
            dir_=dir_,
        )
        for step in range(10):
            info = {'a': step, 'b': step+1}
            logger.write_summary(info)
        logger.finish()


if __name__ == '__main__':
    pytest.main(__file__)
