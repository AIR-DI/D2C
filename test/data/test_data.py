import os
import pytest
import torch
from d2c.data import Data
from d2c.utils.utils import abs_file_path
from d2c.utils.config import ConfigBuilder
from example.benchmark.config.app_config import app_config


class TestData:

    work_abs_dir = abs_file_path(__file__, '../../example/benchmark')
    model_config_path = os.path.join(work_abs_dir, 'config', 'model_config.json5')

    def test_data(self):
        prefix = 'env.external.'
        command_args = {
            prefix+'benchmark_name': 'd4rl',
            prefix+'data_source': 'mujoco',
            prefix+'env_name': 'Hopper-v2',
            prefix+'data_name': 'hopper_random-v2',
            prefix+'state_normalize': True,
        }
        command_args.update({'train.data_loader_name': None})
        cfg_builder = ConfigBuilder(
            app_config=app_config,
            model_config_path=self.model_config_path,
            work_abs_dir=self.work_abs_dir,
            command_args=command_args,
        )
        config = cfg_builder.build_config()
        d4rl_data = Data(config)
        data = d4rl_data.data
        _batch = data.sample_batch(64)
        assert isinstance(_batch['s1'], torch.Tensor)
        assert _batch['s1'].shape == (64, 11)
        assert _batch['a1'].shape == (64, 3)
        shift, scale = d4rl_data.state_shift_scale
        assert shift.shape == (11,)
        assert scale.shape == (11,)
        print(shift, scale)


if __name__ == '__main__':
    pytest.main(__file__)