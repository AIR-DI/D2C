import os

import numpy as np
import pytest
import torch
from d2c.data import Data, DataNoise
from d2c.envs import LeaEnv
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

    def test_data_noise(self):
        prefix = 'env.external.'
        command_args = {
            prefix + 'benchmark_name': 'd4rl',
            prefix + 'data_source': 'mujoco',
            prefix + 'env_name': 'Hopper-v2',
            prefix + 'data_name': 'hopper_random-v2',
            prefix + 'state_normalize': True,
        }
        command_args.update({'train.data_loader_name': None,
                             'train.action_noise': 0.1})
        cfg_builder = ConfigBuilder(
            app_config=app_config,
            model_config_path=self.model_config_path,
            work_abs_dir=self.work_abs_dir,
            command_args=command_args,
        )
        config = cfg_builder.build_config()
        lea_env = LeaEnv(config)
        d4rl_data_noise = DataNoise(config, lea_env)
        data = d4rl_data_noise.data
        _batch = data.get_batch_indices(np.arange(64))
        assert isinstance(_batch['s1'], torch.Tensor)
        assert _batch['s1'].shape == (64, 11)
        assert _batch['a1'].shape == (64, 3)
        assert _batch['a2'].shape == (64, 3)
        shift, scale = d4rl_data_noise.state_shift_scale
        assert shift.shape == (11,)
        assert scale.shape == (11,)
        print(shift, scale)

        d4rl_data = Data(config)
        data1 = d4rl_data.data
        batch1 = data1.get_batch_indices(np.arange(64))

        for i in range(10):
            print(batch1['a1'][i])
            print(_batch['a1'][i], '\n')


if __name__ == '__main__':
    pytest.main(__file__)