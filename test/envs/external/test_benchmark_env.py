import os
import pytest
import numpy as np
from d2c.envs.external import benchmark_env, D4rlEnv
from d2c.utils.utils import abs_file_path
from d2c.utils.config import ConfigBuilder
from example.benchmark.config.app_config import app_config


class TestBMEnv:

    work_abs_dir = abs_file_path(__file__, '../../../example/benchmark')
    model_config_path = os.path.join(work_abs_dir, 'config', 'model_config.json5')
    obs_shift = np.zeros((11,))
    obs_scale = np.ones((11,))

    def test_bm_env(self):
        prefix = 'env.external.'
        command_args = {
            prefix + 'benchmark_name': 'd4rl',
            prefix + 'data_source': 'mujoco',
            prefix + 'env_name': 'Hopper-v2',
        }
        cfg_builder = ConfigBuilder(
            app_config=app_config,
            model_config_path=self.model_config_path,
            work_abs_dir=self.work_abs_dir,
            command_args=command_args,
        )
        config = cfg_builder.build_config()
        env = benchmark_env(config)
        assert isinstance(env, D4rlEnv)
        env = benchmark_env(config, obs_shift=self.obs_shift, obs_scale=self.obs_scale)
        assert isinstance(env, D4rlEnv)

        env = benchmark_env(benchmark_name='d4rl')
        assert env is D4rlEnv


if __name__ == '__main__':
    pytest.main(__file__)