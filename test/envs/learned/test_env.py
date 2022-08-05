import os
import pytest
import numpy as np
from d2c.envs.learned.env import LeaEnv
from d2c.utils.utils import abs_file_path
from d2c.utils.config import ConfigBuilder
from example.benchmark.config.app_config import app_config


class TestLeaEnv:

    work_abs_dir = abs_file_path(__file__, '../../../example/benchmark')
    model_config_path = os.path.join(work_abs_dir, 'config', 'model_config.json5')

    def test_len_env(self):
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
        env = LeaEnv(config)
        obs = env.reset(seed=1)
        assert obs.shape == (1, 11)
        init_s = np.zeros((64, 11))
        obs = env.reset(seed=1, options={'init_s': init_s})
        assert (obs == init_s).all()
        # TODO Some test about the dynamics.


if __name__ == '__main__':
    pytest.main(__file__)
