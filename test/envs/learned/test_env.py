import os
import torch
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
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        prefix = 'env.external.'
        command_args = {
            prefix + 'benchmark_name': 'd4rl',
            prefix + 'data_source': 'mujoco',
            prefix + 'env_name': 'Hopper-v2',
            prefix + 'data_name': 'hopper_medium-v2',
        }
        command_args.update({
            'train.device': device,
            'train.dynamics_ckpt_name': 'dyna-0116',
            'env.learned.dynamic_module_type': 'prob',
            'env.learned.with_reward': True,
        })

        cfg_builder = ConfigBuilder(
            app_config=app_config,
            model_config_path=self.model_config_path,
            work_abs_dir=self.work_abs_dir,
            command_args=command_args,
        )
        config = cfg_builder.build_config()
        model_num = config.model_config.env.learned.prob.model_params[1]
        env = LeaEnv(config)
        env.load()
        obs = env.reset(seed=1)
        assert obs.shape == (1, 11)
        init_s = np.zeros((64, 11))
        obs = env.reset(seed=1, options={'init_s': init_s})
        assert (obs == init_s).all()
        # TODO Some test about the dynamics.
        a = np.random.random((64, 3))
        s_p, r, d, _ = env.step(a)
        assert s_p.shape == (64, 11)
        assert r.shape == (64,)
        assert d.shape == (64,)
        s_p, r, d, dist = env.step_raw(obs, a, with_dist=True)
        assert len(s_p) == len(r) == len(d) == len(dist) == model_num
        assert s_p[0].shape == (64, 11)
        assert r[0].shape == (64,)
        assert d[0].shape == (64,)
        assert dist[0].sample().shape == (64, 11+1)


if __name__ == '__main__':
    pytest.main(__file__)
