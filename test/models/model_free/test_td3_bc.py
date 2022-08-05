import os
import pytest
import torch
import numpy as np
from easydict import EasyDict
from torch.utils.tensorboard import SummaryWriter
from d2c.models.model_free.td3_bc import TD3BCAgent
from d2c.data import Data
from d2c.envs import LeaEnv
from d2c.utils.utils import abs_file_path, maybe_makedirs
from d2c.utils.config import ConfigBuilder
from example.benchmark.config.app_config import app_config


class TestTd3bc:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    work_abs_dir = abs_file_path(__file__, '../../../example/benchmark')
    model_config_path = os.path.join(work_abs_dir, 'config', 'model_config.json5')
    prefix = 'env.external.'
    command_args = {
        prefix + 'benchmark_name': 'd4rl',
        prefix + 'data_source': 'mujoco',
        prefix + 'env_name': 'Hopper-v2',
        prefix + 'data_name': 'hopper_random-v2',
    }
    command_args.update({
        'train.data_loader_name': None,
        'train.device': device,
    })
    cfg_builder = ConfigBuilder(
        app_config=app_config,
        model_config_path=model_config_path,
        work_abs_dir=work_abs_dir,
        command_args=command_args,
    )
    config = cfg_builder.build_config()

    env = LeaEnv(config)
    model_params = EasyDict({'q': [[400, 400], 2], 'p': [[300, 300], ]})
    optimizers = EasyDict({'q': ['adam', 1e-3], 'p': ['adam', 1e-5]})
    train_data = Data(config).data

    def test_td3bc_agent(self):
        if not os.path.exists('./temp/td3_bc'):
            os.makedirs('./temp/td3_bc')
        agent = TD3BCAgent(
            env=self.env,
            model_params=self.model_params,
            optimizers=self.optimizers,
            train_data=self.train_data,
            device=self.device,
        )
        summary_writer = SummaryWriter('./temp/td3_bc/train_log')
        for step in range(20):
            agent.train_step()
            agent.write_train_summary(summary_writer)
            agent.print_train_info()
        agent_ckpt_dir = './temp/td3_bc/agent/agent'
        maybe_makedirs(os.path.dirname(agent_ckpt_dir))
        agent.save(agent_ckpt_dir)
        agent.restore(agent_ckpt_dir)

        policy = agent.test_policies['main']
        obs = np.random.random((64, 11))
        action = policy(obs)
        assert action.shape == (64, 3)


if __name__ == '__main__':
    pytest.main(__file__)
