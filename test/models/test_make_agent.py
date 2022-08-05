import os
import torch
import pytest
from d2c.models import make_agent, BaseAgent
from d2c.data import Data
from d2c.envs import LeaEnv
from d2c.utils.utils import abs_file_path, maybe_makedirs
from d2c.utils.config import ConfigBuilder
from example.benchmark.config.app_config import app_config


class TestMakeAgent:

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    work_abs_dir = abs_file_path(__file__, '../../example/benchmark')
    model_config_path = os.path.join(work_abs_dir, 'config', 'model_config.json5')
    prefix = 'env.external.'
    command_args = {
        prefix + 'benchmark_name': 'd4rl',
        prefix + 'data_source': 'mujoco',
        prefix + 'env_name': 'Hopper-v2',
        prefix + 'data_name': 'hopper_random-v2',
    }
    command_args.update({
        'model.model_name': 'td3_bc',
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
    train_data = Data(config).data

    def test_make_agent(self):
        agent = make_agent(
            config=self.config,
            env=self.env,
            data=self.train_data,
        )
        assert isinstance(agent, BaseAgent)
        agent_ckpt_dir = './model_free/temp/td3_bc/agent/agent'
        self.config.model_config.train.agent_ckpt_dir = agent_ckpt_dir
        maybe_makedirs(os.path.dirname(agent_ckpt_dir))
        agent.save(agent_ckpt_dir)
        agent = make_agent(
            config=self.config,
            env=self.env,
            data=self.train_data,
            restore_agent=True,
        )
        assert isinstance(agent, BaseAgent)


if __name__ == '__main__':
    pytest.main(__file__)


