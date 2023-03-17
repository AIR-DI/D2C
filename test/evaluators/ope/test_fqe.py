import os
import torch
import pytest
from d2c.evaluators.ope.fqe import FQEval
from d2c.models import make_agent
from d2c.envs import LeaEnv, benchmark_env
from d2c.data import Data
from d2c.utils.utils import abs_file_path
from d2c.utils.config import ConfigBuilder
from example.benchmark.config.app_config import app_config


def test_fqe():

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
        'model.model_name': 'td3_bc',
        'train.data_loader_name': None,
        'train.device': device,
        'train.wandb.mode': 'offline',
        'eval.ope.fqe.train_steps': 1000,
        'eval.ope.fqe.model_params': [[256, 256], 1],
    })
    cfg_builder = ConfigBuilder(
        app_config=app_config,
        model_config_path=model_config_path,
        work_abs_dir=work_abs_dir,
        command_args=command_args,
    )
    config = cfg_builder.build_config()
    lea_env = LeaEnv(config)
    agent = make_agent(config, env=lea_env)
    data = Data(config).data
    save_dir = os.path.join(abs_file_path(__file__, './'), 'temp')
    fqe = FQEval(
        agent=agent,
        data=data,
        state_dim=lea_env.observation_space.shape[0],
        action_dim=lea_env.action_space.shape[0],
        save_dir=save_dir,
        wandb_mode='offline',
        device=device,
        train_steps=1000,
        model_params=[[256, 256], 1]
    )
    fqe.eval()

    fqe1 = FQEval.from_config(agent, data, config)
    fqe1.eval()


if __name__ == '__main__':
    pytest.main(__file__)
