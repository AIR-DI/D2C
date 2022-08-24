import os
import torch
import pytest
from d2c.trainers import Trainer
from d2c.models import make_agent
from d2c.data import Data
from d2c.envs import LeaEnv, benchmark_env
from d2c.evaluators.sim import bm_eval
from d2c.utils.utils import abs_file_path, maybe_makedirs
from d2c.utils.config import ConfigBuilder
from example.benchmark.config.app_config import app_config


class TestTrainer:

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
        'train.seed': 1,
        'train.agent_ckpt_dir': './temp/td3_bc/agent/agent',
        'train.total_train_steps': 100,
        'train.summary_freq': 5,
        'train.print_freq': 2,
        'train.eval_freq': 5,
    })
    wandb = {
        'entity': 'd2c',
        'project': 'test_project',
        'name': 'unittest',
        'reinit': False,
        'mode': 'online'
    }
    command_args.update({'train.wandb': wandb})
    cfg_builder = ConfigBuilder(
        app_config=app_config,
        model_config_path=model_config_path,
        work_abs_dir=work_abs_dir,
        command_args=command_args,
    )
    config = cfg_builder.build_config()
    lea_env = LeaEnv(config)
    train_data = Data(config).data
    agent = make_agent(config=config, env=lea_env, data=train_data)
    env = benchmark_env(config)
    eval = bm_eval(agent, env, config)

    def test_trainer(self):

        trainer = Trainer(
            agent=self.agent,
            train_data=self.train_data,
            config=self.config,
            env=self.lea_env,
            evaluator=self.eval,
        )
        trainer.train()


if __name__ == '__main__':
    pytest.main(__file__)

