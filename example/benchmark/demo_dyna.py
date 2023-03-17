import sys
sys.path.append('../../')
import torch
import logging
from d2c.trainers import Trainer
from d2c.models import make_agent
from d2c.envs import LeaEnv
from d2c.data import Data
from example.benchmark.config import make_config

logging.basicConfig(level=logging.INFO)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    prefix = 'env.external.'
    command_args = {
        prefix + 'benchmark_name': 'd4rl',
        prefix + 'data_source': 'mujoco',
        prefix + 'env_name': 'Hopper-v2',
        prefix + 'data_name': 'hopper_medium_expert-v2',
        prefix + 'state_normalize': True,
        prefix + 'score_normalize': True,
    }
    command_args.update({
        'train.data_loader_name': None,
        'train.device': device,
        'train.seed': 0,
        'train.total_train_steps': 1000000,
        'train.batch_size': 256,
        'train.dynamics_ckpt_name': 'dyna-train_1m'
    })
    wandb = {
        'project': 'Train-Dyna',
        'name': 'hopper_medium_expert-train_1m',
        'reinit': False,
        'mode': 'online'
    }
    command_args.update({'train.wandb': wandb})
    command_args.update({'env.learned.dynamic_module_type': 'prob'})

    config = make_config(command_args)
    model_name = config.model_config.model.model_name
    config.model_config.model[model_name].train_schedule = ['d']

    bm_data = Data(config)
    data = bm_data.data
    # Contains dynamics model to be trained.
    lea_env = LeaEnv(config)
    agent = make_agent(config=config, env=lea_env, data=data)
    trainer = Trainer(agent=agent, train_data=data, config=config, env=lea_env)
    trainer.train()


if __name__ == '__main__':
    main()
