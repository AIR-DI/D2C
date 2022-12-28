import sys
sys.path.append('../../')
import torch
import logging
from d2c.trainers import Trainer
from d2c.models import make_agent
from d2c.envs import LeaEnv
from d2c.data import Data
from example.application.config import make_config

logging.basicConfig(level=logging.INFO)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    command_args = {
        'model.model_name': 'bc',
        'model.bc.hyper_params.test_data_ratio': 0.1,
        'model.bc.hyper_params.test_freq': 100,
        'train.data_loader_name': 'app',
        'train.device': device,
        'train.seed': 0,
        'train.total_train_steps': 100000,
        'train.batch_size': 64,
        'train.agent_ckpt_name': '1227'
    }
    wandb = {
        'entity': 'd2c',
        'project': 'idc',
        'name': 'low_level_model',
        'reinit': False,
        'mode': 'online'
    }
    command_args.update({'train.wandb': wandb})

    config = make_config(command_args)
    app_data = Data(config)
    data = app_data.data
    # Contains dynamics model to be trained.
    lea_env = LeaEnv(config)
    agent = make_agent(config=config, env=lea_env, data=data)
    trainer = Trainer(agent=agent, train_data=data, config=config, env=lea_env)
    trainer.train()


if __name__ == '__main__':
    main()
