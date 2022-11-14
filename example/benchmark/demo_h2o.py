import sys
sys.path.append('../../')
import torch
import logging
from d2c.trainers import Trainer
from d2c.models import make_agent
from d2c.envs import benchmark_env, LeaEnv
from d2c.data import Data
from d2c.evaluators import bm_eval
from example.benchmark.config import make_config

logging.basicConfig(level=logging.INFO)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    prefix = 'env.external.'
    command_args = {
        prefix + 'benchmark_name': 'd4rl',
        prefix + 'data_source': 'mujoco',
        prefix + 'env_name': 'HalfCheetah-v2',
        prefix + 'data_name': 'halfcheetah_medium_replay-v2',
        prefix + 'state_normalize': False,
        prefix + 'score_normalize': True,
    }
    command_args.update({
        'model.model_name': 'h2o',
        'train.data_loader_name': None,
        'train.device': device,
        'train.seed': 0,
        'train.total_train_steps': 1000000,
        'train.batch_size': 256,
        'train.agent_ckpt_name': '0817'
    })

    config = make_config(command_args)
    real_dataset = Data(config)
    s_norm = dict(zip(['obs_shift', 'obs_scale'], real_dataset.state_shift_scale))
    data = real_dataset.data

    real_env = benchmark_env(config=config, **s_norm)
    # TODO modify dynamics
    sim_env = benchmark_env(config)
    
    # agent with an empty buffer
    agent = make_agent(config=config, env=sim_env, data=data)
    # envaluate in the real env
    evaluator = bm_eval(agent=agent, env=real_env, config=config)
    # train in the sim env
    trainer = Trainer(agent=agent, train_data=data, config=config, env=sim_env, evaluator=evaluator)
    trainer.train()


if __name__ == '__main__':
    main()
