import sys
sys.path.append('../../')
import torch
import logging
from d2c.trainers import Trainer
from d2c.models import make_agent
from d2c.envs import benchmark_env, LeaEnv
from d2c.data import Data
from d2c.evaluators import bm_eval, make_ope
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
        'model.model_name': 'td3_bc',
        'train.data_loader_name': None,
        'train.device': device,
        'train.seed': 20,
        'train.total_train_steps': 1000,
        'train.batch_size': 256,
        'train.agent_ckpt_name': '230221-train_1k',
        'train.dynamics_ckpt_name': 'dyna-train_1m',
    })
    wandb = {
        'project': 'td3_bc(OPE)-0221',
        'name': 'hopper_medium_expert_train1k',
        'reinit': False,
        'mode': 'online'
    }
    command_args.update({'train.wandb': wandb})

    config = make_config(command_args)
    bm_data = Data(config)
    s_norm = dict(zip(['obs_shift', 'obs_scale'], bm_data.state_shift_scale))
    data = bm_data.data
    # The env of the benchmark to be used for policy evaluation.
    env = benchmark_env(config=config, **s_norm)
    # Contains dynamics model to be trained.
    lea_env = LeaEnv(config)
    agent = make_agent(config=config, env=lea_env, data=data)
    evaluator = bm_eval(agent=agent, env=env, config=config)
    trainer = Trainer(agent=agent, train_data=data, config=config, env=lea_env, evaluator=evaluator)
    trainer.train()
    agent = make_agent(config=config, env=lea_env, data=data, restore_agent=True)
    fqe = make_ope('fqe', from_config=True, agent=agent, data=data, config=config)
    fqe.eval()
    lea_env.load()
    mb_ope = make_ope('mb_ope', from_config=True, agent=agent, data=data, env=lea_env, config=config)
    mb_ope.eval()


if __name__ == '__main__':
    main()
