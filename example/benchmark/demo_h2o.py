import sys
sys.path.append('../../')
import torch
import logging
from d2c.trainers import Trainer
from d2c.models import make_agent
from d2c.envs import benchmark_env, LeaEnv
from d2c.data import Data
from d2c.evaluators import bm_eval
from d2c.utils.utils import update_source_env_gravity, update_source_env_friction, update_source_env_density
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
        prefix + 'unreal_dynamics': 'gravity',
        prefix + 'variety_degree': 2.0,
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
        'train.agent_ckpt_name': '1211'
    })

    config = make_config(command_args)
    real_dataset = Data(config)
    s_norm = dict(zip(['obs_shift', 'obs_scale'], real_dataset.state_shift_scale))
    data = real_dataset.data

    real_env = benchmark_env(config=config, **s_norm)
    # real_env = benchmark_env(config=config)
    real_env_name = config.model_config.env.external.data_name
    if config.model_config.env.external.unreal_dynamics == "gravity":
        update_source_env_gravity(config.model_config.env.external.variety_degree, real_env_name)
    elif config.model_config.env.external.unreal_dynamics == "density":
        update_source_env_density(config.model_config.env.external.variety_degree, real_env_name)
    elif config.model_config.env.external.unreal_dynamics == "friction":
        update_source_env_friction(config.model_config.env.external.variety_degree, real_env_name)
    else:
        raise RuntimeError("Got erroneous unreal dynamics %s" % config.model_config.env.external.unreal_dynamics)
    sim_env = benchmark_env(config, **s_norm)
    if config.model_config.env.external.unreal_dynamics == "gravity":
        update_source_env_gravity(1, real_env_name)
    elif config.model_config.env.external.unreal_dynamics == "density":
        update_source_env_density(1, real_env_name)
    elif config.model_config.env.external.unreal_dynamics == "friction":
        update_source_env_friction(1, real_env_name)
    else:
        raise RuntimeError("Got erroneous unreal dynamics %s" % config.model_config.env.external.unreal_dynamics)
    print("\n-------------Env name: {}, variety: {}, unreal_dynamics: {}-------------".format(config.model_config.env.external.env_name, config.model_config.env.external.variety_degree, config.model_config.env.external.unreal_dynamics))

    
    # agent with an empty buffer
    agent = make_agent(config=config, env=sim_env, data=data)
    # envaluate in the real env
    evaluator = bm_eval(agent=agent, env=real_env, config=config)
    # train in the sim env
    trainer = Trainer(agent=agent, train_data=data, config=config, env=sim_env, evaluator=evaluator)
    trainer.train()


if __name__ == '__main__':
    main()
