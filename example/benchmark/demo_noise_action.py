import sys
sys.path.append('../../')
import fire
import logging
from d2c.trainers import Trainer
from d2c.models import make_agent
from d2c.envs import benchmark_env, LeaEnv
from d2c.data import DataNoise
from d2c.evaluators import bm_eval
from example.benchmark.config import make_config

logging.basicConfig(level=logging.INFO)


def main(**kwargs):
    config = make_config(kwargs)
    # Contains dynamics model to be trained.
    lea_env = LeaEnv(config)
    # Using the noised data.
    bm_data = DataNoise(config, lea_env)
    s_norm = dict(zip(['obs_shift', 'obs_scale'], bm_data.state_shift_scale))
    data = bm_data.data
    # The env of the benchmark to be used for policy evaluation.
    env = benchmark_env(config=config, **s_norm)
    agent = make_agent(config=config, env=lea_env, data=data)
    evaluator = bm_eval(agent=agent, env=env, config=config)
    trainer = Trainer(agent=agent, train_data=data, config=config, env=lea_env, evaluator=evaluator)
    trainer.train()


if __name__ == '__main__':
    fire.Fire(main)
