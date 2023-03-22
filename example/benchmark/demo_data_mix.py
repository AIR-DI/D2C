import sys
sys.path.append('../../')
import fire
import logging
from d2c.trainers import Trainer
from d2c.models import make_agent
from d2c.envs import benchmark_env, LeaEnv
from d2c.data import DataMix
from d2c.evaluators import bm_eval
from example.benchmark.config import make_config

logging.basicConfig(level=logging.INFO)


def main(**kwargs):
    data1 = kwargs.pop('data1')
    ratio1 = kwargs.pop('ratio1')
    data2 = kwargs.pop('data2')
    ratio2 = kwargs.pop('ratio2')
    kwargs['env.external.data_file_path'] = [data1, data2, [ratio1, ratio2]]
    config = make_config(kwargs)
    bm_data = DataMix(config)
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


if __name__ == '__main__':
    fire.Fire(main)
