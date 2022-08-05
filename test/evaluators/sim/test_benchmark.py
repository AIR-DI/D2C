import os
import torch
import pytest
from d2c.evaluators.sim.benchmark import BMEval
from d2c.models import make_agent
from d2c.envs import LeaEnv, benchmark_env
from d2c.utils.utils import abs_file_path
from d2c.utils.config import ConfigBuilder
from d2c.utils import utils
from example.benchmark.config.app_config import app_config


class TestBM:

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
    })
    cfg_builder = ConfigBuilder(
        app_config=app_config,
        model_config_path=model_config_path,
        work_abs_dir=work_abs_dir,
        command_args=command_args,
    )
    config = cfg_builder.build_config()
    lea_env = LeaEnv(config)
    agent = make_agent(config=config, env=lea_env)
    env = benchmark_env(config)
    seed = 1

    def test_bm(self):
        result_dir = './temp'
        utils.maybe_makedirs(result_dir)
        eval_ = BMEval(
            result_dir=result_dir,
            agent=self.agent,
            env=self.env,
            score_normalize=True,
            score_norm_min=0,
            score_norm_max=3000,
            seed=self.seed,
        )
        for step in range(20):
            eval_.eval(step)
        eval_.save_eval_results()


if __name__ == '__main__':
    pytest.main(__file__)



