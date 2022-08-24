import os
import pytest
from d2c.utils.config import ConfigBuilder
from d2c.utils.utils import abs_file_path
from example.benchmark.config.app_config import app_config


class TestCfg:

    work_abs_dir = abs_file_path(__file__, '../../example/benchmark')
    model_config_path = os.path.join(work_abs_dir, 'config', 'model_config.json5')
    command_args = {
        'model.model_name': 'td3_bc',
        'model.td3_bc.hyper_params.policy_noise': 0.5,
        'env.external.benchmark_name': 'd4rl',
        'env.external.data_source': 'mujoco',
        'env.external.env_name': 'Hopper-v2',
        'env.external.score_norm_min': 0.001,
        'train.batch_size': 128,
    }

    def test_cfg_builder(self):
        cfg_builder = ConfigBuilder(
            app_config=app_config,
            model_config_path=self.model_config_path,
            work_abs_dir=self.work_abs_dir,
            command_args=self.command_args,
        )
        config = cfg_builder.build_config()
        assert config.model_config.model.td3_bc.hyper_params.policy_noise == 0.5
        assert config.model_config.train.batch_size == 128
        assert config.model_config.env.basic_info.state_dim == 11
        assert config.model_config.env.basic_info.action_dim == 3
        assert config.model_config.env.external.score_norm_min == 0.001
        assert config.model_config.env.external.score_norm_max == 3234.3
        _ = cfg_builder.main_hyper_params(config.model_config)


if __name__ == '__main__':
    pytest.main(__file__)