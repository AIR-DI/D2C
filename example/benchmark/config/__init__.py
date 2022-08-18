import os
from d2c.utils.config import ConfigBuilder
from d2c.utils.utils import abs_file_path
from example.benchmark.config.app_config import app_config


def make_config(command_args=None):
    work_abs_dir = abs_file_path(__file__, '../')
    model_config_path = os.path.join(work_abs_dir, 'config', 'model_config.json5')
    cfg_builder = ConfigBuilder(
        app_config=app_config,
        model_config_path=model_config_path,
        work_abs_dir=work_abs_dir,
        command_args=command_args,
    )
    return cfg_builder.build_config()