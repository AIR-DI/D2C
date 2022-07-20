import sys
sys.path.append('../../')
import fire
import tensorflow as tf
from AIControlOpt_lib.trainers import Trainer
from AIControlOpt_lib.models import make_model
from AIControlOpt_lib.envs import benchmark_env, Env
from AIControlOpt_lib.data_loader import BenchmarkData
from AIControlOpt_lib.evaluators import EvalBM
from d2c.utils.general_config import build_config
from absl import logging

logging.set_verbosity(logging.INFO)
# gpu_limit(0)
# tf.compat.v1.enable_v2_behavior()
# gpu_options = tf.GPUOptions(allow_growth=True)
# sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

# mujoco
try:
    from local_debug_logger import local_trace
except ImportError:
    local_trace = lambda: None


def main(**kwargs):
    config = build_config(kwargs)
    bm_data = BenchmarkData(config)
    data = bm_data.data
    # The env of the benchmark to be used for policy evaluation.
    bm_env = benchmark_env(config)(config=config, dataset=bm_data)
    # Contains dynamics model to be trained.
    fake_env = Env(config)
    agent = make_model(config=config, env=fake_env, data=data)
    evaluator = EvalBM(agent=agent, env=bm_env, config=config)
    trainer = Trainer(agent=agent, train_data=data, config=config, env=fake_env, evaluator=evaluator)
    trainer.train()


if __name__ == '__main__':
    fire.Fire(main)
