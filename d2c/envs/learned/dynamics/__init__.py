from typing import Union, Any, Callable, Dict
from d2c.envs.learned.dynamics.base import BaseDyna
from d2c.envs.learned.dynamics.mlp import MlpDyna
from d2c.utils.replaybuffer import ReplayBuffer
from d2c.utils.utils import Flags


DYNA_MODULES_DICT = {
    'dnn': MlpDyna,
}


def get_dyna(model_name: str) -> Callable[..., BaseDyna]:
    """Get the dynamics.

    :param str model_name: the name of the dynamics model.
    :return: a dynamics class according to the input.
    """
    return DYNA_MODULES_DICT[model_name]


def make_dynamics(
        config: Union[Flags, Any],
        data: ReplayBuffer = None,
        restore: bool = False
) -> BaseDyna:
    """Construct the Dynamics Agent.

    :param config: the configuration.
    :param data: the data buffer.
    :param bool restore: If restore the dynamics models from the saved model file.
    :return: Dynamics needed.
    """
    model_cfg = config.model_config
    dyna_name = model_cfg.env.learned.dynamic_module_type
    dyna_params = model_cfg.env.learned[dyna_name]
    dyna_args = dict(
        state_dim=model_cfg.env.basic_info.state_dim,
        action_dim=model_cfg.env.basic_info.action_dim,
        train_data=data,
        batch_size=model_cfg.train.batch_size,
        weight_decays=model_cfg.train.weight_decays,
        train_test_ratio=model_cfg.train.train_test_ratio,
        with_reward=model_cfg.env.learned.with_reward,
        device=model_cfg.train.device,
    )
    dyna_args.update(dyna_params)

    dyna = get_dyna(dyna_name)(**dyna_args)
    if restore:
        dyna.restore(model_cfg.train.dynamics_ckpt_dir)

    return dyna
