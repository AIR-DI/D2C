from typing import Union, Any, Dict, Type
from d2c.envs.learned.dynamics.base import BaseDyna
from d2c.envs.learned.dynamics.mlp import MlpDyna
from d2c.envs.learned.dynamics.prob import ProbDyna
from d2c.utils.replaybuffer import ReplayBuffer
from d2c.utils.utils import Flags


DYNA_DICT: Dict[str, Type[BaseDyna]] = {}


def register_dyna(cls: Type[BaseDyna]) -> None:
    """Registering the dynamics class.

    :param cls: Dynamics class inheriting ``BaseDyna``.
    """
    is_registered = cls.TYPE in DYNA_DICT
    assert not is_registered, f'{cls.TYPE} seems to be already registered.'
    DYNA_DICT[cls.TYPE] = cls


register_dyna(ProbDyna)
register_dyna(MlpDyna)


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
        test_data_ratio=model_cfg.train.test_data_ratio,
        with_reward=model_cfg.env.learned.with_reward,
        device=model_cfg.train.device,
    )
    dyna_args.update(dyna_params)

    dyna = DYNA_DICT[dyna_name](**dyna_args)
    if restore:
        dyna.restore(model_cfg.train.dynamics_ckpt_dir)

    return dyna
