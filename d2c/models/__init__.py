from d2c.utils.utils import Flags
from typing import Union, Any, Callable, Dict
from d2c.envs import BaseEnv
from d2c.utils.replaybuffer import ReplayBuffer
from d2c.models.base import BaseAgent
from d2c.models.model_free.td3_bc import TD3BCAgent
from d2c.models.model_free.doge import DOGEAgent
from d2c.models.model_free.h2o import H2OAgent


AGENT_MODULES_DICT = {
    'td3_bc': TD3BCAgent,
    'doge': DOGEAgent,
    'h2o': H2OAgent
}


def get_agent(model_name: str) -> Callable[..., BaseAgent]:
    """Get the RL Agent.

    :param str model_name: the RL algorithm name.
    :return: an Agent corresponding to input name.

    .. note::

        The input name should be in the keys of dict ``AGENT_MODULES_DICT``:

        +------------------+------------------------------------------------+
        |  Imitation       |  'bc',                                         |
        +------------------+------------------------------------------------+
        |  Planning        |  'mopp'                                        |
        +------------------+------------------------------------------------+
        |  Model-free RL   |  'td3_bc', 'doge', 'h2o'                       |
        +------------------+------------------------------------------------+
        |  Model-based RL  |                                                |
        +------------------+------------------------------------------------+
    """
    return AGENT_MODULES_DICT[model_name]


def make_agent(
        config: Union[Flags, Any],
        env: BaseEnv = None,
        data: ReplayBuffer = None,
        restore_agent: bool = False
) -> BaseAgent:
    """Construct the Agent

    Construct an RL Agent with the config and other objects needed.

    :param config: the configuration.
    :param Env env: an Env object.
    :param ReplayBuffer data: the dataset of the batch data.
    :param bool restore_agent: if restore the Agent from the saved model file.
    :return: an Agent constructed with the inputs.

    .. note::

        When training an agent, the parameter "restore_agent" should be ``False``.
        When evaluating a trained policy, the parameter "restore_agent" should be
        ``True`` in "reward eval" mode and ``False`` in "FQE eval" mode.
    """
    model_cfg = config.model_config
    model_name = model_cfg.model.model_name
    agent_config = model_cfg.model[model_name]
    # An empty buffer.
    if config.model_config.train.model_buffer_size is not None:
        _max_size = config.model_config.train.model_buffer_size
    else:
        _max_size = data.capacity
    model_buffer = ReplayBuffer(
        state_dim=model_cfg.env.basic_info.state_dim,
        action_dim=model_cfg.env.basic_info.action_dim,
        max_size=_max_size,
        device=model_cfg.train.device,
    )
    agent_args = dict(
        env=env,
        train_data=data,
        batch_size=model_cfg.train.batch_size,
        weight_decays=model_cfg.train.weight_decays,
        update_freq=model_cfg.train.update_freq,
        update_rate=model_cfg.train.update_rate,
        discount=model_cfg.train.discount,
        empty_dataset=model_buffer,
        device=model_cfg.train.device,
    )
    agent_args.update(agent_config.hyper_params)

    agent = get_agent(model_name)(**agent_args)
    if restore_agent:
        agent.restore(model_cfg.train.agent_ckpt_dir)
    return agent


# def restore(
#         agent: BaseAgent,
#         model_config: Union[Dict, Any]
# ) -> BaseAgent:
#     """Restore an agent from the saved model file.
#
#     :param Agent agent: an initialized agent object.
#     :param model_config: the config for the model.
#     :return: An agent that has been restored.
#     """
#     model_ckpt_dir = [model_config.train.behavior_ckpt_dir,
#                       model_config.train.q_ckpt_dir,
#                       model_config.train.vae_s_ckpt_dir,
#                       model_config.train.agent_ckpt_dir,
#                       ]
#     ckpt_dir_dict = {x: y for x, y in zip(['b', 'q', 'vae_s', 'agent'], model_ckpt_dir)}
#     agent.restore_all(**ckpt_dir_dict)
#     return agent
