from typing import Any, Optional, Callable, Union
from d2c.envs import BaseEnv
from d2c.envs.external.d4rl import D4rlEnv


def d4rl_env(config: Any, **kwargs: Any) -> D4rlEnv:
    env_name = config.model_config.env.external.env_name
    obs_shift = kwargs.get('obs_shift')
    obs_scale = kwargs.get('obs_scale')
    return D4rlEnv(env_name, obs_shift, obs_scale)


ENV_DICT = {
    'd4rl': D4rlEnv,
}


ENV_FUNC_DICT = {
    'd4rl': d4rl_env
}


def benchmark_env(
        config: Optional[Any] = None,
        benchmark_name: Optional[str] = None,
        **kwargs: Any,
) -> Union[BaseEnv, Callable[..., BaseEnv]]:
    """Get the Environment according to the benchmark.

    :param config: the configuration object. When it is not None,
        an instance object of the env class will be returned.
    :param str benchmark_name: the name of the benchmark. When
        `config` is None, and `benchmark_name` is not None, the
        env class will be returned.
    :param kwargs: some parameters like ``obs_shift``, ``obs_scale``
    """
    if config is not None:
        benchmark_name = config.model_config.env.external.benchmark_name
        assert benchmark_name in ENV_FUNC_DICT.keys()
        return ENV_FUNC_DICT[benchmark_name](config, **kwargs)
    else:
        assert benchmark_name is not None
        assert benchmark_name in ENV_DICT.keys()
        return ENV_DICT[benchmark_name]
