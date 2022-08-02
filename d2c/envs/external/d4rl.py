"""An implementation of the Env for D4RL benchmark."""

import gym
import numpy as np
from gym.spaces import Space
from typing import Tuple, Any, Union
from d2c.envs import BaseEnv
from d2c.utils.wrappers import wrapped_norm_obs_env


class D4rlEnv(BaseEnv):
    """The Env for D4RL benchmark.

    :param str env_name: the name of env.
    """

    def __init__(
            self,
            env_name: str,
            obs_shift: np.ndarray = None,
            obs_scale: np.ndarray = None,
    ) -> None:
        super(D4rlEnv, self).__init__()
        self._env_name = env_name
        self._obs_shift = obs_shift
        self._obs_scale = obs_scale
        self._load_model()

    def _load_model(self):
        gym_env = gym.make(self._env_name)
        self._env = wrapped_norm_obs_env(
             gym_env=gym_env,
             shift=self._obs_shift,
             scale=self._obs_scale,
             )

    def _set_action_space(self) -> Space:
        self.action_space = self._env.action_space
        return self.action_space

    def _set_observation_space(self) -> Space:
        self.observation_space = self._env.observation_space
        return self.observation_space

    def step(self, a: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        return self._env.step(a)

    def reset(self, **kwargs: Any) -> Union[np.ndarray, Tuple[np.ndarray, dict]]:
        return self._env.reset(**kwargs)

