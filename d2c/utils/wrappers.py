"""A collection of gym wrappers."""

import gym
import logging
import numpy as np
from gym import spaces


def wrapped_norm_obs_env(
        gym_env: gym.Env,
        shift: np.ndarray,
        scale: np.ndarray
) -> gym.Env:
    """Create a gym environment with normalized observations.

    :param gym_env: the original gym env.
    :param np.ndarray shift: a numpy vector to shift observations.
    :param np.ndarray scale: a numpy vector to scale observations.

    Returns: An initialized gym environment.
    """
    env = check_and_normalize_box_actions(gym_env)

    if shift is not None:
        env = NormalizeStateWrapper(env, shift=shift, scale=scale)

    return env


def check_and_normalize_box_actions(env: gym.Env) -> gym.Env:
    """Wrap env to normalize actions if [low, high] != [-1, 1]."""
    if isinstance(env.action_space, spaces.Box):
        low, high = env.action_space.low, env.action_space.high
        if (np.abs(low + np.ones_like(low)).max() > 1e-6 or
                np.abs(high - np.ones_like(high)).max() > 1e-6):
            logging.info('Normalizing environment actions.')
            return NormalizeBoxActionWrapper(env)

    # Environment does not need to be normalized.
    return env


class NormalizeBoxActionWrapper(gym.ActionWrapper):
    """Rescale the action space of the environment."""

    def __init__(self, env: gym.Env) -> None:
        if not isinstance(env.action_space, spaces.Box):
            raise ValueError('env %s does not use spaces.Box.' % str(env))
        super(NormalizeBoxActionWrapper, self).__init__(env)
        self._max_episode_steps = env._max_episode_steps  # pylint: disable=protected-access

    def action(self, action: np.ndarray) -> np.ndarray:
        # rescale the action
        low, high = self.env.action_space.low, self.env.action_space.high
        scaled_action = low + (action + 1.0) * (high - low) / 2.0
        scaled_action = np.clip(scaled_action, low, high)

        return scaled_action

    def reverse_action(self, scaled_action: np.ndarray) -> np.ndarray:
        low, high = self.env.action_space.low, self.env.action_space.high
        action = (scaled_action - low) * 2.0 / (high - low) - 1.0
        return action


class NormalizeStateWrapper(gym.ObservationWrapper):
    """Wraps an environment to shift and scale observations.
    """

    def __init__(self, env: gym.Env, shift: np.ndarray, scale: np.ndarray) -> None:
        super(NormalizeStateWrapper, self).__init__(env)
        self.shift = shift
        self.scale = scale

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return (observation + self.shift) * self.scale

    @property
    def _max_episode_steps(self):
        return self.env._max_episode_steps  # pylint: disable=protected-access



