"""The base class of Env."""
import gym
from gym.utils import seeding
from typing import Union, Optional, TypeVar, Tuple
from abc import ABC, abstractmethod


ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class BaseEnv(gym.Env, ABC):
    """The main base environment class derived from OpenAI Gym class.

    It encapsulates an environment with arbitrary behind-the-scenes dynamics.
    The dynamics can be a model learned from the batch data. It can also be
    a ready-made external model. Please inherit this class to build these
    two class environments as needed.

    The main API methods that users of this class need to know are:

        * :meth:`~d2c.envs.base.BaseEnv.step`
        * :meth:`~d2c.envs.base.BaseEnv.reset`

    And set the following attributes:

        * ``action_space``: The Space object corresponding to valid actions
        * ``observation_space``: The Space object corresponding to valid observations
        * ``reward_range``: A tuple corresponding to the min and max possible rewards

    Note: a default reward range set to [-inf,+inf] already exists. Set it if you want a narrower range.

    The methods are accessed publicly as "step", "reset", etc...
    """
    def __init__(self):
        self._set_action_space()
        self._set_observation_space()

    @abstractmethod
    def _set_action_space(self):
        pass

    @abstractmethod
    def _set_observation_space(self):
        pass

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.

        Accepts an action and returns a tuple (observation, reward, done, info).

        :param: object action: an action provided by the agent

        :returns: observation, reward, done, info
        """
        raise NotImplementedError

    @abstractmethod
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> Union[ObsType, Tuple[ObsType, dict]]:
        """Resets the environment to an initial state and returns an initial observation.

        This method should also reset the environment's random number
        generator(s) if `seed` is an integer or if the environment has not
        yet initialized a random number generator. If the environment already
        has a random number generator and `reset` is called with `seed=None`,
        the RNG should not be reset.
        Moreover, `reset` should (in the typical use case) be called with an
        integer seed right after initialization and then never again.

        Returns:
            observation (object): the initial observation.
            info (optional dictionary): a dictionary containing extra information, this is only returned if return_info is set to true
        """
        # Initialize the RNG if the seed is manually passed
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)

    @abstractmethod
    def _load_model(self):
        """Load the dynamics model.

        Load a trained dynamics model or an external dynamics model as needed.
        """
        pass

    def render(self, mode="human"):
        pass
