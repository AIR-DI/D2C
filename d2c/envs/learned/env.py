"""The env that contains the dynamics model trained from data set."""
import numpy as np
import torch
from torch import Tensor
from typing import Any, Union, Optional, Tuple, List
from gym.spaces import Space, Box
from d2c.envs import BaseEnv
from d2c.envs.learned.dynamics import DYNA_DICT
from d2c.envs.learned.dynamics import make_dynamics
from d2c.utils import utils


class LeaEnv(BaseEnv):
    """An environment instance that contain the trained dynamics model.

    When training the model-based RL and evaluating the trained RL policy,
    this environment will be used.

    The usage usually is as below:

    1. Train the dynamics model(e.g. a neural network model) with the \
    batch data;
    2. Load the trained dynamics model and use the environment:

    ::

       env = Env(config)
       env.load_model()
       env.reset()
       env.step(a)

    .. seealso::

        Please refer to :class:`~d2c.envs.base.BaseEnv` for other APIs' usage.

    :param config: the configuration that contains the config information of environment.
    """

    def __init__(self, config: Any) -> None:
        self._config = config
        # Config for Env model
        self._env_cfg = config.model_config.env
        super(LeaEnv, self).__init__()
        # The name of the dynamics type.
        self._dyna_type = self._env_cfg.learned.dynamic_module_type
        self._dyna_module = DYNA_DICT[self._dyna_type]
        self._with_reward = self._env_cfg.learned.with_reward  # If the dynamics predict the reward or not.
        self._dynamics_model = None
        self._d_fns = None
        if not self._with_reward:
            try:
                self._r_fn = config.app_config.reward_fn
            except AttributeError:
                print('Please define the reward function first if the dynamics model do not predict reward!')
                raise
        self._done_fn = config.app_config.done_fn
        self._device = config.model_config.train.device
        self.state = None
        self.action_past = None

    def _set_action_space(self) -> Space:
        self._a_dim, a_min, a_max = [self._env_cfg.basic_info[k] for k in ['action_dim', 'action_min', 'action_max']]
        a_min = np.array(a_min) if isinstance(a_min, (list, tuple)) else a_min
        a_max = np.array(a_max) if isinstance(a_max, (list, tuple)) else a_max
        self.action_space = Box(low=a_min, high=a_max, shape=(self._a_dim,), dtype=np.float32)
        return self.action_space

    def _set_observation_space(self) -> Space:
        self._s_dim, s_min, s_max = [self._env_cfg.basic_info[k] for k in ['state_dim', 'state_min', 'state_max']]
        s_min = np.array(s_min) if isinstance(s_min, (list, tuple)) else s_min
        s_max = np.array(s_max) if isinstance(s_max, (list, tuple)) else s_max
        self.observation_space = Box(low=s_min, high=s_max, shape=(self._s_dim,), dtype=np.float32)
        return self.observation_space

    def _load_model(self) -> None:
        """
        Load the trained dynamics models.
        """
        self._dynamics_model = make_dynamics(config=self._config, restore=True)
        self._d_fns = self._dynamics_model.dynamics_fns

    def _dynamics(
            self,
            s: Union[np.ndarray, Tensor],
            a: Union[np.ndarray, Tensor],
            return_dist: bool = False
    ) -> Union[List, Tuple[List, List[Tuple]]]:
        s_ = torch.as_tensor(s, device=self._device, dtype=torch.float32)
        a_ = torch.as_tensor(a, device=self._device, dtype=torch.float32)
        s_p, info = self._d_fns(s_, a_)
        s_dist = None
        if 'dist' in info:
            s_dist = info['dist']
        s_p = [x.cpu().numpy() for x in s_p]
        if return_dist:
            # assert s_dist is not None
            return s_p, s_dist
        else:
            return s_p

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ):
        """Resets the environment to an initial state and returns an initial
        observation. There is difference for RNN dynamics and other dynamics.
        For RNN dynamics model, there is warm-up in this method. Make sure
        the input ``warm_input`` is contained in parameter ``options`` and has
        shape like that ``(batch, timesteps, feature_dim)``, the ``feature_dim``
        is the sum of state-dimension and action-dimension.

        :param int seed: seed for random number generator(s)
        :param bool return_info:
        :param dict options: a dict contain ``init_s`` and ``warm_input``.
            ``init_s``(np.ndarray or Tensor) is the initial state. For RNN dynamics, ``init_s`` is the
            state that is just following the warm_input. ``warm_input``:
            the warm-up input for LSTM dynamics model.
        :return: the initial observation.
        """
        if options is None:
            options = {}
        super(LeaEnv, self).reset(seed=seed)
        init_s = options.get('init_s')
        warm_input = options.get('warm_input')
        self.state = np.zeros((1, self._s_dim), dtype='float32')
        if init_s is not None:
            self.state = utils.to_array_as(init_s, self.state)
            assert len(self.state.shape) == 2
        # Judge that if the dynamics is an RNN model.
        if 'rnn' in self._dyna_type:
            assert warm_input is not None
            assert len(warm_input.shape) == 3
            self._dynamics_model.dynamics_warm(
                torch.as_tensor(warm_input, device=self._device, dtype=torch.float32)
            )
            self.action_past = warm_input[:, -1, self._s_dim:]
        else:
            self.action_past = np.zeros((self.state.shape[0], self._a_dim), dtype='float32')
        if not return_info:
            return self.state
        else:
            return self.state, {}

    def step(
            self,
            a: Union[np.ndarray, Tensor]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        s_p = self._dynamics(self.state, a)
        if not self._with_reward:
            r = [self._r_fn(self.action_past, self.state, a, s2) for s2 in s_p]
        else:
            r = [s[:, -1] for s in s_p]
            s_p = [s[:, :-1] for s in s_p]
        s_p = s_p[np.random.randint(0, len(s_p))]
        r = np.mean(r, axis=0)
        d = self._done_fn(self.action_past, self.state, a, s_p)
        self.state = s_p
        self.action_past = a
        return s_p, r, d, {}

    def step_raw(
            self,
            s: Union[np.ndarray, Tensor],
            a: Union[np.ndarray, Tensor],
            with_dist: bool = False
    ) -> Union[Tuple[List, List, List], Tuple[List, List, List, List]]:
        """ Run one timestep of the environment's dynamics.

        This method is usually used in RL training process.
        Accepts a state and an action, returns a tuple (observation, reward, done,).

        There will be ensemble dynamics models. So every dynamics model will compute
        the results and the returned results will be a list contain all the results.

        :param s: a batch of state
        :param a: a batch of action
        :param bool with_dist: if return the distribution of the predict next states.
        :return: A tuple including three items:

            * ``s_p``: a list, the agent's observation of current environments
            * ``r``: a list, the amount of rewards returned after previous actions
            * ``d``: a list, whether these episodes have ended
        """
        s = s.detach().cpu().numpy() if isinstance(s, Tensor) else s
        a = a.detach().cpu().numpy() if isinstance(a, Tensor) else a
        s_p, s_dist = self._dynamics(s, a, return_dist=True)
        if not self._with_reward:
            r = [self._r_fn(self.action_past, s, a, s2) for s2 in s_p]
        else:
            r = [s[:, -1] for s in s_p]
            s_p = [s[:, :-1] for s in s_p]
        d = [self._done_fn(self.action_past, s, a, s2) for s2 in s_p]
        self.action_past = a
        if not with_dist:
            return s_p, r, d
        else:
            assert s_dist is not None
            return s_p, r, d, s_dist

    def load(self):
        """The API for loading the trained dynamics model."""
        self._load_model()

    def get_dynamics(self):
        """Get the dynamics model."""
        return self._dynamics_model.dyna_nets

    @property
    def r_fn(self):
        """The reward function."""
        return self._r_fn

    @property
    def d_num(self):
        """
        The number of the dynamics models.
        """
        return len(self._dynamics_model.dyna_nets)

    @property
    def dynamics_module(self):
        """
        The dynamics module of the Env.
        """
        return self._dyna_module

    @property
    def dynamics_type(self):
        """
        The type of the dynamics model.
        """
        return self._dyna_type

    @property
    def dynamics_with_reward(self):
        """
        If the dynamics model predict the reward or not.
        """
        return self._with_reward









