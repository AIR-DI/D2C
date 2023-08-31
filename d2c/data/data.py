"""Loading data and constructing the buffer."""

import os
import logging
import numpy as np
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Dict, Callable, Tuple, Union, List
from d2c.envs import BaseEnv, LeaEnv
from d2c.utils.utils import add_gaussian_noise
from d2c.utils.replaybuffer import ReplayBuffer
from d2c.utils.dataloader import D4rlDataLoader, BaseDataLoader, BaseBMLoader


class BaseData(ABC):
    """The basic class of the replay buffer. Inherit this class to build
    data loaders for practical application and benchmark experiments.

    The API methods the user should implement are:

    * :meth:`_build_data`: Construct the data through adding the transitions into the replay buffer.
    * :meth:`_build_data_loader`: Construct the data loader.

    :param str data_path: the file path of the dataset.
    """

    def __init__(
            self,
            data_path: Union[str, List[str]],
    ) -> None:
        self._data_path = data_path
        self._data_loader = None
        self._data = None
        self._buffer_size = None
        self._build_data()

    @abstractmethod
    def _build_data(self) -> None:
        """Construct the data through adding the transitions into the replay buffer."""
        pass

    @abstractmethod
    def _build_data_loader(self) -> None:
        """Construct the data loader."""
        pass

    @property
    def data(self) -> ReplayBuffer:
        """Return the replay buffer."""
        logging.info('='*20 + f'Get a data of size {self._data.size}' + '='*20)
        return self._data


class Data(BaseData):
    """Create the replay buffer with batch data that is from the benchmarks or real-world applications dataset.
    
    It uses different data loaders to get the transitions and adds them into the replay buffer. This class generates
    the dataset for the offline RL training.
    
    The main methods that users of this class need to know are:

    * :meth:`data`: Return the constructed replay buffer;
    * :meth:`_build_data_loader`: Read the batch data and create a data loader. \
        It builds different data loader according to the corresponding parameters \
        in the configuration.
    * :meth:`_build_data`: Create a replay buffer and add the data into it.

    .. note::

        To add a data loader for a new benchmark, you should implement a new method
        `_xxx_data_loader`(like :meth:`_app_data_loader`, :meth:`_d4rl_data_loader`)
        and add this method into :meth:`_data_loader_list`.

    :param config: the configuration.
    """

    def __init__(self, config) -> None:
        app_config = config.app_config
        env_config = config.model_config.env.external
        data_loader_name = config.model_config.train.data_loader_name
        self._split_ratio = config.model_config.train.data_split_ratio  # Data split ratio
        self._device = config.model_config.train.device
        if data_loader_name == 'app':  # For real-world application experiments.
            try:
                data_path = app_config.data_path
            except:
                raise AttributeError('There is no data file for real-world application experiments！')
        elif env_config.data_file_path is not None:
            data_path = env_config.data_file_path
            data_loader_name = env_config.benchmark_name
        else:
            raise ValueError('There is no data file to load!')
        assert isinstance(data_path, str)
        logging.info(f'Loading the offline dataset file from {data_path}.')
        if data_loader_name not in self._data_loader_list.keys():
            raise NotImplementedError(f'There is no data loader for {data_loader_name}!')
        self._app_cfg = app_config
        self._env_cfg = env_config
        self._data_loader_name = data_loader_name
        super(Data, self).__init__(data_path)

    def _build_data(self) -> None:
        self._build_data_loader()
        transitions = self._data_loader.get_transitions(split_ratio=self._split_ratio)
        s1, a1, s2, a2, reward, cost, done = [transitions[k] for k in transitions.keys()]
        self._buffer_size = s1.shape[0]
        state_dim = s1.shape[-1]
        action_dim = a1.shape[-1]
        self._data = ReplayBuffer(
            state_dim,
            action_dim,
            self._buffer_size,
            self._device,
        )
        self._data.add_transitions(
            state=s1,
            action=a1,
            next_state=s2,
            next_action=a2,
            reward=reward,
            done=done,
            cost=cost,
        )

    def _build_data_loader(self) -> None:
        self._data_loader = self._data_loader_list[self._data_loader_name]()

    def _app_data_loader(self):
        raise NotImplementedError

    def _d4rl_data_loader(self) -> D4rlDataLoader:
        state_normalize = self._env_cfg.state_normalize
        reward_normalize = self._env_cfg.reward_normalize
        return D4rlDataLoader(
            self._data_path,
            state_normalize,
            reward_normalize,
        )

    @property
    def _data_loader_list(self) -> Dict[str, Callable[..., BaseDataLoader]]:
        """A dict of the alternative Data loaders."""
        return dict(
            app=self._app_data_loader,
            d4rl=self._d4rl_data_loader,
        )

    @property
    def state_shift_scale(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._data_loader.state_shift_scale


class DataNoise(Data):
    """Construct a dataset that with noised action.

    :param config: the configuration.
    :param BaseEnv env: the env to provide the action information(minimum and maximum).
    """

    def __init__(self, config, env: LeaEnv) -> None:
        self._action_noise = config.model_config.train.action_noise
        self._action_space = env.action_space
        super(DataNoise, self).__init__(config)

    def _build_data(self) -> None:
        self._build_data_loader()
        transitions = self._data_loader.get_transitions()
        s1, a1, s2, a2, reward, cost, done = [transitions[k] for k in transitions.keys()]
        # Add noise to the action.
        a1 = add_gaussian_noise(a1, self._action_space, self._action_noise)
        a2 = add_gaussian_noise(a2, self._action_space, self._action_noise)
        self._buffer_size = s1.shape[0]
        state_dim = s1.shape[-1]
        action_dim = a1.shape[-1]
        self._data = ReplayBuffer(
            state_dim,
            action_dim,
            self._buffer_size,
            self._device,
        )
        self._data.add_transitions(
            state=s1,
            action=a1,
            next_state=s2,
            next_action=a2,
            reward=reward,
            done=done,
            cost=cost,
        )

    def _build_data_loader(self) -> None:
        super(DataNoise, self)._build_data_loader()


class DataMix(BaseData):
    """Construct a dataset by mixing several data."""

    def __init__(self, config) -> None:
        env_config = config.model_config.env.external
        self._device = config.model_config.train.device
        if env_config.data_file_path is not None:
            data_path = env_config.data_file_path
            data_loader_name = env_config.benchmark_name
        else:
            raise ValueError('There is no data file to load!')
        assert isinstance(data_path, list)
        if isinstance(data_path[-1], str):  # There is no mix ratio in the input.
            mix_ratio = [1] * len(data_path)
            data_files = data_path  # The data file list.
        else:
            data_files = data_path[:-1]  # The data file list.
            mix_ratio = data_path[-1]  # The ratio of each data that chosen to be mixed.
            assert isinstance(mix_ratio, list)
        assert len(data_files) == len(mix_ratio)
        for _r in mix_ratio:
            assert 0 < _r <= 1, 'The ratio of the data should between (0, 1]!'
        self._mix_ratio = mix_ratio

        if data_loader_name not in self._data_loader_list.keys():
            raise NotImplementedError(f'There is no data loader for {data_loader_name}!')
        self._env_cfg = env_config
        self._data_loader_name = data_loader_name
        self._obs_shift, self._obs_scale = None, None
        super(DataMix, self).__init__(data_files)

    def _build_data(self) -> None:
        transitions = self._multi_data_load()
        s1, a1, s2, a2, reward, cost, done = [transitions[k] for k in transitions.keys()]
        if self._env_cfg.state_normalize:
            s1, s2, self._obs_shift, self._obs_scale = BaseBMLoader.norm_state(s1, s2)
        if self._env_cfg.reward_normalize:
            reward = BaseBMLoader.norm_reward(reward)
        self._buffer_size = s1.shape[0]
        logging.info(f'The mix data size is {self._buffer_size}.')
        state_dim = s1.shape[-1]
        action_dim = a1.shape[-1]
        self._data = ReplayBuffer(
            state_dim,
            action_dim,
            self._buffer_size,
            self._device,
        )
        self._data.add_transitions(
            state=s1,
            action=a1,
            next_state=s2,
            next_action=a2,
            reward=reward,
            done=done,
            cost=cost,
        )

    def _build_data_loader(self) -> None:
        self._data_loader = self._data_loader_list[self._data_loader_name]

    @staticmethod
    def _d4rl_data_loader(data_path: str) -> D4rlDataLoader:
        state_normalize = False
        reward_normalize = False
        return D4rlDataLoader(
            data_path,
            state_normalize,
            reward_normalize,
        )

    @property
    def _data_loader_list(self) -> Dict[str, Callable[..., BaseDataLoader]]:
        """A dict of the alternative Data loaders."""
        return dict(
            d4rl=self._d4rl_data_loader,
        )

    def _multi_data_load(self) -> OrderedDict:
        self._build_data_loader()
        logging.info('='*10+'Loading the mix datasets'+'='*10)
        transitions = None
        for i, file in enumerate(self._data_path):
            logging.info(f'Mix data {i}: Loading {os.path.basename(file)} | mix ratio: {self._mix_ratio[i]}.')
            data_loader = self._data_loader(file)
            trans = data_loader.get_transitions()
            _num = int(trans['s1'].shape[0] * self._mix_ratio[i])
            logging.info(f'The number of the chosen transitions is {_num}.')
            if i == 0:
                for k, v in trans.items():
                    trans[k] = v[:_num]
                transitions = trans
            else:
                for k, v in trans.items():
                    transitions[k] = np.concatenate([transitions[k], v[:_num]])
        return transitions

    @property
    def state_shift_scale(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._obs_shift, self._obs_scale





