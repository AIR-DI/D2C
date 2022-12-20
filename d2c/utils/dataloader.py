"""Dataloader for loading dataset and generating transitions.

There are data_loaders for benchmarks and real-world applications.
"""

import h5py
import logging
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import List, Tuple, ClassVar, Optional, Callable, Dict, Union
from collections import OrderedDict
from d2c.utils.scalers import create_scaler


class BaseDataLoader(ABC):
    """The base class of the dataset loader.

    Inherit this class to build data loaders for benchmarks and real-world applications.

    The main methods
    """
    @abstractmethod
    def _load_data(self):
        """Load data from the file path and return the transitions elements.

        :return: a tuple like: (s1, a1, s2, a2, reward, cost, done). If there is no cost in the data set, please \
            set the 'cost' to 0.
        """
        pass

    @abstractmethod
    def get_transitions(self, **kwargs):
        """Get the transitions from the dataset.

        :return: A namedtuple that contains the elements of the transitions.
        """
        pass

    def state_shift_scale(self):
        """Get the shift and scale of the state normalization."""
        pass

    @staticmethod
    def _split(ratio: float, shuffle: bool = True, *data: np.ndarray) -> List[np.ndarray]:
        """Split the dataset.

        :param ratio: The split ratio. It should be a float value between (0, 1).
        :param shuffle: If choosing the data randomly.
        :param data: A list contains all the data to be split.
        :return: A List contains all the split data.
        """
        assert 0 < ratio <= 1
        length = data[0].shape[0]
        if shuffle:
            indices = np.random.permutation(length)
            indices = indices[:int(length * ratio)]
        else:
            indices = np.arange(int(length * ratio))
        dataset = [d[indices] for d in data]
        logging.info('=' * 20 + 'Splitting the dataset.' + '=' * 20 + '\n'
                     + f'The splitting ratio: {ratio}, the original data size: {length}, '
                       f'the splitting data size: {dataset[0].shape[0]}. \n')
        return dataset


class AppDataLoader(BaseDataLoader):
    """The data loader for the real-world applications dataset.

    :param str file_path: The path of the benchmark dataset.
    :param np.ndarray state_indices: The indices of the state features.
    :param np.ndarray action_indices: The indices of the action features.
    :param str state_scaler: The scaler type for scaling the state. The available options are \
        `['min_max', 'standard']`. `None` means do not scale the state.
    :param dict state_scaler_params: A dict that contains the parameters of the selected scaler \
        for the state. When chosen the 'min_max' scaler, state_scaler_params should be like \
        `{'minimum': minimum, 'maximum': maximum}`. When chosen the 'standard' scaler, state_scaler_params \
        should be like `{'mean': mean, 'std': std}`.
    :param str action_scaler: The scaler type for scaling the action. The available options are the \
        same with argument `state_scaler`.
    :param dict action_scaler_params: A dict that contains the parameters of the selected scaler \
        for the action. Its form is like the argument `state_scaler_params`.
    :param str reward_scaler: The scaler type for scaling the reward. The available options are the \
        same with argument `state_scaler`.
    :param dict reward_scaler_params: A dict that contains the parameters of the selected scaler \
        for the reward. Its form is like the argument `state_scaler_params`. The difference is that \
        the values of the dict should be float number.
    :param reward_fn: It can be a callable function or an integer number. A function is used to \
        compute the reward. An integer number means that the reward can be read directly from the \
        data according to this index.
    :param cost_fn: It is used to get the cost value and the form is like the argument `reward_fn`.
    :param done_fn: It is used to get the done value and the form is like the argument `reward_fn`.
    """

    TYPE: ClassVar[str] = 'app'
    SCALERS: Dict = {}

    def __init__(
            self,
            file_path: str,
            state_indices: np.ndarray,
            action_indices: np.ndarray,
            state_scaler: Optional[str] = None,
            state_scaler_params: Optional[Dict[str, np.ndarray]] = None,
            action_scaler: Optional[str] = None,
            action_scaler_params: Optional[Dict[str, np.ndarray]] = None,
            reward_scaler: Optional[str] = None,
            reward_scaler_params: Optional[Dict[str, float]] = None,
            reward_fn: Optional[Union[Callable, int]] = None,
            cost_fn: Optional[Union[Callable, int]] = None,
            done_fn: Optional[Union[Callable, int]] = None,
    ) -> None:
        self._file_path = file_path
        self._state_indices = state_indices
        self._action_indices = action_indices
        self._state_scaler = state_scaler
        self._state_scaler_params = state_scaler_params
        self._action_scaler = action_scaler
        self._action_scaler_params = action_scaler_params
        self._reward_scaler = reward_scaler
        self._reward_scaler_params = reward_scaler_params
        self._reward_fn = reward_fn
        self._cost_fn = cost_fn
        self._done_fn = done_fn
        self._get_scaler()

    def _load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        self._data = pd.read_csv(self._file_path)
        state = self._data.iloc[:, self._state_indices].to_numpy()
        action = self._data.iloc[:, self._action_indices].to_numpy()
        timestamp = self._data.loc[:, 'timestamp'].to_numpy()
        s1, s2, a1, a2, reward, cost, done = self._get_dataset(state, action, timestamp)

        return s1, s2, a1, a2, reward, cost, done

    def get_transitions(self, split_ratio: float = None, split_shuffle: bool = True) -> Dict:
        """Get the transitions from the dataset.

        :param float split_ratio: The ratio value for splitting the data.
        :param bool split_shuffle: If choosing the splitting data randomly.
        :return: A namedtuple that contains the elements of the transitions.
        """
        dataset = self._load_data()
        if split_ratio is not None:
            s1, s2, a1, a2, reward, cost, done = self._split(
                split_ratio,
                split_shuffle,
                *dataset,
            )
        else:
            s1, s2, a1, a2, reward, cost, done = dataset

        if self._s_scaler is not None:
            self._s_scaler.fit(s1)
            s1 = self._s_scaler.transform(s1)
            s2 = self._s_scaler.transform(s2)

        if self._a_scaler is not None:
            self._a_scaler.fit(a1)
            a1 = self._a_scaler.transform(a1)
            a2 = self._a_scaler.transform(a2)

        if self._r_scaler is not None:
            self._r_scaler.fit(reward)
            reward = self._r_scaler.transform(reward)

        transitions = OrderedDict(
            s1=s1,
            a1=a1,
            s2=s2,
            a2=a2,
            reward=reward,
            cost=cost,
            done=done,
        )
        return transitions

    def _get_scaler(self) -> None:

        def _create_scaler(name, params):
            if name is not None:
                if params is not None:
                    _scaler = create_scaler(
                        name,
                        **params,
                    )
                else:
                    _scaler = create_scaler(name)
            else:
                _scaler = None
            return _scaler

        self._s_scaler = _create_scaler(
            self._state_scaler,
            self._state_scaler_params,
        )
        self._a_scaler = _create_scaler(
            self._action_scaler,
            self._action_scaler_params,
        )
        self._r_scaler = _create_scaler(
            self._reward_scaler,
            self._reward_scaler_params,
        )
        self.SCALERS['state'] = self._s_scaler
        self.SCALERS['action'] = self._a_scaler
        self.SCALERS['reward'] = self._r_scaler

    def _get_dataset(
            self,
            state: np.ndarray,
            action: np.ndarray,
            timestamp: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Constructing the next state and next action according to the time continuity.

        :param np.ndarray state: The original state data.
        :param np.ndarray action: The original action data.
        :param np.ndarray timestamp: The timestamp of the state and action data.
        """
        s1 = state[:-1]
        s2 = state[1:]
        a1 = action[:-1]
        a2 = action[1:]
        t1 = timestamp[:-1]
        t2 = timestamp[1:]
        time_span = (t2 - t1).tolist()
        t_span = max(set(time_span), key=time_span.count)
        t_delete = np.where(np.array(time_span) > t_span)[0]
        s1 = np.delete(s1, t_delete, axis=0)
        s2 = np.delete(s2, t_delete, axis=0)
        a1 = np.delete(a1, t_delete, axis=0)
        a2 = np.delete(a2, t_delete, axis=0)
        reward, cost, done = self._get_reward(s1, s2, a1, a2, t_delete)
        logging.info('\n' + '='*20 + f'Constructing the transitions.' + '='*20 + '\n'
                     f'The deleted data size is {len(t_delete)}.' + '\n')
        return s1, s2, a1, a2, reward, cost, done

    def _get_reward(
            self,
            s1: np.ndarray,
            s2: np.ndarray,
            a1: np.ndarray,
            a2: np.ndarray,
            t_delete: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Computing the reward, cost and done values.

        :param s1: State.
        :param s2: Next state.
        :param a1: Action.
        :param a2: Next action.
        :param t_delete: The index for deleting the data.
        :return: The computed reward, cost and done.
        """
        def compute_r(fn: Optional[Union[Callable, int]]) -> np.ndarray:
            if fn is not None:
                if isinstance(fn, Callable):
                    r = fn(s1=s1, s2=s2, a1=a1, a2=a2)
                elif isinstance(fn, int):
                    r = self._data.iloc[:, fn].to_numpy()
                    r = np.delete(r, t_delete, axis=0)
                else:
                    raise ValueError(f'The function {fn} is wrong!')
            else:
                r = np.zeros(s1.shape[0])
            return r

        reward = compute_r(self._reward_fn)
        cost = compute_r(self._cost_fn)
        done = compute_r(self._done_fn)

        return reward, cost, done

    def get_scalers(self, name: str) -> Dict:
        """Returning the specific scaler.

        :param str name: The name of the scaler. The available options are `['state', 'action', 'reward']`.
        """
        return self.SCALERS[name]


class BaseBMLoader(BaseDataLoader):
    """The basic class of the benchmark dataset loader.
    Please inherit this class to build data loaders for different benchmarks.

    The main API method the user should implement is:

    * :meth:`_load_data`: load the transitions from the dataset file and return in \
        requested format.

    The main API method that users of this class need to know is:

    * :meth:`get_transitions`: process the transitions from the dataset and return a namedtuple.

    :param str file_path: the path of the benchmark dataset;
    :param bool state_normalize: if normalize the states;
    :param bool reward_normalize: if normalize the rewards.
    """
    def __init__(
            self,
            file_path: str,
            state_normalize: bool = False,
            reward_normalize: bool = False) -> None:
        self._file_path = file_path
        self._state_normalize = state_normalize
        self._reward_normalize = reward_normalize
        self._obs_shift, self._obs_scale = None, None

    def _load_data(self):
        raise NotImplementedError

    def get_transitions(self, split_ratio: float = None, split_shuffle: bool = True) -> OrderedDict:
        """Get the transitions from the dataset.

        :param float split_ratio: The ratio value for splitting the data.
        :param bool split_shuffle: If choosing the splitting data randomly.
        :return: A namedtuple that contains the elements of the transitions.
        """
        dataset = self._load_data()
        if split_ratio is not None:
            demo_s1, demo_a1, demo_s2, demo_a2, demo_r, demo_c, demo_d = self._split(
                split_ratio,
                split_shuffle,
                *dataset,
            )
        else:
            demo_s1, demo_a1, demo_s2, demo_a2, demo_r, demo_c, demo_d = dataset
        if self._state_normalize:
            demo_s1, demo_s2, self._obs_shift, self._obs_scale = self.norm_state(demo_s1, demo_s2)
        if self._reward_normalize:
            demo_r = self.norm_reward(demo_r)
        transitions = OrderedDict(
            s1=demo_s1,
            a1=demo_a1,
            s2=demo_s2,
            a2=demo_a2,
            reward=demo_r,
            cost=demo_c,
            done=demo_d
        )
        return transitions

    @staticmethod
    def norm_state(s1: np.ndarray, s2: np.ndarray)\
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Normalize the states.

        :param np.ndarray s1: the states;
        :param np.ndarray s2: the next states;
        :return: normalized states
        """
        shift = -np.mean(s1, 0)
        scale = 1.0 / (np.std(s1, 0) + 1e-3)
        s1 = (s1 + shift) * scale
        s2 = (s2 + shift) * scale
        return s1, s2, shift, scale

    @staticmethod
    def norm_reward(r: np.ndarray) -> np.ndarray:
        """Normalize the reward.

        :param  np.ndarray r: reward;
        :return: normalized reward.
        """
        r_max = np.max(r)
        r_min = np.min(r)
        r = (r - r_min) / (r_max - r_min)
        return r

    @staticmethod
    def get_keys(h5file: h5py.File) -> List[str]:
        keys = []

        def visitor(name, item):
            if isinstance(item, h5py.Dataset):
                keys.append(name)

        h5file.visititems(visitor)
        return keys

    @property
    def state_shift_scale(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._obs_shift, self._obs_scale


class D4rlDataLoader(BaseBMLoader):
    """Get transitions from the D4RL dataset.
    """

    TYPE: ClassVar[str] = 'd4rl'

    def __init__(
            self,
            file_path: str,
            state_normalize: bool = False,
            reward_normalize: bool = False) -> None:
        super(D4rlDataLoader, self).__init__(
            file_path,
            state_normalize,
            reward_normalize)

    def _load_data(self):
        dataset_file = h5py.File(self._file_path + '.hdf5', 'r')
        offline_dataset = {}
        for k in self.get_keys(dataset_file):
            try:
                # first try loading as an array
                offline_dataset[k] = dataset_file[k][:]
            except ValueError as e:  # try loading as a scalar
                offline_dataset[k] = dataset_file[k][()]
        dataset_file.close()
        dataset_size = len(offline_dataset['observations'])
        offline_dataset['terminals'] = np.squeeze(offline_dataset['terminals'])
        offline_dataset['rewards'] = np.squeeze(offline_dataset['rewards'])
        nonterminal_steps, = np.where(
            np.logical_and(
                np.logical_not(offline_dataset['terminals']),
                np.arange(dataset_size) < dataset_size - 1))
        logging.info('Found %d non-terminal steps out of a total of %d steps.' % (
            len(nonterminal_steps), dataset_size))
        demo_s1 = offline_dataset['observations'][nonterminal_steps]
        demo_s2 = offline_dataset['observations'][nonterminal_steps + 1]
        demo_a1 = offline_dataset['actions'][nonterminal_steps]
        demo_a2 = offline_dataset['actions'][nonterminal_steps + 1]
        demo_r = offline_dataset['rewards'][nonterminal_steps]
        demo_d = offline_dataset['terminals'][nonterminal_steps + 1]
        # set the 'demo_c' to 0.
        demo_c = np.zeros(shape=demo_r.shape)

        return demo_s1, demo_a1, demo_s2, demo_a2, demo_r, demo_c, demo_d
