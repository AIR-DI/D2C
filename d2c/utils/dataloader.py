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


# class AppDataLoader(BaseDataLoader):
#     """The data loader for the real-world applications dataset."""
#     pass


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
