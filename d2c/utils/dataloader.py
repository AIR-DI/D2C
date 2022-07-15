"""Dataloader for loading dataset and generating transitions.

There are data_loaders for benchmarks and real-world applications.
"""

import os
import h5py
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import List, Tuple
from collections import OrderedDict


class BaseDataLoader(ABC):
    """The basic class of the dataset loader.

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
    def get_transitions(self):
        """Get the transitions from the dataset.

        :return: A namedtuple that contains the elements of the transitions.
        """
        pass


class BaseBMLoader(BaseDataLoader):
    """The basic class of the benchmark dataset loader.
    Please inherit this class to build data loaders for different benchmarks.

    The main API method the user should implement is:

    * :meth:`_load_data`: load the transitions from the dataset file and return in \
        requested format.

    The main API method that users of this class need to know is:

    * :meth:`get_transitions`: process the transitions from the dataset and return a namedtuple.

    :param str file_path: the path of the benchmark dataset;
    :param bool normalize_states: if normalize the states;
    :param bool scale_rewards: if normalize the rewards.
    """
    def __init__(
            self,
            file_path: str,
            normalize_states: bool = False,
            scale_rewards: bool = False) -> None:
        self._file_path = file_path
        self._normalize_states = normalize_states
        self._scale_rewards = scale_rewards

    def get_transitions(self) -> Tuple[OrderedDict, np.ndarray, np.ndarray]:
        """Get the transitions from the dataset.

        :return: A namedtuple that contains the elements of the transitions.
        """
        demo_s1, demo_a1, demo_s2, demo_a2, demo_r, demo_c, demo_d = self._load_data()
        if self._normalize_states:
            demo_s1, demo_s2, shift, scale = self._norm_state(demo_s1, demo_s2)
        else:
            shift, scale = None, None
        if self._scale_rewards:
            demo_r = self._scale_r(demo_r)
        transitions = OrderedDict(
            s1=demo_s1,
            a1=demo_a1,
            s2=demo_s2,
            a2=demo_a2,
            reward=demo_r,
            cost=demo_c,
            done=demo_d
        )
        return transitions, shift, scale

    @staticmethod
    def _norm_state(s1: np.ndarray, s2: np.ndarray)\
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
    def _scale_r(r: np.ndarray) -> np.ndarray:
        """Normalize the reward.

        :param  np.ndarray r: reward;
        :return: normalized reward.
        """
        r_max = np.max(r)
        r_min = np.min(r)
        r = (r - r_min) / (r_max - r_min)
        return r

    @staticmethod
    def get_keys(h5file):
        keys = []

        def visitor(name, item):
            if isinstance(item, h5py.Dataset):
                keys.append(name)

        h5file.visititems(visitor)
        return keys