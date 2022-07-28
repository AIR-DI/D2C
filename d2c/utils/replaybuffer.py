"""The replay buffer for RL training."""

import numpy as np
from collections import OrderedDict
from typing import Union


class ReplayBuffer:
    """The base replay buffer.

    The main methods:

    :param int state_dim: the dimension of the state.
    :param int action_dim: the dimension of the action.
    :param int max_size: the maximum size of the buffer.
    """

    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            max_size: int = int(2e6)
    ) -> None:
        self._max_size = max_size
        self._ptr = 0
        self._size = 0
        self._state = np.zeros((max_size, state_dim))
        self._action = np.zeros((max_size, action_dim))
        self._next_state = np.zeros((max_size, state_dim))
        self._next_action = np.zeros((max_size, action_dim))
        self._reward = np.zeros(max_size)
        self._cost = np.zeros(max_size)
        self._done = np.zeros(max_size)
        self._data = OrderedDict(
            s1=self._state,
            a1=self._action,
            s2=self._next_state,
            a2=self._next_action,
            reward=self._reward,
            cost=self._cost,
            done=self._done
        )

    def add(
            self,
            state: np.ndarray,
            action: np.ndarray,
            next_state: np.ndarray,
            next_action: np.ndarray,
            reward: Union[np.ndarray, float, int],
            done: Union[np.ndarray, float, int],
            cost: Union[np.ndarray, float, int] = None
    ) -> None:
        """Add a transition into the buffer.

        :param np.ndarray state: the state with shape (1, state_dim) or (state_dim,)
        :param np.ndarray action: the action with shape (1, action_dim) or (action_dim)
        :param np.ndarray next_state: the next_state with shape (1, state_dim) or (state_dim,)
        :param np.ndarray next_action: the next_action with shape (1, action_dim) or (action_dim)
        :param np.ndarray reward: the reward with shape (1,) or ()
        :param np.ndarray done: the done with shape (1,) or ()
        :param np.ndarray cost: the cost with shape (1,) or ()
        """
        if len(state.shape) > 1:
            assert len(state.shape) == 2
            assert state.shape[0] == 1, 'The shape of the input data is wrong!'
        if cost is None:
            cost = np.zeros([1])
        transition = OrderedDict(
            s1=state,
            a1=action,
            s2=next_state,
            a2=next_action,
            reward=reward,
            cost=cost,
            done=done
        )
        for k, v in self._data.items():
            v[self._ptr] = transition[k]
        # Update the _ptr and _size.
        self._ptr = (self._ptr + 1) % self._max_size
        self._size = min(self._size + 1, self._max_size)

    def sample_batch(self, batch_size: int) -> OrderedDict:
        """Sample a batch of data randomly.

        :param int batch_size: the batch size of the sample data.
        """
        ind = np.random.randint(0, self._size, size=batch_size)

        return OrderedDict((k, np.array(v[ind])) for k, v in self._data.items())  # TODO return torch.Tensor

    def get_batch_indices(self, indices: np.ndarray) -> OrderedDict:
        """Get the batch of data according to the given indices."""
        assert np.max(indices) < self._size, 'There is an index exceeding the size of the buffer.'
        return OrderedDict((k, np.array(v[indices])) for k, v in self._data.items())  # TODO return torch.Tensor

    def add_transitions(
            self,
            state: np.ndarray,
            action: np.ndarray,
            next_state: np.ndarray,
            next_action: np.ndarray,
            reward: np.ndarray,
            done: np.ndarray,
            cost: np.ndarray = None
    ) -> None:
        """Add a batch of transitions into the buffer.

        :param np.ndarray state: the state with shape (batch_size, state_dim)
        :param np.ndarray action: the action with shape (batch_size, action_dim)
        :param np.ndarray next_state: the next_state with shape (batch_size, state_dim)
        :param np.ndarray next_action: the next_action with shape (batch_size, action_dim)
        :param np.ndarray reward: the reward with shape (batch_size,)
        :param np.ndarray done: the done with shape (batch_size,)
        :param np.ndarray cost: the cost with shape (batch_size,)
        """
        batch_size = state.shape[0]
        if cost is None:
            cost = np.zeros(batch_size)
        tail_space = self._max_size - self._ptr
        if batch_size <= tail_space:
            indices = self._ptr + np.arange(batch_size)
        else:
            tail_indices = np.arange(self._ptr, self._max_size)
            head_indices = np.arange(batch_size - tail_space)
            indices = np.concatenate([tail_indices, head_indices])
        transitions = OrderedDict(
            s1=state,
            a1=action,
            s2=next_state,
            a2=next_action,
            reward=reward,
            cost=cost,
            done=done
        )
        for k, v in self._data.items():
            v[indices] = transitions[k]
        # Update the _ptr and _size.
        self._ptr = indices[-1]
        self._size = min(self._size + batch_size, self._max_size)

    @property
    def data(self) -> OrderedDict:
        """All of the transitions in the buffer."""
        return self._data

    @property
    def capacity(self) -> int:
        """The capacity of the replay buffer."""
        return self._max_size

    @property
    def size(self) -> int:
        """The number of the transitions in the replay buffer."""
        return self._size

