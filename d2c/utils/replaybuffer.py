"""The replay buffer for RL training."""

import torch
import numpy as np
from collections import OrderedDict
from typing import Union, Optional


class ReplayBuffer:
    """The base replay buffer.

    :param int state_dim: the dimension of the state.
    :param int action_dim: the dimension of the action.
    :param int max_size: the maximum size of the buffer.
    :param str device: which device to create the data on. Default to 'cpu'.
    """

    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            max_size: int = int(2e6),
            device: Union[str, int, torch.device] = 'cpu',
    ) -> None:
        self._max_size = max_size
        self._device = device
        self._ptr = 0
        self._size = 0
        self._state = torch.empty((max_size, state_dim), dtype=torch.float32, device=self._device)
        self._action = torch.empty((max_size, action_dim), dtype=torch.float32, device=self._device)
        self._next_state = torch.empty((max_size, state_dim), dtype=torch.float32, device=self._device)
        self._next_action = torch.empty((max_size, action_dim), dtype=torch.float32, device=self._device)
        self._reward = torch.empty(max_size, dtype=torch.float32, device=self._device)
        self._cost = torch.empty(max_size, dtype=torch.float32, device=self._device)
        self._done = torch.empty(max_size, dtype=torch.float32, device=self._device)
        self._dsc = torch.empty(max_size, dtype=torch.float32, device=self._device)
        self._data = OrderedDict(
            s1=self._state,
            a1=self._action,
            s2=self._next_state,
            a2=self._next_action,
            reward=self._reward,
            cost=self._cost,
            done=self._done,
            dsc=self._dsc,
        )
        self._shuffle_indices = None

    def add(
            self,
            *,
            state: Union[np.ndarray, torch.Tensor],
            action: Union[np.ndarray, torch.Tensor],
            next_state: Union[np.ndarray, torch.Tensor],
            next_action: Union[np.ndarray, torch.Tensor],
            reward: Union[np.ndarray, torch.Tensor, float, int],
            done: Union[np.ndarray, torch.Tensor, float, int],
            cost: Union[np.ndarray, torch.Tensor, float, int] = None
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
            cost = torch.zeros([1], dtype=torch.float32, device=self._device)
        transition = OrderedDict(
            s1=state,
            a1=action,
            s2=next_state,
            a2=next_action,
            reward=reward,
            cost=cost,
            done=done,
            dsc=1.-done,
        )
        if isinstance(state, np.ndarray):
            for k, v in transition.items():
                transition[k] = torch.as_tensor(v, dtype=torch.float32, device=self._device)
        for k, v in self._data.items():
            v[self._ptr] = transition[k]
        # Update the _ptr and _size.
        self._ptr = (self._ptr + 1) % self._max_size
        self._size = min(self._size + 1, self._max_size)

    def sample_batch(self, batch_size: int) -> OrderedDict:
        """Sample a batch of data randomly.

        :param int batch_size: the batch size of the sample data.
        """
        ind = torch.randint(0, self._size, size=(batch_size,), device=self._device)

        return OrderedDict((k, torch.clone(v[ind])) for k, v in self._data.items())

    def get_batch_indices(self, indices: np.ndarray) -> OrderedDict:
        """Get the batch of data according to the given indices."""
        assert np.max(indices) < self._size, 'There is an index exceeding the size of the buffer.'
        indices = torch.as_tensor(indices, dtype=torch.long, device=self._device)
        return OrderedDict((k, torch.clone(v[indices])) for k, v in self._data.items())

    def add_transitions(
            self,
            *,
            state: Union[np.ndarray, torch.Tensor],
            action: Union[np.ndarray, torch.Tensor],
            next_state: Union[np.ndarray, torch.Tensor],
            next_action: Union[np.ndarray, torch.Tensor],
            reward: Union[np.ndarray, torch.Tensor] = None,
            done: Union[np.ndarray, torch.Tensor] = None,
            cost: Union[np.ndarray, torch.Tensor] = None
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
            cost = torch.zeros(batch_size, dtype=torch.float32, device=self._device)
        if reward is None:
            reward = torch.zeros(batch_size, dtype=torch.float32, device=self._device)
        if done is None:
            done = torch.zeros(batch_size, dtype=torch.float32, device=self._device)
        transitions = OrderedDict(
            s1=state,
            a1=action,
            s2=next_state,
            a2=next_action,
            reward=reward,
            cost=cost,
            done=done,
            dsc=1.-done,
        )
        for k, v in transitions.items():
            transitions[k] = torch.as_tensor(v, dtype=torch.float32, device=self._device)

        tail_space = self._max_size - self._ptr
        if batch_size <= tail_space:
            indices = self._ptr + torch.arange(batch_size, device=self._device)
        else:
            tail_indices = torch.arange(self._ptr, self._max_size, device=self._device)
            head_indices = torch.arange(batch_size - tail_space, device=self._device)
            indices = torch.cat([tail_indices, head_indices])

        for k, v in self._data.items():
            v[indices] = transitions[k]
        # Update the _ptr and _size.
        self._ptr = indices[-1]
        self._size = min(self._size + batch_size, self._max_size)

    @property
    def data(self) -> OrderedDict:
        """All the transitions in the buffer."""
        return self._data

    @property
    def capacity(self) -> int:
        """The capacity of the replay buffer."""
        return self._max_size

    @property
    def size(self) -> int:
        """The number of the transitions in the replay buffer."""
        return self._size

    @property
    def shuffle_indices(self) -> np.ndarray:
        """Returning the shuffled indices of the transitions in the buffer."""
        if self._shuffle_indices is None:
            assert self._size > 0, 'There is no data in buffer!'
            self._shuffle_indices = np.random.permutation(self._size)
        return self._shuffle_indices

