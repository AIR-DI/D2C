import pytest
import torch
import numpy as np
from d2c.utils.replaybuffer import ReplayBuffer


class TestReplayBuffer:

    def test_replaybuffer(self):
        state_dim = 10
        action_dim = 5
        max_size = 1000
        data_size = 500
        batch_size = 64
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        rb = ReplayBuffer(
            state_dim,
            action_dim,
            max_size,
            device,
        )
        s1 = np.ones((data_size, state_dim))
        a1 = np.ones((data_size, action_dim))
        s2 = np.ones((data_size, state_dim))
        a2 = np.ones((data_size, action_dim))
        r = np.ones((data_size,))
        d = np.zeros((data_size,))
        rb.add(
            state=s1[0],
            action=a1[0],
            next_state=s2[0],
            next_action=a2[0],
            reward=r[0],
            done=d[0],
        )
        for _ in range(3):
            rb.add_transitions(
                state=s1,
                action=a1,
                next_state=s2,
                next_action=a2,
                reward=r,
                done=d,
            )
        _batch = rb.sample_batch(batch_size)
        assert isinstance(_batch['s1'], torch.Tensor)
        assert _batch['s1'].shape == (batch_size, state_dim)
        assert _batch['a1'].shape == (batch_size, action_dim)

        indices = np.arange(100, 200)
        _batch = rb.get_batch_indices(indices)
        assert _batch['s1'].shape == (100, state_dim)

        assert rb.capacity == max_size
        assert rb.size == max_size


if __name__ == '__main__':
    pytest.main(__file__)


