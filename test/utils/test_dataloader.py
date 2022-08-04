import pytest
from d2c.utils.dataloader import D4rlDataLoader


class TestDataLoader:

    def test_d4rl_dataloader(self):
        file_path = '../../example/benchmark/data/d4rl/mujoco/hopper_random-v2'  # Without the suffix .hdf5
        normalize_state = True
        scale_reward = True
        dl = D4rlDataLoader(
            file_path,
            normalize_state,
            scale_reward,
        )
        data = dl.get_transitions()
        assert data['s1'].shape[1] == 11
        assert data['a1'].shape[1] == 3

        s_shift_sacle = dl.state_shift_scale
        for s in s_shift_sacle:
            assert s.shape == (11,)


if __name__ == '__main__':
    pytest.main(__file__)
