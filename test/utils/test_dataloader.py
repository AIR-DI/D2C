import numpy as np
import pandas as pd
import pytest
from d2c.utils.dataloader import D4rlDataLoader, AppDataLoader


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

        split_ratio = 0.2
        data_split = dl.get_transitions(split_ratio)
        assert data_split['s1'].shape[1] == 11
        assert data_split['a1'].shape[1] == 3
        assert data_split['s1'].shape[0] == int(data['s1'].shape[0] * split_ratio)
        assert data_split['a1'].shape[0] == int(data['a1'].shape[0] * split_ratio)

    def test_app_dataloader(self):
        file_path = './temp/cold_source-low-free-unit_num_1.csv'
        state_indices = np.arange(0, 9)
        action_indices = np.arange(9, 11)
        app_dataloader = AppDataLoader(file_path, state_indices, action_indices)
        trans = app_dataloader.get_transitions()
        assert trans['s1'].shape == trans['s2'].shape
        assert trans['a1'].shape == trans['a2'].shape
        assert trans['reward'].shape == trans['cost'].shape == trans['done'].shape
        assert trans['s1'].shape[1] == len(state_indices)
        assert trans['a1'].shape[1] == len(action_indices)
        assert trans['s1'].shape[0] == trans['a1'].shape[0] == trans['reward'].shape[0]

        data = pd.read_csv(file_path)
        _min = data.iloc[:, :-1].min(axis=0).to_numpy()
        _min = np.floor(_min)
        _max = data.iloc[:, :-1].max(axis=0).to_numpy()
        _max = np.ceil(_max)
        state_scaler = 'min_max'
        s_min = _min[state_indices]
        s_max = _max[state_indices]
        s_scaler_params = dict(minimum=s_min, maximum=s_max)
        action_scaler = 'min_max'
        a_min = _min[action_indices]
        a_max = _max[action_indices]
        a_scaler_params = dict(minimum=a_min, maximum=a_max)
        reward_scaler = 'min_max'
        r_scaler_params = None

        def r_fn(past_a, s, a, next_s):
            return np.random.random(s.shape[0])

        def d_fn(past_a, s, a, next_s):
            return np.zeros(s.shape[0])

        app_dataloader1 = AppDataLoader(
            file_path=file_path,
            state_indices=state_indices,
            action_indices=action_indices,
            state_scaler=state_scaler,
            state_scaler_params=s_scaler_params,
            action_scaler=action_scaler,
            action_scaler_params=a_scaler_params,
            reward_scaler=reward_scaler,
            reward_scaler_params=r_scaler_params,
            reward_fn=r_fn,
            done_fn=d_fn,
        )
        trans1 = app_dataloader1.get_transitions()

        s1_ref = (trans['s1'] - s_min) / (s_max - s_min)
        s2_ref = (trans['s2'] - s_min) / (s_max - s_min)
        assert np.allclose(trans1['s1'], s1_ref, atol=1e-6)
        assert np.allclose(trans1['s2'], s2_ref)

        a1_ref = (trans['a1'] - a_min) / (a_max - a_min)
        a2_ref = (trans['a2'] - a_min) / (a_max - a_min)
        assert np.allclose(trans1['a1'], a1_ref)
        assert np.allclose(trans1['a2'], a2_ref)

        assert np.all(trans1['reward'] >= 0)
        assert np.all(trans1['reward'] <= 1)
        assert np.all(trans1['done'] == 0)
        assert np.all(trans1['cost'] == 0)

        s_scaler = app_dataloader1.get_scalers('state')
        s_scaler_params_ = s_scaler.get_params()
        assert np.all(s_scaler_params_['minimum'] == s_min)
        assert np.all(s_scaler_params_['maximum'] == s_max)


if __name__ == '__main__':
    pytest.main(__file__)
