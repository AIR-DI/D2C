import pytest
import numpy as np
from d2c.envs.external.d4rl import D4rlEnv


class TestD4rlEnv:
    env_name = 'Hopper-v2'
    obs_shift = np.zeros((11,))
    obs_scale = np.ones((11,))

    def test_d4rl_env(self):
        env = D4rlEnv(
            env_name=self.env_name,
            obs_shift=self.obs_shift,
            obs_scale=self.obs_scale,
        )
        obs = env.reset(seed=1)
        assert obs.shape == (11,)
        for i in range(20):
            action = np.random.uniform(-1, 1, (3,))
            obs, r, d, _ = env.step(action)
            assert obs.shape == (11,)
            assert isinstance(r, float)
            assert isinstance(d, bool)


if __name__ == '__main__':
    pytest.main(__file__)

