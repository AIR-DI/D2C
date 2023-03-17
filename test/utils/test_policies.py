import pytest
import torch
import numpy as np
from gym.spaces import Box
from d2c.utils.policies import DeterministicPolicy, DeterministicSoftPolicy
from d2c.utils.networks import ActorNetworkDet, ActorNetwork


class TestPolicies:

    obs_space = Box(low=-np.inf, high=np.inf, shape=(10,))
    act_space = Box(low=-np.ones((3,)), high=np.ones((3,)))
    layer = [300, 300]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    batch_size = 64
    s1 = np.random.random((batch_size, obs_space.shape[0]))
    s2 = torch.ones((batch_size, obs_space.shape[0]))
    a1 = np.random.random((batch_size, act_space.shape[0]))
    a2 = torch.ones((batch_size, act_space.shape[0]))

    s_dim = obs_space.shape[0]
    a_dim = act_space.shape[0]

    actor_det = ActorNetworkDet(
        observation_space=obs_space,
        action_space=act_space,
        fc_layer_params=layer,
        device=device,
    ).to(device)

    actor = ActorNetwork(
        observation_space=obs_space,
        action_space=act_space,
        fc_layer_params=layer,
        device=device,
    ).to(device)

    def test_det_policy(self):
        det_policy = DeterministicPolicy(
            self.actor_det
        )
        for s in [self.s1, self.s2]:
            a = det_policy(s)
            assert isinstance(a, type(s))
            assert a.shape == (self.batch_size, self.a_dim)

    def test_det_soft_policy(self):
        det_soft_policy = DeterministicSoftPolicy(
            self.actor,
        )
        for s in [self.s1, self.s2]:
            a = det_soft_policy(s)
            assert isinstance(a, type(s))
            assert a.shape == (self.batch_size, self.a_dim)


