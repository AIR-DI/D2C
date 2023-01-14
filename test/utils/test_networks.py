import pytest
import torch
import numpy as np
from gym.spaces import Box
from d2c.utils.networks import (
    ActorNetwork,
    ActorNetworkDet,
    CriticNetwork,
    MLP,
    Discriminator,
    ProbDynamicsNetwork,
)


class TestNet:

    obs_space = Box(low=-np.inf, high=np.inf, shape=(10,))
    act_space = Box(low=-np.ones((3,)), high=np.ones((3,)))
    layer = [300, 300]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    batch_size = 64
    s1 = np.random.random((batch_size, obs_space.shape[0]))
    s2 = torch.tensor(s1)
    a1 = np.random.random((batch_size, act_space.shape[0]))
    a2 = torch.tensor(a1)
    reward = torch.ones(batch_size)

    s_dim = obs_space.shape[0]
    a_dim = act_space.shape[0]

    def test_actor(self):
        actor = ActorNetwork(
            observation_space=self.obs_space,
            action_space=self.act_space,
            fc_layer_params=self.layer,
            device=self.device,
        ).to(self.device)

        for s in [self.s1, self.s2]:
            a, a_sample, log_pi_a = actor(s)
            assert a.shape == (self.batch_size, self.a_dim)
            assert a_sample.shape == (self.batch_size, self.a_dim)
            assert log_pi_a.shape == (self.batch_size, self.a_dim)

        for s, a in zip([self.s1, self.s2], [self.a1, self.a2]):
            log_den = actor.get_log_density(s, a)
            assert log_den.shape == (self.batch_size, self.a_dim)

        for s in [self.s1, self.s2]:
            a, a_sample, log_pi_a = actor.sample_n(s, 5)
            assert a.shape == (self.batch_size, self.a_dim)
            assert a_sample.shape == (5, self.batch_size, self.a_dim)
            assert log_pi_a.shape == (5, self.batch_size, self.a_dim)

        for s in [self.s1, self.s2]:
            a_sample = actor.sample(s)
            assert a_sample.shape == (self.batch_size, self.a_dim)

        assert actor.action_space == self.act_space

    def test_actor_det(self):
        actor_det = ActorNetworkDet(
            observation_space=self.obs_space,
            action_space=self.act_space,
            fc_layer_params=self.layer,
            device=self.device,
        ).to(self.device)
        for s in [self.s1, self.s2]:
            a = actor_det(s)
            assert a.shape == (self.batch_size, self.a_dim)

        assert actor_det.action_space == self.act_space

    def test_critic(self):
        critic = CriticNetwork(
            observation_space=self.obs_space,
            action_space=self.act_space,
            fc_layer_params=self.layer,
            device=self.device,
        ).to(self.device)
        for s, a in zip([self.s1, self.s2], [self.a1, self.a2]):
            q = critic(s, a)
            assert len(q) == self.batch_size

    def test_mlp(self):
        mlp = MLP(
            self.s_dim,
            self.a_dim,
            fc_layer_params=self.layer,
            device=self.device,
        ).to(self.device)
        for s in [self.s1, self.s2]:
            a = mlp(s)
            assert a.shape == (self.batch_size, self.a_dim)

    def test_prob_dynamics(self):
        dynamics1 = ProbDynamicsNetwork(
            self.obs_space.shape[0],
            self.act_space.shape[0],
            self.layer,
            device=self.device,
        ).to(self.device)

        dynamics2 = ProbDynamicsNetwork(
            self.obs_space.shape[0],
            self.act_space.shape[0],
            self.layer,
            local_mode=True,
            device=self.device,
        ).to(self.device)

        for dynamics in [dynamics1, dynamics2]:
            for s, a in zip([self.s1, self.s2], [self.a1, self.a2]):
                mean, sample, dist = dynamics(s, a)
                assert mean.shape == (self.batch_size, self.s_dim)
                assert sample.shape == (self.batch_size, self.s_dim)
                assert dist.sample().shape == (self.batch_size, self.s_dim)

            for s, a in zip([self.s1, self.s2], [self.a1, self.a2]):
                log_density = dynamics.get_log_density(s, a, s)
                assert log_density.shape == (self.batch_size, self.s_dim)

        dynamics = ProbDynamicsNetwork(
            self.obs_space.shape[0],
            self.act_space.shape[0],
            self.layer,
            local_mode=True,
            with_reward=True,
            device=self.device,
        ).to(self.device)
        for s, a in zip([self.s1, self.s2], [self.a1, self.a2]):
            mean, sample, dist = dynamics(s, a)
            assert mean.shape == (self.batch_size, self.s_dim + 1)
            assert sample.shape == (self.batch_size, self.s_dim + 1)
            assert dist.sample().shape == (self.batch_size, self.s_dim + 1)

        _output = torch.cat([self.s2, torch.reshape(self.reward, (-1, 1))], -1)
        log_density = dynamics.get_log_density(self.s2, self.a2, _output)
        assert log_density.shape == (self.batch_size, self.s_dim + 1)

    def test_discriminator(self):
        disc = Discriminator(
            self.obs_space,
            self.act_space,
            self.layer,
            self.device,
        ).to(self.device)

        actor = ActorNetwork(
            observation_space=self.obs_space,
            action_space=self.act_space,
            fc_layer_params=self.layer,
            device=self.device,
        ).to(self.device)

        dynamics = ProbDynamicsNetwork(
            self.obs_space.shape[0],
            self.act_space.shape[0],
            self.layer,
            device=self.device,
        ).to(self.device)

        for s, a in zip([self.s1, self.s2], [self.a1, self.a2]):
            log1 = actor.get_log_density(s, a)
            log2 = dynamics.get_log_density(s, a, s)
            out = disc(s, a, log1, log2)
            assert out.shape == (self.batch_size,)


if __name__ == '__main__':
    pytest.main(__file__)
