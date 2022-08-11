"""
Implementation of TD3+BC (A Minimalist Approach to Offline Reinforcement Learning)
Paper: https://arxiv.org/pdf/2106.06860.pdf
"""
import collections
import torch
import copy
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Union, Tuple, Any, Sequence, Dict, Iterator
from d2c.models.base import BaseAgent, BaseAgentModule
from d2c.utils import networks, utils, policies


class TD3BCAgent(BaseAgent):
    """Implementation of TD3+BC

    :param float policy_noise: the noise used in updating policy network.
    :param int update_actor_freq: the update frequency of actor network.
    :param float noise_clip: the clipping range used in updating policy network.
    :param float alpha: the value of alpha, which controls the weight for TD3 learning
        relative to behavior cloning.

    .. seealso::

        Please refer to :class:`~d2c.models.base.BaseAgent` for more detailed
        explanation.
    """

    def __init__(
            self,
            policy_noise: float = 0.2,
            update_actor_freq: int = 2,
            noise_clip: float = 0.5,
            alpha: float = 2.5,
            **kwargs: Any,
    ) -> None:
        self._policy_noise = policy_noise
        self._update_actor_freq = update_actor_freq
        self._noise_clip = noise_clip
        self._alpha = alpha
        self._p_info = collections.OrderedDict()
        super(TD3BCAgent, self).__init__(**kwargs)

    def _build_fns(self) -> None:
        self._agent_module = AgentModule(modules=self._modules)
        self._q_fns = self._agent_module.q_nets
        self._q_target_fns = self._agent_module.q_target_nets
        self._p_fn = self._agent_module.p_net
        self._p_target_fn = self._agent_module.p_target_net

    def _init_vars(self) -> None:
        pass

    def _build_optimizers(self) -> None:
        opts = self._optimizers
        self._q_optimizer = utils.get_optimizer(opts.q[0])(
            parameters=self._q_fns.parameters(),
            lr=opts.q[1],
            weight_decay=self._weight_decays,
        )
        self._p_optimizer = utils.get_optimizer(opts.p[0])(
            parameters=self._p_fn.parameters(),
            lr=opts.p[1],
            weight_decay=self._weight_decays,
        )

    def _build_q_loss(self, batch: Dict) -> Tuple[Tensor, Dict]:
        s1 = batch['s1']
        s2 = batch['s2']
        a1 = batch['a1']
        r = batch['reward']
        dsc = batch['dsc']
        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                    torch.randn_like(a1) * self._policy_noise
            ).clamp(-self._noise_clip, self._noise_clip)
            next_action = (
                    self._p_target_fn(s2) + noise
            ).clamp(self._a_min, self._a_max)

            # Compute the target Q value
            target_q1 = self._q_target_fns[0](s2, next_action)
            target_q2 = self._q_target_fns[1](s2, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = r + dsc * self._discount * target_q

        # Get current Q estimates
        current_q1 = self._q_fns[0](s1, a1)
        current_q2 = self._q_fns[1](s1, a1)

        # Compute critic loss
        q_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        info = collections.OrderedDict()
        info['Q1'] = current_q1.detach().mean()
        info['Q2'] = current_q2.detach().mean()
        info['Q_target'] = target_q.mean()
        info['Q_loss'] = q_loss
        info['r_mean'] = r.mean()
        info['dsc'] = dsc.mean()
        info['dsc_min'] = dsc.min()
        return q_loss, info

    def _build_p_loss(self, batch: Dict) -> Tuple[Tensor, Dict]:
        s = batch['s1']
        a = batch['a1']
        pi = self._p_fn(s)
        q = self._q_fns[0](s, pi)
        lmbda = self._alpha / q.abs().mean().detach()
        p_loss = -lmbda * q.mean() + F.mse_loss(pi, a)
        info = collections.OrderedDict()
        info['lambda'] = lmbda
        info['actor_loss'] = p_loss
        info['Q_in_actor_loss'] = q.detach().mean()
        return p_loss, info

    def _optimize_q(self, batch: Dict) -> Dict:
        loss, info = self._build_q_loss(batch)
        self._q_optimizer.zero_grad()
        loss.backward()
        self._q_optimizer.step()
        return info

    def _optimize_p(self, batch: Dict) -> Dict:
        loss, info = self._build_p_loss(batch)
        self._p_optimizer.zero_grad()
        loss.backward()
        self._p_optimizer.step()
        return info

    def _optimize_step(self, batch: Dict) -> Dict:
        info = collections.OrderedDict()
        q_info = self._optimize_q(batch)
        if self._global_step % self._update_actor_freq == 0:
            self._p_info = self._optimize_p(batch)
            # Update the target networks.
            self._update_target_fns(self._q_fns, self._q_target_fns)
            self._update_target_fns(self._p_fn, self._p_target_fn)
        info.update(q_info)
        info.update(self._p_info)
        return info

    def _build_test_policies(self) -> None:
        policy = policies.DeterministicPolicy(
            a_network=self._p_fn
        )
        self._test_policies['main'] = policy

    def save(self, ckpt_name: str) -> None:
        torch.save(self._agent_module.state_dict(), ckpt_name + '.pth')
        torch.save(self._agent_module.p_net.state_dict(), ckpt_name + '_policy.pth')

    def restore(self, ckpt_name: str) -> None:
        self._agent_module.load_state_dict(torch.load(ckpt_name + '.pth'))

    def _get_modules(self) -> utils.Flags:
        model_params_q, n_q_fns = self._model_params.q
        model_params_p = self._model_params.p[0]

        def q_net_factory():
            return networks.CriticNetwork(
                observation_space=self._observation_space,
                action_space=self._action_space,
                fc_layer_params=model_params_q,
                device=self._device,
            )

        def p_net_factory():
            return networks.ActorNetworkDet(
                observation_space=self._observation_space,
                action_space=self._action_space,
                fc_layer_params=model_params_p,
                device=self._device,
            )

        modules = utils.Flags(
            q_net_factory=q_net_factory,
            p_net_factory=p_net_factory,
            n_q_fns=n_q_fns,
            device=self._device,
        )
        return modules


class AgentModule(BaseAgentModule):

    def _build_modules(self) -> None:
        device = self._net_modules.device
        self._q_nets = nn.ModuleList()
        n_q_fns = self._net_modules.n_q_fns  # The number of the Q nets.
        for _ in range(n_q_fns):
            self._q_nets.append(self._net_modules.q_net_factory().to(device))
        self._q_target_nets = copy.deepcopy(self._q_nets)
        self._p_net = self._net_modules.p_net_factory().to(device)
        self._p_target_net = copy.deepcopy(self._p_net)

    @property
    def q_nets(self) -> nn.ModuleList:
        return self._q_nets

    @property
    def q_target_nets(self) -> nn.ModuleList:
        return self._q_target_nets

    @property
    def p_net(self) -> nn.Module:
        return self._p_net

    @property
    def p_target_net(self) -> nn.Module:
        return self._p_target_net




