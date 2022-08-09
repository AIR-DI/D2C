"""
Implementation of TD3+BC (A Minimalist Approach to Offline Reinforcement Learning)
Paper: https://arxiv.org/pdf/2106.06860.pdf
"""
import collections
import torch
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
        self._p_fn = self._agent_module.p_current_net
        self._p_target_fn = self._agent_module.p_target_net

    def _init_vars(self) -> None:
        self._q_vars = self._agent_module.q_source_variables
        self._q_target_vars = self._agent_module.q_target_variables
        self._p_vars = self._agent_module.p_current_variables
        self._p_target_vars = self._agent_module.p_target_variables

    def _build_optimizers(self) -> None:
        opts = self._optimizers
        self._q_optimizer = utils.get_optimizer(opts.q[0])(
            parameters=self._q_vars,
            lr=opts.q[1],
            weight_decay=self._weight_decays,
        )
        self._p_optimizer = utils.get_optimizer(opts.p[0])(
            parameters=self._p_vars,
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
            noise = (torch.randn_like(a1, device=self._device) * self._policy_noise).clamp(-self._noise_clip,
                                                                                           self._noise_clip)
            a2_p = (self._p_target_fn(s2) + noise).clamp(self._a_min, self._a_max)
            q2_targets = []
            for _, q_target_fn in self._q_fns:
                q2_targets_ = q_target_fn(s2, a2_p)
                q2_targets.append(q2_targets_)
            q2_target = torch.min(*q2_targets)
            q1_target = r + dsc * self._discount * q2_target
        # q_losses = []
        # for q_fn, _ in self._q_fns:
        #     q1_pred_ = q_fn(s1, a1)
        #     # q_loss_ = nn.MSELoss()(q1_pred_, q1_target)
        #     q_loss_ = F.mse_loss(q1_pred_, q1_target)
        #     q_losses.append(q_loss_)
        # q_loss = torch.add(*q_losses)
        q1_pred1 = self._q_fns[0][0](s1, a1)
        q1_pred2 = self._q_fns[1][0](s1, a1)
        q_loss = F.mse_loss(q1_pred1, q1_target) + F.mse_loss(q1_pred2, q1_target)

        info = collections.OrderedDict()
        info['q_loss'] = q_loss
        info['r_mean'] = torch.mean(r)
        info['dsc_mean'] = torch.mean(dsc)
        info['q1_mean'] = q1_pred1.detach().mean()
        info['q2_mean'] = q1_pred2.detach().mean()
        info['q_target_mean'] = torch.mean(q1_target)
        return q_loss, info

    def _build_p_loss(self, batch: Dict) -> Tuple[Tensor, Dict]:
        s = batch['s1']
        a = batch['a1']
        a_p = self._p_fn(s)
        q_pred = self._q_fns[0][0](s, a_p)
        lmbda = self._alpha / q_pred.abs().mean().detach()
        # bc_loss = nn.MSELoss()(a_p, a)
        bc_loss = F.mse_loss(a_p, a)
        p_loss = -lmbda * q_pred.mean() + bc_loss
        # info
        info = collections.OrderedDict()
        info['p_loss'] = p_loss
        info['bc_loss'] = bc_loss
        info['lambda'] = lmbda
        info['Q_in_p_loss'] = q_pred.detach().mean()
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
            self._update_target_fns(self._q_vars, self._q_target_vars)
            self._update_target_fns(self._p_vars, self._p_target_vars)
        info.update(q_info)
        info.update(self._p_info)
        return info

    def _build_test_policies(self) -> None:
        policy = policies.DeterministicPolicy(
            a_network=self._agent_module.p_current_net
        )
        self._test_policies['main'] = policy

    def save(self, ckpt_name: str) -> None:
        torch.save(self._agent_module.state_dict(), ckpt_name + '.pth')
        torch.save(self._agent_module.p_current_net.state_dict(), ckpt_name + '_policy.pth')

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
        n_q_fns = self._net_modules.n_q_fns
        for _ in range(n_q_fns):
            self._q_nets.append(
                nn.ModuleList(
                    [self._net_modules.q_net_factory().to(device),
                     self._net_modules.q_net_factory().to(device)]
                )  # source and target
            )
        self._p_nets = nn.ModuleList(
            [self._net_modules.p_net_factory().to(device),
             self._net_modules.p_net_factory().to(device)]
        )  # source and target

    @property
    def q_nets(self) -> nn.ModuleList:
        return self._q_nets

    @property
    def q_source_variables(self) -> Iterator:
        """The parameters of all the source Q networks."""
        vars_ = []
        for q_net, _ in self._q_nets:
            vars_ += list(q_net.parameters())
        return vars_

    @property
    def q_target_variables(self) -> Iterator:
        """The parameters of all the target Q networks."""
        vars_ = []
        for _, q_net in self._q_nets:
            vars_ += list(q_net.parameters())
        return vars_

    @property
    def p_current_net(self) -> nn.Module:
        return self._p_nets[0]

    @property
    def p_current_variables(self) -> Iterator:
        return self._p_nets[0].parameters()

    @property
    def p_target_net(self) -> nn.Module:
        return self._p_nets[1]

    @property
    def p_target_variables(self) -> Iterator:
        return self._p_nets[1].parameters()




