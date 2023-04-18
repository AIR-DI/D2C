"""
Implementation of IQL (Offline Reinforcement Learning with Implicit Q-Learning)
Paper: https://arxiv.org/pdf/2110.06169.pdf
"""
import collections
import torch
import copy
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Tuple, Any, Dict
from d2c.models.base import BaseAgent, BaseAgentModule
from d2c.utils import networks, utils, policies


class IQLAgent(BaseAgent):
    """Implementation of IQL

    :param float temperature: the value of temperature, which controls the weight for
        maximum of the Q-function to behavior cloning.
    :param float expectile: the hyperparameter of expectile regression.

    .. seealso::

        Please refer to :class:`~d2c.models.base.BaseAgent` for more detailed
        explanation.
    """

    def __init__(
            self,
            temperature: float = 2.0,
            expectile: float = 0.8,
            **kwargs: Any,
    ) -> None:
        self._temperature = temperature
        self._expectile = expectile
        self._p_info = collections.OrderedDict()
        super(IQLAgent, self).__init__(**kwargs)
    
    def _build_fns(self) -> None:
        self._agent_module = AgentModule(modules=self._modules)
        self._v_fn = self._agent_module.v_net
        self._q_fns = self._agent_module.q_nets
        self._q_target_fns = self._agent_module.q_target_nets
        self._p_fn = self._agent_module.p_net  

    def _init_vars(self) -> None:
        pass

    def _build_optimizers(self) -> None:
        opts = self._optimizers
        self._v_optimizer = utils.get_optimizer(opts.v[0])(
            parameters=self._v_fn.parameters(),
            lr=opts.v[1],
            weight_decay=self._weight_decays,
        )
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

    def _build_v_loss(self, batch: Dict) -> Tuple[Tensor, Dict]:
        s = batch['s1']
        a = batch['a1']
        with torch.no_grad():
            q1 = self._q_target_fns[0](s, a) 
            q2 = self._q_target_fns[1](s, a)
            q = torch.minimum(q1, q2)

        # Compute expectile loss
        v = self._v_fn(s)
        diff = q - v
        weight = torch.where(diff > 0, self._expectile, (1 - self._expectile))
        v_loss = (weight * (diff**2)).mean()
        info = collections.OrderedDict()
        info['V'] = v.detach().mean()
        info['v_loss'] = v_loss
        return v_loss, info

    def _build_q_loss(self, batch: Dict) -> Tuple[Tensor, Dict]:
        s1 = batch['s1']
        s2 = batch['s2']
        a1 = batch['a1']
        r = batch['reward']
        dsc = batch['dsc']
        with torch.no_grad():
            # Compute the target Q value
            next_v = self._v_fn(s2)
            target_q = r + dsc * self._discount * next_v
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
        with torch.no_grad():
            q1 = self._q_target_fns[0](s, a) 
            q2 = self._q_target_fns[1](s, a)
            q = torch.minimum(q1, q2)
            v = self._v_fn(s)

        exp_a = torch.exp((q - v) * self._temperature)
        exp_a = torch.min(exp_a, torch.FloatTensor([100.0]).to(self._device))

        # Compute policy loss
        log_prob = self._p_fn.get_log_density(s, a)
        p_loss = (-(exp_a.unsqueeze(-1) * log_prob)).mean()
        info = collections.OrderedDict()
        info['actor_loss'] = p_loss
        info['Q_in_actor_loss'] = q.detach().mean()
        return p_loss, info

    def _optimize_v(self, batch: Dict) -> Dict:
        loss, info = self._build_v_loss(batch)
        self._v_optimizer.zero_grad()
        loss.backward()
        self._v_optimizer.step()
        return info

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
        v_info = self._optimize_v(batch)
        self._p_info = self._optimize_p(batch)
        q_info = self._optimize_q(batch)
        self._update_target_fns(self._q_fns, self._q_target_fns)
        info.update(v_info)
        info.update(self._p_info)
        info.update(q_info)
        return info

    def _build_test_policies(self) -> None:
        policy = policies.DeterministicSoftPolicy(
            a_network=self._p_fn
        )
        self._test_policies['main'] = policy

    def save(self, ckpt_name: str) -> None:
        torch.save(self._agent_module.state_dict(), ckpt_name + '.pth')
        torch.save(self._agent_module.p_net.state_dict(), ckpt_name + '_policy.pth')

    def restore(self, ckpt_name: str) -> None:
        self._agent_module.load_state_dict(torch.load(ckpt_name + '.pth'))

    def _get_modules(self) -> utils.Flags:
        model_params_v = self._model_params.v[0]
        model_params_q, n_q_fns = self._model_params.q
        model_params_p = self._model_params.p[0]

        def v_net_factory():
            return networks.ValueNetwork(
                observation_space=self._observation_space,
                fc_layer_params=model_params_v,
                device=self._device,
            )

        def q_net_factory():
            return networks.CriticNetwork(
                observation_space=self._observation_space,
                action_space=self._action_space,
                fc_layer_params=model_params_q,
                device=self._device,
            )

        def p_net_factory():
            return networks.ActorNetwork(
                observation_space=self._observation_space,
                action_space=self._action_space,
                fc_layer_params=model_params_p,
                device=self._device,
            )

        modules = utils.Flags(
            v_net_factory=v_net_factory,
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
        self._v_net = self._net_modules.v_net_factory().to(device)
        n_q_fns = self._net_modules.n_q_fns  # The number of the Q nets.
        for _ in range(n_q_fns):
            self._q_nets.append(self._net_modules.q_net_factory().to(device))
        self._q_target_nets = copy.deepcopy(self._q_nets)
        self._p_net = self._net_modules.p_net_factory().to(device)

    @property
    def v_net(self) -> nn.Module:
        return self._v_net
    
    @property
    def q_nets(self) -> nn.ModuleList:
        return self._q_nets

    @property
    def q_target_nets(self) -> nn.ModuleList:
        return self._q_target_nets

    @property
    def p_net(self) -> nn.Module:
        return self._p_net


