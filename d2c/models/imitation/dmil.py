"""
Implementation of DMIL (Discriminator-Guided Model-Based Offline Imitation Learning)
Paper: https://arxiv.org/abs/2207.00244
"""
import collections
import torch
from torch import nn, Tensor
from typing import Any, Dict
from typing import Tuple, Type
from d2c.models.base import BaseAgent, BaseAgentModule
from d2c.utils import networks, utils, policies

ModuleType = Type[nn.Module]
LOG_STD_MIN = -5
LOG_STD_MAX = 0
LAMBDA_MIN = 1
LAMBDA_MAX = 100


class DMILAgent(BaseAgent):
    """Implementation of DMIL.

    :param float alpha1: The hyperparameter alpha for policy.
    :param float alpha2: The hyperparameter alpha for dynamics model.
    :param float train_f_steps: The total training steps to train the dynamics model.
    :param int rollout_freq: The frequency value for the dynamics model rollout.
    :param int rollout_size: The size of the rollout data.

    .. seealso::

        Please refer to :class:`~d2c.models.base.BaseAgent` for more detailed
        explanation.
    """

    def __init__(
            self,
            alpha1: float = 10,
            alpha2: float = 10,
            train_f_steps: int = int(1e+3),
            rollout_freq: int = 1000,
            rollout_size: int = None,
            **kwargs: Any,
    ) -> None:
        self._alpha1 = alpha1
        self._alpha2 = alpha2
        self._train_f_steps = train_f_steps
        self._rollout_freq = rollout_freq
        super(DMILAgent, self).__init__(**kwargs)
        if rollout_size is None:
            self._rollout_size = int(self._train_data.size / 10)
        else:
            self._rollout_size = rollout_size
        self._p_info = collections.OrderedDict()

    def _build_fns(self) -> None:
        self._agent_module = AgentModule(modules=self._modules)
        self._f_fn = self._agent_module.f_net
        self._p_fn = self._agent_module.p_net
        self._d_fn = self._agent_module.d_net

    def _init_vars(self) -> None:
        pass

    def _build_optimizers(self) -> None:
        opts = self._optimizers
        self._f_optimizer = utils.get_optimizer(opts.f[0])(
            parameters=self._f_fn.parameters(),
            lr=opts.f[1],
            weight_decay=self._weight_decays,
        )
        self._p_optimizer = utils.get_optimizer(opts.p[0])(
            parameters=self._p_fn.parameters(),
            lr=opts.p[1],
            weight_decay=self._weight_decays,
        )
        self._d_optimizer = utils.get_optimizer(opts.d[0])(
            parameters=self._d_fn.parameters(),
            lr=opts.d[1],
            weight_decay=self._weight_decays,
        )

    def _build_p_loss(self, batch: Dict, batch2: Dict) -> Tuple[Tensor, Dict]:
        s1 = batch['s1']
        a1 = batch['a1']
        s12 = batch['s2']
        s2 = batch2['s1']
        a2 = batch2['a1']
        s22 = batch2['s2']
        logpi = self._p_fn.get_log_density(s1, a1)
        logpi_d = self._p_fn.get_log_density(s2, a2)
        with torch.no_grad():
            lossf1 = self._f_fn.get_log_density(s1, a1, s12)
            lossf2 = self._f_fn.get_log_density(s2, a2, s22)
            output = self._d_fn(s1, a1, logpi, lossf1)
            output = torch.clamp(output, 0.1, 0.9)
            output_d = self._d_fn(s2, a2, logpi_d, lossf2)
            output_d = torch.clamp(output_d, 0.1, 0.9)
        p_loss = self._alpha1 * torch.mean(-torch.sum(logpi, 1)) - torch.mean(-torch.sum(logpi, 1) / output) + \
                 torch.mean(-torch.sum(logpi_d, 1) / (1 - output_d))
        info = collections.OrderedDict()
        info['p_loss'] = p_loss

        return p_loss, info

    def _build_f_loss0(self, batch: Dict) -> Tuple[Tensor, Dict]:
        s1 = batch['s1']
        s12 = batch['s2']
        a1 = batch['a1']
        lossf1 = self._f_fn.get_log_density(s1, a1, s12)
        f_loss = torch.mean(-torch.sum(lossf1, 1))
        info = collections.OrderedDict()
        info['f_loss_pre'] = f_loss

        return f_loss, info

    def _build_f_loss(self, batch: Dict, batch2: Dict) -> Tuple[Tensor, Dict]:
        s1 = batch['s1']
        s12 = batch['s2']
        a1 = batch['a1']
        s2 = batch2['s1']
        s22 = batch2['s2']
        a2 = batch2['a1']
        lossf1 = self._f_fn.get_log_density(s1, a1, s12)
        lossf2 = self._f_fn.get_log_density(s2, a2, s22)
        with torch.no_grad():
            logpi = self._p_fn.get_log_density(s1, a1)
            logpi_d = self._p_fn.get_log_density(s2, a2)
            output = self._d_fn(s1, a1, logpi, lossf1)
            output = torch.clamp(output, 0.1, 0.9)
            output_d = self._d_fn(s2, a2, logpi_d, lossf2)
            output_d = torch.clamp(output_d, 0.1, 0.9)
        f_loss = self._alpha2 * torch.mean(-torch.sum(lossf1, 1)) - torch.mean(-torch.sum(lossf1, 1) / output) + \
                 torch.mean(-torch.sum(lossf2, 1) / (1 - output_d))
        info = collections.OrderedDict()
        info['f_loss'] = f_loss

        return f_loss, info

    def _build_d_loss(self, batch: Dict, batch2: Dict) -> Tuple[Tensor, Dict]:
        s1 = batch['s1']
        a1 = batch['a1']
        s12 = batch['s2']
        s2 = batch2['s1']
        a2 = batch2['a1']
        s22 = batch2['s2']
        with torch.no_grad():
            lossf1 = self._f_fn.get_log_density(s1, a1, s12)
            lossf2 = self._f_fn.get_log_density(s2, a2, s22)
            logpi = self._p_fn.get_log_density(s1, a1)
            logpi_d = self._p_fn.get_log_density(s2, a2)
        out1 = self._d_fn(s1, a1, logpi, lossf1)
        out1 = torch.clamp(out1, 0.1, 0.9)
        out2 = self._d_fn(s2, a2, logpi_d, lossf2)
        out2 = torch.clamp(out2, 0.1, 0.9)
        d_loss = torch.mean(-torch.log(out1)) + torch.mean(-torch.log(1 - out2))
        info = collections.OrderedDict()
        info['d_loss'] = d_loss

        return d_loss, info

    def _optimize_d(self, batch: Dict, batch2: Dict) -> Dict:
        loss, info = self._build_d_loss(batch, batch2)
        self._d_optimizer.zero_grad()
        loss.backward()
        self._d_optimizer.step()
        return info

    def _optimize_p(self, batch: Dict, batch2: Dict) -> Dict:
        loss, info = self._build_p_loss(batch, batch2)
        self._p_optimizer.zero_grad()
        loss.backward()
        self._p_optimizer.step()
        return info

    def _optimize_f(self, batch: Dict, batch2: Dict) -> Dict:
        loss, info = self._build_f_loss(batch, batch2)
        self._f_optimizer.zero_grad()
        loss.backward()
        self._f_optimizer.step()
        return info

    def _optimize_f0(self, batch: Dict) -> Dict:
        loss, info = self._build_f_loss0(batch)
        self._f_optimizer.zero_grad()
        loss.backward()
        self._f_optimizer.step()
        return info

    def _optimize_step(self, batch: Dict) -> Dict:
        info = collections.OrderedDict()
        if self._global_step < self._train_f_steps:
            lf_info = self._optimize_f0(batch)
            info.update(lf_info)
        elif self._global_step == self._train_f_steps:
            _batch = self._train_data.sample_batch(self._empty_dataset.capacity)
            self.generate_rollout(_batch)
        else:
            batch2 = self._empty_dataset.sample_batch(self._batch_size)
            d_info = self._optimize_d(batch, batch2)
            p_info = self._optimize_p(batch, batch2)
            f_info = self._optimize_f(batch, batch2)
            if self._global_step % self._rollout_freq == 0:
                _batch = self._train_data.sample_batch(self._rollout_size)
                self.generate_rollout(_batch)
            info.update(d_info)
            info.update(p_info)
            info.update(f_info)
        return info

    def generate_rollout(self, batch: Dict) -> None:
        s1 = batch['s1']
        a1 = batch['a1']
        with torch.no_grad():
            s2, _, _ = self._f_fn(s1, a1)
            a2, _, _ = self._p_fn(s2)
            s3, _, _ = self._f_fn(s2, a2)
            a3, _, _ = self._p_fn(s3)
        self._empty_dataset.add_transitions(state=s2, action=a2, next_state=s3, next_action=a3)

    def _build_test_policies(self) -> None:
        policy = policies.DeterministicSoftPolicy(
            a_network=self._p_fn
        )
        self._test_policies['main'] = policy

    def save(self, ckpt_name: str) -> None:
        torch.save(self._agent_module.state_dict(), ckpt_name + '.pth')
        torch.save(self._agent_module.p_net.state_dict(), ckpt_name + '_policy.pth')
        torch.save(self._agent_module.d_net.state_dict(), ckpt_name + '_distance.pth')

    def restore(self, ckpt_name: str) -> None:
        self._agent_module.load_state_dict(torch.load(ckpt_name + '.pth'))

    def _get_modules(self) -> utils.Flags:
        model_params_f = self._model_params.f[0]
        model_params_p = self._model_params.p[0]
        model_params_d = self._model_params.d[0]

        def f_net_factory():
            return networks.ProbDynamicsNetwork(
                state_dim=self._observation_space.shape[0],
                action_dim=self._action_space.shape[0],
                fc_layer_params=model_params_f,
                device=self._device,
            )

        def p_net_factory():
            return networks.ActorNetwork(
                observation_space=self._observation_space,
                action_space=self._action_space,
                fc_layer_params=model_params_p,
                device=self._device,
            )

        def d_net_factory():
            return networks.Discriminator(
                observation_space=self._observation_space,
                action_space=self._action_space,
                fc_layer_params=model_params_d,
                device=self._device,
            )

        modules = utils.Flags(
            f_net_factory=f_net_factory,
            p_net_factory=p_net_factory,
            d_net_factory=d_net_factory,
            device=self._device,
        )
        return modules


class AgentModule(BaseAgentModule):

    def _build_modules(self) -> None:
        device = self._net_modules.device
        self._p_net = self._net_modules.p_net_factory().to(device)
        self._f_net = self._net_modules.f_net_factory().to(device)
        self._d_net = self._net_modules.d_net_factory().to(device)

    @property
    def f_net(self) -> nn.ModuleList:
        return self._f_net

    @property
    def p_net(self) -> nn.Module:
        return self._p_net

    @property
    def d_net(self) -> nn.Module:
        return self._d_net


