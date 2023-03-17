import collections
import torch
import numpy as np
from typing import ClassVar, Any, Dict, Tuple, Union, List
from torch import nn, Tensor
from d2c.envs.learned.dynamics.base import BaseDyna, BaseDynaModule
from d2c.utils import networks
from d2c.utils import utils


class ProbDyna(BaseDyna):
    """Implementation of dynamics with probabilistic neural network.

    Use the deep fully-connected network as the dynamics model.
    The inputs are the current state and action and the outputs
    is ``(mean, std)`` of the distribution of the predict next state.

    :param bool local_mode: `local_mode` means that this dynamics model predicts
        the difference to the current state.
    """

    TYPE: ClassVar[str] = 'prob'

    def __init__(self, local_mode: bool = True, **kwargs: Any):
        self._local_mode = local_mode
        super(ProbDyna, self).__init__(**kwargs)

    def _build_fns(self) -> None:
        self._dyna_module = DynaModule(modules=self._modules)
        self._d_fns = self._dyna_module.d_nets

    def _build_optimizers(self) -> None:
        opt = self._optimizers
        self._optimizer = utils.get_optimizer(opt[0])(
            parameters=self._d_fns.parameters(),
            lr=opt[1],
            weight_decay=self._weight_decays,
        )

    def _build_loss(self, batch: Dict) -> Tuple[Tensor, Dict]:
        s1 = batch['s1']
        a1 = batch['a1']
        s2 = batch['s2']
        if not self._with_reward:
            _target = s2
        else:
            r = batch['reward']
            r = torch.reshape(r, (-1, 1))
            _target = torch.cat([s2, r], -1)
        losses = []
        for d_fn in self._d_fns:
            log_pi_s2 = d_fn.get_log_density(s1, a1, _target)
            loss = - log_pi_s2.mean()
            loss += 0.01 * d_fn.max_logstd.mean() - 0.01 * d_fn.min_logstd.mean()
            losses.append(loss)
        loss = 0
        for l in losses:
            loss = loss + l
        info = collections.OrderedDict()
        info['d_loss(-log_prob)'] = loss
        return loss, info

    def _build_test_loss(self, batch: Dict) -> Dict:
        s1 = batch['s1']
        a1 = batch['a1']
        s2 = batch['s2']
        if not self._with_reward:
            _target = s2
        else:
            r = batch['reward']
            r = torch.reshape(r, (-1, 1))
            _target = torch.cat([s2, r], -1)
        test_data_size = len(s1)
        test_mse = [[]] * len(self._d_fns)
        test_neg_log_prob = [[]] * len(self._d_fns)
        with torch.no_grad():
            for i in range(test_data_size // self._batch_size + (test_data_size % self._batch_size > 0)):
                _s1 = s1[i * self._batch_size:(i + 1) * self._batch_size]
                _a1 = a1[i * self._batch_size:(i + 1) * self._batch_size]
                _tar = _target[i * self._batch_size:(i + 1) * self._batch_size]
                for j, d_fn in enumerate(self._d_fns):
                    s_pred, _, s_dist = d_fn(_s1, _a1)
                    _loss_mse = ((s_pred - _tar) ** 2).mean()
                    _loss_neg_log_prob = - s_dist.log_prob(_tar).mean()
                    _loss_neg_log_prob += 0.01 * d_fn.max_logstd.mean() - 0.01 * d_fn.min_logstd.mean()
                    test_mse[j].append(_loss_mse)
                    test_neg_log_prob[j].append(_loss_neg_log_prob)
            test_mse = [torch.as_tensor(x).mean() for x in test_mse]
            test_mse = torch.sum(torch.as_tensor(test_mse))
            test_neg_log_prob = [torch.as_tensor(x).mean() for x in test_neg_log_prob]
            test_neg_log_prob = torch.sum(torch.as_tensor(test_neg_log_prob))
            info = collections.OrderedDict()
            info['test(mse)'] = test_mse
            info['test(-log_prob)'] = test_neg_log_prob
        return info

    def _optimize_step(self, batch: Dict) -> Dict:
        loss, info = self._build_loss(batch)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        return info

    def _get_modules(self) -> utils.Flags:
        model_params, n_d_fns = self._model_params

        def d_net_factory():
            return networks.ProbDynamicsNetwork(
                state_dim=self._state_dim,
                action_dim=self._action_dim,
                fc_layer_params=model_params,
                local_mode=self._local_mode,
                with_reward=self._with_reward,
                device=self._device,
            )

        modules = utils.Flags(
            n_d_fns=n_d_fns,
            d_net_factory=d_net_factory,
            device=self._device,
        )
        return modules

    def dynamics_fns(
            self,
            s: Union[np.ndarray, Tensor],
            a: Union[np.ndarray, Tensor]
    ) -> Tuple[List, Dict]:
        s = torch.as_tensor(s, device=self._device, dtype=torch.float32)
        a = torch.as_tensor(a, device=self._device, dtype=torch.float32)
        s_pred, s_dist = [], []
        with torch.no_grad():
            for d_fn in self._d_fns:
                _mean, _, _dist = d_fn(s, a)
                s_pred.append(_mean)
                s_dist.append(_dist)
        return s_pred, {'dist': s_dist}


class DynaModule(BaseDynaModule):

    def _build_modules(self) -> None:
        self._d_nets = nn.ModuleList()
        n_d_fns = self._net_modules.n_d_fns
        device = self._net_modules.device
        for _ in range(n_d_fns):
            self._d_nets.append(self._net_modules.d_net_factory().to(device))

    @property
    def d_nets(self) -> nn.ModuleList:
        return self._d_nets

