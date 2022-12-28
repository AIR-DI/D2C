"""
Behavior cloning via maximum likelihood.
"""
import collections
import torch
import numpy as np
from torch import nn, Tensor
from typing import Tuple, Any, Optional, Dict
from d2c.models.base import BaseAgent, BaseAgentModule
from d2c.utils import networks, utils, policies


class BCAgent(BaseAgent):

    """Implementation of Behavior cloning via maximum likelihood.

    :param float test_data_ratio: The ratio of the test data in the training data.
    :param float test_freq: The frequency of validation.
    """

    def __init__(
            self,
            test_data_ratio: float = 0.0,
            test_freq: Optional[int] = None,
            **kwargs: Any,
    ) -> None:
        self._test_data_ratio = test_data_ratio
        self._test_freq = test_freq
        super(BCAgent, self).__init__(**kwargs)

    def _build_fns(self) -> None:
        self._agent_module = AgentModule(modules=self._modules)
        self._p_fn = self._agent_module.p_net

    def _build_optimizers(self) -> None:
        opts = self._optimizers
        self._p_optimizer = utils.get_optimizer(opts.p[0])(
            parameters=self._p_fn.parameters(),
            lr=opts.p[1],
            weight_decay=self._weight_decays,
        )

    def _build_p_loss(self, batch: Dict) -> Tuple[Tensor, Dict]:
        s1 = batch['s1']
        a1 = batch['a1']
        log_prob = self._p_fn.get_log_density(s1, a1)
        loss = - log_prob.mean()
        info = collections.OrderedDict()
        info['p_loss'] = loss
        return loss, info

    def _optimize_p(self, batch: Dict) -> Dict:
        loss, info = self._build_p_loss(batch)
        self._p_optimizer.zero_grad()
        loss.backward()
        self._p_optimizer.step()
        return info

    def _optimize_step(self, batch: Dict) -> Dict:
        p_info = self._optimize_p(batch)
        if self._test_freq:
            if self._global_step % self._test_freq == 0:
                test_info = self._test_step()
                for k, v in test_info.items():
                    self._train_info[k] = v
        return p_info

    def _get_train_batch(self) -> Dict:
        shuffle_indices = self._train_data.shuffle_indices
        train_size = int(self._train_data.size * (1 - self._test_data_ratio))
        train_indices = shuffle_indices[:train_size]
        batch_indices = np.random.choice(train_indices, self._batch_size)
        return self._train_data.get_batch_indices(batch_indices)

    def _get_test_batch(self) -> Dict:
        shuffle_indices = self._train_data.shuffle_indices
        train_size = int(self._train_data.size * (1 - self._test_data_ratio))
        test_indices = shuffle_indices[train_size:]
        return self._train_data.get_batch_indices(test_indices)

    def _test_step(self) -> Dict:
        test_batch = self._get_test_batch()
        s1 = test_batch['s1']
        a1 = test_batch['a1']
        test_data_size = len(s1)
        test_loss = []
        test_log_prob = []
        with torch.no_grad():
            for i in range(test_data_size // self._batch_size + (test_data_size % self._batch_size > 0)):
                _s1 = s1[i*self._batch_size:(i+1)*self._batch_size]
                _a1 = a1[i*self._batch_size:(i+1)*self._batch_size]
                a_pred, _, _ = self._p_fn(_s1)
                _loss = ((a_pred - _a1) ** 2).mean().item()
                test_loss.append(_loss)
                log_prob = self._p_fn.get_log_density(_s1, _a1)
                test_log_prob.append(log_prob.mean().item())
        test_loss = np.mean(test_loss)
        test_log_prob = np.mean(test_log_prob)
        info = collections.OrderedDict()
        info['test_loss(mse)'] = test_loss
        info['test_loss(-log_prob)'] = - test_log_prob
        return info

    def _build_test_policies(self) -> None:
        policy = policies.DeterministicSoftPolicy(
            a_network=self._p_fn,
        )
        self._test_policies['main'] = policy

    def save(self, ckpt_name: str) -> None:
        torch.save(self._agent_module.state_dict(), ckpt_name + '.pth')

    def restore(self, ckpt_name: str) -> None:
        self._agent_module.load_state_dict(torch.load(ckpt_name + '.pth'))

    def _get_modules(self) -> utils.Flags:
        model_params_p = self._model_params.p[0]

        def p_net_factory():
            return networks.ActorNetwork(
                observation_space=self._observation_space,
                action_space=self._action_space,
                fc_layer_params=model_params_p,
                device=self._device,
            )

        modules = utils.Flags(
            p_net_factory=p_net_factory,
            device=self._device,
        )
        return modules


class AgentModule(BaseAgentModule):

    def _build_modules(self) -> None:
        device = self._net_modules.device
        self._p_net = self._net_modules.p_net_factory().to(device)

    @property
    def p_net(self) -> nn.Module:
        return self._p_net
