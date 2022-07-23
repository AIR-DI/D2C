"""
Implementation of TD3+BC (A Minimalist Approach to Offline Reinforcement Learning)
Paper: https://arxiv.org/pdf/2106.06860.pdf
"""

import torch
from torch import nn, Tensor
from typing import Union, Tuple, Any, Sequence, Dict, Iterator
from d2c.models.base import BaseAgent, BaseAgentModule
from d2c.utils import networks, utils


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
        super(TD3BCAgent, self).__init__(**kwargs)

    def _build_fns(self) -> None:
        self._agent_module = AgentModule(modules=self._modules)
        self._q_fns = self._agent_module.q_nets
        self._p_fn = self._agent_module.p_net

    def _init_vars(self) -> None:
        self._q_vars = self._agent_module.q_source_variables
        self._p_vars = self._agent_module.p_variables

    def _get_source_target_vars(self) -> Tuple[Sequence[Tensor], Sequence[Tensor]]:
        return (self._agent_module.q_source_variables,
                self._agent_module.q_target_variables)

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

    def _build_q_loss(self, batch: Dict):
        s1 = batch['s1']
        s2 = batch['s2']
        a1 = batch['a1']
        r = batch['reward']
        dsc = batch['dsc']



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
        )
        return modules


class AgentModule(BaseAgentModule):

    def _build_modules(self) -> None:
        self._q_nets = []
        n_q_fns = self._modules.n_q_fns
        for _ in range(n_q_fns):
            self._q_nets.append(
                [self._modules.q_net_factory(),
                 self._modules.q_net_factory()]  # source and target
            )
        self._p_net = self._modules.p_net_factory()

    @property
    def q_nets(self) -> Sequence[Sequence[nn.Module]]:
        return self._q_nets

    @property
    def q_source_variables(self) -> Tuple:
        """The parameters of all the source Q networks."""
        vars_ = []
        for q_net, _ in self._q_nets:
            vars_ += list(q_net.parameters())
        return tuple(vars_)

    @property
    def q_target_variables(self) -> Tuple:
        """The parameters of all the target Q networks."""
        vars_ = []
        for _, q_net in self._q_nets:
            vars_ += list(q_net.parameters())
        return tuple(vars_)

    @property
    def p_net(self) -> nn.Module:
        return self._p_net

    @property
    def p_variables(self) -> Iterator:
        return self._p_net.parameters()




