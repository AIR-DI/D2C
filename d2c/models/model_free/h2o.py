"""
Implementation of H2O (Dynamics-Aware Hybrid Offline-and-Online Reinforcement Learning)
Paper: https://arxiv.org/abs/2206.13464.pdf
"""
import collections
import torch
import copy
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Union, Tuple, Any, Sequence, Dict, Iterator
from d2c.models.base import BaseAgent, BaseAgentModule
from d2c.utils import networks, utils, policies

LAMBDA_MIN = 1
LAMBDA_MAX = 100

class H2OAgent(BaseAgent):
    """Implementation of H2O

    :param float policy_noise: the noise used in updating policy network.
    :param int update_actor_freq: the update frequency of actor network.
    :param float noise_clip: the clipping range used in updating policy network.
    :param float alpha: the value of alpha, which controls the weight for TD3 learning relative to behavior cloning.
    :param int N: the number of noise samples to train distance function
    :param float initial_lambda: the value of initial Lagrangian multiplier
    :param float lambda_lr: the update step size of Lagrangian multiplier
    :param float train_d_steps: the total training steps to train distance function
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
            N: int = 20,
            initial_lambda: float = 5,
            lambda_lr: float = 3e-4,
            train_d_steps: int = int(1e+5),
            automatic_entropy_tuning: bool = True,
            **kwargs: Any,
    ) -> None:
        self._policy_noise = policy_noise
        self._update_actor_freq = update_actor_freq
        self._noise_clip = noise_clip
        self._alpha = alpha
        self._N = N
        self._initial_lambda = initial_lambda
        self._train_d_steps = train_d_steps
        self._lambda_lr = lambda_lr
        self._automatic_entropy_tuning = automatic_entropy_tuning
        self._p_info = collections.OrderedDict()
        super(H2OAgent, self).__init__(**kwargs)

    def _build_fns(self) -> None:
        self._agent_module = AgentModule(modules=self._modules)
        self._q_fns = self._agent_module.q_nets
        self._q_target_fns = self._agent_module.q_target_nets
        self._p_fn = self._agent_module.p_net
        self._p_target_fn = self._agent_module.p_target_net
        self._dsa_fn = self._agent_module.dsa_net
        self._dsas_fn = self._agent_module.dsas_net

    def _init_vars(self) -> None:
        self._auto_lmbda = torch.tensor(self._initial_lambda, dtype=torch.float32, device=self._device)

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
        self._dsa_optimizer = utils.get_optimizer(opts.dsa[0])(
            parameters=self._dsa_fn.parameters(),
            lr=opts.dsa[1],
            weight_decay=self._weight_decays,
        )
        self._dsas_optimizer = utils.get_optimizer(opts.dsas[0])(
            parameters=self._dsas_fn.parameters(),
            lr=opts.dsas[1],
            weight_decay=self._weight_decays,
        )

    def _build_dsa_loss(self, real_batch: Dict, sim_batch: Dict) -> Tuple[Tensor, Dict]:
        real_state = real_batch['s1']
        real_action = real_batch['a1']
        real_next_state = real_batch['s2']
        
        sim_state = sim_batch['s1']
        sim_action = sim_batch['a1']
        sim_next_state = sim_batch['s2']
        
        real_sa_logits = self._dsa_fn(real_state, real_action)
        real_sa_prob = F.softmax(real_sa_logits, dim=1)
        sim_sa_logits = self._dsa_fn(sim_state, sim_action)
        sim_sa_prob = F.softmax(sim_sa_logits, dim=1)
        
        real_adv_logits = self._dsas_fn(real_state, real_action, real_next_state)
        real_sas_prob = F.softmax(real_adv_logits + real_sa_logits, dim=1)
        sim_adv_logits = self._dsas_fn(sim_state, sim_action, sim_next_state)
        sim_sas_prob = F.softmax(sim_adv_logits + sim_sa_logits, dim=1)
        
        dsa_loss = (- torch.log(real_sa_prob[:, 0]) - torch.log(sim_sa_prob[:, 1])).mean()
        dsas_loss = (- torch.log(real_sas_prob[:, 0]) - torch.log(sim_sas_prob[:, 1])).mean()

        info = collections.OrderedDict()
        info['dsa_loss'] = dsa_loss
        info['dsas_loss'] = dsas_loss
        return dsa_loss, dsas_loss, info

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
    
    def _build_p_loss(self, real_batch: Dict, sim_batch: Dict, bc=False) -> Tuple[Tensor, Dict]:
        real_state = real_batch['s1']
        real_action = real_batch['a1']
        
        sim_state = sim_batch['s1']
        sim_action = sim_batch['a1']
        
        df_state = torch.cat([real_state, sim_state], dim=0)
        df_action = torch.cat([real_action, sim_action], dim=0)
        
        _, df_new_action, df_log_pi = self._p_fn(df_state)
        
        if self._automatic_entropy_tuning:
            # TODO
            alpha_loss = -(self.log_alpha() * (df_log_pi + self.config.target_entropy).detach()).mean()
            alpha = self.log_alpha().exp() * self.config.alpha_multiplier
        else:
            alpha_loss = df_state.new_tensor(0.0)
            alpha = df_state.new_tensor(self.config.alpha_multiplier)
            
        if bc:
            log_probs = self._p_fn.get_log_density(df_state, df_action)
            p_loss = (alpha * df_log_pi - log_probs).mean()
        else:
            q_new_action = torch.min(
                self._q_fns[0](df_state, df_new_action),
                self._q_fns[1](df_state, df_new_action),
            )
            p_loss = (alpha * df_log_pi - q_new_action).mean() 

        info = collections.OrderedDict()
        info['actor_loss'] = p_loss
        info['alpha_loss'] = alpha_loss

        return p_loss, info

    def _optimize_q(self, batch: Dict) -> Dict:
        loss, info = self._build_q_loss(batch)
        self._q_optimizer.zero_grad()
        loss.backward()
        self._q_optimizer.step()
        return info

    def _optimize_dsa(self, batch: Dict):
        dsa_loss, info = self._build_dsa_loss(batch)
        self._dsa_optimizer.zero_grad()
        dsa_loss.backward()
        self._dsa_optimizer.step()
        return info
    
    def _optimize_dsas(self, batch: Dict):
        dsas_loss, info = self._build_dsas_loss(batch)
        self._dsas_optimizer.zero_grad()
        dsas_loss.backward()
        self._dsas_optimizer.step()
        return info

    def _optimize_step(self, batch: Dict) -> Dict:
        info = collections.OrderedDict()
        q_info = self._optimize_q(batch)
        if self._global_step <= self._train_d_steps:
            distance_info = self._optimize_distance(batch)
            info.update(distance_info)
        if self._global_step % self._update_actor_freq == 0:
            # self._p_info = self._optimize_p_alpha(batch)
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
        torch.save(self._agent_module.dis_net.state_dict(), ckpt_name + '_distance.pth')

    def restore(self, ckpt_name: str) -> None:
        self._agent_module.load_state_dict(torch.load(ckpt_name + '.pth'))

    def _get_modules(self) -> utils.Flags:
        model_params_q, n_q_fns = self._model_params.q
        model_params_p = self._model_params.p[0]
        model_params_dsa = self._model_params.dsa[0]
        model_params_dsas = self._model_params.dsas[0]

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

        def dsa_net_factory():
            return networks.ConcatDiscriminator(
                input_dim=self._observation_space + self._action_space,
                output_dim=2,
                fc_layer_params=model_params_dsa,
                device=self._device,
            )
            
        def dsas_net_factory():
            return networks.ConcatDiscriminator(
                input_dim=2 * self._observation_space + self._action_space,
                output_dim=2,
                fc_layer_params=model_params_dsas,
                device=self._device,
            )

        modules = utils.Flags(
            q_net_factory=q_net_factory,
            p_net_factory=p_net_factory,
            dsa_net_factory=dsa_net_factory,
            dsas_net_factory=dsas_net_factory,
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
        self._dsa_net = self._net_modules.dsa_net_factory().to(device)
        self._dsas_net = self._net_modules.dsas_net_factory().to(device)

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

    @property
    def dsa_net(self) -> nn.Module:
        return self._dsa_net
    
    @property
    def dsas_net(self) -> nn.Module:
        return self._dsas_net




