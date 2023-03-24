"""
Implementation of H2O (Dynamics-Aware Hybrid Offline-and-Online Reinforcement Learning)
Paper: https://arxiv.org/abs/2206.13464.pdf
"""
import collections
import torch
from tqdm import trange
import numpy as np
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

    :param int update_actor_freq: the update frequency of actor network.
    :param int rollout_sim_freq: the rollout frequency of simulation samples.
    :param int rollout_sim_num: number of simulation samples per rollout.
    :param bool automatic_entropy_tuning: whether to adopt automatic tuning of entropy coefficient (alpha) in entropy-regularized RL algorithms.
    :param float log_alpha_init_value: initialization value for log alpha.
    :param float log_alpha_prime_init_value: initialization value for log alpha prime.
    :param float target_entropy: target entropy value (from CQL).
    :param bool backup_entropy: whether to apply entropy backup (from CQL).
    :param float alpha_multiplier: alpha multiplier (from CQL).
    :param int sampling_n_next_states: the number of s' resampled from certain s,a pair when performing dynamics gap quantification.
    :param float s_prime_std_ratio: the multiplier on the standard deviation of s' when performing dynamics gap quantification.
    :param float noise_std_discriminator: the standard deviation of noise applied on discriminator training.
    :param bool cql_lagrange: whether to apply alpha prime (from CQL).
    :param float cql_target_action_gap: lagrange threshold (from CQL).
    :param float cql_temp: temperature coefficient of regularization term in solving the inner-loop maximization problem.
    :param int cql_clip_diff_min: min value of value regularizaion term (q_diff).
    :param int cql_clip_diff_max: max value of value regularizaion term (q_diff).
    :param float min_q_weight: multiplier on value regularization term (beta).
    
    :param bool use_td_target_ratio: whether to use dynamics ratio to fix bellman error.
    :param bool use_value_regularization: whether to use value regularization.
    :param bool use_adaptive_weighting: whether to use adaptive weight (omega).
    :param bool use_variant: whether to use H2O-variant.
    :param float clip_dynamics_ratio_min: min value of dynamics ratio.
    :param float clip_dynamics_ratio_max: max value of dynamics ratio.
    :param float adaptive_weighting_min: min value of adaptive weight (omega).
    :param float adaptive_weighting_max: max value of adaptive weight (omega).
    :param float joint_noise_std: the standard deviation of joint noise (to introduce dynamics gap).
    :param int max_traj_length: the maximum length of sampled trajectories.

    .. seealso::

        Please refer to :class:`~d2c.models.base.BaseAgent` for more detailed
        explanation.
    """

    def __init__(
            self,
            update_actor_freq: int = 1,
            rollout_sim_freq: int = 1000,
            rollout_sim_num: int = 1000,
            automatic_entropy_tuning: bool = True,
            log_alpha_init_value: float = 0.0,
            log_alpha_prime_init_value: float = 1.0,
            target_entropy: float = 0.0,
            backup_entropy: bool = False,
            alpha_multiplier: float = 1.0,
            sampling_n_next_states: int = 10,
            s_prime_std_ratio: float = 1.0,
            noise_std_discriminator: float = 0.1,
            cql_lagrange: bool = False,
            cql_target_action_gap: float = 1.0,
            cql_temp: float = 1.0,
            cql_clip_diff_min: int = -1000,
            cql_clip_diff_max: int = 1000,
            min_q_weight: float = 0.01,
            use_td_target_ratio: bool = True,
            use_value_regularization: bool = True,
            use_adaptive_weighting: bool = True,
            use_variant: bool = False,
            clip_dynamics_ratio_min: float = 1e-5,
            clip_dynamics_ratio_max: float = 1.0,
            adaptive_weighting_min: float = 1e-45,
            adaptive_weighting_max: float = 10,
            joint_noise_std: float = 0.0,
            max_traj_length: int = 1000,
            **kwargs: Any,
    ) -> None:
        self._update_actor_freq = update_actor_freq
        self._rollout_sim_freq = rollout_sim_freq
        self._rollout_sim_num = rollout_sim_num

        self._automatic_entropy_tuning = automatic_entropy_tuning
        self._log_alpha_init_value = log_alpha_init_value
        self._log_alpha_prime_init_value = log_alpha_prime_init_value
        self._target_entropy = target_entropy
        self._alpha_multiplier = alpha_multiplier
        self._backup_entropy = backup_entropy

        self._cql_target_action_gap = cql_target_action_gap
        self._cql_temp = cql_temp
        self._cql_lagrange = cql_lagrange
        self._cql_clip_diff_min = cql_clip_diff_min
        self._cql_clip_diff_max = cql_clip_diff_max
        self._min_q_weight = min_q_weight

        self._use_td_target_ratio = use_td_target_ratio
        self._use_value_regularization = use_value_regularization
        self._use_adaptive_weighting = use_adaptive_weighting
        self._use_variant = use_variant
        self._sampling_n_next_states = sampling_n_next_states
        self._s_prime_std_ratio = s_prime_std_ratio
        self._noise_std_discriminator = noise_std_discriminator
        self._clip_dynamics_ratio_min = clip_dynamics_ratio_min
        self._clip_dynamics_ratio_max = clip_dynamics_ratio_max
        self._adaptive_weighting_min = adaptive_weighting_min
        self._adaptive_weighting_max = adaptive_weighting_max
        self._joint_noise_std = joint_noise_std
        self._max_traj_length = max_traj_length
        self._p_info = collections.OrderedDict()
        super(H2OAgent, self).__init__(**kwargs)

        _state_data = self._train_data.data['s1']
        self.mean = _state_data.mean(0, keepdims=True)
        self.std = _state_data.std(0, keepdims=True)

    def _build_fns(self) -> None:
        self._agent_module = AgentModule(modules=self._modules)
        self._q_fns = self._agent_module.q_nets
        self._q_target_fns = self._agent_module.q_target_nets
        self._p_fn = self._agent_module.p_net
        self._p_target_fn = self._agent_module.p_target_net
        self._dsa_fn = self._agent_module.dsa_net
        self._dsas_fn = self._agent_module.dsas_net
        if self._automatic_entropy_tuning:
            self._log_alpha_fn = self._agent_module.log_alpha_net
        if self._cql_lagrange:
            self._log_alpha_prime_fn = self._agent_module.log_alpha_prime_net

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
        if self._automatic_entropy_tuning:
            self._alpha_optimizer = utils.get_optimizer(opts.alpha[0])(
                parameters=self._log_alpha_fn.parameters(),
                lr=opts.alpha[1],
                weight_decay=self._weight_decays,
            )
        if self._cql_lagrange:
            self._alpha_prime_optimizer = utils.get_optimizer(opts.alpha_prime[0])(
                parameters=self._log_alpha_prime_fn.parameters(),
                lr=opts.alpha_prime[1],
                weight_decay=self._weight_decays,
            )

    def _build_dsa_dsas_loss(self, batch: Tuple) -> Tuple[Tensor, Tensor, Dict]:
        real_batch, sim_batch = batch
        real_state = real_batch['s1']
        real_action = real_batch['a1']
        real_next_state = real_batch['s2']

        sim_state = sim_batch['s1']
        sim_action = sim_batch['a1']
        sim_next_state = sim_batch['s2']

        # input noise: prevents overfitting
        if self._noise_std_discriminator > 0:
            real_state += torch.randn(real_state.shape, device=self._device) * self._noise_std_discriminator
            real_action += torch.randn(real_action.shape, device=self._device) * self._noise_std_discriminator
            real_next_state += torch.randn(real_next_state.shape, device=self._device) * self._noise_std_discriminator
            sim_state += torch.randn(sim_state.shape, device=self._device) * self._noise_std_discriminator
            sim_action += torch.randn(sim_action.shape, device=self._device) * self._noise_std_discriminator
            sim_next_state += torch.randn(sim_next_state.shape, device=self._device) * self._noise_std_discriminator

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

    def _build_q_alpha_prime_loss(self, batch: Tuple[Dict, Dict]) -> Tuple:
        real_batch, sim_batch = batch
        real_state = real_batch['s1']
        real_action = real_batch['a1']
        real_r = real_batch['reward']
        real_next_state = real_batch['s2']
        real_dsc = real_batch['dsc']

        sim_state = sim_batch['s1']
        sim_action = sim_batch['a1']
        sim_r = sim_batch['reward']
        sim_next_state = sim_batch['s2']
        sim_dsc = sim_batch['dsc']

        real_qf1_pred = self._q_fns[0](real_state, real_action)
        real_qf2_pred = self._q_fns[1](real_state, real_action)
        sim_qf1_pred = self._q_fns[0](sim_state, sim_action)
        sim_qf2_pred = self._q_fns[1](sim_state, sim_action)

        _, real_new_next_action, real_next_log_pi = self._p_fn(real_next_state)
        real_target_q_values = torch.min(
            self._q_target_fns[0](real_next_state, real_new_next_action),
            self._q_target_fns[1](real_next_state, real_new_next_action),
        )

        _, sim_new_next_action, sim_next_log_pi = self._p_fn(sim_next_state)
        sim_target_q_values = torch.min(
            self._q_target_fns[0](sim_next_state, sim_new_next_action),
            self._q_target_fns[1](sim_next_state, sim_new_next_action),
        )

        if self._backup_entropy:
            real_target_q_values = real_target_q_values - self.alpha * real_next_log_pi
            sim_target_q_values = sim_target_q_values - self.alpha * sim_next_log_pi

        real_td_target = real_r + real_dsc * self._discount * real_target_q_values
        sim_td_target = sim_r + sim_dsc * self._discount * sim_target_q_values

        real_qf1_loss = F.mse_loss(real_qf1_pred, real_td_target.detach())
        real_qf2_loss = F.mse_loss(real_qf2_pred, real_td_target.detach())

        if self._use_td_target_ratio:
            sqrt_IS_ratio = torch.clamp(self.real_sim_dynacmis_ratio(sim_state, sim_action, sim_next_state),
                                        self._clip_dynamics_ratio_min, self._clip_dynamics_ratio_max).sqrt()
        else:
            sqrt_IS_ratio = torch.ones((sim_state.shape[0],)).to(self._device)

        sim_qf1_loss = F.mse_loss(sqrt_IS_ratio * sim_qf1_pred, sqrt_IS_ratio * sim_td_target.detach())
        sim_qf2_loss = F.mse_loss(sqrt_IS_ratio * sim_qf2_pred, sqrt_IS_ratio * sim_td_target.detach())
        qf1_loss = real_qf1_loss + sim_qf1_loss
        qf2_loss = real_qf2_loss + sim_qf2_loss

        if not self._use_value_regularization:
            q_loss = qf1_loss + qf2_loss
        else:
            if self._use_adaptive_weighting:
                u_sa = self.kl_sim_divergence(sim_state, sim_action, sim_next_state)
            else:
                u_sa = torch.ones(sim_r.shape[0], device=self._device)

            omega = u_sa / u_sa.sum()
            std_omega = omega.std()

            if self._use_variant:
                sim_qf1_gap = (omega * sim_qf1_pred).sum()
                sim_qf2_gap = (omega * sim_qf2_pred).sum()
            else:
                sim_qf1_pred += torch.log(omega)
                sim_qf2_pred += torch.log(omega)
                sim_qf1_gap = torch.logsumexp(sim_qf1_pred / self._cql_temp, dim=0) * self._cql_temp
                sim_qf2_gap = torch.logsumexp(sim_qf2_pred / self._cql_temp, dim=0) * self._cql_temp

            qf1_diff = torch.clamp(
                sim_qf1_gap - real_qf1_pred.mean(),
                self._cql_clip_diff_min,
                self._cql_clip_diff_max,
            )
            qf2_diff = torch.clamp(
                sim_qf2_gap - real_qf2_pred.mean(),
                self._cql_clip_diff_min,
                self._cql_clip_diff_max,
            )

            if self._cql_lagrange:
                alpha_prime = torch.clamp(torch.exp(self._log_alpha_prime_fn()), min=0.0, max=1000000.0)
                min_qf1_loss = alpha_prime * self._min_q_weight * (qf1_diff - self._cql_target_action_gap)
                min_qf2_loss = alpha_prime * self._min_q_weight * (qf2_diff - self._cql_target_action_gap)

                self._alpha_prime_optimizer.zero_grad()
                alpha_prime_loss = (- min_qf1_loss - min_qf2_loss) * 0.5
                alpha_prime_loss.backward(retain_graph=True)

                self._alpha_prime_optimizer.step()
            else:
                min_qf1_loss = qf1_diff * self._min_q_weight
                min_qf2_loss = qf2_diff * self._min_q_weight
                df_state = torch.cat([real_state, sim_state], dim=0)
                alpha_prime_loss = df_state.new_tensor(0.0)
                alpha_prime = df_state.new_tensor(0.0)

            q_loss = qf1_loss + qf2_loss + min_qf1_loss + min_qf2_loss

        info = collections.OrderedDict()
        info['real_Q1'] = real_qf1_pred.detach().mean()
        info['real_Q2'] = real_qf2_pred.detach().mean()
        info['sim_Q1'] = sim_qf1_pred.detach().mean()
        info['sim_Q2'] = sim_qf2_pred.detach().mean()
        info['real_Q_target'] = real_target_q_values.mean()
        info['sim_Q_target'] = sim_target_q_values.mean()

        info['real_Q1_loss'] = real_qf1_loss.detach().mean()
        info['real_Q2_loss'] = real_qf2_loss.detach().mean()
        info['sim_Q1_loss'] = sim_qf1_loss.detach().mean()
        info['sim_Q2_loss'] = sim_qf2_loss.detach().mean()
        info['Q1_loss'] = qf1_loss.detach().mean()
        info['Q2_loss'] = qf2_loss.detach().mean()
        info['Q_loss'] = q_loss

        info['u_sa'] = u_sa.detach().mean()
        info['std_omega'] = std_omega.detach().mean()
        info['min_qf1_loss'] = min_qf1_loss.detach().mean()
        info['min_qf2_loss'] = min_qf2_loss.detach().mean()
        info['qf1_diff'] = qf1_diff.detach().mean()
        info['qf2_diff'] = qf2_diff.detach().mean()
        info['sim_qf1_gap'] = sim_qf1_gap.detach().mean()
        info['sim_qf2_gap'] = sim_qf2_gap.detach().mean()
        info['alpha_prime_loss'] = alpha_prime_loss.detach().mean()
        info['alpha_prime'] = alpha_prime
        if self._cql_lagrange:
            info['alpha_prime_loss'] = alpha_prime_loss
            return q_loss, alpha_prime_loss, info
        else:
            return q_loss, 0, info

    def _build_p_alpha_loss(self, batch: Tuple[Dict, Dict], bc: bool = False) -> Tuple:
        real_batch, sim_batch = batch
        real_state = real_batch['s1']
        real_action = real_batch['a1']

        sim_state = sim_batch['s1']
        sim_action = sim_batch['a1']

        df_state = torch.cat([real_state, sim_state], dim=0)
        df_action = torch.cat([real_action, sim_action], dim=0)

        _, df_new_action, df_log_pi = self._p_fn(df_state)

        if self._automatic_entropy_tuning:
            alpha_loss = -(self._log_alpha_fn() * (df_log_pi + self._target_entropy).detach()).mean()
            self.alpha = self._log_alpha_fn().exp() * self._alpha_multiplier
        else:
            alpha_loss = df_state.new_tensor(0.0)
            self.alpha = df_state.new_tensor(self._alpha_multiplier)

        if bc:
            log_prob = self._p_fn.get_log_density(df_state, df_action)
            p_loss = (self.alpha * df_log_pi - log_prob).mean()
        else:
            q_new_action = torch.min(
                self._q_fns[0](df_state, df_new_action),
                self._q_fns[1](df_state, df_new_action),
            )
            sum_log_pi = df_log_pi.sum(dim=-1)
            p_loss = (self.alpha * sum_log_pi - q_new_action).mean()

        info = collections.OrderedDict()
        info['actor_loss'] = p_loss
        info['log_pi'] = sum_log_pi.mean()
        info['alpha'] = self.alpha
        if self._automatic_entropy_tuning:
            info['alpha_loss'] = alpha_loss
            return p_loss, alpha_loss, info
        else:
            return p_loss, 0, info

    def _optimize_p_alpha(self, batch: Tuple[Dict, Dict]) -> Dict:
        p_loss, alpha_loss, info = self._build_p_alpha_loss(batch)

        self._p_optimizer.zero_grad()
        p_loss.backward()
        self._p_optimizer.step()

        if self._automatic_entropy_tuning:
            self._alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self._alpha_optimizer.step()

        return info

    def _optimize_q_alpha_prime(self, batch: Tuple[Dict, Dict]) -> Dict:
        q_loss, alpha_prime_loss, info = self._build_q_alpha_prime_loss(batch)

        self._q_optimizer.zero_grad()
        q_loss.backward()
        self._q_optimizer.step()

        if self._cql_lagrange:
            self._alpha_prime_optimizer.zero_grad()
            alpha_prime_loss.backward(retain_graph=True)
            self._alpha_prime_optimizer.step()

        return info

    def _optimize_dsa_dsas(self, batch: Tuple[Dict, Dict]) -> Dict:
        dsa_loss, dsas_loss, info = self._build_dsa_dsas_loss(batch)

        self._dsa_optimizer.zero_grad()
        dsa_loss.backward(retain_graph=True)

        self._dsas_optimizer.zero_grad()
        dsas_loss.backward()

        self._dsa_optimizer.step()
        self._dsas_optimizer.step()
        return info

    def _get_train_batch(self) -> Tuple[Dict, Dict]:
        """Sample two batches of transitions from real dataset and sim replay buffer respectively."""
        # periodically rollout transitions from sim env
        if self._global_step % self._rollout_sim_freq == 0:
            with torch.no_grad():
                self._traj_steps = 0
                self._current_state = self._env.reset()
                for _ in trange(self._rollout_sim_num):
                    self._traj_steps += 1
                    state = self._current_state
                    _, action, _ = self._p_fn(state)
                    action = action.cpu().numpy()
                    if self._joint_noise_std > 0:
                        next_state, reward, done, __ = self._env.step(
                            action + np.random.randn(action.shape[0], ) * self._joint_noise_std)
                    else:
                        next_state, reward, done, __ = self._env.step(action)

                    self._empty_dataset.add(state=state, action=action, next_state=next_state, next_action=0,
                                            reward=reward, done=done)
                    self._current_state = next_state

                    if done or self._traj_steps >= self._max_traj_length:
                        self._traj_steps = 0
                        self._current_state = self._env.reset()

        _real_batch = self._train_data.sample_batch(self._batch_size)
        _sim_batch = self._empty_dataset.sample_batch(self._batch_size)

        return _real_batch, _sim_batch

    def _optimize_step(self, batch: Tuple[Dict, Dict]) -> Dict:
        info = collections.OrderedDict()
        # dis_real_batch = self._train_data.sample_batch(self._batch_size)
        # dis_sim_batch = self._empty_dataset.sample_batch(self._batch_size
        q_info = self._optimize_q_alpha_prime(batch)
        d_info = self._optimize_dsa_dsas(batch)
        if self._global_step % self._update_actor_freq == 0:
            self._p_info = self._optimize_p_alpha(batch)
            # Update the target networks.
            self._update_target_fns(self._q_fns, self._q_target_fns)
            self._update_target_fns(self._p_fn, self._p_target_fn)
        info.update(q_info)
        info.update(d_info)
        info.update(self._p_info)
        return info

    def _build_test_policies(self) -> None:
        policy = policies.DeterministicSoftPolicy(
            a_network=self._p_fn
        )
        self._test_policies['main'] = policy

    def real_sim_dynacmis_ratio(self, states: Tensor, actions: Tensor, next_states: Tensor) -> Tensor:
        sa_logits = self._dsa_fn(states, actions)
        sa_prob = F.softmax(sa_logits, dim=1)
        adv_logits = self._dsas_fn(states, actions, next_states)
        sas_prob = F.softmax(adv_logits + sa_logits, dim=1)

        with torch.no_grad():
            ratio = (sas_prob[:, 0] * sa_prob[:, 1]) / (sas_prob[:, 1] * sa_prob[:, 0])

        return ratio

    def log_sim_real_dynacmis_ratio(self, states: Tensor, actions: Tensor, next_states: Tensor) -> Tensor:
        sa_logits = self._dsa_fn(states, actions)
        sa_prob = F.softmax(sa_logits, dim=1)
        adv_logits = self._dsas_fn(states, actions, next_states)
        sas_prob = F.softmax(adv_logits + sa_logits, dim=1)

        with torch.no_grad():
            # clipped pM^/pM
            log_ratio = - torch.log(sas_prob[:, 0]) \
                        + torch.log(sas_prob[:, 1]) \
                        + torch.log(sa_prob[:, 0]) \
                        - torch.log(sa_prob[:, 1])

        return log_ratio

    def kl_sim_divergence(self, states: Tensor, actions: Tensor, next_states: Tensor) -> Tensor:
        states = torch.repeat_interleave(states, self._sampling_n_next_states, dim=0)
        actions = torch.repeat_interleave(actions, self._sampling_n_next_states, dim=0)
        next_states = torch.repeat_interleave(next_states, self._sampling_n_next_states, dim=0)
        # TODO: data std
        next_states += torch.randn(next_states.shape, device=self._device) * self.std * self._s_prime_std_ratio
        log_ratio = self.log_sim_real_dynacmis_ratio(states, actions, next_states).reshape(
            (-1, self._sampling_n_next_states))

        return torch.clamp(log_ratio.mean(dim=1), self._adaptive_weighting_min, self._adaptive_weighting_max)

    def save(self, ckpt_name: str) -> None:
        torch.save(self._agent_module.state_dict(), ckpt_name + '.pth')
        torch.save(self._agent_module.q_nets.state_dict(), ckpt_name + '_q.pth')
        torch.save(self._agent_module.p_net.state_dict(), ckpt_name + '_policy.pth')
        # torch.save(self._agent_module.dsa_net.state_dict(), ckpt_name + '_dsa.pth')
        # torch.save(self._agent_module.dsas_net.state_dict(), ckpt_name + '_dsa.pth')

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
            return networks.ConcatClassifier(
                input_dim=self._observation_space.shape[0] + self._action_space.shape[0],
                output_dim=2,
                fc_layer_params=model_params_dsa,
                device=self._device,
            )

        def dsas_net_factory():
            return networks.ConcatClassifier(
                input_dim=2 * self._observation_space.shape[0] + self._action_space.shape[0],
                output_dim=2,
                fc_layer_params=model_params_dsas,
                device=self._device,
            )

        def log_alpha_net_factory():
            return networks.Scalar(
                init_value=self._log_alpha_init_value,
                device=self._device
            )

        def log_alpha_prime_net_factory():
            return networks.Scalar(
                init_value=self._log_alpha_prime_init_value,
                device=self._device
            )

        modules = utils.Flags(
            q_net_factory=q_net_factory,
            p_net_factory=p_net_factory,
            dsa_net_factory=dsa_net_factory,
            dsas_net_factory=dsas_net_factory,
            n_q_fns=n_q_fns,
            log_alpha_net_factory=log_alpha_net_factory,
            log_alpha_prime_net_factory=log_alpha_prime_net_factory,
            device=self._device,
            automatic_entropy_tuning=self._automatic_entropy_tuning,
            cql_lagrange=self._cql_lagrange
        )

        return modules


class AgentModule(BaseAgentModule):

    def _build_modules(self) -> None:
        device = self._net_modules.device
        automatic_entropy_tuning = self._net_modules.automatic_entropy_tuning
        cql_lagrange = self._net_modules.cql_lagrange
        self._q_nets = nn.ModuleList()
        n_q_fns = self._net_modules.n_q_fns  # The number of the Q nets.
        for _ in range(n_q_fns):
            self._q_nets.append(self._net_modules.q_net_factory().to(device))
        self._q_target_nets = copy.deepcopy(self._q_nets)
        self._p_net = self._net_modules.p_net_factory().to(device)
        self._p_target_net = copy.deepcopy(self._p_net)
        self._dsa_net = self._net_modules.dsa_net_factory().to(device)
        self._dsas_net = self._net_modules.dsas_net_factory().to(device)
        if automatic_entropy_tuning:
            self._log_alpha_net = self._net_modules.log_alpha_net_factory().to(device)
        if cql_lagrange:
            self._log_alpha_prime_net = self._net_modules.log_alpha_prime_net_factory().to(device)

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

    @property
    def log_alpha_net(self) -> nn.Module:
        return self._log_alpha_net

    @property
    def log_alpha_prime_net(self) -> nn.Module:
        return self._log_alpha_prime_net
