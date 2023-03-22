import os
import torch
import collections
import logging
import copy
import numpy as np
import torch.nn.functional as F
from datetime import datetime
from torch import nn, Tensor
from torch.utils.tensorboard import SummaryWriter
from typing import Union, Optional, Dict, Any, List, Tuple, ClassVar
from d2c.models.base import BaseAgent
from d2c.evaluators.base import BaseEval
from d2c.utils.logger import WandbLogger
from d2c.utils.replaybuffer import ReplayBuffer
from d2c.utils import utils, logger, networks


class FQE:
    """Fitted Q Evaluation.

    FQE is an off-policy evaluation method that approximates a Q function
    :math:`Q_\theta (s, a)` with the trained policy :math:`\pi_\phi(s)`.

    References:
        * `Le et al., Batch Policy Learning under Constraints.
          <https://arxiv.org/abs/1903.08738>`_

    :param nn.Module policy: the policy to be evaluated.
    :param int state_dim: dimension of the state.
    :param int action_dim: the dimension of the action.
    :param ReplayBuffer train_data: the dataset of the batch data.
    :param List model_params: the parameters for constructing the critic network. It can be a list like \
        `[[256, 256], 2]``. ``[256, 256]`` means a two-layer FC network with 256 units in each layer and \
        the number ``2`` means the number of the Q nets for ensemble.
    :param List optimizers: the parameters for create the optimizers. It can be a dict like \
        ``['adam', 1e-4]``. It contains the type of the optimizer and the learning rate for \
        the critic network model.
    :param int batch_size: the size of data batch for training.
    :param float weight_decays: L2 regularization coefficient of the networks.
    :param int update_freq: the frequency of update the parameters of the target network.
    :param float update_rate: the rate of update the parameters of the target network.
    :param float discount: the discount factor for computing the cumulative reward.
    :param device: which device to create this model on. Default to None.
    """

    def __init__(
            self,
            policy: nn.Module,
            state_dim: int,
            action_dim: int,
            train_data: ReplayBuffer,
            model_params: Union[List, Tuple] = ([1024, 1024, 1024, 1024], 1),
            optimizers: Union[List, Tuple] = ('adam', 1e-4),
            batch_size: int = 256,
            weight_decays: float = 0.0,
            update_freq: int = 100,
            update_rate: float = 1,
            discount: float = 0.99,
            device: Optional[Union[str, int, torch.device]] = None,
    ) -> None:
        self._policy = policy
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._train_data = train_data
        self._model_params = model_params
        self._optimizers = optimizers
        self._batch_size = batch_size
        self._weight_decays = weight_decays
        self._update_freq = update_freq
        self._update_rate = update_rate
        self._discount = discount
        self._device = device
        self._modules = self._get_modules()
        self._build_critic()
        self._build_optimizers()
        self._global_step = 0
        self._train_info = collections.OrderedDict()

    def _build_critic(self) -> None:
        self._critic_module = CriticModule(self._modules)
        self._q_fns = self._critic_module.q_nets
        self._q_target_fns = self._critic_module.q_target_nets

    def _build_optimizers(self) -> None:
        opts = self._optimizers
        self._q_optimizer = utils.get_optimizer(opts[0])(
            parameters=self._q_fns.parameters(),
            lr=opts[1],
            weight_decay=self._weight_decays,
        )

    def _build_loss(self, batch: Dict) -> Tuple[Tensor, Dict]:
        s1 = batch['s1']
        s2 = batch['s2']
        a1 = batch['a1']
        r = batch['reward']
        dsc = batch['dsc']
        with torch.no_grad():
            next_action = self._policy(s2)
            # Compute the target Q value
            value_list = []
            for q_tar_fn in self._q_target_fns:
                target = q_tar_fn(s2, next_action)
                value_list.append(torch.unsqueeze(target, 0))
            values = torch.cat(value_list, dim=0)
            target_q = torch.min(values, dim=0).values
            target_q = r + dsc * self._discount * target_q
        td_sum = torch.tensor(
            0.0,
            dtype=torch.float32,
            device=self._device,
        )
        cur_q_list = []
        for q_fn in self._q_fns:
            current_q = q_fn(s1, a1)
            cur_q_list.append(current_q.detach().mean())
            loss = F.mse_loss(current_q, target_q)
            td_sum = td_sum + loss
        info = collections.OrderedDict()
        info['Q_mean'] = torch.mean(torch.as_tensor(cur_q_list))
        info['Q_target'] = target_q.mean()
        info['Q_loss'] = td_sum
        return td_sum, info

    def _optimize(self, batch: Dict) -> Dict:
        loss, info = self._build_loss(batch)
        self._q_optimizer.zero_grad()
        loss.backward()
        self._q_optimizer.step()
        return info

    def _optimize_step(self, batch: Dict) -> Dict:
        info = self._optimize(batch)
        if self._global_step % self._update_freq == 0:
            # Update the target networks.
            self._update_target_fns(self._q_fns, self._q_target_fns)
        return info

    def train_step(self) -> None:
        """Train the agent for one step."""
        train_batch = self._get_train_batch()
        info = self._optimize_step(train_batch)
        for key, val in info.items():
            self._train_info[key] = val.item()
        self._global_step += 1

    def _update_target_fns(
            self,
            source_module: nn.Module,
            target_module: nn.Module,
    ) -> None:
        tau = self._update_rate
        for tar, sou in zip(target_module.parameters(), source_module.parameters()):
            tar.data.copy_(sou.data * tau + tar.data * (1.0 - tau))

    def _get_train_batch(self) -> Dict:
        """Samples a batch of transitions from the training data set."""
        _batch = self._train_data.sample_batch(self._batch_size)

        return _batch

    def print_train_info(self) -> None:
        """Print the training information in training process."""
        info = self._train_info
        step = self._global_step
        summary_str = utils.get_summary_str(step, info)
        logging.info(summary_str)

    def write_train_summary(self, summary_writer: SummaryWriter) -> None:
        """Record the training information.

        :param SummaryWriter summary_writer: a file writer.
        """
        info = self._train_info
        step = self._global_step
        logger.write_summary_tensorboard(summary_writer, step, info)
        _info = {}
        _info.update(global_step=step)
        _info.update(info)
        logger.WandbLogger.write_summary(_info)

    def save(self, ckpt_name: str) -> None:
        torch.save(self._critic_module.state_dict(), ckpt_name + '.pth')

    def restore(self, ckpt_name: str) -> None:
        self._critic_module.load_state_dict(torch.load(ckpt_name + '.pth'))

    @property
    def global_step(self) -> int:
        """The global training step."""
        return self._global_step

    def _get_modules(self) -> utils.Flags:
        model_params_q, n_q_fns = self._model_params

        def q_net_factory():
            return networks.CriticNetwork(
                observation_space=self._state_dim,
                action_space=self._action_dim,
                fc_layer_params=model_params_q,
                device=self._device,
            )

        modules = utils.Flags(
            q_net_factory=q_net_factory,
            n_q_fns=n_q_fns,
            device=self._device,
        )
        return modules

    def get_q(self, s: Tensor, a: Tensor) -> Tensor:
        q = []
        for q_fn in self._q_fns:
            q.append(q_fn(s, a))
        q = [x.unsqueeze(0) for x in q]
        q = torch.mean(torch.cat(q, 0), dim=0)
        return q


class CriticModule(nn.Module):

    def __init__(
            self,
            modules: Union[utils.Flags, Any],
    ) -> None:
        super(CriticModule, self).__init__()
        self._net_modules = modules
        self._build_modules()

    def _build_modules(self) -> None:
        device = self._net_modules.device
        self._q_nets = nn.ModuleList()
        n_q_fns = self._net_modules.n_q_fns  # The number of the Q nets.
        for _ in range(n_q_fns):
            self._q_nets.append(self._net_modules.q_net_factory().to(device))
        self._q_target_nets = copy.deepcopy(self._q_nets)

    @property
    def q_nets(self) -> nn.ModuleList:
        return self._q_nets

    @property
    def q_target_nets(self) -> nn.ModuleList:
        return self._q_target_nets


class FQEval(BaseEval):
    """Evaluator with fitted-Q evaluation.

    The main implementation of this method:

    1. Load the policy model that is to be evaluated;
    2. Train the Q-net using FQE with respect to the loaded policy;
    3. Compare the Q-value computed by the trained Q-nets.

    :param BaseAgent agent: The agent object that contains the trained policy to be evaluated.
    :param ReplayBuffer data: The dataset used to train the Q function in FQE.
    :param int state_dim: Dimension of the state.
    :param int action_dim: The dimension of the action.
    :param str save_dir: The absolute path of the folder to save the Q function model and the evaluating results.
    :param int train_steps: The number of steps of training the Q function.
    :param int print_freq: The frequency of printing the training metric information.
    :param int summary_freq: The frequency of recording the training metric information.
    :param List model_params: The parameters for constructing the critic network. It can be a list like \
        `[[256, 256], 2]``. ``[256, 256]`` means a two-layer FC network with 256 units in each layer and \
        the number ``2`` means the number of the Q nets for ensemble.
    :param List optimizers: The parameters for create the optimizers. It can be a dict like \
        ``['adam', 1e-4]``. It contains the type of the optimizer and the learning rate for \
        the critic network model.
    :param int batch_size: The size of data batch for training.
    :param float weight_decays: L2 regularization coefficient of the networks.
    :param int update_freq: The frequency of update the parameters of the target network.
    :param float update_rate: The rate of update the parameters of the target network.
    :param float discount: The discount factor for computing the cumulative reward.
    :param device: Which device to create this model on. Default to None.
    :param str wandb_project: The project of the W&B logger for recoding the training and evaluating information.
    :param str wandb_name: W&B run name.
    :param str wandb_mode: Can be `"online"`, `"offline"` or `"disabled"`. Defaults to online.
    :param int start: The index of the start point of the evaluating data in the whole dataset.
    :param int steps: The number of the evaluating times beginning with the start point.
    """

    TYPE: ClassVar[str] = 'fqe'

    def __init__(
            self,
            agent: BaseAgent,
            data: ReplayBuffer,
            state_dim: int,
            action_dim: int,
            save_dir: str,
            train_steps: int = 250000,
            print_freq: int = 1000,
            summary_freq: int = 100,
            model_params: Union[List, Tuple] = ([1024, 1024, 1024, 1024], 1),
            optimizers: Union[List, Tuple] = ('adam', 1e-4),
            batch_size: int = 256,
            weight_decays: float = 0.0,
            update_freq: int = 100,
            update_rate: float = 1,
            discount: float = 0.99,
            device: Optional[Union[str, int, torch.device]] = None,
            wandb_project: Optional[str] = None,
            wandb_name: Optional[str] = None,
            wandb_mode: Optional[str] = 'online',
            start: int = 0,
            steps: int = 100,
    ) -> None:
        self._policy = agent.test_policies['main']
        self._save_dir = save_dir
        self._train_steps = train_steps
        self._print_freq = print_freq
        self._summary_freq = summary_freq
        self._fqe = FQE(
            policy=self._policy,
            state_dim=state_dim,
            action_dim=action_dim,
            train_data=data,
            model_params=model_params,
            optimizers=optimizers,
            batch_size=batch_size,
            weight_decays=weight_decays,
            update_freq=update_freq,
            update_rate=update_rate,
            discount=discount,
            device=device,
        )
        _time = datetime.now()
        if wandb_project is None:
            self._wandb_project = 'FQE-' + str(_time.date())
        else:
            self._wandb_project = wandb_project
        if wandb_name is None:
            self._wandb_name = str(_time)
        else:
            self._wandb_name = wandb_name
        self._wandb_mode = wandb_mode
        self._data = data
        self._batch_size = batch_size
        self._start = start
        self._steps = steps
        self._train_critic()

    def _train_critic(self) -> None:
        print('\n' + '='*20 + 'Beginning the FQE procedure.' + '='*20)
        self._train_summary_writer, self._eval_summary_writer = self._build_summary()
        wandb_name = '(FQE-train)' + self._wandb_name
        train_wandb_logger = WandbLogger(
            project=self._wandb_project,
            name=wandb_name,
            mode=self._wandb_mode,
            dir_=self._train_summary_dir,
        )
        for i in range(self._train_steps):
            self._fqe.train_step()
            step = self._fqe.global_step
            if step % self._print_freq == 0:
                self._fqe.print_train_info()
            if step % self._summary_freq == 0 or step == self._train_steps:
                self._fqe.write_train_summary(self._train_summary_writer)
        self._fqe.save(self._q_ckpt_dir)
        self._train_summary_writer.close()
        train_wandb_logger.finish()

    def _build_summary(self) -> Tuple[SummaryWriter, SummaryWriter]:
        log_dir = self._save_dir
        self._q_ckpt_dir = os.path.join(log_dir, 'fqe_Q/q')
        self._train_summary_dir = self._q_ckpt_dir + '_train_log'
        utils.maybe_makedirs(os.path.dirname(self._train_summary_dir))
        train_summary_writer = SummaryWriter(self._train_summary_dir)
        self._eval_summary_dir = os.path.join(log_dir, 'fqe_result')
        eval_summary_writer = SummaryWriter(self._eval_summary_dir)
        return train_summary_writer, eval_summary_writer

    def _eval_policies(self, test_obs: Tensor) -> Dict:
        info = collections.OrderedDict()
        with torch.no_grad():
            a = self._policy(test_obs)
            estimate_q = self._fqe.get_q(test_obs, a)
        info['estimate_q'] = estimate_q.mean().item()
        return info

    def eval(self) -> None:
        assert self._start < self._data.size
        wandb_name = '(FQE-eval)' + self._wandb_name
        eval_wandb_logger = WandbLogger(
            project=self._wandb_project,
            name=wandb_name,
            mode=self._wandb_mode,
            dir_=self._eval_summary_dir,
        )
        for i in range(self._steps):
            data_index = np.arange(
                self._start + i * self._batch_size,
                self._start + (i+1) * self._batch_size,
            )
            if data_index[-1] > self._data.size:
                break
            test_data = self._data.get_batch_indices(data_index)
            test_obs = test_data['s1']
            eval_info = self._eval_policies(test_obs)
            logger.write_summary_tensorboard(self._eval_summary_writer, i, eval_info)
            logger.WandbLogger.write_summary(eval_info)
        self._eval_summary_writer.close()
        eval_wandb_logger.finish()

    @classmethod
    def from_config(cls, agent: BaseAgent, data: ReplayBuffer, config: Union[utils.Flags, Any]):
        model_cfg = config.model_config
        eval_cfg = model_cfg.eval.ope.fqe
        save_dir = os.path.join(model_cfg.train.agent_ckpt_dir + '_eval', cls.TYPE)
        return cls(
            agent=agent,
            data=data,
            state_dim=model_cfg.env.basic_info.state_dim,
            action_dim=model_cfg.env.basic_info.action_dim,
            save_dir=save_dir,
            train_steps=eval_cfg.train_steps,
            print_freq=model_cfg.train.print_freq,
            summary_freq=model_cfg.train.summary_freq,
            model_params=eval_cfg.model_params,
            optimizers=eval_cfg.optimizers,
            batch_size=model_cfg.train.batch_size,
            weight_decays=model_cfg.train.weight_decays,
            update_freq=eval_cfg.update_freq,
            update_rate=eval_cfg.update_rate,
            discount=model_cfg.train.discount,
            device=model_cfg.train.device,
            wandb_project=model_cfg.train.wandb.project,
            wandb_name=model_cfg.train.wandb.name,
            wandb_mode=model_cfg.train.wandb.mode,
            start=eval_cfg.start,
            steps=eval_cfg.eval_steps,
        )









