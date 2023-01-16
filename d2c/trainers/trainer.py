"""Trainer for RL models."""

import os
import time
import logging
import json5
import copy
from typing import Any, Union, Optional, Callable, Tuple, Dict
from torch.utils.tensorboard import SummaryWriter
from d2c.trainers.base import BaseTrainer
from d2c.models import BaseAgent
from d2c.envs import LeaEnv
from d2c.evaluators import BaseEval
from d2c.utils.replaybuffer import ReplayBuffer
from d2c.utils import utils
from d2c.envs.learned.dynamics import make_dynamics
from d2c.utils.logger import WandbLogger
from d2c.utils.config import ConfigBuilder


class Trainer(BaseTrainer):
    """Implementation of the trainer.

    :param evaluator: the evaluation for testing the training polices. It should \
        contain an external perfect env such as :class:`~d2c.evaluators.sim.benchmark.BMEval`. \
        You can input an evaluator when training in the benchmark experiments.

    .. seealso::

        Please refer to :class:`~d2c.trainers.base.BaseTrainer`
        for more detailed explanation.
    """

    def __init__(
            self,
            agent: Union[BaseAgent, Any],
            train_data: ReplayBuffer,
            config: Union[Any, utils.Flags],
            env: LeaEnv = None,
            evaluator: Union[Any, BaseEval] = None
    ) -> None:
        super(Trainer, self).__init__(agent, env, train_data, config)
        self._train_steps = self._train_cfg.total_train_steps
        self._summary_freq = self._train_cfg.summary_freq
        self._print_freq = self._train_cfg.print_freq
        self._save_freq = self._train_cfg.save_freq
        self._agent_name = self._model_cfg.model.model_name
        self._evaluator = evaluator
        self._eval_freq = self._train_cfg.eval_freq

    def train(self) -> None:
        _custom_train = self._build_train_schedule()
        _custom_train()

    def _train_behavior(self) -> None:
        b_ckpt_dir = self._train_cfg.behavior_ckpt_dir
        train_summary_writer, _ = self.check_ckpt(b_ckpt_dir)
        if train_summary_writer is not None:
            # Train the behavior
            for i in range(self._train_steps):
                train_b_info = self._agent.train_behavior_step()
                if i % self._print_freq == 0:
                    logging.info(utils.get_summary_str(step=i, info=train_b_info))
                if i % self._summary_freq == 0 or i == self._train_steps:
                    self._agent.write_b_train_summary(train_summary_writer, i, train_b_info)
            self._agent.save_behavior_model(b_ckpt_dir)
        self._agent.restore_behavior_model(b_ckpt_dir)
        train_summary_writer.close()

    def _train_dynamics(self) -> None:
        d_ckpt_dir = self._train_cfg.dynamics_ckpt_dir
        train_summary_writer, train_summary_dir = self.check_ckpt(d_ckpt_dir)
        logger_name = '(Dyna)' + self._model_cfg.train.wandb.name
        _config = copy.deepcopy(self._model_cfg.env.learned)
        _keys = list(_config.keys())
        for k in _keys:
            _config.pop(k) if k not in ['dynamic_module_type', 'with_reward', _config.dynamic_module_type] else None
        wandb_logger = self._build_wandb_logger(dir_=train_summary_dir, name=logger_name, _config=_config)
        if train_summary_writer is not None:
            # Train the dynamics
            dyna = make_dynamics(self._config, self._train_data)
            step = dyna.global_step
            while step < self._train_steps:
                dyna.train_step()
                step = dyna.global_step
                if step % self._print_freq == 0:
                    dyna.test_step()
                    dyna.print_train_info()
                if step % self._summary_freq == 0 or step == self._train_steps:
                    dyna.test_step()
                    dyna.write_train_summary(train_summary_writer)
            dyna.save(d_ckpt_dir)
            train_summary_writer.close()
            wandb_logger.finish()
        self._env.load()

    def _train_q(self) -> None:
        q_ckpt_dir = self._train_cfg.q_ckpt_dir
        train_summary_writer, _ = self.check_ckpt(q_ckpt_dir)
        if train_summary_writer is not None:
            # Train the Q-value function
            for i in range(self._train_steps):
                train_q_info = self._agent.train_q_step(i)
                if i % self._print_freq == 0:
                    logging.info(utils.get_summary_str(step=i, info=train_q_info))
                if i % self._summary_freq == 0 or i == self._train_steps:
                    self._agent.write_q_train_summary(train_summary_writer, i, train_q_info)
            self._agent.save_q_model(q_ckpt_dir)
        self._agent.restore_q_model(q_ckpt_dir)
        train_summary_writer.close()

    def _train_vae_s(self) -> None:
        vae_s_ckpt_dir = self._train_cfg.vae_s_ckpt_dir
        train_summary_writer, _ = self.check_ckpt(vae_s_ckpt_dir)
        if train_summary_writer is not None:
            for i in range(self._train_steps):
                train_vae_s_info = self._agent.train_vae_s_step()
                if i % self._print_freq == 0:
                    logging.info(utils.get_summary_str(step=i, info=train_vae_s_info))
                if i % self._summary_freq == 0 or i == self._train_steps:
                    self._agent.write_vaes_train_summary(train_summary_writer, i, train_vae_s_info)
            self._agent.save_vae_s_model(vae_s_ckpt_dir)
        self._agent.restore_vae_s_model(vae_s_ckpt_dir)
        train_summary_writer.close()

    def _train_agent(self) -> None:
        agent_ckpt_dir = self._train_cfg.agent_ckpt_dir
        utils.maybe_makedirs(os.path.dirname(agent_ckpt_dir))
        train_summary_dir = agent_ckpt_dir + '_train_log'
        train_summary_writer = SummaryWriter(train_summary_dir)
        wandb_logger = self._build_wandb_logger(dir_=train_summary_dir)

        time_st_total = time.time()
        step = self._agent.global_step
        while step < self._train_steps:
            self._agent.train_step()
            step = self._agent.global_step
            if step % self._summary_freq == 0 or step == self._train_steps:
                self._agent.write_train_summary(train_summary_writer)
            if step % self._print_freq == 0 or step == self._train_steps:
                self._agent.print_train_info()
            if step % self._eval_freq == 0 or step == self._train_steps:
                if self._evaluator is not None:
                    try:
                        eval_info = self._evaluator.eval(step)
                    except:
                        logging.info('Something wrong when evaluating the policy!')
                    else:
                        eval_info.update(global_step=step)
                        wandb_logger.write_summary(eval_info)
                    if step == self._train_steps:
                        self._evaluator.save_eval_results()
            if step % self._save_freq == 0:
                self._agent.save(agent_ckpt_dir)
                logging.info(f'Agent saved at {agent_ckpt_dir}.')
        self._agent.save(agent_ckpt_dir)
        train_summary_writer.close()
        wandb_logger.finish()
        time_cost = time.time() - time_st_total
        logging.info('Training finished, time cost %.4gs.', time_cost)

    @staticmethod
    def check_ckpt(_model_ckpt_dir: str) -> Tuple[Optional[SummaryWriter], str]:
        """Determine if the model files exist.

        When calling the :meth:`train` method, it will check if the models have been trained
        and decide if to create a file writer.

        :param str _model_ckpt_dir: the file path of the model that will be trained.

        :return: a file_writer for recording the model training information. If the
            model has already been trained, it will return ``None``.

        """
        _train_summary_dir = _model_ckpt_dir+'_train_log'
        if os.path.exists(f'{_model_ckpt_dir}.pth'):
            logging.info(f'Checkpoint found at {_model_ckpt_dir}')
            train_summary_writer = None
        else:
            logging.info(f'No trained checkpoint, train the {_model_ckpt_dir}')
            utils.maybe_makedirs(os.path.dirname(_model_ckpt_dir))
            train_summary_writer = SummaryWriter(
                _train_summary_dir
            )
        return train_summary_writer, _train_summary_dir

    def _build_train_schedule(self) -> Callable:
        train_fn_dict = dict(
            b=self._train_behavior,
            d=self._train_dynamics,
            q=self._train_q,
            vae_s=self._train_vae_s,
            agent=self._train_agent,
        )
        train_sche = self._model_cfg.model[self._agent_name].train_schedule

        def custom_train():
            for x in train_sche:
                train_fn_dict[x]()

        return custom_train

    def _build_wandb_logger(
            self,
            dir_: Optional[str] = None,
            name: Optional[str] = None,
            _config: Optional[Dict] = None,
    ) -> WandbLogger:
        _params = copy.deepcopy(self._model_cfg.train.wandb)
        if dir_ is not None:
            utils.maybe_makedirs(dir_)
            _params.update(dir_=dir_)
        if name is not None:
            _params.update(name=name)
        if _config is None:
            _config = ConfigBuilder.main_hyper_params(self._model_cfg)
        _params.update(config=_config)
        return WandbLogger(**_params)





