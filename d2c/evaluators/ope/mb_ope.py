import os
import collections
import numpy as np
from datetime import datetime
from torch import nn, Tensor
from torch.utils.tensorboard import SummaryWriter
from typing import Union, Optional, Dict, Any, ClassVar
from d2c.evaluators.base import BaseEval
from d2c.models import BaseAgent
from d2c.envs import LeaEnv
from d2c.utils.logger import WandbLogger
from d2c.utils.replaybuffer import ReplayBuffer
from d2c.utils import utils, logger


class MBOPE(BaseEval):
    """Model-based off-policy evaluation.
    Using the trained dynamics model to evaluate the policy through predicting the cumulative reward.

    :param BaseAgent agent: The agent with the trained policy.
    :param ReplayBuffer data: The dataset of the batch data.
    :param LeaEnv env: The env object that contains the trained dynamics.
    :param str save_dir: The directory for saving the evaluating results.
    :param float discount: The discount coefficient for computing the cumulative reward.
    :param int episode_steps: The number of steps for dynamics rollout.
    :param int eval_size: The batch size of the evaluating data.
    :param Optional[str] wandb_project: The wandb project name.
    :param Optional[str] wandb_name: The name of the wandb logger object.
    :param Optional[str] wandb_mode: The mode of the wandb logger. Default to be 'online'.
    :param int start: The start point of the evaluating data.
    :param int steps: The evaluating data size.
    """

    TYPE: ClassVar[str] = 'mb_ope'

    def __init__(
            self,
            agent: BaseAgent,
            data: ReplayBuffer,
            env: LeaEnv,
            save_dir: str,
            discount: float = 0.99,
            episode_steps: int = 20,
            eval_size: int = 256,
            wandb_project: Optional[str] = None,
            wandb_name: Optional[str] = None,
            wandb_mode: Optional[str] = 'online',
            start: int = 0,
            steps: int = 100,
    ):
        self._policies = agent.test_policies
        self._data = data
        self._env = env
        self._save_dir = save_dir
        self._discount = discount
        self._episode_steps = episode_steps
        self._eval_size = eval_size
        _time = datetime.now()
        if wandb_project is None:
            self._wandb_project = 'MBOPE-' + str(_time.date())
        else:
            self._wandb_project = wandb_project
        if wandb_name is None:
            self._wandb_name = str(_time)
        else:
            self._wandb_name = wandb_name
        self._wandb_mode = wandb_mode
        self._start = start
        self._steps = steps
        self._eval_summary_writer = SummaryWriter(self._save_dir)

    def _eval_policies(self, test_obs: Tensor) -> Dict:
        info = collections.OrderedDict()
        for name, policy in self._policies.items():
            self._env.reset(options={'init_s': test_obs})
            reward = 0
            obs = test_obs
            for t in range(self._episode_steps):
                action = policy(obs)
                obs, r, _, _ = self._env.step(action)
                reward += self._discount ** t * r
            info[f'({name})MBOPE'] = np.mean(reward)
        return info

    def eval(self) -> None:
        print('\n' + '=' * 20 + 'Beginning the model-based OPE procedure.' + '=' * 20)
        assert self._start < self._data.size
        wandb_name = '(MBOPE-eval)' + self._wandb_name
        eval_wandb_logger = WandbLogger(
            project=self._wandb_project,
            name=wandb_name,
            mode=self._wandb_mode,
            dir_=self._save_dir,
        )
        for i in range(self._steps):
            data_index = np.arange(
                self._start + i * self._eval_size,
                self._start + (i+1) * self._eval_size,
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
    def from_config(
            cls,
            agent: BaseAgent,
            data: ReplayBuffer,
            env: LeaEnv,
            config: Union[utils.Flags, Any],
    ):
        model_cfg = config.model_config
        eval_cfg = model_cfg.eval.ope.mb_ope
        save_dir = os.path.join(model_cfg.train.agent_ckpt_dir + '_eval', cls.TYPE)
        return cls(
            agent=agent,
            data=data,
            env=env,
            save_dir=save_dir,
            discount=eval_cfg.discount,
            episode_steps=eval_cfg.episode_steps,
            eval_size=model_cfg.train.batch_size,
            wandb_project=model_cfg.train.wandb.project,
            wandb_name=model_cfg.train.wandb.name,
            wandb_mode=model_cfg.train.wandb.mode,
            start=eval_cfg.start,
            steps=eval_cfg.eval_steps,
        )
