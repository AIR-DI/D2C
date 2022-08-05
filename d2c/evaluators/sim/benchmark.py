"""Use the simulators from the benchmarks to evaluate the policy."""

import logging
import os
import collections
import numpy as np
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Tuple, List
from d2c.evaluators.base import BaseEval
from d2c.utils import utils
from d2c.utils.logger import write_summary_tensorboard
from d2c.models import BaseAgent
from d2c.envs import BaseEnv


class BMEval(BaseEval):
    """The evaluator for the benchmark experiments.

    The main API methods for users are:

    * :meth:`eval`
    * :meth:`save_eval_results`

    :param str result_dir: the path of the folder for saving the evaluating results.
    :param BaseAgent agent: The agent to be evaluated.
    :param BaseEnv env: An env for the benchmark dataset.
    :param int n_eval_episodes: the number of evaluating episodes.
    :param bool score_normalize: if normalizing the evaluating score.
    :param float score_norm_min: the minimum value for normalizing the score.
    :param float score_norm_max: the maximum value for normalizing the score.
    :param int seed: the random seed for env.
    """
    def __init__(
            self,
            result_dir: str,
            agent: BaseAgent,
            env: BaseEnv,
            n_eval_episodes: int = 20,
            score_normalize: bool = False,
            score_norm_min: float = None,
            score_norm_max: float = None,
            seed: int = None,
    ) -> None:
        self._agent = agent
        self._env = env
        self._env.reset(seed=seed)
        self._n_episodes = n_eval_episodes
        self._score_norm = score_normalize
        self._score_norm_min = score_norm_min
        self._score_norm_max = score_norm_max
        self._policies = self._agent.test_policies
        self._eval_summary_dir = result_dir
        self._eval_summary_writers = self._build_writer()
        # To restore the evaluation results of the polices in the training process
        self._eval_r_results = []

    def _eval_policy_episodes(self, policy: nn.Module) -> Tuple[float, float, np.ndarray]:
        results = []
        for i in range(self._n_episodes):
            observation = self._env.reset()
            done = None
            total_rewards = 0.0
            while not done:
                action = policy(observation)
                observation, reward, done, _ = self._env.step(action)
                total_rewards += reward
            results.append(total_rewards)
        logging.info('='*20+f' Complete evaluation of {self._n_episodes} episodes! '+'='*20)
        results = np.array(results)
        return float(np.mean(results)), float(np.std(results)), results

    def _eval_policies(self) -> Tuple[List, Dict]:
        results_episode_return = []
        results_std = []
        complete_results = []
        infos = collections.OrderedDict()
        for name, policy in self._policies.items():
            mean, std, comp_result = self._eval_policy_episodes(policy)
            if self._score_norm:
                mean = 100 * (mean - self._score_norm_min) / (self._score_norm_max - self._score_norm_min)
                comp_result = 100 * (comp_result - self._score_norm_min) / (self._score_norm_max - self._score_norm_min)
                std = float(np.std(comp_result))
            results_episode_return.append(mean)
            results_std.append(std)
            complete_results.append(comp_result)
            infos[name] = collections.OrderedDict()
            infos[name]['episode_mean'] = mean
        results = [results_episode_return] + [results_std] + [complete_results]
        return results, infos

    def eval(self, step: int) -> None:
        """The evaluation API methods for policies evaluation.

        :param int step: The step number of the agent training process.
        """
        eval_r_result, eval_r_info = self._eval_policies()
        self._eval_r_results.append([step] + eval_r_result)
        for policy_key, policy_info in eval_r_info.items():
            logging.info(utils.get_summary_str(
                step=None, info=policy_info, prefix=policy_key + ': '
            ))
            write_summary_tensorboard(self._eval_summary_writers[policy_key], step, policy_info)
        logging.info(f'Testing at step {step}.')

    def save_eval_results(self) -> None:
        """Save the whole evaluation results across the agent training process.

        Call it when agent training finished.
        """
        eval_r_results = np.array(self._eval_r_results)
        results_file = os.path.join(self._eval_summary_dir, 'results_reward.npy')
        np.save(results_file, eval_r_results)
        logging.info(f'The results have been saved in {results_file}.')
        # Close the summary writer.
        for writer in self._eval_summary_writers.values():
            writer.close()
        logging.info('The summary writers of the evaluator have been closed.')

    def _build_writer(self) -> Dict[str, SummaryWriter]:
        eval_summary_writers = collections.OrderedDict()
        for policy_key in self._agent.test_policies.keys():
            eval_summary_writer = SummaryWriter(
                os.path.join(self._eval_summary_dir, policy_key)
            )
            eval_summary_writers[policy_key] = eval_summary_writer
        return eval_summary_writers


