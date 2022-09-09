from typing import Any, Union
from d2c.evaluators.sim.benchmark import BMEval
from d2c.models import BaseAgent
from d2c.envs import BaseEnv
from d2c.utils.utils import Flags


def bm_eval(
        agent: BaseAgent,
        env: BaseEnv,
        config: Union[Any, Flags]
) -> BMEval:
    """The API of building the evaluators with simulator.

    :param BaseAgent agent: The agent to be evaluated.
    :param BaseEnv env: An env to evaluate the policy.
    :param config: The configuration object.
    """
    n_eval_episodes = config.model_config.eval.n_eval_episodes
    score_normalize = config.model_config.env.external.score_normalize
    score_norm_min = config.model_config.env.external.score_norm_min
    score_norm_max = config.model_config.env.external.score_norm_max
    seed = config.model_config.train.seed
    log_dir = config.model_config.eval.log_dir
    agent_dir = config.model_config.train.agent_ckpt_dir
    result_dir = agent_dir + '_' + log_dir
    return BMEval(result_dir=result_dir, agent=agent, env=env, n_eval_episodes=n_eval_episodes,
                  score_normalize=score_normalize, score_norm_min=score_norm_min,
                  score_norm_max=score_norm_max, seed=seed)
