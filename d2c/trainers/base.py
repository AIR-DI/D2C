"""Trainers for all models in the RL algorithms"""
from abc import ABC, abstractmethod
from d2c.utils import utils



class BaseTrainer(ABC):
    """The base class for RL trainer.

    We aim to modularizing the RL training. There are all training methods for
    the models used in the RL algorithms, like Q-value function model, behavior
    policy model, policy model, dynamics model.

    We recommend that descendants of `BaseTrainer` implement the following methods:

    * ``__init__()``: initialize the Trainer, including getting several configurations \
        from the config and creating the folders for saving models;
    * ``train()``: the main interface for training. Train the models needed in turn \
        according to the configuration;
    * ``_train_behavior()``: train the behavior policy model in advance using the batch data;
    * ``_train_dynamics()``: train the dynamics model in advance using the batch data. \
        The dynamics trained here are used in the model-based RL and the env learned;
    * ``_train_q()``: train the Q-value function model with respect to behavior policy in \
        advance using Fitted-Q Evaluation with the batch data;
    * ``_train_vae_s()``: train the VAE model that is about the state;
    * ``_train_agent()``: train the RL agent. The main training process of the most RL algorithms \
        is here.

    :param Agent agent: the agent of the RL algorithm.
    :param Env env: the env with the dynamics which will be trained.
    :param FileReplayBuffer train_data: the replay buffer that contains the batch data.
    :param config: the configuration.
    """

    def __init__(self, agent, env, train_data, config):
        self._agent = agent
        self._env = env
        self._train_data = train_data
        self._config = config
        self._app_config = config.app_config
        self._model_config = config.model_config
        self._train_config = config.model_config.train
        utils.maybe_makedirs(self._train_config.model_dir)
        utils.set_seed(self._train_config.seed)  # set the random seed

    @abstractmethod
    def train(self):
        """Training the models needed."""
        pass

    @abstractmethod
    def _train_behavior(self):
        """Training the behavior models."""
        pass

    @abstractmethod
    def _train_dynamics(self):
        """Training the dynamics models."""
        pass

    @abstractmethod
    def _train_q(self):
        """Training the Q-value function model."""
        pass

    @abstractmethod
    def _train_vae_s(self):
        pass

    @abstractmethod
    def _train_agent(self):
        pass

