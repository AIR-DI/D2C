"""The base class of RL agent."""
import torch
import collections
import logging
from torch import nn, Tensor
from abc import ABC, abstractmethod
from easydict import EasyDict
from typing import Union, Optional, List, Tuple, Dict, Sequence, Any, Iterator
from torch.utils.tensorboard import SummaryWriter
from d2c.envs import LeaEnv
from d2c.utils.replaybuffer import ReplayBuffer
from d2c.utils import utils, logger


class BaseAgent(ABC):
    """The base class for learning policy and interacting with environment.

    We aim to modularizing RL algorithms. It comes into 4 classes of offline
    RL algorithms in D2C. All the RL algorithms must inherit
    :class:`~d2c.models.base.BaseAgent`.

    An agent class typically has the following parts:

    * ``train_step()``: train the policy for one step;
    * ``save()``: save the trained models;
    * ``restore()``: restore the trained models;
    * ``_get_modules()``: create the network factories needed for building the models \
        that construct an agent;
    * ``test_policies()``: return the trained policy of this agent

    :param BaseEnv env: the environment learned that contains the dynamics model. It provides the information of the \
        environment, like observation information and action information. It can also provide the trained dynamics \
        models for the model-based RL algorithms.
    :param Dict model_params: the parameters for constructing all the models of the algorithm. It can be a dict like \
        ``{q: [[256, 256], 2], p: [[256, 256],]}`` that contains the parameters of the Q net and the actor(policy) \
        net. ``[256, 256]`` means a two-layer FC network with 256 units in each layer and the number ``2`` means the \
        number of the Q nets.
    :param Dict optimizers: the parameters for create the optimizers. It can be a dict like \
        ``{q: ['adam', 3e-4], p: ['adam', 3e-4]}``. It contains the type of the optimizer and the learning rate for \
        every network model.
    :param ReplayBuffer train_data: the dataset of the batch data.
    :param int batch_size: the size of data batch for training.
    :param float weight_decays: L2 regularization coefficient of the networks.
    :param int update_freq: the frequency of update the parameters of the target network.
    :param float update_rate: the rate of update the parameters of the target network.
    :param float discount: the discount factor for computing the cumulative reward.
    :param ReplayBuffer empty_dataset: a replay buffer for storing the generated virtual
        data by the simulator. It should be empty and the training beginning.
    :param device: which device to create this model on. Default to None.
    """

    def __init__(
            self,
            env: LeaEnv,
            model_params: Union[Dict, EasyDict, Any],
            optimizers: Union[Dict, EasyDict, Any],
            train_data: ReplayBuffer,
            batch_size: int = 64,
            weight_decays: float = 0.0,
            update_freq: int = 1,
            update_rate: float = 0.005,
            discount: float = 0.99,
            empty_dataset: Optional[ReplayBuffer] = None,
            device: Optional[Union[str, int, torch.device]] = None,
    ) -> None:
        # the Env with the dynamics model pre-trained
        self._env = env
        self._observation_space = env.observation_space
        self._action_space = env.action_space
        self._a_max = torch.tensor(self._action_space.high, device=device, dtype=torch.float32)
        self._a_min = torch.tensor(self._action_space.low, device=device, dtype=torch.float32)
        self._a_dim = self._action_space.shape[0]
        self._model_params = model_params
        self._optimizers = optimizers
        self._batch_size = batch_size
        self._weight_decays = weight_decays
        self._train_data = train_data
        self._update_freq = update_freq
        self._update_rate = update_rate
        self._discount = discount
        self._empty_dataset = empty_dataset
        self._device = device
        self._modules = self._get_modules()
        self._build_agent()

    def _build_agent(self) -> None:
        """Builds agent components."""
        self._build_fns()
        self._init_vars()
        self._build_optimizers()
        self._global_step = 0
        self._train_info = collections.OrderedDict()
        self._test_policies = collections.OrderedDict()
        self._build_test_policies()

    @abstractmethod
    def _build_fns(self) -> None:
        """Build all the models of this RL algorithm."""
        pass

    def _init_vars(self):
        """Initialize the variables of all models."""
        raise NotImplementedError

    @abstractmethod
    def _build_optimizers(self) -> None:
        """Build optimizers for all the models."""
        pass

    def _build_loss(self, batch: Dict):
        pass

    def _get_train_batch(self) -> Dict:
        """Samples a batch of transitions from the training data set."""
        _batch = self._train_data.sample_batch(self._batch_size)

        return _batch

    @abstractmethod
    def _optimize_step(self, batch: Dict):
        """Build the optimizing schedule for all models."""
        pass

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

    @abstractmethod
    def _build_test_policies(self):
        """Build the policies for testing."""
        pass

    @property
    def test_policies(self) -> Dict:
        """The trained policy."""
        return self._test_policies

    @abstractmethod
    def save(self, ckpt_name: str) -> None:
        """Save the whole agent.

        :param str ckpt_name: the file path for model saving.
        """
        pass

    @abstractmethod
    def restore(self, ckpt_name: str) -> None:
        """Restore the agent from the saved model file.

        :param str ckpt_name: the file path of the model saved.
        """
        pass

    @property
    def global_step(self) -> int:
        """The global training step."""
        return self._global_step

    @abstractmethod
    def _get_modules(self):
        """Construct the network factories for building the models."""
        pass

    def _build_fqe_loss(self, q_fns, p_fn, batch):
        """
        The loss for Fitted-Q evaluation
        :param q_fns: Q nets
        :param p_fn: policy to be evaluated
        :return: The loss for Fitted-Q evaluation
        """
        pass


class BaseAgentModule(nn.Module, ABC):
    """The base class for AgentModule of any agent.

    Build the models for the Agent according to the input network factories.

    The following method should be implementation:

    * ``_build_modules()``: build the models needed using the input network factories.

    :param modules: the network factories that generated by an Agent method
        :meth:`~AIControlOpt_lib.models.agent.Agent._get_modules`.
    """

    def __init__(
            self,
            modules: Union[utils.Flags, Any],
    ) -> None:
        super(BaseAgentModule, self).__init__()
        self._net_modules = modules
        self._build_modules()

    @abstractmethod
    def _build_modules(self) -> None:
        pass

