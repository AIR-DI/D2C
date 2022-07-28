"""Base class of the dynamics model."""

import torch
import collections
import numpy as np
from torch import nn, Tensor
from absl import logging
from abc import ABC, abstractmethod
from easydict import EasyDict
from typing import Union, Optional, List, Tuple, Dict, Sequence, Any, Iterator
from d2c.utils import utils, logger
from d2c.utils.replaybuffer import ReplayBuffer


class BaseDyna(ABC):
    """The base class for learning dynamics.

    It comes into different classes of dynamics with different network structure.
    All the dynamics model must inherit
    :class:`~d2c.envs.learned.dynamics.base.BaseDyna`.

    A dynamic class typically has the following parts:

    * :meth:`train_step`: train the dynamic for one step;
    * :meth:`save`: save the trained models;
    * :meth:`restore`: restore the trained models.

    :param int state_dim: the dimension of the state.
    :param int action_dim: the dimension of the action.
    :param model_params: the parameters for construct the models.
    :param optimizers: the parameters for create the optimizers.
    :param ReplayBuffer train_data: the dataset of the batch data.
    :param int batch_size: the size of data batch for training.
    :param float weight_decays: L2 regularization coefficient of the networks.
    :param float train_test_ratio: the ratio of the training dataset.
    :param bool with_reward: if the output of the dynamics contains the reward or not.
    :param device: which device to create this model on. Default to None.
    """

    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            model_params: Union[Dict, EasyDict, Any],
            optimizers: Union[Dict, EasyDict, Any],
            train_data: ReplayBuffer,
            batch_size: int = 64,
            weight_decays: float = 0.0,
            train_test_ratio: float = 0.99,
            with_reward: bool = False,
            device: Optional[Union[str, int, torch.device]] = None,
    ) -> None:
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._model_params = model_params
        self._optimizers = optimizers
        self._train_data = train_data
        self._batch_size = batch_size
        self._weight_decays = weight_decays
        self._train_test_ratio = train_test_ratio
        self._with_reward = with_reward
        self._device = device
        self._modules = self._get_modules()
        self._build_agent()

    def _build_agent(self):
        """Builds dynamics components."""
        self._build_fns()
        self._init_vars()
        self._build_optimizers()
        self._global_step = 0
        self._train_info = collections.OrderedDict()

    @abstractmethod
    def _build_fns(self) -> None:
        """Build all the models of the dynamics."""
        self._agent_module = BaseDynaModule(modules=self._modules)

    @abstractmethod
    def _init_vars(self):
        """Initialize the variables of all models."""
        self._var = None

    def _build_optimizers(self) -> None:
        """Build optimizers for all the models."""
        opt = self._optimizers
        self._optimizer = utils.get_optimizer(opt[0])(
            parameters=self._var,
            lr=opt[1],
            weight_decay=self._weight_decays,
        )

    @abstractmethod
    def _build_loss(self, batch: Dict):
        """Build the loss for dynamics model training."""
        pass

    @abstractmethod
    def _build_test_loss(self, batch: Dict):
        """Build the loss for model testing."""
        pass

    def _get_batch(self, batch_indices: np.ndarray) -> Dict:
        """Samples and constructs batch of transitions from the training data set."""
        _batch = self._train_data.get_batch_indices(batch_indices)
        _batch['dsc'] = 1 - _batch['done']  # done -> discount

        return _batch

    def _get_train_batch(self) -> Dict:
        """Samples and constructs batch of transitions from the training data set"""
        batch_indices = np.random.choice(
            int(self._train_data.size*self._train_test_ratio),
            self._batch_size
        )

        return self._get_batch(batch_indices)

    def _get_test_batch(self) -> Dict:
        """Samples and constructs batch of transitions from the testing data set"""
        batch_indices = np.arange(
            int(self._train_data.size*self._train_test_ratio),
            self._train_data.size
        )

        return self._get_batch(batch_indices)

    def _optimize_step(self, batch: Dict) -> Dict:
        loss, info = self._build_loss(batch)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        return info

    def train_step(self) -> None:
        """Train the dynamics model for one step."""
        train_batch = self._get_train_batch()
        info = self._optimize_step(train_batch)
        for key, val in info.items():
            self._train_info[key] = val.item()
        self._global_step += 1

    def test_step(self) -> None:
        """Test the model with test dataset."""
        test_batch = self._get_test_batch()
        test_loss, info = self._build_test_loss(test_batch)
        for key, val in info.items():
            self._train_info[key] = val.item()

    def print_train_info(self) -> None:
        """Print the training information in training process."""
        info = self._train_info
        step = self._global_step
        summary_str = utils.get_summary_str(step, info)
        logging.info(summary_str)

    def write_train_summary(self, summary_writer) -> None:
        """Record the training information.

        :param summary_writer: a tf file writer.
        """
        info = self._train_info
        step = self._global_step
        logger.write_summary_tensorboard(summary_writer, step, info)
        logger.write_summary_wandb(info)

    def save(self, ckpt_name) -> None:
        """Save the dynamics model.

        :param str ckpt_name: the file path for model saving.
        """
        torch.save(self._agent_module.state_dict(), ckpt_name + '.pth')

    def restore(self, ckpt_name: str) -> None:
        """Restore the dynamics model.

        :param str ckpt_name: the file path of the model saved.
        """
        self._agent_module.load_state_dict(torch.load(ckpt_name + '.pth'))

    @property
    def global_step(self) -> int:
        """The global training step."""
        return self._global_step

    @abstractmethod
    def _get_modules(self) -> utils.Flags:
        """Construct the network factories for building the models."""
        pass

    @abstractmethod
    def dynamics_fns(
            self,
            s: Union[np.ndarray, Tensor],
            a: Union[np.ndarray, Tensor]
    ) -> Any:
        """Predict the next state.

        :param s: the input state.
        :param a: the input action."""
        pass


class BaseDynaModule(ABC, nn.Module):
    """The base class for Module of any dynamics.

    Build the models for the dynamics according to the input network factories.

    The following method should be implementation:

    * ``_build_modules()``: build the models needed using the input network factories.

    :param modules: the network factories that generated by an BaseDyna method
        :meth:`~d2c.envs.learned.dynamics.base.BaseDyna._get_modules`.
    """

    def __init__(
            self,
            modules: utils.Flags = None,
    ) -> None:
        super(BaseDynaModule, self).__init__()
        self._modules = modules
        self._build_modules()

    @abstractmethod
    def _build_modules(self) -> None:
        pass
