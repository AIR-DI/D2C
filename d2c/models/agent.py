"""The base class of RL agent."""

import torch
import collections
import numpy as np
from absl import logging
from typing import Union, Optional, List, Tuple, Dict
from d2c.envs.base import BaseEnv
from d2c.utils.replaybuffer import ReplayBuffer
from d2c.utils import utils


class Agent(object):
    """The base class for learning policy and interacting with environment.

    We aim to modularizing RL algorithms. It comes into 4 classes of offline
    RL algorithms in D2C. All of the RL algorithms must inherit
    :class:`~d2c.models.agent.Agent`.

    A agent class typically has the following parts:

    * ``train_step()``: train the policy for one step;
    * ``save()``: save the trained models;
    * ``restore()``: restore the trained models;
    * ``_get_modules()``: create the network factories needed for building the models \
        that construct an agent;
    * ``test_policies()``: return the trained policy of this agent

    :param BaseEnv env: the environment learned that contains the trained dynamics.
    :param Dict model_params: the parameters for construct the models.
    :param Dict optimizers: the parameters for create the optimizers.
    :param ReplayBuffer train_data: the dataset of the batch data.
    :param int batch_size: the size of data batch for training.
    :param float weight_decays: L2 regularization coefficient of the networks.
    :param int update_freq: the frequency of update the parameters of the target network.
    :param float update_rate: the rate of update the parameters of the target network.
    :param float discount: the discount factor for computing the cumulative reward.
    :param ReplayBuffer empty_dataset: a replay buffer for storing the generated virtual
        data by the simulator.
    :param device: which device to create this model on. Default to None.

    """

    def __init__(
            self,
            env: BaseEnv,
            model_params: Dict,
            optimizers: Dict,
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
        self._observation_space = env.observation_space()
        self._action_space = env.action_space()
        self._a_max = self._action_space.high
        self._a_dim = self._action_space.shape[0]
        self._model_params = model_params
        self._modules = self._get_modules()
        self._optimizers = optimizers
        self._batch_size = batch_size
        self._weight_decays = weight_decays
        self._train_data = train_data
        self._update_freq = update_freq
        self._update_rate = update_rate
        self._discount = discount
        self._empty_dataset = empty_dataset
        self._device = device
        self._build_agent()

    def _build_agent(self):
        """Builds agent components."""
        self._build_fns()
        self._build_optimizers()
        self._global_step = tf.Variable(0)
        self._train_info = collections.OrderedDict()
        self._checkpointer = self._build_checkpointer()
        self._test_policies = collections.OrderedDict()
        self._build_test_policies()
        self._online_policy = self._build_online_policy()
        if self._train_data is not None:
            train_batch = self._get_train_batch()
            self._init_vars(train_batch)

    def _build_fns(self):
        self._agent_module = AgentModule(modules=self._modules)

    def _get_vars(self):
        return []

    def _build_optimizers(self):
        opt = self._optimizers[0]
        opt_fn = utils.get_optimizer(opt[0])
        self._optimizer = opt_fn(lr=opt[1])

    def _build_loss(self, batch):
        raise NotImplementedError

    def _build_checkpointer(self):
        return tf.train.Checkpoint(
            agent=self._agent_module,
            global_step=self._global_step,
        )

    def _build_test_policies(self):
        return None
        # raise NotImplementedError

    def _build_online_policy(self):
        return None

    @property
    def test_policies(self):
        """The trained policy."""
        return self._test_policies

    @property
    def online_policy(self):
        return self._online_policy

    def _get_batch(self, batch_indices):
        batch_ = self._train_data.get_batch_indices(batch_indices)
        # Transform the done batch
        batch_[-1] = np.abs(batch_[-1] - 1)
        batch_ = [tf.convert_to_tensor(x, dtype=tf.float32) for x in batch_]
        batch = {x: y for x, y in zip(['s1', 'a1', 's2', 'a2', 'r', 'c', 'dsc'],
                                      [batch_[i] for i in range(len(batch_))])}
        return batch

    def _get_train_batch(self):
        """Samples and constructs batch of transitions from the training data set."""
        batch_indices = np.random.choice(
            len(self._train_data),
            self._batch_size
        )

        return self._get_batch(batch_indices)

    def _optimize_step(self, batch):
        with tf.GradientTape() as tape:
            loss, info = self._build_loss(batch)
        trainable_vars = self._get_vars()
        grads = tape.gradient(loss, trainable_vars)
        grads_and_vars = tuple(zip(grads, trainable_vars))
        self._optimizer.apply_gradients(grads_and_vars)
        return info

    def train_step(self):
        """Train the agent for one step."""
        train_batch = self._get_train_batch()
        info = self._optimize_step(train_batch)
        for key, val in info.items():
            self._train_info[key] = val.numpy()
        self._global_step.assign_add(1)

    def _init_vars(self, batch):
        pass

    def _get_source_target_vars(self):
        return [], []

    def _update_target_fns(self, source_vars, target_vars):
        utils.soft_variables_update(
            source_vars,
            target_vars,
            tau=self._update_rate)

    def print_train_info(self):
        """Print the training information in training process."""
        info = self._train_info
        step = self._global_step.numpy()
        summary_str = utils.get_summary_str(step, info)
        logging.info(summary_str)

    def write_train_summary(self, summary_writer):
        """Record the training information.

        :param summary_writer: a tf file writer.
        """
        info = self._train_info
        step = self._global_step.numpy()
        utils.write_summary(summary_writer, step, info)

    def save(self, ckpt_name):
        """Save the whole agent.

        :param str ckpt_name: the file path for model saving.
        """
        self._checkpointer.write(ckpt_name)

    def restore(self, ckpt_name):
        """Restore the agent from the saved model file.

        :param str ckpt_name: the file path of the model saved.
        """
        self._checkpointer.restore(ckpt_name)

    @property
    def global_step(self):
        """The global training step."""
        return self._global_step.numpy()

    def _get_modules(self):
        """
        :return: The network factories for building the models
        """
        raise NotImplementedError

    def _build_fqe_loss(self, q_fns, p_fn, batch):
        """
        The loss for Fitted-Q evaluation
        :param q_fns: Q nets
        :param p_fn: policy to be evaluated
        :return: The loss for Fitted-Q evaluation
        """
        s1 = batch['s1']
        s2 = batch['s2']
        a1 = batch['a1']
        r = batch['r']
        dsc = batch['dsc']
        _, a2_p, _ = p_fn(s2)
        # Compute q target
        q1_targets = []
        q1_preds = []
        for q_fn, q_fn_target in q_fns:
            q2_target_ = q_fn_target(s2, a2_p)
            q1_pred = q_fn(s1, a1)
            q1_preds.append(q1_pred)
            q1_target_ = tf.stop_gradient(r + dsc * self._discount * q2_target_)
            q1_targets.append(q1_target_)
        # q loss
        q_losses = []
        for q1_pred, q1_target in zip(q1_preds, q1_targets):
            q_loss_ = tf.reduce_mean(tf.square(q1_pred - q1_target))
            q_losses.append(q_loss_)
        q_loss = tf.add_n(q_losses)
        # get q weight norm
        q_weights = []
        for q_fn, _ in q_fns:
            q_weights += q_fn.weights
        norms = []
        for w in q_weights:
            norm = tf.reduce_sum(tf.square(w))
            norms.append(norm)
        q_w_norm = tf.add_n(norms)
        norm_loss = self._weight_decays * q_w_norm
        loss = q_loss + norm_loss
        info = collections.OrderedDict()
        info['q_loss'] = q_loss
        info['q_norm'] = q_w_norm
        info['r_mean'] = tf.reduce_mean(r)
        info['dsc_mean'] = tf.reduce_mean(dsc)
        info['q1_target_mean'] = tf.reduce_mean(q1_targets[0])
        info['q2_target_mean'] = tf.reduce_mean(q1_targets[1])
        return loss, info


class AgentModule(tf.Module):
    """The base class for AgentModule of any agent.

    Build the models for the Agent according to the input network factories.

    The following method should be implementation:

    * ``_build_modules()``: build the models needed using the input network factories.

    :param modules: the network factories that generated by an Agent method
        :meth:`~AIControlOpt_lib.models.agent.Agent._get_modules`.
    """

    def __init__(
            self,
            modules=None,
    ):
        super(AgentModule, self).__init__()
        self._modules = modules
        self._build_modules()

    def _build_modules(self):
        pass

