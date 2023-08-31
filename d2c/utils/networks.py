"""Neural networks for RL models."""

import torch
import numpy as np
import torch.nn.functional as F
from gym.spaces import Box, Space
from torch import nn, Tensor
from typing import Tuple, List, Union, Type, Optional, Sequence
from torch.distributions import Normal, TransformedDistribution, Distribution
from torch.distributions.transforms import AffineTransform, SigmoidTransform


ModuleType = Type[nn.Module]
LOG_STD_MIN = -5
LOG_STD_MAX = 2


def miniblock(
        input_size: int,
        output_size: int = 0,
        norm_layer: Optional[ModuleType] = None,
        activation: Optional[ModuleType] = None,
        linear_layer: Type[nn.Linear] = nn.Linear,
) -> List[nn.Module]:
    """Construct a miniblock with given input/output-size, norm layer and \
    activation."""
    layers: List[nn.Module] = [linear_layer(input_size, output_size)]
    if norm_layer is not None:
        layers += [norm_layer(output_size)]  # type: ignore
    if activation is not None:
        layers += [activation()]
    return layers


def get_spec_means_mags(
        space: Box,
        device: Optional[Union[str, int, torch.device]] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    means = (space.high + space.low) / 2.0
    mags = (space.high - space.low) / 2.0
    means = torch.as_tensor(means, device=device, dtype=torch.float32)
    mags = torch.as_tensor(mags, device=device, dtype=torch.float32)
    return means, mags


class ActorNetwork(nn.Module):
    """Stochastic Actor network.

    :param Box observation_space: the observation space information. It is an instance
        of class: ``gym.spaces.Box``.
    :param Box action_space: the action space information. It is an instance
        of class: ``gym.spaces.Box``.
    :param tuple fc_layer_params: the network parameter. For example:
        ``(300, 300)`` means a 2-layer network with 300 units in each layer.
    :param device: which device to create this model on. Default to 'cpu'.
    """

    def __init__(
            self,
            observation_space: Union[Box, Space],
            action_space: Union[Box, Space],
            fc_layer_params: Sequence[int] = (),
            device: Union[str, int, torch.device] = 'cpu',
    ) -> None:
        super(ActorNetwork, self).__init__()
        self._device = device
        state_dim = observation_space.shape[0]
        self._action_space = action_space
        self._action_dim = action_space.shape[0]
        self._layers = []
        hidden_sizes = [state_dim] + list(fc_layer_params)
        for in_dim, out_dim in zip(hidden_sizes[:-1], hidden_sizes[1:]):
            self._layers += miniblock(in_dim, out_dim, None, nn.ReLU)
        output_dim = self._action_dim * 2
        self._layers += [nn.Linear(hidden_sizes[-1], output_dim)]
        self._model = nn.Sequential(*self._layers)
        self._action_means, self._action_mags = get_spec_means_mags(
            self._action_space, self._device)

    def _get_output(self, state: Union[np.ndarray, torch.Tensor]) \
            -> Tuple[Distribution, torch.Tensor]:
        state = torch.as_tensor(state, device=self._device, dtype=torch.float32)
        h = self._model(state)
        mean, log_std = torch.split(h, split_size_or_sections=[self._action_dim, self._action_dim], dim=-1)
        a_tanh_mode = torch.tanh(mean) * self._action_mags + self._action_means
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        std = torch.exp(log_std)
        # TanhTransform()is equivalent to \
        # ComposeTransform([AffineTransform(0., 2.), SigmoidTransform(), AffineTransform(-1., 2.)])
        a_distribution = TransformedDistribution(
            base_distribution=Normal(
                loc=mean,
                scale=std,
            ),
            transforms=[
                AffineTransform(0., 2.),
                SigmoidTransform(),
                AffineTransform(-1., 2.),
                AffineTransform(loc=self._action_means, scale=self._action_mags)
            ],
        )
        return a_distribution, a_tanh_mode

    def forward(self, state: Union[np.ndarray, torch.Tensor])\
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        a_dist, a_tanh_mode = self._get_output(state)
        a_sample = a_dist.rsample()
        log_pi_a = a_dist.log_prob(a_sample)
        return a_tanh_mode, a_sample, log_pi_a

    def get_log_density(self, state: Tensor, action: Tensor) -> Tensor:
        a_dist, _ = self._get_output(state)
        action = torch.as_tensor(action, dtype=torch.float32, device=self._device)
        log_density = a_dist.log_prob(action)
        return log_density

    def sample_n(self, state: Union[np.ndarray, Tensor], n: int = 1)\
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        a_dist, a_tanh_mode = self._get_output(state)
        a_sample = a_dist.rsample([n])
        log_pi_a = a_dist.log_prob(a_sample)
        return a_tanh_mode, a_sample, log_pi_a

    def sample(self, state: Union[np.ndarray, Tensor]) -> Tensor:
        return self.sample_n(state, n=1)[1][0]

    @property
    def action_space(self) -> Box:
        return self._action_space


class ActorNetworkDet(nn.Module):
    """Deterministic Actor network.

    :param Box observation_space: the observation space information. It is an instance
        of class: ``gym.spaces.Box``.
    :param Box action_space: the action space information. It is an instance
        of class: ``gym.spaces.Box``.
    :param tuple fc_layer_params: the network parameter. For example:
        ``(300, 300)`` means a 2-layer network with 300 units in each layer.
    :param device: which device to create this model on. Default to 'cpu'.
    """

    def __init__(
            self,
            observation_space: Union[Box, Space],
            action_space: Union[Box, Space],
            fc_layer_params: Sequence[int] = (),
            device: Union[str, int, torch.device] = 'cpu',
    ) -> None:
        super(ActorNetworkDet, self).__init__()
        self._device = device
        state_dim = observation_space.shape[0]
        self._action_space = action_space
        self._action_dim = action_space.shape[0]
        self._layers = []
        hidden_sizes = [state_dim] + list(fc_layer_params)
        for in_dim, out_dim in zip(hidden_sizes[:-1], hidden_sizes[1:]):
            self._layers += miniblock(in_dim, out_dim, None, nn.ReLU)
        self._layers += [nn.Linear(hidden_sizes[-1], self._action_dim)]
        self._model = nn.Sequential(*self._layers)
        self._action_means, self._action_mags = get_spec_means_mags(
            self._action_space, self._device)

    def forward(self, state: Union[np.ndarray, Tensor]) -> Tensor:
        state = torch.as_tensor(state, device=self._device, dtype=torch.float32)
        a = self._model(state)
        return torch.tanh(a) * self._action_mags + self._action_means

    @property
    def action_space(self) -> Box:
        return self._action_space


class ProbDynamicsNetwork(nn.Module):
    """Stochastic Dynamics network(Probabilistic dynamics model).

    :param int state_dim: the observation space dimension.
    :param int action_dim: the action space dimension.
    :param tuple fc_layer_params: the network parameter. For example:
        ``(300, 300)`` means a 2-layer network with 300 units in each layer.
    :param bool local_mode: `local_mode` means that this model predicts the difference to the current state.
    :param bool with_reward: if the output of the dynamics contains the reward or not.
    :param device: which device to create this model on. Default to 'cpu'.
    """

    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            fc_layer_params: Sequence[int] = (),
            local_mode: bool = False,
            with_reward: bool = False,
            device: Union[str, int, torch.device] = 'cpu',
    ) -> None:
        super(ProbDynamicsNetwork, self).__init__()
        self._local_mode = local_mode
        self._with_reward = with_reward
        self._device = device
        self._action_dim = action_dim
        self._state_dim = state_dim
        self._layers = []
        hidden_sizes = [state_dim+self._action_dim] + list(fc_layer_params)
        for in_dim, out_dim in zip(hidden_sizes[:-1], hidden_sizes[1:]):
            self._layers += miniblock(in_dim, out_dim, None, nn.ReLU)
        output_dim = (self._state_dim + with_reward) * 2
        self._layers += [nn.Linear(hidden_sizes[-1], output_dim)]
        self._model = nn.Sequential(*self._layers)
        # logstd bounds
        init_max = torch.empty(1, state_dim + with_reward, dtype=torch.float32).fill_(2.0)
        init_min = torch.empty(1, state_dim + with_reward, dtype=torch.float32).fill_(-10.0)
        self._max_logstd = nn.Parameter(init_max)
        self._min_logstd = nn.Parameter(init_min)

    def _get_output(
            self,
            state: Union[np.ndarray, torch.Tensor],
            action: Union[np.ndarray, torch.Tensor],
    ) -> Tuple[Distribution, torch.Tensor]:
        state = torch.as_tensor(state, device=self._device, dtype=torch.float32)
        action = torch.as_tensor(action, device=self._device, dtype=torch.float32)
        h = torch.cat([state, action], 1)
        h = self._model(h)
        mean, log_std = torch.split(h, split_size_or_sections=self._state_dim + self._with_reward, dim=-1)
        log_std = self._max_logstd - F.softplus(self._max_logstd - log_std)
        log_std = self._min_logstd + F.softplus(log_std - self._min_logstd)
        std = torch.exp(log_std)
        if self._local_mode:
            if self._with_reward:
                s_p, reward = torch.split(mean, [self._state_dim, 1], dim=-1)
                s_p = s_p + state
                mean = torch.cat([s_p, reward], -1)
            else:
                mean = mean + state
        dist = Normal(loc=mean, scale=std)
        return dist, mean

    def forward(
            self,
            state: Union[np.ndarray, torch.Tensor],
            action: Union[np.ndarray, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Distribution]:
        dist, mean = self._get_output(state, action)
        sample = dist.rsample()
        return mean, sample, dist

    def get_log_density(
            self,
            state: Tensor,
            action: Tensor,
            output: Tensor
    ) -> Tensor:
        assert output.shape[-1] == self._state_dim + self._with_reward, 'Wrong target dimension!'
        dist, _ = self._get_output(state, action)
        output = torch.as_tensor(output, dtype=torch.float32, device=self._device)
        log_density = dist.log_prob(output)
        return log_density

    @property
    def max_logstd(self):
        return self._max_logstd

    @property
    def min_logstd(self):
        return self._min_logstd


class CriticNetwork(nn.Module):
    """Critic Network.

    :param gym.spaces.Box or int observation_space: the observation space information. It is an instance
        of class: ``gym.spaces.Box``. `observation_space` can also be an integer which
        represents the dimension of the observation.
    :param gym.spaces.Box or int action_space: the action space information. It is an instance
        of class: ``gym.spaces.Box``. `action_space` can also be an integer which
        represents the dimension of the action.
    :param tuple fc_layer_params: the network parameter. For example:
        ``(300, 300)`` means a 2-layer network with 300 units in each layer.
    :param device: which device to create this model on. Default to 'cpu'.
    """

    def __init__(
            self,
            observation_space: Union[Box, Space, int],
            action_space: Union[Box, Space, int],
            fc_layer_params: Sequence[int] = (),
            device: Union[str, int, torch.device] = 'cpu',
    ) -> None:
        super(CriticNetwork, self).__init__()
        self._device = device
        if isinstance(observation_space, int):
            state_dim = observation_space
        else:
            state_dim = observation_space.shape[0]
        if isinstance(action_space, int):
            action_dim = action_space
        else:
            action_dim = action_space.shape[0]
        self._layers = []
        hidden_sizes = [state_dim + action_dim] + list(fc_layer_params)
        for in_dim, out_dim in zip(hidden_sizes[:-1], hidden_sizes[1:]):
            self._layers += miniblock(in_dim, out_dim, None, nn.ReLU)
        self._layers += [nn.Linear(hidden_sizes[-1], 1)]
        self._model = nn.Sequential(*self._layers)

    def forward(
            self,
            state: Union[np.ndarray, Tensor],
            action: Union[np.ndarray, Tensor]
    ) -> Tensor:
        state = torch.as_tensor(state, device=self._device, dtype=torch.float32)
        action = torch.as_tensor(action, device=self._device, dtype=torch.float32)
        h = torch.cat([state, action], dim=-1)
        h = self._model(h)
        return torch.reshape(h, [-1])


class MLP(nn.Module):
    """Multi-layer Perceptron.

    :param int input_dim: the dimension of the input.
    :param int output_dim: the dimension of the output.
    :param tuple fc_layer_params: the network parameter. For example:
        ``(300, 300)`` means a 2-layer network with 300 units in each layer.
    :param device: which device to create this model on. Default to 'cpu'.
    """
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            fc_layer_params: Sequence[int] = (),
            device: Union[str, int, torch.device] = 'cpu',
    ) -> None:
        super(MLP, self).__init__()
        self._device = device
        self._layers = []
        hidden_sizes = [input_dim] + list(fc_layer_params)
        for in_dim, out_dim in zip(hidden_sizes[:-1], hidden_sizes[1:]):
            self._layers += miniblock(in_dim, out_dim, None, nn.ReLU)
        self._layers += [nn.Linear(hidden_sizes[-1], output_dim)]
        self._model = nn.Sequential(*self._layers)

    def forward(self, inputs: Union[np.ndarray, Tensor]) -> Tensor:
        inputs = torch.as_tensor(inputs, device=self._device, dtype=torch.float32)
        return self._model(inputs)


class Classifier(nn.Module):
    """ based on Multi-layer Perceptron. Discriminator network for H2O.

    :param int input_dim: the dimension of the input.
    :param int output_dim: the dimension of the output.
    :param tuple fc_layer_params: the network parameter. For example:
        ``(300, 300)`` means a 2-layer network with 300 units in each layer.
    :param device: which device to create this model on. Default to 'cpu'.
    """
    def __init__(
            self,
            input_dim: int,
            output_dim: int = 2,
            fc_layer_params: Sequence[int] = (),
            device: Union[str, int, torch.device] = 'cpu',
    ) -> None:
        super(Classifier, self).__init__()
        self._device = device
        self._layers = []
        hidden_sizes = [input_dim] + list(fc_layer_params)
        for in_dim, out_dim in zip(hidden_sizes[:-1], hidden_sizes[1:]):
            self._layers += miniblock(in_dim, out_dim, None, nn.ReLU)
        self._layers += miniblock(hidden_sizes[-1], output_dim, None, nn.Tanh)
        self._model = nn.Sequential(*self._layers)

    def forward(self, inputs: Union[np.ndarray, Tensor]) -> Tensor:
        inputs = torch.as_tensor(inputs, device=self._device, dtype=torch.float32)
        return self._model(inputs) * 2


class ConcatClassifier(Classifier):
    """Concatenate inputs along dimension and then pass through MLP.

    :param int dim: concatenate inputs in row or column (0 or 1).
    """
    def __init__(
            self,
            dim: int = 1,
            *args,
            **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.dim = dim

    def forward(self, *inputs: Union[np.ndarray, Tensor]) -> Tensor:
        flat_inputs = torch.cat(inputs, dim=self.dim)
        return super().forward(flat_inputs)


class Scalar(nn.Module):
    """ Scalar network

    :param float init_value: initialized value for the scalar
    """
    def __init__(
        self, 
        init_value: float,
        device: Union[str, int, torch.device] = 'cpu'
    ) -> None:
        super().__init__()
        self._device = device
        self.constant = nn.Parameter(
            torch.tensor(init_value, dtype=torch.float32).to(self._device)
        )

    def forward(self) -> Tensor:
        return self.constant


class Discriminator(nn.Module):
    """A Discriminator Network(for DMIL).

    :param Box observation_space: the observation space information. It is an instance
        of class: ``gym.spaces.Box``.
    :param Box action_space: the action space information. It is an instance
        of class: ``gym.spaces.Box``.
    :param tuple fc_layer_params: the network parameter. For example:
        ``(300, 300)`` means a 2-layer network with 300 units in each layer.
    :param device: which device to create this model on. Default to 'cpu'.
    """

    def __init__(
            self,
            observation_space: Union[Box, Space],
            action_space: Union[Box, Space],
            fc_layer_params: Sequence[int] = (),
            device: Union[str, int, torch.device] = 'cpu',
    ) -> None:
        super(Discriminator, self).__init__()
        self._device = device
        state_dim = observation_space.shape[0]
        action_dim = action_space.shape[0]
        self._layers = []
        hidden_sizes = [2 * state_dim + 2 * action_dim] + list(fc_layer_params)
        for in_dim, out_dim in zip(hidden_sizes[:-1], hidden_sizes[1:]):
            self._layers += miniblock(in_dim, out_dim, None, nn.ReLU)
        self._layers += miniblock(hidden_sizes[-1], 1, None, nn.Sigmoid)
        self._model = nn.Sequential(*self._layers)

    def forward(
            self,
            state: Union[np.ndarray, Tensor],
            action: Union[np.ndarray, Tensor],
            logpi: Union[np.ndarray, Tensor],
            lossf: Union[np.ndarray, Tensor]
    ) -> Tensor:
        state = torch.as_tensor(state, device=self._device, dtype=torch.float32)
        action = torch.as_tensor(action, device=self._device, dtype=torch.float32)
        logpi = torch.as_tensor(logpi, device=self._device, dtype=torch.float32)
        lossf = torch.as_tensor(lossf, device=self._device, dtype=torch.float32)
        h = torch.cat([state, action, logpi, lossf], dim=-1)
        h = self._model(h)
        return torch.reshape(h, [-1])


class ValueNetwork(nn.Module):
    """Value Network.

    :param gym.spaces.Box or int observation_space: The observation space information. It is an instance
        of class: ``gym.spaces.Box``. It can also be an integer which represents the dimension of the observation.
    :param tuple fc_layer_params: the network parameter. For example:
        ``(300, 300)`` means a 2-layer network with 300 units in each layer.
    :param device: which device to create this model on. Default to 'cpu'.
    """

    def __init__(
            self,
            observation_space: Union[Box, Space, int],
            fc_layer_params: Sequence[int] = (),
            device: Union[str, int, torch.device] = 'cpu',
    ) -> None:
        super(ValueNetwork, self).__init__()
        self._device = device
        if isinstance(observation_space, int):
            state_dim = observation_space
        else:
            state_dim = observation_space.shape[0]
        output_dim = 1
        self._layers = []
        hidden_sizes = [state_dim] + list(fc_layer_params)
        for in_dim, out_dim in zip(hidden_sizes[:-1], hidden_sizes[1:]):
            self._layers += miniblock(in_dim, out_dim, None, nn.ReLU)
        self._layers += [nn.Linear(hidden_sizes[-1], output_dim)]
        self._model = nn.Sequential(*self._layers)

    def forward(self, inputs: Union[np.ndarray, Tensor]) -> Tensor:
        inputs = torch.as_tensor(inputs, device=self._device, dtype=torch.float32)
        h = self._model(inputs)
        return torch.reshape(h, [-1])