import torch
import numpy as np
from typing import Any, ClassVar, Dict, Optional, Type
from abc import ABC, abstractmethod


class Scaler(ABC):

    TYPE: ClassVar[str] = 'none'

    @abstractmethod
    def fit(self, data: np.ndarray) -> None:
        """Estimating the scaling parameters from the data.

        :param data: The data for estimating the scaling parameters.
        """
        pass

    @abstractmethod
    def transform(self, x: np.ndarray) -> np.ndarray:
        """Returning the processed data.

        :param x: The data to be scaled.
        :return: The scaled data.
        """
        pass

    @abstractmethod
    def reverse_transform(self, x: np.ndarray) -> np.ndarray:
        """Returning the reversely transformed data.

        :param x: The data to be reversely transformed.
        :return: The reversely scaled data.
        """
        pass

    @abstractmethod
    def transform_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """Returning the processed data in torch Tensor.

        :param x: The data to be scaled.
        :return: The scaled data.
        """
        pass

    @abstractmethod
    def reverse_transform_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """Returning the reversely transformed data in torch Tensor.

        :param x: The data to be reversely transformed.
        :return: The reversely scaled data.
        """
        pass

    def get_type(self) -> str:
        """Returning the scaler type.

        :return: Scaler type.
        """
        return self.TYPE

    @abstractmethod
    def get_params(self, deep: bool = False) -> Dict[str, Any]:
        """Returning the scaling parameters.

        :param deep: Flag to deeply copy the objects.
        :return: Scaler parameters.
        """
        pass


class MinMaxScaler(Scaler):
    """Min-Max normalization processing.

    .. math::

        x' = (x - \min{x}) / (\max{x} - \min{x})

    .. code-block:: python

        # initialize with dataset
        scaler = MinMaxScaler(dataset)

        # initialize manually
        minimum = observations.min(axis=0)
        maximum = observations.max(axis=0)
        scaler = MinMaxScaler(minimum=minimum, maximum=maximum)

    :param np.ndarray data: The data to be processed.
    :param np.ndarray minimum: The minimum values for each feature.
    :param np.ndarray maximum: The maximum values for each feature.
    """

    TYPE: ClassVar[str] = 'min_max'
    _minimum: Optional[np.ndarray]
    _maximum: Optional[np.ndarray]

    def __init__(
            self,
            data: Optional[np.ndarray] = None,
            minimum: Optional[np.ndarray] = None,
            maximum: Optional[np.ndarray] = None,
    ):
        self._minimum = None
        self._maximum = None
        if data is not None:
            self.fit(data)
        elif minimum is not None and maximum is not None:
            self._minimum = np.asarray(minimum)
            self._maximum = np.asarray(maximum)

    def fit(self, data: np.ndarray) -> None:
        if self._minimum is not None and self._maximum is not None:
            return

        self._minimum = np.min(data, axis=0)
        self._maximum = np.max(data, axis=0)

    def transform(self, x: np.ndarray) -> np.ndarray:
        assert self._minimum is not None and self._maximum is not None
        return (x - self._minimum) / (self._maximum - self._minimum)

    def reverse_transform(self, x: np.ndarray) -> np.ndarray:
        assert self._minimum is not None and self._maximum is not None
        return x * (self._maximum - self._minimum) + self._minimum

    def transform_tensor(self, x: torch.Tensor) -> torch.Tensor:
        assert self._minimum is not None and self._maximum is not None
        minimum = torch.tensor(
            self._minimum,
            dtype=torch.float32,
            device=x.device,
        )
        maximum = torch.tensor(
            self._maximum,
            dtype=torch.float32,
            device=x.device,
        )
        return (x - minimum) / (maximum - minimum)

    def reverse_transform_tensor(self, x: torch.Tensor) -> torch.Tensor:
        assert self._minimum is not None and self._maximum is not None
        minimum = torch.tensor(
            self._minimum,
            dtype=torch.float32,
            device=x.device,
        )
        maximum = torch.tensor(
            self._maximum,
            dtype=torch.float32,
            device=x.device,
        )
        return x * (maximum - minimum) + minimum

    def get_params(self, deep: bool = False) -> Dict[str, Any]:
        if self._minimum is not None:
            minimum = self._minimum.copy() if deep else self._minimum
        else:
            minimum = None

        if self._maximum is not None:
            maximum = self._maximum.copy() if deep else self._maximum
        else:
            maximum = None

        return {'minimum': minimum, 'maximum': maximum}


class StandardScaler(Scaler):
    """Standardization processing.

    .. math::

        x' = (x - \mu) / \sigma

    .. code-block:: python

        # initialize with dataset
        scaler = StandardScaler(dataset)

        # initialize manually
        mean = observations.mean(axis=0)
        std = observations.std(axis=0)
        scaler = StandardScaler(mean=mean, std=std)

    :param
    """

    TYPE: ClassVar[str] = 'standard'
    _mean: Optional[np.ndarray]
    _std: Optional[np.ndarray]
    _eps: float

    def __init__(
            self,
            data: Optional[np.ndarray] = None,
            mean: Optional[np.ndarray] = None,
            std: Optional[np.ndarray] = None,
            eps: float = 1e-3,
    ) -> None:
        self._mean = None
        self._std = None
        self._eps = eps
        if data is not None:
            self.fit(data)
        elif mean is not None and std is not None:
            self._mean = np.asarray(mean)
            self._std = np.asarray(std)

    def fit(self, data: np.ndarray) -> None:
        if self._mean is not None and self._std is not None:
            return

        self._mean = np.mean(data, axis=0)
        self._std = np.std(data, axis=0)

    def transform(self, x: np.ndarray) -> np.ndarray:
        assert self._mean is not None and self._std is not None
        return (x - self._mean) / (self._std + self._eps)

    def reverse_transform(self, x: np.ndarray) -> np.ndarray:
        assert self._mean is not None and self._std is not None
        return x * (self._std + self._eps) + self._mean

    def transform_tensor(self, x: torch.Tensor) -> torch.Tensor:
        assert self._mean is not None and self._std is not None
        mean = torch.tensor(self._mean, dtype=torch.float32, device=x.device)
        std = torch.tensor(self._std, dtype=torch.float32, device=x.device)
        return (x - mean) / (std + self._eps)

    def reverse_transform_tensor(self, x: torch.Tensor) -> torch.Tensor:
        assert self._mean is not None and self._std is not None
        mean = torch.tensor(self._mean, dtype=torch.float32, device=x.device)
        std = torch.tensor(self._std, dtype=torch.float32, device=x.device)
        return x * (std + self._eps) + mean

    def get_params(self, deep: bool = False) -> Dict[str, Any]:
        if self._mean is not None:
            mean = self._mean.copy() if deep else self._mean
        else:
            mean = None

        if self._std is not None:
            std = self._std.copy() if deep else self._std
        else:
            std = None

        return {"mean": mean, "std": std, "eps": self._eps}


SCALER_LIST: Dict[str, Type[Scaler]] = {}


def register_scaler(cls: Type[Scaler]) -> None:
    """Registering the scaler class.

    :param cls: Scaler class inheriting ``Scaler``.
    """
    is_registered = cls.TYPE in SCALER_LIST
    assert not is_registered, f'{cls.TYPE} seems to be already registered.'
    SCALER_LIST[cls.TYPE] = cls


def create_scaler(name: str, **kwargs: Any) -> Scaler:
    """Creating a registered scaler object.

    :param name: The name of the registered scaler type.
    :param kwargs: The scaler arguments.
    :return: A scaler object.
    """
    assert name in SCALER_LIST, f'{name} seems not to be registered.'
    scaler = SCALER_LIST[name](**kwargs)
    assert isinstance(scaler, Scaler)
    return scaler


register_scaler(MinMaxScaler)
register_scaler(StandardScaler)

