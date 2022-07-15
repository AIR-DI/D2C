"""Loading data and constructing the buffer."""

from abc import ABC, abstractmethod


class BaseData(ABC):
    """The basic class of the replay buffer. Inherit this class to build
    data loaders for practical application and benchmark experiments.

    The API methods the user should implement are:

    * :meth:`_build_data`:
    * :meth:`_build_data_loader`:

    :param str data_path: the file path of the dataset.

    """

    def __init__(
            self,
            data_path: str,
    ) -> None:
        self._data_path = data_path
        self._data_loader = None
        self._data = None
        self._buffer_size = None
        self._build_data()

    @abstractmethod
    def _build_data(self) -> None:
        pass

    @abstractmethod
    def _build_data_loader(self) -> None:
        pass

    @property
    def data(self) —> :
        """Return the replay buffer."""
        return self._data
