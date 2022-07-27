"""Loading data and constructing the buffer."""

from abc import ABC, abstractmethod
from absl import logging
from typing import Dict, Callable
from d2c.utils.replaybuffer import ReplayBuffer
from d2c.utils.dataloader import AppDataLoader, D4rlDataLoader, BaseDataLoader


class BaseData(ABC):
    """The basic class of the replay buffer. Inherit this class to build
    data loaders for practical application and benchmark experiments.

    The API methods the user should implement are:

    * :meth:`_build_data`: Construct the data through adding the transitions into the replay buffer.
    * :meth:`_build_data_loader`: Construct the data loader.

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
        """Construct the data through adding the transitions into the replay buffer."""
        pass

    @abstractmethod
    def _build_data_loader(self) -> None:
        """Construct the data loader."""
        pass

    @property
    def data(self) -> ReplayBuffer:
        """Return the replay buffer."""
        return self._data


class Data(BaseData):
    """Create the replay buffer with batch data that is from the benchmarks or real-world applications dataset.
    
    It uses different data loaders to get the transitions and adds them into the replay buffer. This class 
    generates the dataset for the offline RL training.
    
    The main methods that users of this class need to know are:

    * :meth:`data`: Return the constructed replay buffer;
    * :meth:`_build_data_loader`: Read the batch data and create a data loader. \
        It builds different data loader according to the corresponding parameters \
        in the configuration.
    * :meth:`_build_data`: Create a replay buffer and add the data into it.

    .. note::

        To add a data loader for a new benchmark, you should implement a new method
        `_xxx_data_loader`(like :meth:`_app_data_loader`, :meth:`_d4rl_data_loader`)
        and add this method into :meth:`_data_loader_list`.

    :param config: the configuration.
    """

    def __init__(self, config) -> None:
        app_config = config.app_config
        env_config = config.model_config.env.env_external
        data_loader_name = config.model_config.train.data_loader_name
        if data_loader_name is 'app':  # For real-world application experiments.
            try:
                data_path = app_config.data_path
            except:
                raise AttributeError('There is no data file for real-world application experiments！')
        elif env_config.data_file_path is not None:
            data_path = env_config.data_file_path
            data_loader_name = env_config.benchmark_name
        else:
            raise ValueError('There is no data file to load!')
        assert isinstance(data_path, str)
        logging.info(f'Loading the offline dataset file from {data_path}.')
        if data_loader_name not in self._data_loader_list.keys():
            raise NotImplementedError(f'There is no data loader for {data_loader_name}!')
        self._app_cfg = app_config
        self._env_cfg = env_config
        self._data_loader_name = data_loader_name
        super(Data, self).__init__(data_path)

    def _build_data(self) -> None:
        self._build_data_loader()
        transitions, self._shift, self._scale = self._data_loader.get_transitions()
        s1, a1, s2, a2, reward, cost, done = [transitions[k] for k in transitions.keys()]
        self._buffer_size = s1.shape[0]
        state_dim = s1.shape[-1]
        action_dim = a1.shape[-1]
        self._data = ReplayBuffer(
            state_dim,
            action_dim,
            self._buffer_size
        )
        self._data.add_transitions(s1, a1, s2, a2, reward, cost, done)

    def _build_data_loader(self) -> None:
        self._data_loader = self._data_loader_list[self._data_loader_name]()

    def _app_data_loader(self):
        raise NotImplementedError

    def _d4rl_data_loader(self) -> D4rlDataLoader:
        state_normalize = self._env_cfg.state_normalize
        scale_rewards = self._env_cfg.scale_rewards
        return D4rlDataLoader(
            self._data_path,
            state_normalize,
            scale_rewards,
        )

    @property
    def _data_loader_list(self) -> Dict[str, Callable[..., BaseDataLoader]]:
        """A dict of the alternative Data loaders."""
        return dict(
            app=self._app_data_loader,
            d4rl=self._d4rl_data_loader,
        )





