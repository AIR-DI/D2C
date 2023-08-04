Create Dataset
=================


:class:`~d2c.data.data.Data` uses the Dataloader to load and parse the raw offline data and obtain the transition data containing (observation, action, reward, done), and then stores the obtained transition into the ReplayBuffer, and finally obtains the Dataset required for offline RL training. The base class of Data, :class:`~d2c.data.data.BaseData`, contains the following parts:

- :meth:`~d2c.data.data.BaseData.__init__`: Contains some initialization information, such as data file path, dataloader attributes, Buffer size, etc.;

- :meth:`~d2c.data.data.BaseData._build_data_loader`: Construct DataLoader;

- :meth:`~d2c.data.data.BaseData._build_data`: Add the transitions obtained from Dataloader to Buffer to obtain Dataset;

- :meth:`~d2c.data.data.BaseData.data`: Call this interface to return the final Replay Buffer, which is the constructed dataset.


Data
------------------
:class:`~d2c.data.data.Data` inherits from :class:`~d2c.data.data.BaseData` and can use different Dataloaders to load offline data from different sources and application types to construct the dataset required for training. Here are the introductions of each part:

- :meth:`~d2c.data.data.Data.__init__`: Based on the parameter information in the configuration file, determine the type of Dataloader;

- :meth:`~d2c.data.data.Data._app_data_loader`: Construct a Dataloader for real-world scenario data;

- :meth:`~d2c.data.data.Data._d4rl_data_loader`: Construct a Dataloader for the D4RL dataset;

- :meth:`~d2c.data.data.Data._data_loader_list`: Put the above-mentioned Dataloaders into a dictionary;

In addition to the two Dataloaders mentioned above, you can also add custom Dataloaders for other benchmark datasets, just add a new dataloader construction method in Data, and add the new method to the dictionary in :meth:`~d2c.data.data.Data._data_loader_list`.

In addition, you can also customize other Data according to your needs to achieve customized requirements, such as :class:`~d2c.data.data.DataNoise` inherits from :class:`~d2c.data.data.Data` and re-implements the :meth:`~d2c.data.data._build_data` method to construct a Dataset that adds noise to the action. Another example is :class:`~d2c.data.data.DataMix`, which inherits from :class:`~d2c.data.data.BaseData` and implements a dataset that mixes multiple data sources.


Customize Dataloader
----------------------
By inheriting from :class:`~d2c.utils.dataloader.BaseDataLoader`, different types of Dataloaders can be implemented to import offline data from different sources. The BaseDataLoader mainly contains the following parts:

- :meth:`~d2c.utils.dataloader.BaseDataLoader._load_data`: Load data from the raw data file and return transitions elements;

- :meth:`~d2c.utils.dataloader.BaseDataLoader.get_transitions`: Process data and generate transitions;

- :meth:`~d2c.utils.dataloader.BaseDataLoader.state_shift_scale`: Get the shift and scale of the state normalization.

When customizing the Dataloader for benchmark datasets, you can inherit from the base class :class:`~d2c.utils.dataloader.BaseBMLoader` and implement :meth:`~d2c.utils.dataloader.BaseBMLoader._load_data` to construct Dataloaders for different benchmark datasets. :class:`~d2c.utils.dataloader.D4rlDataLoader` is a Dataloader constructed for the D4RL dataset, which can be used as a reference.

For offline data from real-world scenarios, they are usually saved as ``.csv`` files. ``AppDataLoader`` is specially designed to load offline datasets from real-world scenarios.