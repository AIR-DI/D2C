Configuration
=================
All the hyperparameters involved in this project are managed uniformly using the config object. When instantiating objects such as ``Data``, ``Env``, ``Agent``, ``Evaluator``, ``Trainer``, etc., the config object is used as the parameter input. The class :class:`~d2c.utils.config.ConfigBuilder` is used to construct the config, and the required parameters include:

- ``app_config``: Contains the configuration information related to the real-world scenario application, which needs to be configured when performing real-world scenario application experiments;

- ``model_config_path``: The file path of ``model_config.json5``. This json file organizes and manages all the hyperparameters involved in the algorithm model in a key-value pair form, which will be introduced in detail later;

- ``work_abs_dir``: The absolute path of the working folder, such as the folder ``example/benchmark/`` in this project, which contains executable scripts, data folders, model folders, etc.;

- ``command_args``: A dictionary that can be constructed based on command line parameter input to modify the parameters in the ``model_config.json5`` file. The key in this dictionary should match the index path of the relevant parameter in ``model_config.json5``, such as:

::

    prefix = 'env.external.'
    command_args = {
        prefix + 'benchmark_name': 'd4rl',
        prefix + 'data_source': 'mujoco',
        prefix + 'env_name': 'Hopper-v2',
        prefix + 'data_name': 'hopper_medium_expert-v2',
        prefix + 'state_normalize': True,
        prefix + 'score_normalize': True,
    }
    command_args.update({
        'model.model_name': 'td3_bc',
        'train.data_loader_name': None,
        'train.device': device,
        'train.seed': 20,
        'train.total_train_steps': 1000,
        'train.batch_size': 256,
        'train.agent_ckpt_name': '230221-train_1k',
        'train.dynamics_ckpt_name': 'dyna-train_1m',
    })

- ``experiment_type``: This parameter has two choices 'benchmark', 'application'. 'benchmark' means that the experiment is a benchmark dataset experiment, and 'application' means that the experiment is a real-world scenario application experiment.


Usage
-----------
The usage of the Config object is as follows:

We use the parameter ``command_args`` what mentioned above to build the config.

::

    import os
    from d2c.utils.config import ConfigBuilder
    from d2c.utils.utils import abs_file_path
    from example.benchmark.config.app_config import app_config

    # 'work_abs_dir' is the absolute path to the working folder
    model_config_path = os.path.join(work_abs_dir, 'config', 'model_config.json5')
    cfg_builder = ConfigBuilder(
        app_config=app_config,
        model_config_path=model_config_path,
        work_abs_dir=work_abs_dir,
        command_args=command_args,
    )
    config = cfg_builder.build_config()

Then we build other objects based on this ``config``.

- Constructing dataset:

::

    from d2c.data import Data
    bm_data = Data(config)
    s_norm = dict(zip(['obs_shift', 'obs_scale'], bm_data.state_shift_scale))
    data = bm_data.data

- Constructing Env:

::

    from d2c.envs import benchmark_env, LeaEnv
    # The env of the benchmark to be used for policy evaluation.
    env = benchmark_env(config=config, **s_norm)
    # The learned env that contains dynamics model.
    lea_env = LeaEnv(config)

- Constructing agent and trainer:

::

    from d2c.models import make_agent
    from d2c.evaluators import bm_eval
    from d2c.trainers import Trainer
    agent = make_agent(config=config, env=lea_env, data=data)
    evaluator = bm_eval(agent=agent, env=env, config=config)
    trainer = Trainer(agent=agent, train_data=data, config=config, env=lea_env, evaluator=evaluator)
    trainer.train()

- Constructing evaluator for OPE:

::

    from d2c.evaluators import make_ope
    agent = make_agent(config=config, env=lea_env, data=data, restore_agent=True)
    fqe = make_ope('fqe', from_config=True, agent=agent, data=data, config=config)
    fqe.eval()
    lea_env.load()
    mb_ope = make_ope('mb_ope', from_config=True, agent=agent, data=data, env=lea_env, config=config)
    mb_ope.eval()


Model config
-------------------
``model_config`` is an important part of the Config object. It exists in the form of the file ``model_config.json5``, which contains all the hyperparameters related to the algorithm model. For an example of the file, please refer to the project file ``example/benchmark/config/model_config.json5``. In model_config, it mainly contains the following parts of content:

model
*********************
Hyperparameters related to the RL algorithm.

- ``model_name`` indicates the selected RL algorithm, here we take the ``td3+bc`` algorithm as an example.

- ``train_schedule`` indicates the process of model training, ``['agent']`` means only training the RL agent, while ``['d', 'b', 'q', 'agent']`` means training dynamics, behavior, Q separately first, and then training the agent at last.

- ``hyper_params`` are used for algorithm initialization, which are the parameters passed in when instantiating :class:`~d2c.models.model_free.td3_bc.TD3BCAgent`, and can be added or deleted as needed.

::

    td3_bc: {
      train_schedule: ['agent'],
      hyper_params: {
        model_params: {q: [[256, 256], 2], p: [[256, 256],]},
        optimizers: {q: ['adam', 3e-4], p: ['adam', 3e-4]}
      }

env
**********************
Hyperparameters related to Env.

- ``basic_info`` indicates the basic information of the environment, including the dimensions of observation and action, and the upper and lower bounds of each dimension. Using the d4rl mujoco dataset as an example, when the ``basic_info`` information is not provided here, :class:`~d2c.utils.config.ConfigBuilder` will use the predefined environment information in the file ``example/data/d4rl/mujoco/__init__.py`` under the dataset folder to set the ``basic_info`` when constructing the config.

- ``external`` contains the hyperparameters related to external env, including the name of the benchmark, the name of the environment, the name of the offline dataset, and other information.

- ``learned`` contains the hyperparameters related to learned env, including the type of the dynamics model, and the hyperparameters of the model.

train
************************
Hyperparameters related to Trainer.

- This section contains the save path of the model files in the algorithm. ``agent_ckpt_dir`` indicates the save path of the RL Agent. If this is not given, it will be automatically generated in :meth:`~d2c.utils.config.ConfigBuilder._update_model_dir` of :class:`~d2c.utils.config.ConfigBuilder`. For other model file save paths, please refer to :meth:`~d2c.utils.config.ConfigBuilder._update_model_dir` for automatic generation.

- ``wandb`` contains parameters that can be customized when using wandb logger. If you want to use wandb logger, please set the corresponding parameters here.

eval
************************
Hyperparameters related to Evaluators.


Using CLI to modify parameters
----------------------------------
You can also modify the parameters in the command line. Here we use ``example/benchmark/demo.py`` file as an example. Here we use the fire package, Python Fire is a library for creating CLIs from absolutely any Python object.

::

    python demo.py
        --train.agent_ckpt_name='221228'
        --model.model_name='bc'
        --train.batch_size=256
        --env.external.benchmark_name=$BM_NAME
        --env.external.env_name=$ENV
        --env.external.data_name=$DATA
        --env.external.data_source=$DATA_SOURCE
        --env.external.state_normalize=True
        --env.external.score_normalize=True
        --train.total_train_steps=$TRAIN_STEPS
        --train.seed=$SEED
        --train.wandb.entity='d2c'
        --train.wandb.project='test_bc'
        --train.wandb.name='bc-'$DATA'-seed'$SEED

The name of the parameter to be modified in the command line should correspond to the full index key of the corresponding parameter in ``model_config`` file.
This way you can modify any parameter in ``model_config`` when training the model. For more information, please refer to the file ``example/benchmark/run.sh``.