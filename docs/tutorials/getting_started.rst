Getting Started
===============

Install
-------

You can install ``D2C`` via GitHub repository::

  $ git clone https://github.com/AIR-DI/D2C.git
  $ cd d2c
  $ pip install -e .

.. note::

  ``D2C`` supports Python 3.7+. Make sure which version you use.

.. note::

  If you use GPU, please setup CUDA first.

.. note::

  If you use MuJoCo environment, please install mujoco correctly('mujoco-py==2.1.2.14').

Preparation
---------------

Here, we train a offline RL algorithm TD3+BC based on the mujoco dataset from D4RL.
First, you should construct a working folder as structure below::

  working_folder/
    |- config/
      |- __init__.py
      |- app_config.py
      |- model_config.json5
    |- data/
      |- d4rl/
        |- mujoco/
          |- __init__.py
          |- ...

In following example, we use the folder ``example/benchmark`` in d2c repository as the working folder.

Prepare Data
----------------

You can download the mujoco dataset from D4RL_ and place the data files in ``/benchmark/data/d4rl/mujoco/``.

.. _D4RL: http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/?spm=a2cl9.codeup_devops2020_goldlog_projectFiles.0.0.2658334bxqxjOf

Configuration
---------------

There are two configuration files in ``benchmark/config``:

- ``app_config.py``: The configuration for customize application(like the real-world application). See more documents at :doc:`./configuration`.

- ``model_config.json5``: The summary configuration for the RL algorithms, environment, trainer and other modules.

We construct a configuration object for all components in the workflow.

.. code-block:: python

  from example.benchmark.config import make_config
  command_args = {
        'model.model_name': 'td3_bc',
        'env.external.benchmark_name': 'd4rl',
        'env.external.data_source': 'mujoco',
        'env.external.env_name': 'Hopper-v2',
        'env.external.data_name': 'hopper_medium_expert-v2',
        'train.device': 'cuda',
        'train.total_train_steps': 1000000,
        'train.batch_size': 256,
        'train.agent_ckpt_name': 'xxxx'
    }
  config = make_config(command_args)

Most of the hyper-parameters have been set up well in the configuration file. You can also modify the hyper-parameters like above.
The keys in the dict ``command_args`` correspond to the full index names of the hyper-parameters in the configuration file ``model_config.json5``.

Make the Dataset
-----------------

In the ``config`` we build above, we set up the choice of the dataset which is ``hopper_medium_expert-v2``.
We build the data object as below:

.. code-block:: python

  from d2c.data import Data
  bm_data = Data(config)
  s_norm = dict(zip(['obs_shift', 'obs_scale'], bm_data.state_shift_scale))
  data = bm_data.data

Make Environments
-------------------

Here, we setup the mujoco env for policy evaluation.

.. code-block:: python

  from d2c.envs import benchmark_env, LeaEnv
  # The env of the benchmark to be used for policy evaluation.
  env = benchmark_env(config=config, **s_norm)
  # Contains dynamics model to be trained.
  lea_env = LeaEnv(config)

``lea_env`` is a dummy env which contains dynamics model to be trained(if needed).
Besides, it can provides some information of the environment like ``observation_space`` and ``action_space``.

Setup Algorithm
----------------
There are many offline RL algorithms available in D2C. In ``config``, we have setup the algorithm named ``td3+bc``.
Setup the agent and the evaluator:

.. code-block:: python

  from d2c.models import make_agent
  from d2c.evaluators import bm_eval
  agent = make_agent(config=config, env=lea_env, data=data)
  evaluator = bm_eval(agent=agent, env=env, config=config)

Start Training
---------------

Now, you can setup the :class:`~d2c.trainers.Trainer` and start data-driven training.

.. code-block:: python

  from d2c.trainers import Trainer
  trainer = Trainer(agent=agent, train_data=data, config=config, env=lea_env, evaluator=evaluator)
  trainer.train()

Off-policy Evaluation
-----------------------

D2C provides several off-policy evaluation methods. You can use fitted Q evaluation when the agent has been trained.

.. code-block:: python

  from d2c.evaluators import make_ope
  fqe = make_ope('fqe', from_config=True, agent=agent, data=data, config=config)
  fqe.eval()

Save and Load
--------------

D2C saves the models in training procedure automatically.
You can load a trained agent like this:

.. code-block:: python

  agent = make_agent(config=config, env=lea_env, data=data, restore_agent=True)

