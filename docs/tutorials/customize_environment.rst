Customize Environment
=======================
Env are built based on :class:`~d2c.envs.BaseEnv`, which has an interface consistent with gym Env. It mainly includes:

- :meth:`~d2c.envs.BaseEnv.__init__`: Set the environment's observation and action information;

- :meth:`~d2c.envs.BaseEnv._set_action_space`: Set the action space information;

- :meth:`~d2c.envs.BaseEnv._set_observation_space`: Set the observation space information;

- :meth:`~d2c.envs.BaseEnv.step`: Run the environment dynamics model for one time step;

- :meth:`~d2c.envs.BaseEnv.reset`: Reset the environment to an initial state and return an initial observation;

- :meth:`~d2c.envs.BaseEnv._load_model`: Load the dynamics model, which may be a trained neural network-based dynamics model, or a ready-made dynamics model obtained from the outside world, depending on the specific situation.

Env are mainly divided into two categories: :ref:`learned env <learned-env-label>`  and :ref:`external env <external-env-label>`. The former constructs env based on the offline trained dynamics model, while the latter constructs env based on the accessed ready-made simulation environment or dynamics model from the outside world. External env can customize different environments for different application scenarios. Here are some further introductions.


Learned Env
-----------------------
The core of :class:`~d2c.envs.LeaEnv` is a dynamics model of the current task environment, which is usually a neural network model trained based on offline data. LeaEnv is generally mainly used for model-based RL training and offline policy evaluation. Usually when using LeaEnv, the corresponding dynamics model needs to be trained in advance. If the LeaEnv instance does not load the dynamics model, then the Env can only provide the environment's observation and action information. LeaEnv mainly contains the following parts:

- :meth:`~d2c.envs.LeaEnv.__init__`: Use the passed-in config object to initialize the Env, which mainly involves the `env` part under model_config (model_config.json5) in config;

- :meth:`~d2c.envs.LeaEnv._load_model`: Load the trained dynamics model (refer to the dynamics section below);

- :meth:`~d2c.envs.LeaEnv.step`: Input action to make the environment run forward one step, which will call the dynamics model for calculation;

- :meth:`~d2c.envs.LeaEnv.step_raw`: It also makes the environment's dynamics model run for one time step, but unlike the `step` method, this method will return more raw information, such as when performing RL training, if there are multiple ensemble dynamics models in the algorithm, then this will return the results of all dynamics models, that is, the output object is `List`;

- :meth:`~d2c.envs.LeaEnv.d_num`: Return the number of dynamics models.

The usage usually is as below:

::

    env = Env(config)
    env.load_model()
    env.reset()
    env.step(a)


Dynamics
***********************
You can customize new dynamics models by inheriting from the base class :class:`~d2c.envs.learned.dynamics.base.BaseDyna`, and you need to customize the required network structure and loss function, etc. :class:`~d2c.envs.learned.dynamics.prob.ProbDyna` is an example of a dynamics model using a probabilistic neural network. After completing a custom dynamics model class, please use the function :ref:`register_dyna <register-dyna-label>` to register it, refer to ``register_dyna(ProbDyna)``. When constructing a dynamics model instance, please use the function :ref:`make_dynamics <make-dynamics-label>`. ``make_dynamics`` contains three input parameters:

- ``config``: config object, pay attention to the ``env.learned`` part in model_config(model_config.json5), which contains the type selection and model parameters of the dynamics model;

- ``data``: The data required for model training, if only loading the model, no input is required;

- ``restore``: Set to ``True`` when loading the dynamics model, the model file path parameter is located in ``train.dynamics_ckpt_dir`` in ``model_config.json5``.


External Env
------------------------
:ref:`External Env <external-env-label>` is based on existing environments or dynamics models, inherits from :class:`~d2c.envs.base.BaseEnv` and implements the relevant abstract methods. :class:`~d2c.envs.external.d4rl.D4rlEnv` is an Env built for the D4RL benchmark dataset, which uses the relevant environments in gym and constructs the environment in the method :meth:`~d2c.envs.external.d4rl.D4rlEnv._load_model`. You can customize the required external env class according to your needs. After customizing an external env class, you need to create a constructor that instantiates the Env based on the config object, refer to :func:`d2c.envs.external.__init__.d4rl_env`, and then register the external env class and its constructor, refer to:

::

    ENV_DICT = {'d4rl': D4rlEnv,}
    ENV_FUNC_DICT = {'d4rl': d4rl_env}

Finally, use the function :ref:`benchmark_env <benchmark-env-label>` to instantiate the Env. The parameters of benchmark_env are:

- ``config``: config object, pay attention to the ``model_config.env.external`` part, where ``benchmark_name`` indicates the name of the benchmark dataset, and ``env_name`` indicates the name of the environment to be built;

- ``benchmark_name``: If you do not need to instantiate Env, but just need to get the Env class, you can omit the config parameter and only input the benchmark_name parameter, which can return the corresponding Env class.

Instantiate an environment:

::

    config = make_config(kwargs)
    bm_data = Data(config)
    s_norm = dict(zip(['obs_shift', 'obs_scale'], bm_data.state_shift_scale))
    # The env of the benchmark to be used for policy evaluation.
    env = benchmark_env(config=config, **s_norm)

Get the external env class of the benchmark D4RL:

::

    env = benchmark_env(benchmark_name='d4rl')