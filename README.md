# AIControlOpt_lib

**AIControlOpt_lib** is an intelligent engine for solving various control problems in real-world. This engine is based on all kinds of **offline reinforcement learning** algorithms that developed by JD iCity. The main feature of our engine is that it takes advantage of huge volumes of offline data to automatically learn the control policy. It makes offline reinforcement learning technology be applied in the real world application more possibly and simplistically.

The overall framework of the engine is as below:

![overall_framework](./res/overall_framework.png "Overall Framework")

Here are other features of this engine:

- Include a large collection of offline reinforcement learning algorithms: model-free offline RL, model-based offline RL, planning method and imitation learning. There are our self-developed algorithms and other advanced algorithms in each category.
- Support many types of control tasks and cover control problems that have diverse scale and objectives.
- Policy training process is automatic. It simplifies the processes of problem definition and model training so that it is possible for the developers from business domain can use this engine to create value easily. The following is the overall workflow of this engine:

![workflow](./res/workflow.png "workflow")

## Implemented Offline RL Algorithms
### Model-free RL methods
- **SABER**: "**S**afe **A**ctor-Critic with **Be**havior **R**egularization."
- **BCQ**: Fujimoto, Scott, et al. "Off-Policy Deep Reinforcement Learning without Exploration." International Conference on Machine Learning, 2018, pp. 2052–2062. [paper](https://arxiv.org/abs/1812.02900) [code](https://github.com/sfujim/BCQ)
- **BEAR**: Kumar, Aviral, et al. "Stabilizing Off-Policy Q-Learning via Bootstrapping Error Reduction." Advances in Neural Information Processing Systems, 2019. [paper](https://arxiv.org/abs/1906.00949) [code](https://github.com/aviralkumar2907/BEAR)
- **BRAC**: Wu, Yifan, et al. "Behavior Regularized Offline Reinforcement Learning." arXiv preprint arXiv:1911.11361 (2019) [paper](https://arxiv.org/abs/1911.11361) [code](https://github.com/google-research/google-research/tree/master/behavior_regularized_offline_rl)
- **CQL**: Kumar, Aviral, et al. "Conservative Q-Learning for Offline Reinforcement Learning." Advances in Neural Information Processing Systems, vol. 33, 2020. [paper](https://arxiv.org/abs/2006.04779) [code](https://github.com/aviralkumar2907/CQL)
- **SBQ**: Xu, Haoran, et al. "Soft Behavior-constrained Q-Learning for Offline Reinforcement Learning."
### Model-based RL methods
- **MORE**: Zhan, Xianyuan, et al. "DeepThermal: Combustion Optimization for Thermal Power Generating Units Using Offline Reinforcement Learning." arXiv preprint arxiv.org/abs/2102.11492 (2021) [paper](https://arxiv.org/abs/2102.11492)
- **MOPO**: Yu, Tianhe, et al. "MOPO: Model-Based Offline Policy Optimization." Advances in Neural Information Processing Systems, vol. 33, 2020. [paper](https://papers.nips.cc/paper/2020/hash/a322852ce0df73e204b7e67cbbef0d0a-Abstract.html) [code](https://github.com/tianheyu927/mopo)
### Planning methods
- **MOPP**: Zhan, Xianyuan, et al. "Model-Based Offline Planning with Trajectory Pruning." arXiv preprint arxiv.org/abs/2105.07351 (2021) [paper](https://arxiv.org/abs/2105.07351)
- **MBOP**: Argenson, Arthur, et al. "Model-based offline planning." arXiv preprint arXiv:2008.05556 (2020). [paper](https://arxiv.org/abs/2008.05556)
- **PDDM**: Nagabandi, A, et al. "Deep Dynamics Models for Learning Dexterous Manipulation."
Conference on Robot Learning, 2020. [paper](https://arxiv.org/abs/1909.11652)
### Imitation learning methods
- **BC**: [(Behavior Cloning)](http://www.cse.unsw.edu.au/~claude/papers/MI15.pdf)
- **ValueDice**: Kostrikov, Ilya, et al. "Imitation Learning via Off-Policy Distribution Matching." International Conference on Learning Representations, 2020. [paper](https://arxiv.org/abs/1912.05032) [code](https://github.com/google-research/google-research/tree/master/value_dice)
- **ORIL**: Zolna, Konrad, et al. "Offline learning from demonstrations and unlabeled experience." arXiv preprint arXiv:2011.13885(2020). [paper](https://arxiv.org/abs/2011.13885)

## Documentation
- The API documentation can be found [here](/doc). You can browse the documentation locally by
opening the file ``doc/build/html/index.html``.

- The example scripts are under [test/](/test) folder and [example/](/example) folder.

- The details of the code structure are [here](res/README.md).

- Now, our library supports offline RL benchmark experiments. The details about all the benchmarks are [here](AIControlOpt_lib/envs/envs_external/README.md).

## Quick Start

------
#### Setup configuration
For a RL training task, there are usually 3 necessary files. Follow's configurations
shows a minimal example that can be found in [example](/example/application0) directory.
- Model parameters configuration.
Here is an exmaple for the model hyper parameters configuration. A complete file is 
[here](example/application0/config/model_config.json5) in example folder. More detailed 
description wiht these parameters of agent, algorithms, environment, training and evaluation 
can be found [here](example/application0/config/parameters_description.md).
```json5
{
  model: {
    // choose the offline RL algorithm.
    model_name: 'more',
    // Hyper Parameters for the chosen model 
    more: {
      train_schedule: ['d', 'vae_s', 'agent'],
      hyper_params: {
        model_params: {
          q: [[400, 400], 2],
          p: [[300, 300],],
          vae_s: [[750, 750],],
        },
        optimizers: [['adam',1e-3], ['adam', 1e-5], ['adam', 1e-4], ['adam', 1e-3]],
      }
    },
  },
  env: {
    // Parameters for dynamics model.
    dynamic_module_type: 'dnn',
    // parameters for each type of dynmamics models
    dnn: {
      model_params: [[200, 200], 3],
      optimizers: [['adam', 1e-3],],
    },
  },
  // Hyper parameters for model training.
  train: {
    train_test_ratio: 0.99,
    batch_size: 64,
    model_buffer_size: 1000000,
    weight_decays: 1e-5,
    update_freq: 1,
    update_rate: 0.005,
    discount: 0.99,
    total_train_steps: 1000,
    summary_freq: 100,
    print_freq: 100,
    save_freq: 1000,
    // parameters for model files
    model_dir: 'models',
  },
  // Parameters for policy evaluation.
  eval: {
    episode_step: 10,
    log_dir: 'eval',
    start: 0.,
    steps: 100,
  },
  interface: {
    policy_file: null,
    log_path: null,
  }
}
```
- Application parameters configuration.  
The reward function, cost function, done function and some parameters of the practical 
application should be in the application configuration file. An example is [here](example/application0/config/app_config.py).
- Processed data file.
The offline data should be CSV file with this format:  

    |feature name0|feature name1|feature name2| ...   |
    | ----------- | ----------- | ----------- | ----- |
    | value       | value       | value       | ...   |
    | ...         | ...         | ...         | ...   |
    
These two configuration files should be placed in ``example/application0/config/`` folder and
the data file should be placed in ``example/application0/data/`` folder.

#### Start training task
```bash
cd example/application0/
python main.py
```

#### Evaluate local trained model
```bash
cd example/application0/
python eval.py
```
Now, there are two offline policy evaluation methods. For example, the two functions
``eval_reward()``, ``eval_fqe()`` in [evaluation script](/example/application0/eval.py)
respectively represent these two methods: ``reward evaluation`` and ``FQE``. 
FQE can be used for evaluating the policy that is trained from model-free RL or model-based RL. 
The policy that is trained by planning methods and imitation methods usually cannot be evaluated with FQE.


#### Run with CLI
Could setup the hyper parameters of the chosen RL algorithm in command line. These setting 
parameters will replace the parameters in the model parameters configuration file above.
The folder to save the models will be named with the corresponding model name parameters from 
the command line.
```bash
# train the RL model.
cd example/application0/
python main.py \
  --model.model_name='bcq' \
  --train.total_train_steps=1000 \
  --train.seed=1 \
  --train.agent_ckpt_name='test_202202' \

# evaluate the trained model.
cd example/application0/
python eval.py \
  --model.model_name='bcq' \
  --train.seed=1 \
  --train.agent_ckpt_name='test_202202' \

```


