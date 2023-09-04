# D2C

[![Documentation Status](https://readthedocs.org/projects/air-d2c/badge/?version=latest)](https://air-d2c.readthedocs.io/en/latest/)
![MIT](https://img.shields.io/badge/license-MIT-blue)

D2C(**D**ata-**d**riven **C**ontrol Library) is a library for data-driven control based on reinforcement learning. It is a platform for conducting reinforcement learning experiments and solving various control problems in real-world scenarios. It has two main advantages: first, it makes the RL experiments fast and convenient; second, it enables the application of offline reinforcement learning technology in real-world settings more easily and simply.

The supported RL algorithms include:

- [Twin Delayed DDPG with Behavior Cloning (TD3+BC)](https://arxiv.org/pdf/2106.06860.pdf)
- [Distance-Sensitive Offline Reinforcement Learning (DOGE)](https://arxiv.org/abs/2205.11027.pdf)
- [Dynamics-Aware Hybrid Offline-and-Online Reinforcement Learning (H2O)](https://arxiv.org/abs/2206.13464.pdf)
- [Sparse Q-learning (SQL)](https://arxiv.org/abs/2303.15810)
- [Policy-guided Offline RL (POR)](https://arxiv.org/abs/2210.08323)
- [Offline Reinforcement Learning with Implicit Q-Learning (IQL)](https://arxiv.org/pdf/2110.06169.pdf)
- [Discriminator-Guided Model-Based Offline Imitation Learning (DMIL)](https://arxiv.org/abs/2207.00244)
- [Behavior Cloning (BC)](http://www.cse.unsw.edu.au/~claude/papers/MI15.pdf)

## Features:

- It includes a large collection of offline reinforcement learning algorithms: model-free offline RL, model-based offline RL, planning methods and imitation learning. In each category, there are our self-developed algorithms and other advanced algorithms.

- It is highly modular and extensible. You can easily build custom algorithms and conduct experiments with it.

- It automates the policy training process in real-world scenario applications. It simplifies the steps of problem definition, model training, policy evaluation and model deployment.

## Documentation

The tutorials and API documentation are hosted on [air-d2c.readthedocs.io](https://air-d2c.readthedocs.io/).

The example scripts are under [example/](./example/benchmark) folder and [test/](./test) folder.

## Installation
D2C interface can be installed as follows:
```commandline
git clone https://github.com/AIR-DI/D2C.git
cd d2c
pip install -e .
```

## Usage
Here is an example of TD3+BC. The full script can be found at [example/benchmark/demo_td3_bc.py](./example/benchmark/demo_td3_bc.py).

First, a configuration file [model_config.json5](./example/benchmark/config/model_config.json5) which contains all the hyper-parameters of the RL algorithms should be placed in [example/benchmark/config](./example/benchmark/config). Please check this file for detail about the hyper-parameters.

Then, the offline data for the algorithm should be placed in [example/benchmark/data](./example/benchmark/data). Here, we use the mujoco dataset from D4RL([Download link](http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/)) and placed the data files in [example/benchmark/data/d4rl/mujoco/](./example/benchmark/data/d4rl/mujoco).

Import some relevant packages:
```
import torch
from d2c.trainers import Trainer
from d2c.models import make_agent
from d2c.envs import benchmark_env, LeaEnv
from d2c.data import Data
from d2c.evaluators import bm_eval
from example.benchmark.config import make_config
```

Set the hyper-parameters and generate the `config`. Most of the hyper-parameters have been set up well in the configuration file. You can also modify the hyper-parameters like this:
```
command_args = {
        'model.model_name': 'td3_bc',
        'train.data_loader_name': None,
        'train.device': device,
        'train.seed': 0,
        'train.total_train_steps': 1000000,
        'train.batch_size': 256,
        'train.agent_ckpt_name': '0810'
    }
config = make_config(command_args)
```
The keys in the dict ``command_args`` are the names of the hyper-parameters in the configuration file [model_config.json5](./example/benchmark/config/model_config.json5).

Make the dataset:
```
bm_data = Data(config)
s_norm = dict(zip(['obs_shift', 'obs_scale'], bm_data.state_shift_scale))
data = bm_data.data
```

Make environments:
```
# The env of the benchmark to be used for policy evaluation.
env = benchmark_env(config=config, **s_norm)
# Contains dynamics model to be trained.
lea_env = LeaEnv(config)
```

Setup the agent and the evaluator:
```
agent = make_agent(config=config, env=lea_env, data=data)
evaluator = bm_eval(agent=agent, env=env, config=config)
```

Let's train it:
```
trainer = Trainer(agent=agent, train_data=data, config=config, env=lea_env, evaluator=evaluator)
trainer.train()
```

You can also run it in command line. Please refer to the file [example/benchmark/run.sh](./example/benchmark/run.sh) for details.

## Support

## Roadmap

## Contributing
D2C is under development. More new RL algorithms are going to be added and we will keep improving D2C. We always welcome contributions to help make D2C better with us together.

## Acknowledgment
Show your appreciation to those who have contributed to the project.
