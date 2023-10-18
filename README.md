<p align="center"><img align="center" width="300px" src="docs/_static/images/d2c-logo.png"></p>

# D2C: Data-Driven Control Library

[![Documentation Status](https://readthedocs.org/projects/air-d2c/badge/?version=latest)](https://air-d2c.readthedocs.io/en/latest/)
![MIT](https://img.shields.io/badge/license-MIT-blue)

D2C(**D**ata-**d**riven **C**ontrol Library) is a library for data-driven decision-making & control based on state-of-the-art offline reinforcement learning (RL), offline imitation learning (IL), and offline planning algorithms. It is a platform for solving various decision-making & control problems in real-world scenarios. D2C is designed to offer fast and convenient algorithm performance development and testing, as well as providing easy-to-use toolchains to accelerate the real-world deployment of SOTA data-driven decision-making methods.

The current supported offline RL/IL algorithms include (**more to come**):

- [Twin Delayed DDPG with Behavior Cloning (TD3+BC)](https://arxiv.org/pdf/2106.06860.pdf)
- [Distance-Sensitive Offline Reinforcement Learning (DOGE)](https://arxiv.org/abs/2205.11027.pdf)
- [Dynamics-Aware Hybrid Offline-and-Online Reinforcement Learning (H2O)](https://arxiv.org/abs/2206.13464.pdf)
- [Sparse Q-learning (SQL)](https://arxiv.org/abs/2303.15810)
- [Policy-guided Offline RL (POR)](https://arxiv.org/abs/2210.08323)
- [Offline Reinforcement Learning with Implicit Q-Learning (IQL)](https://arxiv.org/pdf/2110.06169.pdf)
- [Discriminator-Guided Model-Based Offline Imitation Learning (DMIL)](https://arxiv.org/abs/2207.00244)
- [Behavior Cloning (BC)](http://www.cse.unsw.edu.au/~claude/papers/MI15.pdf)

## Features:

- D2C includes a large collection of offline RL and IL algorithms: model-free and model-based offline RL/IL algorithms, as well as planning methods. 

- D2C is highly modular and extensible. You can easily build custom algorithms and conduct experiments with it.

- D2C automates the development process in real-world control applications. It simplifies the steps of problem definition/mathematical formulation, policy training, policy evaluation and model deployment.

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

[//]: # (Tell people where they can go to for help. It can be any combination of an issue tracker, a chat room, an email address, etc.)
| Channel | Link                                                 |
|:-|:-----------------------------------------------------|
| Issues | [GitHub Issues](https://github.com/AIR-DI/D2C/issues) |
|E-mail | zhuxiangyu@air.tsinghua.edu.cn |

[//]: # (## Roadmap)

## Contributing
D2C is under development. More new RL algorithms are going to be added and we will keep improving D2C. We always welcome contributions to help make D2C better with us together.

## Citation:

To cite this repository:

```
@misc{d2c,
  author = {Xiangyu Zhu and Jianxiong Li and Wenjia Zhang and Haoyi Niu and Yinan Zheng and Haoran Xu and Xianyuan Zhan},
  title = {{D2C}: Data-driven Control Library},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/AIR-DI/D2C}},
}
```

## Acknowledgment
D2C is supported by [AIR-DREAM Lab](https://air-dream.netlify.app/), which is a research group at [Institute for AI Industry Research (AIR), Tsinghua University](https://air.tsinghua.edu.cn/en/).
