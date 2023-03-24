# D2C

D2C(**D**ata-**d**riven **C**ontrol Library) is a Data-driven Control Library based on reinforcement learning. It is a platform for reinforcement learning experiments and solving various control problems in the real-world Scenario. On the on hand, It makes the RL experiments Fast and convenient. On the other hand, It makes offline reinforcement learning technology be applied in the real world application more possibly and simplistically.

The supported RL algorithms include:

- [Twin Delayed DDPG with Behavior Cloning (TD3+BC)](https://arxiv.org/pdf/2106.06860.pdf)
- [Distance-Sensitive Offline Reinforcement Learning (DOGE)](https://arxiv.org/abs/2205.11027.pdf)
- [Dynamics-Aware Hybrid Offline-and-Online Reinforcement Learning (H2O)](https://arxiv.org/abs/2206.13464.pdf)
- [Discriminator-Guided Model-Based Offline Imitation Learning (DMIL)](https://arxiv.org/abs/2207.00244)
- [Behavior Cloning (BC)](http://www.cse.unsw.edu.au/~claude/papers/MI15.pdf)

## Features:

- Include a large collection of offline reinforcement learning algorithms: model-free offline RL, model-based offline RL, planning method and imitation learning. There are our self-developed algorithms and other advanced algorithms in each category.
- It is highly modular and extensible. It is easy for you to build custom algorithms and conduct experiments.
- Policy training process is automatic in real-world scenario applications. It simplifies the processes of problem definition, model training, policy evaluation and model deployment.

## Badges
On some READMEs, you may see small images that convey metadata, such as whether or not all the tests are passing for the project. You can use Shields to add some to your README. Many services also have instructions for adding a badge.

## Documentation

The tutorials and API documentation are hosted on [d2c.readthedocs.io](https://zackxiangyu-d2c.readthedocs-hosted.com).

The example scripts are under [example/](https://gitlab.com/air_rl/algorithms-library/d2c/-/tree/dev/example/benchmark) folder and [test/](https://gitlab.com/air_rl/algorithms-library/d2c/-/tree/dev/test) folder.

## Installation
D2C interface can be installed as follows:
```commandline
git clone https://gitlab.com/air_rl/algorithms-library/d2c.git
cd d2c
pip install -e .
```

## Usage
Here is an example of TD3+BC. The full script can be found at [example/benchmark/demo_td3_bc.py](https://gitlab.com/air_rl/algorithms-library/d2c/-/tree/dev/example/benchmark/demo_td3_bc.py).

First, a configuration file [model_config.json5](https://gitlab.com/air_rl/algorithms-library/d2c/-/tree/dev/example/benchmark/config/model_config.json5) which contains all the hyper-parameters of the RL algorithms should be placed in [example/benchmark/config/](https://gitlab.com/air_rl/algorithms-library/d2c/-/tree/dev/example/benchmark/config). Please check this file for detail about the hyper-parameters.

Then, the offline data for the algorithm should be placed in [example/benchmark/data/](https://gitlab.com/air_rl/algorithms-library/d2c/-/tree/dev/example/benchmark/data/). Here, we use the mujoco dataset from D4RL([Download link](http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/)) and placed the data files in [example/benchmark/data/d4rl/mujoco/](https://gitlab.com/air_rl/algorithms-library/d2c/-/tree/dev/example/benchmark/data/d4rl/mujoco/).

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
The keys in the dict ``command_args`` are the names of the hyper-parameters in the configuration file [model_config.json5](https://gitlab.com/air_rl/algorithms-library/d2c/-/tree/dev/example/benchmark/config/model_config.json5).

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

You can also run it in command line. Please refer to the file [example/benchmark/run.sh](https://gitlab.com/air_rl/algorithms-library/d2c/-/tree/dev/example/benchmark/run.sh) for details.

## Support
Tell people where they can go to for help. It can be any combination of an issue tracker, a chat room, an email address, etc.

## Roadmap
If you have ideas for releases in the future, it is a good idea to list them in the README.

## Contributing
D2C is under development.

## Authors and acknowledgment
Show your appreciation to those who have contributed to the project.
