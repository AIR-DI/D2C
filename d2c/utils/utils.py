"""A collection of some little utils."""
import os
import random
import re
import time
from typing import Dict, Generator, List

import gym
import numpy as np
import torch
from typing import Dict, Generator, List, Callable, Union
from gym.spaces import Space, Box


def get_summary_str(step: int = None, info: Dict = None, prefix: str = '') -> str:
    summary_str = prefix
    if step is not None:
        summary_str += 'Step %d; ' % step
    for key, val in info.items():
        if isinstance(val, (int, np.int32, np.int64)):
            summary_str += '%s %d; ' % (key, val)
        elif isinstance(val, (float, np.float32, np.float64)):
            summary_str += '%s %.4g; ' % (key, val)
    return summary_str


def get_optimizer(name: str) -> Callable:
    """Get an optimizer generator that returns an optimizer according to lr."""
    if name == 'adam':
        def adam_opt_(parameters, lr, weight_decay=0.0):
            return torch.optim.Adam(params=parameters, lr=lr, weight_decay=weight_decay)

        return adam_opt_
    else:
        raise ValueError('Unknown optimizer %s.' % name)
    
# generate xml assets path: gym_xml_path
def generate_xml_path() -> str:
    import os
    import gym
    xml_path = os.path.join(gym.__file__[:-11], 'envs/mujoco/assets')

    assert os.path.exists(xml_path)
    print("gym_xml_path: ",xml_path)

    return xml_path


gym_xml_path = generate_xml_path()


# update env xml with different dynamics
def update_xml(index: str, env_name: str) -> None:
    xml_name = parse_xml_name(env_name)
    os.system('cp ./xml_path/{0}/{1} {2}/{1}}'.format(index, xml_name, gym_xml_path))

    time.sleep(0.2)

# translate env name into xml file name
def parse_xml_name(env_name: str) -> str:
    if 'walker' in env_name.lower():
        xml_name = "walker2d.xml"
    elif 'hopper' in env_name.lower():
        xml_name = "hopper.xml"
    elif 'halfcheetah' in env_name.lower():
        xml_name = "half_cheetah.xml"
    elif "ant" in env_name.lower():
        xml_name = "ant.xml"
    else:
        raise RuntimeError("No available environment named \'%s\'" % env_name)

    return xml_name

# generate target env
def update_target_env(env_name: str) -> None:
    xml_name = parse_xml_name(env_name)

    os.system(
        'cp xml_path/target_file/{0} {1}/{0}'.format(xml_name, gym_xml_path))

    time.sleep(0.2)

# change gravity
def update_source_env_gravity(variety_degree: float, env_name: str) -> None:
    old_xml_name = parse_xml_name(env_name)
    # create new xml 
    xml_name = "{}_gravityx{}.xml".format(old_xml_name.split(".")[0], variety_degree)

    with open('xml_path/target_file/{}'.format(old_xml_name), "r+") as f:

        new_f = open('xml_path/source_file/{}'.format(xml_name), "w+")
        for line in f.readlines():
            if "gravity" in line:
                pattern = re.compile(r"gravity=\"(.*?)\"")
                a = pattern.findall(line)
                gravity_list = a[0].split(" ")
                new_gravity_list = []
                for num in gravity_list:
                    new_gravity_list.append(variety_degree * float(num))

                replace_num = " ".join(str(i) for i in new_gravity_list)
                replace_num = "gravity=\"" + replace_num + "\""
                sub_result = re.sub(pattern, str(replace_num), line)

                new_f.write(sub_result)
            else:
                new_f.write(line)

        new_f.close()

    # replace the default gym env with newly-revised env
    os.system(
        'cp xml_path/source_file/{0} {1}/{2}'.format(xml_name, gym_xml_path, old_xml_name))

    time.sleep(0.2)

# change density
def update_source_env_density(variety_degree: float, env_name: str) -> None:
    old_xml_name = parse_xml_name(env_name)
    # create new xml 
    xml_name = "{}_densityx{}.xml".format(old_xml_name.split(".")[0], variety_degree)

    with open('xml_path/target_file/{}'.format(old_xml_name), "r+") as f:

        new_f = open('xml_path/source_file/{}'.format(xml_name), "w")
        for line in f.readlines():
            if "density" in line:
                pattern = re.compile(r'(?<=density=")\d+\.?\d*')
                a = pattern.findall(line)
                current_num = float(a[0])
                replace_num = current_num * variety_degree
                sub_result = re.sub(pattern, str(replace_num), line)

                new_f.write(sub_result)
            else:
                new_f.write(line)

        new_f.close()

    # replace the default gym env with newly-revised env
    os.system(
        'cp xml_path/source_file/{0} {1}/{2}'.format(xml_name, gym_xml_path, old_xml_name))

    time.sleep(0.2)

# change friction
def update_source_env_friction(variety_degree: float, env_name: str) -> None:
    old_xml_name = parse_xml_name(env_name)
    # create new xml 
    xml_name = "{}_frictionx{}.xml".format(old_xml_name.split(".")[0], variety_degree)

    with open('xml_path/target_file/{}'.format(old_xml_name), "r+") as f:

        new_f = open('xml_path/source_file/{}'.format(xml_name), "w")
        for line in f.readlines():
            if "friction" in line:
                pattern = re.compile(r"friction=\"(.*?)\"")
                a = pattern.findall(line)
                friction_list = a[0].split(" ")
                new_friction_list = []
                for num in friction_list:
                    new_friction_list.append(variety_degree * float(num))

                replace_num = " ".join(str(i) for i in new_friction_list)
                replace_num = "friction=\"" + replace_num + "\""
                sub_result = re.sub(pattern, str(replace_num), line)

                new_f.write(sub_result)
            else:
                new_f.write(line)

        new_f.close()

    # replace the default gym env with newly-revised env
    os.system(
        'cp xml_path/source_file/{0} {1}/{2}'.format(xml_name, gym_xml_path, old_xml_name))

    time.sleep(0.2)


class Flags(object):

    def __init__(self, **kwargs) -> None:
        for key, val in kwargs.items():
            setattr(self, key, val)


def chain_gene(*args: List[Generator]) -> Generator:
    """Connect several Generator objects into one Generator object."""
    for x in args:
        yield from x


def maybe_makedirs(log_dir: str) -> None:
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)


def set_seed(seed: int) -> None:
    seed %= 4294967294
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    print("Using seed {}".format(seed))


def abs_file_path(file, relative_path):
    return os.path.abspath(os.path.join(os.path.split(os.path.abspath(file))[0], relative_path))


def add_gaussian_noise(data: np.ndarray, space: Union[Box, Space], std: float) -> np.ndarray:
    noise = space.high * np.random.normal(loc=0, scale=std, size=data.shape)
    return np.clip(data + noise, space.low, space.high)


def to_array_as(x, y):
    if isinstance(x, torch.Tensor) and isinstance(y, np.ndarray):
        return x.detach().cpu().numpy().astype(y.dtype)
    elif isinstance(x, np.ndarray) and isinstance(y, torch.Tensor):
        return torch.as_tensor(x).to(y)
    else:
        return x

