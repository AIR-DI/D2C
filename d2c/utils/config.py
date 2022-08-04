"""The general config that integrates the app_config and model_config"""

import os
import json5
import importlib
import numpy as np
from easydict import EasyDict
from typing import Union, Optional, Dict, Any, Tuple
from d2c.utils.utils import Flags
from d2c.envs import benchmark_env


def read_config_from_json(
        config_file: str,
        encoding: Optional[str] = None,
        easydict: bool = False
) -> Union[Dict, EasyDict]:
    with open(config_file, "r", encoding=encoding) as f:
        config_dict = json5.load(f)
    if easydict:
        config = EasyDict(config_dict)
    else:
        config = config_dict
    return config


def update_config(
        config_file: str,
        hyper_params: Optional[Dict] = None,
        encoding: Optional[str] = None
) -> EasyDict:
    """read from config_file and update it by hyper_params"""
    if hyper_params is None:
        hyper_params = {}
    config_dict = read_config_from_json(config_file, encoding=encoding)
    config_dict = update_nested_dict_by_dict(hyper_params, config_dict)
    config = EasyDict(config_dict)
    return config


def update_nested_dict_by_kv(
        dic: Dict,
        keys: str,
        value: Any
) -> None:
    """make use of the shallow copy of dict to update value."""
    for key in keys[:-1]:
        dic = dic.setdefault(key, {})
    dic[keys[-1]] = value


def update_nested_dict_by_dict(
        from_dict: Dict,
        to_dict: Dict
) -> Dict:
    for k, v in from_dict.items():
        update_nested_dict_by_kv(to_dict, k.split("."), v)
    return to_dict


class ConfigBuilder:
    """Builder the complete configuration with app_config and model_config, and set the parameters
    according to the CLI input.

    The main API method is:

    * :meth:`build_config`: get the complete configuration.

    :param app_config: the app_config;
    :param model_config_path: the model_config file path;
    :param str model_config_path: the absolute path of the work dir that contains the \
        `run` script, `data` dir and `models` dir.
    :param dict command_args: the CLI parameters input;
    :return: a complete configuration that can be used by main function.
    """

    def __init__(
            self,
            app_config: Flags,
            model_config_path: str,
            work_abs_dir: str,
            command_args: Dict
    ) -> None:
        self._command_args = command_args
        self._app_cfg = app_config
        self._model_cfg_path = model_config_path
        self._work_abs_dir = work_abs_dir
        self._env_info = None
        self._model_cfg = None
        self._update_model_cfg()

    def _update_model_cfg(self) -> None:
        self._model_cfg = update_config(self._model_cfg_path, self._command_args)
        self._env_info = self._get_env_info()
        # update env parameters
        self._update_env_info()
        # create all the models saving paths
        self._update_model_dir()
        print('=' * 10 + 'The config of this experiment' + '=' * 10)
        print(json5.dumps(self._model_cfg, indent=2, ensure_ascii=False))

    def build_config(self) -> Flags:
        """The API to build the final config."""
        config = Flags(
            app_config=self._app_cfg,
            model_config=self._model_cfg
        )
        return config

    @property
    def main_hyper_params(self) -> Dict:
        """Get the main hyperparameters of this experiment."""
        _dict = {}
        _dict.update(model_name=self._model_cfg.model.model_name)
        model_hyper_params = self._model_cfg.model[_dict['model_name']].hyper_params
        model_hyper_params['model_params'] = str(model_hyper_params.model_params)
        model_hyper_params['optimizers'] = str(model_hyper_params.optimizers)
        _dict.update(model_hyper_params)

        train_params = ['device', 'train_test_ratio', 'batch_size',
                        'update_freq', 'update_rate', 'discount',
                        'total_train_steps', 'seed']
        for k in train_params:
            _dict.update({k: self._model_cfg.train[k]})

        print('='*10 + 'The main hyperparameters of this experiment' + '='*10)
        print(json5.dumps(_dict, indent=2, ensure_ascii=False))

        return _dict

    def _get_env_info(self) -> Flags:
        try:
            self._env_ext = self._model_cfg.env.external
            # sys.path.append('../../example/benchmark/')
            import_path = '.'.join(('example.benchmark.data', self._env_ext.benchmark_name, self._env_ext.data_source))
            module = importlib.import_module(import_path)
            domain = self._env_ext.env_name.split('-')[0]
            env_info = Flags(
                norm_min=getattr(module, (domain + '_random_score').upper()),
                norm_max=getattr(module, (domain + '_expert_score').upper()),
                state_info=getattr(module, (domain + '_state').upper()),
                action_info=getattr(module, (domain + '_action').upper()),
            )
        except:
            benchmark_name = self._env_ext.benchmark_name
            data_source = self._env_ext.data_source
            env_name = self._env_ext.env_name
            kwargs = dict()
            if 'combined_challenge' in self._env_ext:
                kwargs.update({'combined_challenge': self._env_ext.combined_challenge})
            state_info, action_info = self._get_env_space(
                benchmark_name,
                data_source,
                env_name,
                **kwargs,
            )
            env_info = Flags(
                norm_min=None,
                norm_max=None,
                state_info=state_info,
                action_info=action_info,
            )
        return env_info

    def _get_env_space(
            self,
            benchmark_name: str,
            data_source: str,
            env_name: str,
            **kwargs: Any
    ) -> Tuple:
        env_class = benchmark_env(benchmark_name=benchmark_name)
        environment_space = env_class.make_env_space(
            data_source=data_source,
            env_name=env_name,
            **kwargs,
        )
        observation_space = environment_space.observation
        action_space = environment_space.action
        try:
            state_info = (observation_space.shape[0], observation_space.low,
                          observation_space.high)  # (dimension, minimum, maximum)
        except:
            state_info = (observation_space.shape[0], -np.inf, np.inf)  # (dimension, minimum, maximum)
        action_info = (action_space.shape[0], action_space.low, action_space.high)

        return state_info, action_info

    def _update_env_info(self) -> None:
        # update env parameters
        data_file_path = os.path.join(
            self._work_abs_dir,
            'data',
            self._env_ext.benchmark_name,
            self._env_ext.data_source,
            self._env_ext.data_name
        )
        # Update the env basic_info
        if not self._model_cfg.env.basic_info.state_dim:
            self._model_cfg.env.basic_info.update(dict([('state_dim', self._env_info.state_info[0]),
                                                        ('state_min', self._env_info.state_info[1]),
                                                        ('state_max', self._env_info.state_info[2])]))
        if not self._model_cfg.env.basic_info.action_dim:
            self._model_cfg.env.basic_info.update(dict([('action_dim', self._env_info.action_info[0]),
                                                        ('action_min', self._env_info.action_info[1]),
                                                        ('action_max', self._env_info.action_info[2])]))
        # Update the external env info
        temp_dict = dict([('score_norm_min', self._env_info.norm_min),
                          ('score_norm_max', self._env_info.norm_max),
                          ('data_file_path', data_file_path)])
        for k, v in temp_dict.items():
            if not self._model_cfg.env.external[k]:
                self._model_cfg.env.external[k] = v

    def _update_model_dir(self) -> None:
        """Construct the models' file path"""
        model_dir = self._model_cfg.train.model_dir
        model_dir = os.path.join(
            self._work_abs_dir,
            model_dir,
            self._env_ext.benchmark_name,
            self._env_ext.data_source,
            self._env_ext.env_name,
            self._env_ext.data_name,
            str(self._env_ext.num_transitions)+'_'+str(self._env_ext.state_normalize),
        )
        model_name = self._model_cfg.model.model_name
        behavior_ckpt_dir = os.path.join(
            model_dir,
            'behavior',
            self._model_cfg.train.behavior_ckpt_name,
        )
        dynamics_ckpt_dir = os.path.join(
            model_dir,
            'dynamics',
            self._model_cfg.env.learned.dynamic_module_type,
            self._model_cfg.train.dynamics_ckpt_name,
        )
        q_ckpt_dir = os.path.join(model_dir, 'Q', self._model_cfg.train.q_ckpt_name)
        vae_s_ckpt_dir = os.path.join(model_dir, 'vae_s', self._model_cfg.train.vae_s_ckpt_name)
        agent_ckpt_dir = os.path.join(
            model_dir,
            'agent',
            model_name,
            str(self._model_cfg.train.agent_ckpt_name),
            str(self._model_cfg.train.seed),
            'agent',
        )
        model_dir_dict = dict(
            behavior_ckpt_dir=behavior_ckpt_dir,
            dynamics_ckpt_dir=dynamics_ckpt_dir,
            q_ckpt_dir=q_ckpt_dir,
            vae_s_ckpt_dir=vae_s_ckpt_dir,
            agent_ckpt_dir=agent_ckpt_dir,
        )
        self._model_cfg.train.update(model_dir_dict)




