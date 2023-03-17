"""The general config that integrates the app_config and model_config"""

import os
import copy
import json5
import logging
import importlib
import inspect
import numpy as np
from easydict import EasyDict
from typing import Union, Optional, Dict, Any, Tuple, Generator, Callable, List
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


def flat_dict(x: Dict) -> Generator:
    for key, value in x.items():
        if isinstance(value, dict):
            for k, v in flat_dict(value):
                k = '.'.join([key, k])
                yield k, v
        else:
            yield key, value


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
    :param str experiment_type: the available options are `['benchmark', 'application']`.
    :return: a complete configuration that can be used by main function.
    """

    def __init__(
            self,
            app_config: Any,
            model_config_path: str,
            work_abs_dir: str,
            command_args: Optional[Dict] = None,
            experiment_type: str = 'benchmark',
    ) -> None:
        self._command_args = command_args
        self._exp_type = experiment_type
        self._check_command_args()
        self._app_cfg = app_config
        self._check_app_config()
        self._model_cfg_path = model_config_path
        self._work_abs_dir = work_abs_dir
        self._env_info = None
        self._model_cfg = None
        self._update_model_cfg()

    def _check_command_args(self) -> None:
        """Check the input command parameters."""
        assert isinstance(self._command_args, dict)
        for k in self._command_args.keys():
            _k = k.split('.')[0]
            if _k not in ['model', 'env', 'train', 'eval', 'interface']:
                raise KeyError(f'The key {_k} is not in the model_config!')

    def _check_app_config(self) -> None:
        """Check the elements in app_config."""
        essential_attrs = [
            'state_indices',
            'action_indices',
        ]
        optional_attrs = [
            'state_scaler',
            'state_scaler_params',
            'action_scaler',
            'action_scaler_params',
            'reward_scaler',
            'reward_scaler_params',
            'reward_fn',
            'cost_fn',
            'done_fn',
        ]

        for attr in essential_attrs:
            if not hasattr(self._app_cfg, attr):
                raise AttributeError(f'The app_config lacks the essential attribute named {attr}!')

        miss_attrs = []
        for attr in optional_attrs:
            if not hasattr(self._app_cfg, attr):
                setattr(self._app_cfg, attr, None)
                miss_attrs.append(attr)
        if len(miss_attrs) > 0:
            logging.warning(f'The app_config lacks the attributes: {miss_attrs} and all of them have been set to None.')

        def inspect_fn_params(fn: Callable) -> List[str]:
            sig = inspect.signature(fn)
            return [x for x in sig.parameters.keys()]

        for fn_name in ['reward_fn', 'cost_fn', 'done_fn']:
            fn = getattr(self._app_cfg, fn_name)
            if fn is not None and isinstance(fn, Callable):
                params = inspect_fn_params(fn)
                params_required = ['past_a', 's', 'a', 'next_s']
                assert params == params_required, f'The parameters of the function {fn_name} should be set like ' \
                                                  f'{params_required}!'

    def _update_model_cfg(self) -> None:
        self._model_cfg = update_config(self._model_cfg_path, self._command_args)
        self._env_info = self._get_env_info()
        # update env parameters
        self._update_env_info()
        # create all the models saving paths
        self._update_model_dir()
        logging.debug('=' * 20 + 'The config of this experiment' + '=' * 20)
        _m_cfg = copy.deepcopy(self._model_cfg)
        _dict = {}
        for k, v in flat_dict(_m_cfg):
            if isinstance(v, np.ndarray):
                _dict.update({k: v.tolist()})
        _m_cfg = update_nested_dict_by_dict(_dict, _m_cfg)
        logging.debug(json5.dumps(_m_cfg, indent=2, ensure_ascii=False))

    def build_config(self) -> Flags:
        """The API to build the final config."""
        config = Flags(
            app_config=self._app_cfg,
            model_config=self._model_cfg
        )
        return config

    @staticmethod
    def main_hyper_params(_model_cfg: Union[Dict, EasyDict, Any]) -> Dict:
        """Get the main hyperparameters of this experiment.

        :param dict _model_cfg: the model_config.
        """
        _model_cfg = copy.deepcopy(_model_cfg)
        _dict = {}
        _dict.update(model_name=_model_cfg.model.model_name)
        model_hyper_params = _model_cfg.model[_dict['model_name']].hyper_params
        for k, v in model_hyper_params.items():
            model_hyper_params[k] = str(v)
        _dict.update(model_hyper_params)

        _dict.update(env_external=_model_cfg.env.external)

        train_params = ['device', 'test_data_ratio', 'batch_size',
                        'update_freq', 'update_rate', 'discount',
                        'total_train_steps', 'seed', 'action_noise']
        for k in train_params:
            _dict.update({k: _model_cfg.train[k]})

        print('='*20 + 'The main hyperparameters of this experiment' + '='*20)
        _d = {}
        for k, v in flat_dict(_dict):
            if isinstance(v, np.ndarray):
                _d.update({k: v.tolist()})
        _dict = update_nested_dict_by_dict(_d, _dict)
        print(json5.dumps(_dict, indent=2, ensure_ascii=False))

        return _dict

    def _get_env_info(self) -> Flags:
        if self._exp_type == 'benchmark':
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
        elif self._exp_type == 'application':
            state_dim = len(self._app_cfg.state_indices)
            if self._app_cfg.state_scaler == 'min_max':
                state_min = 0.0
                state_max = 1.0
            else:
                state_min, state_max = -np.inf, np.inf
            action_dim = len(self._app_cfg.action_indices)
            if self._app_cfg.action_scaler == 'min_max':
                action_min = 0.0
                action_max = 1.0
            else:
                action_min, action_max = -np.inf, np.inf
            env_info = Flags(
                norm_min=None,
                norm_max=None,
                state_info=(state_dim, state_min, state_max),
                action_info=(action_dim, action_min, action_max),
            )
        else:
            raise ValueError(f'The value of the parameter experiment_type is wrong!')
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
        # Update the env basic_info
        self._update_env_basic_info()
        if self._exp_type == 'benchmark':
            # update env parameters
            data_file_path = os.path.join(
                self._work_abs_dir,
                'data',
                self._env_ext.benchmark_name,
                self._env_ext.data_source,
                self._env_ext.data_name
            )
            # Update the external env info
            temp_dict = dict([('score_norm_min', self._env_info.norm_min),
                              ('score_norm_max', self._env_info.norm_max),
                              ('data_file_path', data_file_path)])
            for k, v in temp_dict.items():
                if self._model_cfg.env.external[k] is None:
                    self._model_cfg.env.external[k] = v

    def _update_env_basic_info(self) -> None:
        state_dim = self._env_info.state_info[0]
        state_min = self._env_info.state_info[1]
        state_max = self._env_info.state_info[2]
        action_dim = self._env_info.action_info[0]
        action_min = self._env_info.action_info[1]
        action_max = self._env_info.action_info[2]

        if not self._model_cfg.env.basic_info.state_dim:
            self._model_cfg.env.basic_info.update(dict([('state_dim', state_dim),
                                                        ('state_min', state_min),
                                                        ('state_max', state_max)]))
        if not self._model_cfg.env.basic_info.action_dim:
            self._model_cfg.env.basic_info.update(dict([('action_dim', action_dim),
                                                        ('action_min', action_min),
                                                        ('action_max', action_max)]))

    def _update_model_dir(self) -> None:
        """Construct the models' file path"""
        model_dir = self._model_cfg.train.model_dir
        model_name = self._model_cfg.model.model_name
        if self._exp_type == 'benchmark':
            model_dir = os.path.join(
                self._work_abs_dir,
                model_dir,
                self._env_ext.benchmark_name,
                self._env_ext.data_source,
                model_name,
                self._env_ext.env_name + '_' + self._env_ext.data_name,
                # 's_norm_'+str(self._env_ext.state_normalize),
            )
        elif self._exp_type == 'application':
            model_dir = os.path.join(
                self._work_abs_dir,
                model_dir,
                model_name,
            )
        else:
            raise ValueError(f'The value of the parameter experiment_type is wrong!')

        if not self._model_cfg.train.get('behavior_ckpt_dir'):
            behavior_ckpt_dir = os.path.join(
                model_dir,
                'behavior',
                self._model_cfg.train.behavior_ckpt_name,
            )
            self._model_cfg.train.update(behavior_ckpt_dir=behavior_ckpt_dir)

        if not self._model_cfg.train.get('dynamics_ckpt_dir'):
            dynamics_ckpt_dir = os.path.join(
                model_dir,
                'dynamics',
                self._model_cfg.env.learned.dynamic_module_type,
                self._model_cfg.train.dynamics_ckpt_name,
            )
            self._model_cfg.train.update(dynamics_ckpt_dir=dynamics_ckpt_dir)

        if not self._model_cfg.train.get('q_ckpt_dir'):
            q_ckpt_dir = os.path.join(model_dir, 'Q', self._model_cfg.train.q_ckpt_name)
            self._model_cfg.train.update(q_ckpt_dir=q_ckpt_dir)

        if not self._model_cfg.train.get('vae_s_ckpt_dir'):
            vae_s_ckpt_dir = os.path.join(model_dir, 'vae_s', self._model_cfg.train.vae_s_ckpt_name)
            self._model_cfg.train.update(vae_s_ckpt_dir=vae_s_ckpt_dir)

        if not self._model_cfg.train.get('agent_ckpt_dir'):
            agent_ckpt_dir = os.path.join(
                model_dir,
                'agent',
                str(self._model_cfg.train.agent_ckpt_name),
                'seed'+str(self._model_cfg.train.seed),
                'agent',
            )
            self._model_cfg.train.update(agent_ckpt_dir=agent_ckpt_dir)
