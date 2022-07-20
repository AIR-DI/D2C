#!/usr/bin/env python
# coding: utf-8
"""The general config that integrats the app_config and model_config"""

import os
import sys
import importlib
import numpy as np
from example.benchmark.config.app_config import app_config
from AIControlOpt_lib.utils.config import update_config
from AIControlOpt_lib.utils.utils import Flags
from AIControlOpt_lib.utils.utils import abs_file_path
from AIControlOpt_lib.envs import benchmark_env


def build_config(command_args=None):
    model_config_path = abs_file_path(__file__, './model_config.json5')
    conf = ConfigBuilder(app_config, model_config_path, command_args)
    return conf.build_config()


class ConfigBuilder:
    """Builder the complete configuration with app_config and model_config, and set the parameters
    according to the CLI input.

    The main API method is:

    * :meth:`build_config`: get the complete configuration.

    :param app_conf: the app_config;
    :param model_conf_path: the model_config file path;
    :param dict command_args: the CLI parameters input;
    :return: a complete configuration that can be used by main function.
    """

    def __init__(self, app_conf, model_conf_path, command_args):
        self._command_args = command_args
        self._app_conf = app_conf
        self._model_conf_path = model_conf_path
        self._env_info = None
        self._model_conf = None
        self._update_model_conf()
        self._update_app_conf()

    def _update_model_conf(self):
        self._model_conf = update_config(self._model_conf_path, self._command_args)
        self._env_info = self._get_env_info()
        # update env_external parameters
        self._update_env_external()
        # create all the models saving paths
        self._update_model_dir()
        print(self._model_conf)

    def _update_app_conf(self):
        self._app_conf.state_dim, self._app_conf.state_min, self._app_conf.state_max = self._env_info.state_info
        self._app_conf.action_dim, self._app_conf.action_min, self._app_conf.action_max = self._env_info.action_info

    def build_config(self):
        config = Flags(
            app_config=self._app_conf,
            model_config=self._model_conf)
        return config

    def _get_env_info(self):
        try:
            self._env_ext = self._model_conf.env.env_external
            sys.path.append('../../example/benchmark/')
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
            state_info, action_info = self._get_env_spec(
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

    def _get_env_spec(self, benchmark_name, data_source, env_name, **kwargs):
        env_class = benchmark_env(benchmark_name=benchmark_name)
        environment_spec = env_class.make_env_spec(
            data_source=data_source,
            env_name=env_name,
            **kwargs,
        )
        observation_spec = environment_spec.observation
        action_spec = environment_spec.action
        try:
            state_info = (observation_spec.shape[0], observation_spec.minimum,
                          observation_spec.maximum)  # (dimension, minimum, maximum)
        except:
            state_info = (observation_spec.shape[0], -np.inf, np.inf)  # (dimension, minimum, maximum)
        action_info = (action_spec.shape[0], action_spec.minimum, action_spec.maximum)

        return state_info, action_info

    def _update_env_external(self):
        # update env_external parameters
        data_file_path = os.path.join(
            'data',
            self._env_ext.benchmark_name,
            self._env_ext.data_source,
            self._env_ext.data_name
        )
        data_file_path = abs_file_path(__file__, '../' + data_file_path)
        self._model_conf.env.env_external.update(dict([('score_norm_min', self._env_info.norm_min),
                                                       ('score_norm_max', self._env_info.norm_max),
                                                       ('data_file_path', data_file_path)]))

    def _update_model_dir(self):
        """Construct the models' file path"""
        model_dir = self._model_conf.train.model_dir
        model_dir = abs_file_path(__file__, '../' + model_dir)
        model_dir = os.path.join(
            model_dir,
            self._env_ext.benchmark_name,
            self._env_ext.data_source,
            self._env_ext.env_name,
            self._env_ext.data_name,
            str(self._env_ext.num_transitions)+'_'+str(self._env_ext.state_normalize),
        )
        model_name = self._model_conf.model.model_name
        behavior_ckpt_dir = os.path.join(
            model_dir,
            'behavior',
            self._model_conf.train.behavior_ckpt_name,
        )
        dynamics_ckpt_dir = os.path.join(
            model_dir,
            'dynamics',
            self._model_conf.env.dynamic_module_type,
            self._model_conf.train.dynamics_ckpt_name,
        )
        q_ckpt_dir = os.path.join(model_dir, 'Q', self._model_conf.train.q_ckpt_name)
        vae_s_ckpt_dir = os.path.join(model_dir, 'vae_s', self._model_conf.train.vae_s_ckpt_name)
        agent_ckpt_dir = os.path.join(
            model_dir,
            'agent',
            model_name,
            str(self._model_conf.train.agent_ckpt_name),
            str(self._model_conf.train.seed),
            'agent',
        )
        model_dir_dict = dict(
            behavior_ckpt_dir=behavior_ckpt_dir,
            dynamics_ckpt_dir=dynamics_ckpt_dir,
            q_ckpt_dir=q_ckpt_dir,
            vae_s_ckpt_dir=vae_s_ckpt_dir,
            agent_ckpt_dir=agent_ckpt_dir,
        )
        self._model_conf.train.update(model_dir_dict)




