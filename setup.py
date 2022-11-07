#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from setuptools import setup
from setuptools import find_packages

setup(
    name='d2c',
    description="D2C is a Data-driven Control Library based on reinforcement learning.",
    url='https://gitlab.com/air_rl/algorithms-library/d2c.git',
    python_requires=">=3.7",
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
        'gym',
        'numpy',
        'pandas',
        'torch',
        'tqdm',
        'json5',
        'wandb',
        'tensorboard',
        'easydict',
        'h5py',
    ],
    extras_require={
        'mujoco': ['mujoco-py']
    },
)
