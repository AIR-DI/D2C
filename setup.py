#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from setuptools import setup
from setuptools import find_packages

setup(
    name='d2c',
    description="D2C is a Data-driven Control Library based on reinforcement learning.",
    url='https://github.com/AIR-DI/D2C.git',
    python_requires=">=3.7",
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'gym==0.23.1',
        'numpy',
        'pandas',
        'torch',
        'tqdm',
        'json5',
        'wandb',
        'tensorboard',
        'easydict',
        'h5py',
        'fire',
    ],
    extras_require={
        'mujoco': ['mujoco-py==2.1.2.14']
    },
)
