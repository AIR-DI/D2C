import os
import pytest
import torch
import numpy as np
from d2c.utils.scalers import (
    MinMaxScaler,
    StandardScaler,
    create_scaler,
)


state = np.random.random((32, 10)) * 100
minimum = np.zeros(10)
maximum = np.ones(10) * 100
mean = np.mean(state, 0)
std = np.std(state, 0)
eps = 1e-3


def test_create_scaler():
    scaler1 = create_scaler('min_max')
    scaler2 = create_scaler('min_max', data=state)
    scaler3 = create_scaler('min_max', minimum=minimum, maximum=maximum)
    for s in [scaler1, scaler2, scaler3]:
        assert isinstance(s, MinMaxScaler)
    scaler4 = create_scaler('standard')
    scaler5 = create_scaler('standard', data=state)
    scaler6 = create_scaler('standard', mean=mean, std=std)
    for s in [scaler4, scaler5, scaler6]:
        assert isinstance(s, StandardScaler)


def test_min_max_scaler():
    scaler1 = MinMaxScaler(data=state)
    s = scaler1.transform(state)
    assert np.all(s >= 0.0)
    assert np.all(s <= 1.0)
    _min = np.min(state, 0)
    _max = np.max(state, 0)
    ref_s = (state - _min) / (_max - _min)
    assert np.allclose(s, ref_s)

    reversed_s = scaler1.reverse_transform(s)
    assert np.allclose(state, reversed_s)

    s_tensor = scaler1.transform_tensor(torch.tensor(state))
    assert np.allclose(ref_s, s_tensor.numpy())

    reversed_s_tensor = scaler1.reverse_transform_tensor(s_tensor)
    assert np.allclose(state, reversed_s_tensor.numpy())

    scaler1_param = scaler1.get_params()
    assert np.all(scaler1_param['minimum'] == _min)
    assert np.all(scaler1_param['maximum'] == _max)


def test_min_max_scaler_with_min_max():
    scaler = MinMaxScaler(minimum=minimum, maximum=maximum)
    s = scaler.transform(state)
    assert np.all(s >= 0.0)
    assert np.all(s <= 1.0)
    ref_s = (state - minimum) / (maximum - minimum)
    assert np.allclose(s, ref_s)

    reversed_s = scaler.reverse_transform(s)
    assert np.allclose(state, reversed_s)

    s_tensor = scaler.transform_tensor(torch.tensor(state))
    assert np.allclose(ref_s, s_tensor.numpy())

    reversed_s_tensor = scaler.reverse_transform_tensor(s_tensor)
    assert np.allclose(state, reversed_s_tensor.numpy())

    scaler1_param = scaler.get_params()
    assert np.all(scaler1_param['minimum'] == minimum)
    assert np.all(scaler1_param['maximum'] == maximum)


def test_standard_scaler():
    scaler = StandardScaler(data=state)
    _mean = np.mean(state, 0)
    _std = np.std(state, 0)
    s = scaler.transform(state)
    ref_s = (state - _mean) / (_std + eps)
    assert np.allclose(s, ref_s)

    reversed_s = scaler.reverse_transform(ref_s)
    assert np.allclose(state, reversed_s)

    s_tensor = scaler.transform_tensor(torch.tensor(state))
    assert np.allclose(ref_s, s_tensor.numpy(), atol=1e-6)

    reversed_s_tensor = scaler.reverse_transform_tensor(s_tensor)
    assert np.allclose(state, reversed_s_tensor.numpy())

    scaler_param = scaler.get_params()
    assert np.all(scaler_param['mean'] == _mean)
    assert np.all(scaler_param['std'] == _std)


def test_standard_scaler_with_mean_std():
    scaler = StandardScaler(mean=mean, std=std)
    s = scaler.transform(state)
    ref_s = (state - mean) / (std + eps)
    assert np.allclose(s, ref_s)

    reversed_s = scaler.reverse_transform(ref_s)
    assert np.allclose(state, reversed_s)

    s_tensor = scaler.transform_tensor(torch.tensor(state))
    assert np.allclose(ref_s, s_tensor.numpy(), atol=1e-6)

    reversed_s_tensor = scaler.reverse_transform_tensor(s_tensor)
    assert np.allclose(state, reversed_s_tensor.numpy())

    scaler_param = scaler.get_params()
    assert np.all(scaler_param['mean'] == mean)
    assert np.all(scaler_param['std'] == std)


if __name__ == '__main__':
    pytest.main(__file__)




