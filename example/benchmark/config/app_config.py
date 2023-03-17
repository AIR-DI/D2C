"""The config for the application"""

import numpy as np
from d2c.utils.utils import Flags


# def reward_function(past_a, s, a, next_s):
#     r = a[:, 0] + next_s[:, 0]
#     return r
#
#
# def cost_function(past_a, s, a, next_s):
#     c = a[:, 0] - past_a[:, 0]
#     return c
#
#
def done_function(past_a, s, a, next_s):
    return np.zeros(past_a.shape[0], dtype='float32')


def pack_func(init_func):
    def func(past_a, s, a, next_s):
        input_dim = len(past_a.shape)
        args = [past_a, s, a, next_s]
        if input_dim == 1:
            args = [x.reshape(1, -1) for x in args]
            return init_func(*args)[0]
        elif input_dim == 2:
            return init_func(*args)
        raise ValueError('Wrong shape for inputs!')
    return func


app_config = Flags(
    data_path=None,
    state_indices=None,
    action_indices=None,
    state_scaler='min_max',
    state_scaler_params=None,
    action_scaler='min_max',
    action_scaler_params=None,
    reward_scaler='min_max',
    reward_scaler_params=None,
    reward_fn=None,
    cost_fn=None,
    done_fn=done_function,
)

