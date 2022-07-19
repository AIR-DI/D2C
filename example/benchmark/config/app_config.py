"""The config for the application"""

import numpy as np
from AIControlOpt_lib.utils.utils import Flags


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
    # state and action
    state_dim=None,
    state_max=np.inf,
    state_min=-np.inf,
    action_dim=None,
    action_max=None,
    action_min=None,
    done_function=pack_func(done_function),
    # data
    split_data_buffer=False,
)

