"""The config for the application"""

import os
import numpy as np
from d2c.utils.utils import Flags
from d2c.utils.utils import abs_file_path


work_abs_dir = abs_file_path(__file__, '../')
data_path = os.path.join(work_abs_dir, 'data', 'cold_source-low-refrig-unit_num_1.csv')
s_min = [0, 5, 5, 5, 200, 200, 0, 5, 5, 5]
s_max = [3, 40, 40, 40, 900, 900, 100, 40, 40, 40]
a_min = [0, 0]
a_max = [50, 50]


app_config = Flags(
    data_path=data_path,
    state_indices=np.arange(0, 10),
    action_indices=np.arange(10, 12),
    state_scaler='min_max',
    state_scaler_params=dict(minimum=s_min, maximum=s_max),
    action_scaler='min_max',
    action_scaler_params=dict(minimum=a_min, maximum=a_max),
    reward_scaler=None,
    reward_scaler_params=None,
    reward_fn=None,
    cost_fn=None,
    done_fn=None,
)

