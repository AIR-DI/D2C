import numpy as np


HOPPER_RANDOM_SCORE = -19.5
HOPPER_EXPERT_SCORE = 3234.3
HOPPER_STATE = (11, -np.inf, np.inf)  # (dimension, minimum, maximum)
HOPPER_ACTION = (3, -1, 1)  # (dimension, minimum, maximum)

WALKER2D_RANDOM_SCORE = -0.48
WALKER2D_EXPERT_SCORE = 4592.3
WALKER2D_STATE = (17, -np.inf, np.inf)  # (dimension, minimum, maximum)
WALKER2D_ACTION = (6, -1, 1)  # (dimension, minimum, maximum)

HALFCHEETAH_RANDOM_SCORE = -278.6
HALFCHEETAH_EXPERT_SCORE = 12135.0
HALFCHEETAH_STATE = (17, -np.inf, np.inf)  # (dimension, minimum, maximum)
HALFCHEETAH_ACTION = (6, -1, 1)  # (dimension, minimum, maximum)
