from typing import ClassVar
from d2c.envs.learned.dynamics import BaseDyna


class MlpDyna(BaseDyna):
    """MLP dynamics model."""
    TYPE: ClassVar[str] = 'mlp'
    pass
