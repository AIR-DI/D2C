"""
Base evaluator for trained policy.
"""
from abc import ABC, abstractmethod
from typing import ClassVar


class BaseEval(ABC):
    """The base class of the evaluator.

    This is used for policy evaluation. There are two main classes
    of evaluation methods:

    1. Use the simulator to evaluate the policy;
    2. Use offline policy evaluation methods to evaluate the policy.
    """

    TYPE: ClassVar[str] = 'none'

    @abstractmethod
    def _eval_policies(self, *args):
        pass

    @abstractmethod
    def eval(self, *args):
        """The API for evaluating."""
        pass