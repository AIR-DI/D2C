from typing import Dict, Type
from d2c.evaluators.base import BaseEval
from d2c.evaluators.ope.fqe import FQEval
from d2c.evaluators.ope.mb_ope import MBOPE


OPE_DICT: Dict[str, Type[BaseEval]] = {}


def register_ope(cls: Type[BaseEval]) -> None:
    """Registering the OPE methods.

    :param cls: OPE class inheriting ``BaseEval``.
    """
    is_registered = cls.TYPE in OPE_DICT
    assert not is_registered, f'{cls.TYPE} seems to be already registered.'
    OPE_DICT[cls.TYPE] = cls


register_ope(FQEval)
register_ope(MBOPE)


def make_ope(name: str, from_config: bool, **kwargs) -> BaseEval:
    """Creating the OPE.

    :param str name: The name of the registered OPE type. The available types are: 'fqe'.
    :param bool from_config: If using the config to create the OPE.
    :param kwargs: The OPE arguments.
    :return: An OPE object.
    """
    assert name in OPE_DICT, f'{name} seems not to be registered.'
    if from_config:
        ope = OPE_DICT[name].from_config(**kwargs)
    else:
        ope = OPE_DICT[name](**kwargs)
    assert isinstance(ope, BaseEval)
    return ope





