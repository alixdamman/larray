from abc import ABC
from typing import Union, Sized

import numpy as np


# define abstract base classes to enable isinstance type checking on our objects
# idea taken from https://github.com/pandas-dev/pandas/blob/master/pandas/core/dtypes/generic.py
class ABCAxis(ABC, Sized):
    # @gdm: axis name can be an int ?
    name: Union[None, int, str]
    labels: np.ndarray


class ABCAxisReference(ABCAxis):
    pass


class ABCArray(ABC):
    data: np.ndarray
