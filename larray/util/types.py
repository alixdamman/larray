from typing import Union, TypeVar

from numpy import generic

T = TypeVar('T')
T_co = TypeVar('T_co', covariant=True)
Scalar = Union[bool, int, float, str, bytes, generic]
