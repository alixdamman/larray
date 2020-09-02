from typing import Any

import numpy as np


def isscalar(element: Any) -> bool:
    """
    Returns `True` if the type of element is a scalar type.

    Parameters
    ----------
    element: any
        Input argument, can be of any type and shape.

    Returns
    -------
    bool
        `True` if `element` is a scalar type, `False` if it is not.

    Examples
    --------
    >>> isscalar(3.1)
    True
    >>> isscalar(np.array(3.1))
    False
    >>> isscalar([3.1])
    False
    >>> isscalar(False)
    True
    >>> isscalar('larray')
    True
    >>> from larray import ndtest
    >>> isscalar(ndtest((2, 2)))
    False
    """
    return np.isscalar(element)
