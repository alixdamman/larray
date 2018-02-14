from __future__ import absolute_import, division, print_function

import os
import sys
from unittest import TestCase

import pytest
import numpy as np
import pandas as pd

from larray import zeros, Session, to_session


@to_session
class Input():
    def __init__(self):
        self.arr0 = zeros((2, 2))
        self.arr1 = zeros((3, 3))
        self.arr2 = zeros((4, 4))
    def another_method(self):
        pass

i = Input()

print(dir(i))


