# copy-pasted from SQLAlchemy util/_collections.py

# Copyright (C) 2005-2015 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: http://www.opensource.org/licenses/mit-license.php

from typing import List, Iterator, Any

from larray.util.misc import unique_list


class OrderedSet(set):
    def __init__(self, d=None) -> None:
        set.__init__(self)
        if d is not None:
            self._list: List[Any] = unique_list(d)
            set.update(self, self._list)
        else:
            self._list = []

    def add(self, element) -> None:
        if element not in self:
            self._list.append(element)
        set.add(self, element)

    def remove(self, element) -> None:
        set.remove(self, element)
        self._list.remove(element)

    def insert(self, pos, element) -> None:
        if element not in self:
            self._list.insert(pos, element)
        set.add(self, element)

    def discard(self, element) -> None:
        if element in self:
            self._list.remove(element)
            set.remove(self, element)

    def clear(self) -> None:
        set.clear(self)
        self._list = []

    def __getitem__(self, key) -> Any:
        return self._list[key]

    def __iter__(self) -> Iterator[Any]:
        return iter(self._list)

    def __repr__(self) -> str:
        return '%s(%r)' % (self.__class__.__name__, self._list)

    __str__ = __repr__

    # added comment type: ignore because mypy expects the signature:
    # update(self, *iterable: Iterable[_T]) -> None
    def update(self, iterable) -> 'OrderedSet':                 # type: ignore
        for e in iterable:
            if e not in self:
                self._list.append(e)
                set.add(self, e)
        return self
    __ior__ = update

    def union(self, other) -> 'OrderedSet':                     # type: ignore
        result = self.__class__(self)
        result.update(other)
        return result
    __or__ = union
    __add__ = union

    def intersection(self, other) -> 'OrderedSet':              # type: ignore
        other = set(other)
        return self.__class__(a for a in self if a in other)
    __and__ = intersection

    def symmetric_difference(self, other) -> 'OrderedSet':      # type: ignore
        other = set(other)
        result = self.__class__(a for a in self if a not in other)
        result.update(a for a in other if a not in self)
        return result
    __xor__ = symmetric_difference

    def difference(self, other) -> 'OrderedSet':                # type: ignore
        other = set(other)
        return self.__class__(a for a in self if a not in other)
    __sub__ = difference

    def intersection_update(self, other) -> 'OrderedSet':       # type: ignore
        other = set(other)
        set.intersection_update(self, other)
        self._list = [a for a in self._list if a in other]
        return self
    __iand__ = intersection_update

    def symmetric_difference_update(self, other) -> 'OrderedSet':   # type: ignore
        set.symmetric_difference_update(self, other)
        self._list = [a for a in self._list if a in self]
        self._list += [a for a in other._list if a in self]
        return self
    __ixor__ = symmetric_difference_update

    def difference_update(self, other) -> 'OrderedSet':         # type: ignore
        set.difference_update(self, other)
        self._list = [a for a in self._list if a in self]
        return self
    __isub__ = difference_update
