from __future__ import absolute_import, division, print_function

from unittest import TestCase
import unittest

import numpy as np
from larray import Session, Axis, LArray, ndrange, isnan, view, local_arrays
from larray.tests.test_la import assert_array_nan_equal


class TestSession(TestCase):
    def setUp(self):
        self.a = Axis('a', [])
        self.b = Axis('b', [])
        self.e = ndrange((2, 3)).rename(0, 'a0').rename(1, 'a1')
        self.f = ndrange((3, 2)).rename(0, 'a0').rename(1, 'a1')
        self.g = ndrange((2, 4)).rename(0, 'a0').rename(1, 'a1')
        self.session = Session(self.a, self.b, c='c', d={},
                               e=self.e, f=self.f, g=self.g)

    def test_getitem(self):
        s = self.session
        self.assertIs(s['a'], self.a)
        self.assertIs(s['b'], self.b)
        self.assertEqual(s['c'], 'c')
        self.assertEqual(s['d'], {})

    def test_getitem_list(self):
        s = self.session
        self.assertEqual(list(s[[]]), [])
        self.assertEqual(list(s[['a', 'b']]), [self.a, self.b])
        self.assertEqual(list(s[['a', 'e', 'g']]), [self.a, self.e, self.g])

    def test_getitem_larray(self):
        s1 = self.session.filter(kind=LArray)
        s2 = Session({'e': self.e + 1, 'f': self.f})
        res_eq = s1[s1 == s2]
        res_neq = s1[s1 != s2]
        self.assertEqual(list(res_eq), [self.f])
        self.assertEqual(list(res_neq), [self.e, self.g])

    def test_setitem(self):
        s = self.session
        s['g'] = 'g'
        self.assertEqual(s['g'], 'g')

    def test_getattr(self):
        s = self.session
        self.assertIs(s.a, self.a)
        self.assertIs(s.b, self.b)
        self.assertEqual(s.c, 'c')
        self.assertEqual(s.d, {})

    def test_setattr(self):
        s = self.session
        s.h = 'h'
        self.assertEqual(s.h, 'h')

    def test_add(self):
        s = self.session
        h = Axis('h', [])
        s.add(h, i='i')
        self.assertTrue(h.equals(s.h))
        self.assertEqual(s.i, 'i')

    def test_iter(self):
        self.assertEqual(list(self.session), [self.a, self.b, 'c', {},
                                              self.e, self.f, self.g])

    def test_filter(self):
        s = self.session
        s.ax = 'ax'
        self.assertEqual(list(s.filter()), [self.a, 'ax', self.b, 'c', {},
                                            self.e, self.f, self.g])
        self.assertEqual(list(s.filter('a')), [self.a, 'ax'])
        self.assertEqual(list(s.filter('a', dict)), [])
        self.assertEqual(list(s.filter('a', str)), ['ax'])
        self.assertEqual(list(s.filter('a', Axis)), [self.a])
        self.assertEqual(list(s.filter(kind=Axis)), [self.a, self.b])
        self.assertEqual(list(s.filter(kind=LArray)), [self.e, self.f, self.g])
        self.assertEqual(list(s.filter(kind=dict)), [{}])

    def test_names(self):
        s = self.session
        self.assertEqual(s.names, ['a', 'b', 'c', 'd', 'e', 'f', 'g'])
        # add them in the "wrong" order
        s.add(i='i')
        s.add(h='h')
        self.assertEqual(s.names, ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'])

    def test_dump(self):
        self.session.dump('test_session.h5')
        self.session.dump('test_session.xlsx')
        self.session.dump('test_session_ef.xlsx', ['e', 'f'])
        self.session.dump_excel('test_session2.xlsx')
        self.session.dump_csv('test_session_csv')

    def test_load(self):
        s = Session()
        s.load('test_session.h5', ['e', 'f'])
        self.assertEqual(s.names, ['e', 'f'])

        s = Session()
        s.load('test_session.h5')
        self.assertEqual(s.names, ['e', 'f', 'g'])

        s = Session()
        s.load('test_session_ef.xlsx')
        self.assertEqual(s.names, ['e', 'f'])

        s = Session()
        s.load('test_session_csv', engine='pandas_csv')
        self.assertEqual(s.names, ['e', 'f', 'g'])

    def test_eq(self):
        sess = self.session.filter(kind=LArray)
        expected = Session([('e', self.e), ('f', self.f), ('g', self.g)])
        self.assertTrue(all(sess == expected))

        other = Session({'e': self.e, 'f': self.f})
        res = sess == other
        self.assertEqual(res.ndim, 1)
        self.assertEqual(res.axes.names, ['name'])
        self.assertTrue(np.array_equal(res.axes.labels[0], ['e', 'f', 'g']))
        self.assertEqual(list(res), [True, True, False])

        e2 = self.e.copy()
        e2.i[1, 1] = 42
        other = Session({'e': e2, 'f': self.f})
        res = sess == other
        self.assertEqual(res.axes.names, ['name'])
        self.assertTrue(np.array_equal(res.axes.labels[0], ['e', 'f', 'g']))
        self.assertEqual(list(res), [False, True, False])

    def test_ne(self):
        sess = self.session.filter(kind=LArray)
        expected = Session([('e', self.e), ('f', self.f), ('g', self.g)])
        self.assertFalse(any(sess != expected))

        other = Session({'e': self.e, 'f': self.f})
        res = sess != other
        self.assertEqual(res.axes.names, ['name'])
        self.assertTrue(np.array_equal(res.axes.labels[0], ['e', 'f', 'g']))
        self.assertEqual(list(res), [False, False, True])

        e2 = self.e.copy()
        e2.i[1, 1] = 42
        other = Session({'e': e2, 'f': self.f})
        res = sess != other
        self.assertEqual(res.axes.names, ['name'])
        self.assertTrue(np.array_equal(res.axes.labels[0], ['e', 'f', 'g']))
        self.assertEqual(list(res), [True, False, True])

    def test_sub(self):
        sess = self.session.filter(kind=LArray)
        other = Session({'e': self.e - 1, 'f': 1})
        diff = sess - other
        assert_array_nan_equal(diff['e'], np.full((2, 3), 1, dtype=np.int32))
        assert_array_nan_equal(diff['f'], np.arange(-1, 5).reshape(3, 2))
        self.assertTrue(isnan(diff['g']).all())

    def test_div(self):
        sess = self.session.filter(kind=LArray)
        other = Session({'e': self.e - 1, 'f': self.f + 1})
        res = sess / other

        flat_e = np.arange(6) / np.arange(-1, 5)
        assert_array_nan_equal(res['e'], flat_e.reshape(2, 3))

        flat_f = np.arange(6) / np.arange(1, 7)
        assert_array_nan_equal(res['f'], flat_f.reshape(3, 2))
        self.assertTrue(isnan(res['g']).all())

    def test_init(self):
        s = Session('test_session.h5')
        self.assertEqual(s.names, ['e', 'f', 'g'])

        s = Session('test_session_ef.xlsx')
        self.assertEqual(s.names, ['e', 'f'])

        # TODO: format autodetection does not work in this case
        # s = Session('test_session_csv')
        # self.assertEqual(s.names, ['e', 'f', 'g'])

if __name__ == "__main__":
    # import doctest
    # doctest.testmod(larray.core)
    unittest.main()
