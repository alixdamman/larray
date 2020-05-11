from __future__ import absolute_import, division, print_function

import os
import shutil
from datetime import date, time, datetime

import numpy as np
import pandas as pd
import pytest

from larray.tests.common import (assert_array_nan_equal, inputpath, tmp_path, meta,
                                 needs_xlwings, needs_pytables, needs_xlrd)
from larray.inout.common import _supported_scalars_types
from larray import (Session, Axis, AxisCollection, Array, Group, isnan, zeros_like, ndtest, ones_like, ones, full,
                    full_like, stack, local_arrays, global_arrays, arrays, ConstrainedSession, ConstrainedArray)
from larray.util.compat import pickle, PY2


def equal(o1, o2):
    if isinstance(o1, Array) or isinstance(o2, Array):
        return o1.equals(o2)
    elif isinstance(o1, Axis) or isinstance(o2, Axis):
        return o1.equals(o2)
    else:
        return o1 == o2


def assertObjListEqual(got, expected):
    assert len(got) == len(expected)
    for e1, e2 in zip(got, expected):
        assert equal(e1, e2), "{} != {}".format(e1, e2)


a = Axis('a=a0..a2')
a2 = Axis('a=a0..a4')
a3 = Axis('a=a0..a3')
anonymous = Axis(4)
a01 = a['a0,a1'] >> 'a01'
ano01 = a['a0,a1']
b = Axis('b=0..4')
b2 = Axis('b=b0..b4')
b024 = b[[0, 2, 4]] >> 'b024'
c = 'c'
d = {}
e = ndtest([(2, 'a'), (3, 'b')])
_e = ndtest((3, 3))
f = ndtest((Axis(3), Axis(2)), dtype=float)
g = ndtest([(2, 'a'), (4, 'b')])
h = ndtest((a3, b2))
k = ndtest((3, 3))

# ########################### #
#           SESSION           #
# ########################### #


@pytest.fixture()
def session():
    return Session([('b', b), ('b024', b024), ('a', a), ('a2', a2), ('anonymous', anonymous),
                    ('a01', a01), ('ano01', ano01), ('c', c), ('d', d), ('e', e), ('g', g), ('f', f), ('h', h)])


def test_init_session(meta):
    s = Session(b, b024, a, a01, a2=a2, anonymous=anonymous, ano01=ano01, c=c, d=d, e=e, g=g, f=f, h=h)
    assert list(s.keys()) == ['b', 'b024', 'a', 'a01', 'a2', 'anonymous', 'ano01', 'c', 'd', 'e', 'g', 'f', 'h']

    # TODO: format autodetection does not work in this case
    # s = Session('test_session_csv')
    # assert list(s.keys()) == ['e', 'f', 'g']

    # metadata
    s = Session(b, b024, a, a01, a2=a2, anonymous=anonymous, ano01=ano01, c=c, d=d, e=e, f=f, g=g, h=h, meta=meta)
    assert s.meta == meta


@needs_xlwings
def test_init_session_xlsx():
    s = Session(inputpath('demography_eurostat.xlsx'))
    assert list(s.keys()) == ['population', 'population_benelux', 'population_5_countries',
                              'births', 'deaths', 'immigration']


@needs_pytables
def test_init_session_hdf():
    s = Session(inputpath('test_session.h5'))
    assert list(s.keys()) == ['e', 'f', 'g', 'h', 'a', 'a2', 'anonymous', 'b', 'a01', 'ano01', 'b024']


def test_getitem(session):
    assert session['a'] is a
    assert session['a2'] is a2
    assert session['anonymous'] is anonymous
    assert session['b'] is b
    assert session['a01'] is a01
    assert session['ano01'] is ano01
    assert session['b024'] is b024
    assert session['c'] == 'c'
    assert session['d'] == {}
    assert equal(session['e'], e)
    assert equal(session['h'], h)


def test_getitem_list(session):
    assert list(session[[]]) == []
    assert list(session[['b', 'a']]) == [b, a]
    assert list(session[['a', 'b']]) == [a, b]
    assert list(session[['a', 'a2']]) == [a, a2]
    assert list(session[['anonymous', 'ano01']]) == [anonymous, ano01]
    assert list(session[['b024', 'a']]) == [b024, a]
    assert list(session[['e', 'a01']]) == [e, a01]
    assert list(session[['a', 'e', 'g']]) == [a, e, g]
    assert list(session[['g', 'a', 'e']]) == [g, a, e]


def test_getitem_larray(session):
    s1 = session.filter(kind=Array)
    s2 = Session({'e': e + 1, 'f': f})
    res_eq = s1[s1.element_equals(s2)]
    res_neq = s1[~(s1.element_equals(s2))]
    assert list(res_eq) == [f]
    assert list(res_neq) == [e, g, h]


def test_setitem(session):
    s = session.copy()
    s['g'] = 'g'
    assert s['g'] == 'g'


def test_getattr(session):
    assert session.a is a
    assert session.a2 is a2
    assert session.anonymous is anonymous
    assert session.b is b
    assert session.a01 is a01
    assert session.ano01 is ano01
    assert session.b024 is b024
    assert session.c == 'c'
    assert session.d == {}


def test_setattr(session):
    s = session.copy()
    s.i = 'i'
    assert s.i == 'i'


def test_add(session):
    i = Axis('i=i0..i2')
    i01 = i['i0,i1'] >> 'i01'
    session.add(i, i01, j='j')
    assert i.equals(session.i)
    assert i01 == session.i01
    assert session.j == 'j'


def test_iter(session):
    expected = [b, b024, a, a2, anonymous, a01, ano01, c, d, e, g, f, h]
    assertObjListEqual(session, expected)


def test_filter(session):
    session.ax = 'ax'
    assertObjListEqual(session.filter(), [b, b024, a, a2, anonymous, a01, ano01, 'c', {}, e, g, f, h, 'ax'])
    assertObjListEqual(session.filter('a*'), [a, a2, anonymous, a01, ano01, 'ax'])
    assert list(session.filter('a*', dict)) == []
    assert list(session.filter('a*', str)) == ['ax']
    assert list(session.filter('a*', Axis)) == [a, a2, anonymous]
    assert list(session.filter(kind=Axis)) == [b, a, a2, anonymous]
    assert list(session.filter('a01', Group)) == [a01]
    assert list(session.filter(kind=Group)) == [b024, a01, ano01]
    assertObjListEqual(session.filter(kind=Array), [e, g, f, h])
    assert list(session.filter(kind=dict)) == [{}]
    assert list(session.filter(kind=(Axis, Group))) == [b, b024, a, a2, anonymous, a01, ano01]


def test_names(session):
    assert session.names == ['a', 'a01', 'a2', 'ano01', 'anonymous', 'b', 'b024',
                             'c', 'd', 'e', 'f', 'g', 'h']
    # add them in the "wrong" order
    session.add(j='j')
    session.add(i='i')
    assert session.names == ['a', 'a01', 'a2', 'ano01', 'anonymous', 'b', 'b024',
                             'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']


def _test_io(tmpdir, session, meta, engine, ext):
    filename = f"test_{engine}.{ext}" if 'csv' not in engine else f"test_{engine}{ext}"
    fpath = tmp_path(tmpdir, filename)

    is_excel_or_csv = 'excel' in engine or 'csv' in engine

    kind = Array if is_excel_or_csv else (Axis, Group, Array) + _supported_scalars_types
    session = session.filter(kind=kind)

    session.meta = meta

    # save and load
    session.save(fpath, engine=engine)
    s = Session()
    s.load(fpath, engine=engine)
    # use Session.names instead of Session.keys because CSV, Excel and HDF do *not* keep ordering
    assert s.names == session.names
    assert s.equals(session)
    if not PY2 and not is_excel_or_csv:
        for key in s.filter(kind=Axis).keys():
            assert s[key].dtype == session[key].dtype
    if engine != 'pandas_excel':
        assert s.meta == meta

    # update a Group + an Axis + an array (overwrite=False)
    a4 = Axis('a=0..3')
    a4_01 = a3['0,1'] >> 'a01'
    e2 = ndtest((a4, 'b=b0..b2'))
    h2 = full_like(h, fill_value=10)
    Session(a=a4, a01=a4_01, e=e2, h=h2).save(fpath, overwrite=False, engine=engine)
    s = Session()
    s.load(fpath, engine=engine)
    if engine == 'pandas_excel':
        # Session.save() via engine='pandas_excel' always overwrite the output Excel files
        assert s.names == ['e', 'h']
    elif is_excel_or_csv:
        assert s.names == ['e', 'f', 'g', 'h']
    else:
        assert s.names == session.names
        assert s['a'].equals(a4)
        assert s['a01'].equals(a4_01)
    assert_array_nan_equal(s['e'], e2)
    if engine != 'pandas_excel':
        assert s.meta == meta

    # load only some objects
    session.save(fpath, engine=engine)
    s = Session()
    names_to_load = ['e', 'f'] if is_excel_or_csv else ['a', 'a01', 'a2', 'anonymous', 'e', 'f', 's_bool', 's_int']
    s.load(fpath, names=names_to_load, engine=engine)
    assert s.names == names_to_load
    if engine != 'pandas_excel':
        assert s.meta == meta

    return fpath


def _add_scalars_to_session(s):
    # 's' for scalar
    s['s_int'] = 5
    s['s_float'] = 5.5
    s['s_bool'] = True
    s['s_str'] = 'string'
    s['s_date'] = date(2020, 1, 10)
    s['s_time'] = time(11, 23, 54)
    s['s_datetime'] = datetime(2020, 1, 10, 11, 23, 54)
    return s


@needs_pytables
def test_h5_io(tmpdir, session, meta):
    session = _add_scalars_to_session(session)
    _test_io(tmpdir, session, meta, engine='pandas_hdf', ext='h5')


@needs_xlrd
def test_xlsx_pandas_io(tmpdir, session, meta):
    _test_io(tmpdir, session, meta, engine='pandas_excel', ext='xlsx')


@needs_xlwings
def test_xlsx_xlwings_io(tmpdir, session, meta):
    _test_io(tmpdir, session, meta, engine='xlwings_excel', ext='xlsx')


def test_csv_io(tmpdir, session, meta):
    try:
        fpath = _test_io(tmpdir, session, meta, engine='pandas_csv', ext='csv')

        names = Session({k: v for k, v in session.items() if isinstance(v, Array)}).names

        # test loading with a pattern
        pattern = os.path.join(fpath, '*.csv')
        s = Session(pattern)
        assert s.names == names
        assert s.meta == meta

        # create an invalid .csv file
        invalid_fpath = os.path.join(fpath, 'invalid.csv')
        with open(invalid_fpath, 'w') as f:
            f.write(',",')

        # try loading the directory with the invalid file
        with pytest.raises(pd.errors.ParserError) as e_info:
            s = Session(pattern)

        # test loading a pattern, ignoring invalid/unsupported files
        s = Session()
        s.load(pattern, ignore_exceptions=True)
        assert s.names == names
        assert s.meta == meta
    finally:
        shutil.rmtree(fpath)


def test_pickle_io(tmpdir, session, meta):
    session = _add_scalars_to_session(session)
    _test_io(tmpdir, session, meta, engine='pickle', ext='pkl')


def test_pickle_roundtrip(session, meta):
    original = session.filter(kind=Array)
    original.meta = meta
    s = pickle.dumps(original)
    res = pickle.loads(s)
    assert res.equals(original)
    assert res.meta == meta


def test_element_equals(session):
    session_cls = session.__class__
    other_session = session_cls([(key, value) for key, value in session.items()])

    keys = [key for key, value in session.items() if isinstance(value, (Axis, Group, Array))]
    expected_res = full(Axis(keys, 'name'), fill_value=True, dtype=bool)

    # ====== same sessions ======
    res = session.element_equals(other_session)
    assert res.axes == expected_res.axes
    assert res.equals(expected_res)

    # ====== session with missing/extra items ======
    # delete some items
    for deleted_key in ['b', 'b024', 'g']:
        del other_session[deleted_key]
        expected_res[deleted_key] = False
    # add one item
    other_session['k'] = k
    expected_res = expected_res.append('name', False, label='k')

    res = session.element_equals(other_session)
    assert res.axes == expected_res.axes
    assert res.equals(expected_res)

    # ====== session with a modified array ======
    h2 = h.copy()
    h2['a1', 'b1'] = 42
    other_session['h'] = h2
    expected_res['h'] = False

    res = session.element_equals(other_session)
    assert res.axes == expected_res.axes
    assert res.equals(expected_res)


def to_boolean_array_eq(res):
    return stack([(key, item.all() if isinstance(item, Array) else item)
                  for key, item in res.items()], 'name')


def test_eq(session):
    session_cls = session.__class__
    other_session = session_cls([(key, value) for key, value in session.items()])
    expected_res = full(Axis(list(session.keys()), 'name'), fill_value=True, dtype=bool)

    # ====== same sessions ======
    res = session == other_session
    res = to_boolean_array_eq(res)
    assert res.axes == expected_res.axes
    assert res.equals(expected_res)

    # ====== session with missing/extra items ======
    del other_session['g']
    expected_res['g'] = False
    other_session['k'] = k
    expected_res = expected_res.append('name', False, label='k')

    res = session == other_session
    res = to_boolean_array_eq(res)
    assert res.axes == expected_res.axes
    assert res.equals(expected_res)

    # ====== session with a modified array ======
    h2 = h.copy()
    h2['a1', 'b1'] = 42
    other_session['h'] = h2
    expected_res['h'] = False

    res = session == other_session
    assert res['h'].equals(session['h'] == other_session['h'])
    res = to_boolean_array_eq(res)
    assert res.axes == expected_res.axes
    assert res.equals(expected_res)


def to_boolean_array_ne(res):
    return stack([(key, item.any() if isinstance(item, Array) else item)
                  for key, item in res.items()], 'name')


def test_ne(session):
    session_cls = session.__class__
    other_session = session_cls([(key, value) for key, value in session.items()])
    expected_res = full(Axis(list(session.keys()), 'name'), fill_value=False, dtype=bool)

    # ====== same sessions ======
    res = session != other_session
    res = to_boolean_array_ne(res)
    assert res.axes == expected_res.axes
    assert res.equals(expected_res)

    # ====== session with missing/extra items ======
    del other_session['g']
    expected_res['g'] = True
    other_session['k'] = k
    expected_res = expected_res.append('name', True, label='k')

    res = session != other_session
    res = to_boolean_array_ne(res)
    assert res.axes == expected_res.axes
    assert res.equals(expected_res)

    # ====== session with a modified array ======
    h2 = h.copy()
    h2['a1', 'b1'] = 42
    other_session['h'] = h2
    expected_res['h'] = True

    res = session != other_session
    assert res['h'].equals(session['h'] != other_session['h'])
    res = to_boolean_array_ne(res)
    assert res.axes == expected_res.axes
    assert res.equals(expected_res)


def test_sub(session):
    sess = session

    # session - session
    other = Session({'e': e, 'f': f})
    other['e'] = e - 1
    other['f'] = ones_like(f)
    diff = sess - other
    assert_array_nan_equal(diff['e'], np.full((2, 3), 1, dtype=np.int32))
    assert_array_nan_equal(diff['f'], f - ones_like(f))
    assert isnan(diff['g']).all()
    assert diff.a is a
    assert diff.a01 is a01
    assert diff.c is c

    # session - scalar
    diff = sess - 2
    assert_array_nan_equal(diff['e'], e - 2)
    assert_array_nan_equal(diff['f'], f - 2)
    assert_array_nan_equal(diff['g'], g - 2)
    assert diff.a is a
    assert diff.a01 is a01
    assert diff.c is c

    # session - dict(Array and scalar)
    other = {'e': ones_like(e), 'f': 1}
    diff = sess - other
    assert_array_nan_equal(diff['e'], e - ones_like(e))
    assert_array_nan_equal(diff['f'], f - 1)
    assert isnan(diff['g']).all()
    assert diff.a is a
    assert diff.a01 is a01
    assert diff.c is c

    # session - array
    axes = [a, b]
    other = Session([('a', a), ('a01', a01), ('c', c), ('e', ndtest((a, b))),
                     ('f', full((a, b), fill_value=3)), ('g', ndtest('c=c0..c2'))])
    diff = other - ones(axes)
    assert_array_nan_equal(diff['e'], other['e'] - ones(axes))
    assert_array_nan_equal(diff['f'], other['f'] - ones(axes))
    assert_array_nan_equal(diff['g'], other['g'] - ones(axes))
    assert diff.a is a
    assert diff.a01 is a01
    assert diff.c is c


def test_rsub(session):
    sess = session

    # scalar - session
    diff = 2 - sess
    assert_array_nan_equal(diff['e'], 2 - e)
    assert_array_nan_equal(diff['f'], 2 - f)
    assert_array_nan_equal(diff['g'], 2 - g)
    assert diff.a is a
    assert diff.a01 is a01
    assert diff.c is c

    # dict(Array and scalar) - session
    other = {'e': ones_like(e), 'f': 1}
    diff = other - sess
    assert_array_nan_equal(diff['e'], ones_like(e) - e)
    assert_array_nan_equal(diff['f'], 1 - f)
    assert isnan(diff['g']).all()
    assert diff.a is a
    assert diff.a01 is a01
    assert diff.c is c


def test_div(session):
    sess = session
    session_cls = session.__class__

    other = session_cls({'e': e, 'f': f})
    other['e'] = e - 1
    other['f'] = f + 1

    with pytest.warns(RuntimeWarning) as caught_warnings:
        res = sess / other
    assert len(caught_warnings) == 1
    assert caught_warnings[0].message.args[0] == "divide by zero encountered during operation"
    assert caught_warnings[0].filename == __file__

    with np.errstate(divide='ignore', invalid='ignore'):
        flat_e = np.arange(6) / np.arange(-1, 5)
    assert_array_nan_equal(res['e'], flat_e.reshape(2, 3))

    flat_f = np.arange(6) / np.arange(1, 7)
    assert_array_nan_equal(res['f'], flat_f.reshape(3, 2))
    assert isnan(res['g']).all()


def test_rdiv(session):
    sess = session

    # scalar / session
    res = 2 / sess
    assert_array_nan_equal(res['e'], 2 / e)
    assert_array_nan_equal(res['f'], 2 / f)
    assert_array_nan_equal(res['g'], 2 / g)
    assert res.a is a
    assert res.a01 is a01
    assert res.c is c

    # dict(Array and scalar) - session
    other = {'e': e, 'f': f}
    res = other / sess
    assert_array_nan_equal(res['e'], e / e)
    assert_array_nan_equal(res['f'], f / f)
    assert res.a is a
    assert res.a01 is a01
    assert res.c is c


def test_to_globals(session):
    with pytest.warns(RuntimeWarning) as caught_warnings:
        session.to_globals()
    assert len(caught_warnings) == 1
    assert caught_warnings[0].message.args[0] == "Session.to_globals should usually only be used in interactive " \
                                                 "consoles and not in scripts. Use warn=False to deactivate this " \
                                                 "warning."
    assert caught_warnings[0].filename == __file__

    assert a is session.a
    assert b is session.b
    assert c is session.c
    assert d is session.d
    assert e is session.e
    assert f is session.f
    assert g is session.g

    # test inplace
    backup_dest = e
    backup_value = session.e.copy()
    session.e = zeros_like(e)
    session.to_globals(inplace=True, warn=False)
    # check the variable is correct (the same as before)
    assert e is backup_dest
    assert e is not session.e
    # check the content has changed
    assert_array_nan_equal(e, session.e)
    assert not e.equals(backup_value)
    # reset e to its original value
    e[:] = backup_value


def test_local_arrays():
    h = ndtest(2)
    _h = ndtest(3)

    # exclude private local arrays
    s = local_arrays()
    s_expected = Session([('h', h)])
    assert s.equals(s_expected)

    # all local arrays
    s = local_arrays(include_private=True)
    s_expected = Session([('h', h), ('_h', _h)])
    assert s.equals(s_expected)


def test_global_arrays():
    # exclude private global arrays
    s = global_arrays()
    s_expected = Session([('e', e), ('f', f), ('g', g), ('h', h), ('k', k)])
    assert s.equals(s_expected)

    # all global arrays
    s = global_arrays(include_private=True)
    s_expected = Session([('e', e), ('_e', _e), ('f', f), ('g', g), ('h', h), ('k', k)])
    assert s.equals(s_expected)


def test_arrays():
    i = ndtest(2)
    _i = ndtest(3)

    # exclude private arrays
    s = arrays()
    s_expected = Session([('e', e), ('f', f), ('g', g), ('h', h), ('i', i), ('k', k)])
    assert s.equals(s_expected)

    # all arrays
    s = arrays(include_private=True)
    s_expected = Session([('_e', _e), ('_i', _i), ('e', e), ('f', f), ('g', g), ('h', h), ('i', i), ('k', k)])
    assert s.equals(s_expected)


# ############################ #
#      CONSTRAINEDSESSION      #
# ############################ #

class TestConstrainedSession(ConstrainedSession):
    b = b
    b024 = b024
    a: Axis
    a2: Axis
    anonymous = anonymous
    a01: Group
    ano01 = ano01
    c: str = c
    d = dict()
    e: Array
    g: Array
    f: ConstrainedArray((Axis(3), Axis(2)))
    h: ConstrainedArray((a3, b2), dtype=int)


@pytest.fixture()
def constrainedsession():
    return TestConstrainedSession(a=a, a2=a2, a01=a01, e=e, g=g, f=f, h=h)


def test_create_constrainedsession_instance(meta):
    declared_variable_keys = {'b', 'b024', 'a', 'a2', 'anonymous', 'a01', 'ano01', 'c', 'd', 'e', 'g', 'f', 'h'}

    cs = TestConstrainedSession(a, a01, a2=a2, e=e, f=f, g=g, h=h)
    assert set(cs.keys()) == declared_variable_keys
    assert cs.b.equals(b)
    assert cs.b024.equals(b024)
    assert cs.a.equals(a)
    assert cs.a2.equals(a2)
    assert cs.anonymous.equals(anonymous)
    assert cs.a01.equals(a01)
    assert cs.ano01.equals(ano01)
    assert cs.c == c
    assert cs.d == d
    assert cs.e.equals(e)
    assert cs.g.equals(g)
    assert cs.f.equals(f)
    assert cs.h.equals(h)

    # metadata
    cs = TestConstrainedSession(a, a01, a2=a2, e=e, f=f, g=g, h=h, meta=meta)
    assert cs.meta == meta

    # passing a scalar to set all elements a ConstrainedArray
    cs = TestConstrainedSession(a, a01, a2=a2, e=e, f=f, g=g, h=5)
    assert cs.h.axes == AxisCollection((a3, b2))
    assert cs.h.equals(full(axes=(a3, b2), fill_value=5))

    # passing an array with wrong dtype to set a ConstrainedArray
    with pytest.warns(UserWarning) as caught_warnings:
        cs = TestConstrainedSession(a, a01, a2=a2, e=e, f=f, g=g, h=ones((a3, b2)))
    assert caught_warnings[0].message.args[0] == "Expected array or scalar of dtype int32 for the array 'h' " \
                                                 "but got array or scalar of dtype float64"

    # add the undeclared variable 'i'
    with pytest.warns(UserWarning) as caught_warnings:
        cs = TestConstrainedSession(a, a01, a2=a2, i=5, e=e, f=f, g=g, h=h)
    assert caught_warnings[0].message.args[0] == f"'i' is not declared in '{cs.__class__.__name__}'"
    assert set(cs.keys()) == declared_variable_keys | {'i'}


@needs_pytables
def test_init_constrainedsession_hdf():
    cs = TestConstrainedSession(inputpath('test_session.h5'))
    assert set(cs.keys()) == {'b', 'b024', 'a', 'a2', 'anonymous', 'a01', 'ano01', 'c', 'd', 'e', 'g', 'f', 'h'}


def test_getitem_cs(constrainedsession):
    test_getitem(constrainedsession)


def test_setitem_cs(constrainedsession):
    cs = constrainedsession

    # only change values of an array -> OK
    cs['h'] = zeros_like(h)

    # trying to add an undeclared variable -> prints a warning message
    with pytest.warns(UserWarning) as caught_warnings:
        cs['i'] = ndtest((3, 3))
    assert len(caught_warnings) == 1
    assert caught_warnings[0].message.args[0] == f"'i' is not declared in '{cs.__class__.__name__}'"

    # trying to set a variable with an object of different type -> should fail
    # a) type given explicitly
    # -> Axis
    expected_error_msg = "instance of Axis expected"
    with pytest.raises(TypeError) as error:
        cs['a'] = 0
    assert str(error.value) == expected_error_msg
    # -> ConstrainedArray
    expected_error_msg = "Expected object of type 'Array' or a scalar for the variable 'h' " \
                         "but got object of type 'ndarray'"
    with pytest.raises(TypeError) as error:
        cs['h'] = h.data
    assert str(error.value) == expected_error_msg
    # b) type deduced from the given default value
    expected_error_msg = "instance of Axis expected"
    with pytest.raises(TypeError) as error:
        cs['b'] = ndtest((3, 3))
    assert str(error.value) == expected_error_msg

    # trying to set a ConstrainedArray variable using a scalar -> OK
    cs['h'] = 5

    # trying to set a ConstrainedArray variable using an array with axes in different order -> OK
    cs['h'] = h.transpose()

    # trying to set a ConstrainedArray variable using an array with wrong axes -> should fail
    # a) missing axis
    expected_error_msg = "Array 'h' was declared with axes {a, b} but got array with missing axis {b}"
    with pytest.raises(ValueError) as error:
        cs['h'] = ndtest(a3)
    assert str(error.value) == expected_error_msg
    # b) extra axis
    expected_error_msg = "Array 'h' was declared with axes {a, b} but got array with extra axis {c}"
    with pytest.raises(ValueError) as error:
        cs['h'] = ndtest((a3, b2, 'c=c0..c2'))
    assert str(error.value) == expected_error_msg
    # c) incompatible axis
    expected_error_msg = """\
Incompatible axis for array 'h':
Axis(['a0', 'a1', 'a2', 'a3', 'a4'], 'a')
vs
Axis(['a0', 'a1', 'a2', 'a3'], 'a')"""
    with pytest.raises(ValueError) as error:
        cs['h'] = h.append('a', 0, 'a4')
    assert str(error.value) == expected_error_msg

    # trying to set a ConstrainedArray variable using an array with wrong dtype -> print a warning
    with pytest.warns(UserWarning) as caught_warnings:
        cs['h'] = ndtest((a3, b2), dtype=float)
    assert len(caught_warnings) == 1
    assert caught_warnings[0].message.args[0] == "Expected array or scalar of dtype int32 for the array 'h' " \
                                                 "but got array or scalar of dtype float64"


def test_getattr_cs(constrainedsession):
    test_getattr(constrainedsession)


def test_setattr_cs(constrainedsession):
    cs = constrainedsession

    # only change values of an array -> OK
    cs.h = zeros_like(h)

    # trying to add an undeclared variable -> prints a warning message
    with pytest.warns(UserWarning) as caught_warnings:
        cs.i = ndtest((3, 3))
    assert len(caught_warnings) == 1
    assert caught_warnings[0].message.args[0] == f"'i' is not declared in '{cs.__class__.__name__}'"

    # trying to set a variable with an object of different type -> should fail
    # a) type given explicitly
    # -> Axis
    expected_error_msg = "instance of Axis expected"
    with pytest.raises(TypeError) as error:
        cs.a = 0
    assert str(error.value) == expected_error_msg
    # -> ConstrainedArray
    expected_error_msg = "Expected object of type 'Array' or a scalar for the variable 'h' " \
                         "but got object of type 'ndarray'"
    with pytest.raises(TypeError) as error:
        cs.h = h.data
    assert str(error.value) == expected_error_msg
    # b) type deduced from the given default value
    expected_error_msg = "instance of Axis expected"
    with pytest.raises(TypeError) as error:
        cs.b = ndtest((3, 3))
    assert str(error.value) == expected_error_msg

    # trying to set a ConstrainedArray variable using a scalar -> OK
    cs.h = 5

    # trying to set a ConstrainedArray variable using an array with axes in different order -> OK
    cs.h = h.transpose()

    # trying to set a ConstrainedArray variable using an array with wrong axes -> should fail
    # a) missing axis
    expected_error_msg = "Array 'h' was declared with axes {a, b} but got array with missing axis {b}"
    with pytest.raises(ValueError) as error:
        cs.h = ndtest(a3)
    assert str(error.value) == expected_error_msg
    # b) extra axis
    expected_error_msg = "Array 'h' was declared with axes {a, b} but got array with extra axis {c}"
    with pytest.raises(ValueError) as error:
        cs.h = ndtest((a3, b2, 'c=c0..c2'))
    assert str(error.value) == expected_error_msg
    # c) incompatible axis
    expected_error_msg = """\
Incompatible axis for array 'h':
Axis(['a0', 'a1', 'a2', 'a3', 'a4'], 'a')
vs
Axis(['a0', 'a1', 'a2', 'a3'], 'a')"""
    with pytest.raises(ValueError) as error:
        cs.h = h.append('a', 0, 'a4')
    assert str(error.value) == expected_error_msg

    # trying to set a ConstrainedArray variable using an array with wrong dtype -> print a warning
    with pytest.warns(UserWarning) as caught_warnings:
        cs.h = ndtest((a3, b2), dtype=float)
    assert len(caught_warnings) == 1
    assert caught_warnings[0].message.args[0] == "Expected array or scalar of dtype int32 for the array 'h' " \
                                                 "but got array or scalar of dtype float64"


def test_add_cs(constrainedsession):
    cs = constrainedsession
    cs_class_name = cs.__class__.__name__

    with pytest.warns(UserWarning) as caught_warnings:
        test_add(cs)
    assert len(caught_warnings) == 3
    assert caught_warnings[0].message.args[0] == f"'i' is not declared in '{cs_class_name}'"
    assert caught_warnings[1].message.args[0] == f"'i01' is not declared in '{cs_class_name}'"
    assert caught_warnings[2].message.args[0] == f"'j' is not declared in '{cs_class_name}'"


def test_iter_cs(constrainedsession):
    test_iter(constrainedsession)


def test_filter_cs(constrainedsession):
    test_filter(constrainedsession)


def test_names_cs(constrainedsession):
    assert constrainedsession.names == ['a', 'a01', 'a2', 'ano01', 'anonymous', 'b', 'b024',
                                        'c', 'd', 'e', 'f', 'g', 'h']


def _test_io_cs(tmpdir, meta, engine, ext):
    filename = f"test_{engine}.{ext}" if 'csv' not in engine else f"test_{engine}{ext}"
    fpath = tmp_path(tmpdir, filename)

    is_excel_or_csv = 'excel' in engine or 'csv' in engine

    # Save and load
    # -------------

    # a) - all typed variables have a defined value
    #    - no extra variables are added
    csession = TestConstrainedSession(a=a, a2=a2, a01=a01, d=d, e=e, g=g, h=h, meta=meta)
    cls_name = csession.__class__.__name__
    csession.save(fpath, engine=engine)
    cs = TestConstrainedSession()
    cs.load(fpath, engine=engine)
    # --- keys ---
    assert list(cs.keys()) == list(csession.keys())
    # --- constant variables ---
    assert cs.b.equals(b)
    assert cs.b024.equals(b024)
    assert cs.anonymous.equals(anonymous)
    assert cs.ano01.equals(ano01)
    assert cs.f.equals(f)
    # --- typed variables ---
    # dict is not supported by any format
    assert cs.d is NOT_LOADED
    # Array is support by all formats
    assert cs.e.equals(e)
    assert cs.g.equals(g)
    assert cs.h.equals(h)
    # Axis and Group are not supported by the Excel and CSV formats
    if is_excel_or_csv:
        assert cs.a is NOT_LOADED
        assert cs.a2 is NOT_LOADED
        assert cs.a01 is NOT_LOADED
    else:
        assert cs.a.equals(a)
        assert cs.a2.equals(a2)
        assert cs.a01.equals(a01)
    # --- dtype of Axis variables ---
    if not is_excel_or_csv:
        for key in cs.filter(kind=Axis).keys():
            assert cs[key].dtype == csession[key].dtype
    # --- metadata ---
    if engine != 'pandas_excel':
        assert cs.meta == meta

    # b) - not all typed variables have a defined value
    #    - no extra variables are added
    csession = TestConstrainedSession(a=a, d=d, e=e, h=h, meta=meta)
    if 'csv' in engine:
        import shutil
        shutil.rmtree(fpath)

    with pytest.warns(UserWarning) as caught_warnings:
        csession.save(fpath, engine=engine)
    assert len(caught_warnings) == 3
    for i, var_name in enumerate(['a2', 'a01', 'g']):
        assert caught_warnings[i].message.args[0] == f"The variable '{var_name}' is declared in the '{cls_name}' " \
                                                     f"class definition but was not set."
    cs = TestConstrainedSession()
    cs.load(fpath, engine=engine)
    # --- keys ---
    assert list(cs.keys()) == list(csession.keys())
    # --- constant variables ---
    assert cs.b.equals(b)
    assert cs.b024.equals(b024)
    assert cs.anonymous.equals(anonymous)
    assert cs.ano01.equals(ano01)
    assert cs.f.equals(f)
    # --- typed variables ---
    # dict is not supported by any format
    assert cs.d is NOT_LOADED
    # Array is support by all formats
    assert cs.e.equals(e)
    assert cs.g is NOT_LOADED
    assert cs.h.equals(h)
    # Axis and Group are not supported by the Excel and CSV formats
    if is_excel_or_csv:
        assert cs.a is NOT_LOADED
        assert cs.a2 is NOT_LOADED
        assert cs.a01 is NOT_LOADED
    else:
        assert cs.a.equals(a)
        assert cs.a2 is NOT_LOADED
        assert cs.a01 is NOT_LOADED

    # c) - all typed variables have a defined value
    #    - extra variables are added
    with pytest.warns(UserWarning) as caught_warnings:
        csession = TestConstrainedSession(a=a, a2=a2, a01=a01, d=d, e=e, g=g, h=h,
                                          k=ndtest((2, 2)), j=ndtest((3, 3)), i=ndtest((6)), meta=meta)
        csession.save(fpath, engine=engine)
    assert len(caught_warnings) == 3
    for i, var_name in enumerate(['k', 'j', 'i']):
        assert caught_warnings[i].message.args[0] == f"'{var_name}' is not declared in '{cls_name}'"

    cs = TestConstrainedSession()
    cs.load(fpath, engine=engine)
    # --- names ---
    # use .names instead of .keys() because CSV, Excel and HDF do *not* keep ordering.
    # This may change the order of undeclared variables
    assert cs.names == csession.names

    # Update a Group + an Axis + an array (overwrite=False)
    # -----------------------------------------------------
    csession = TestConstrainedSession(a=a, a2=a2, a01=a01, d=d, e=e, g=g, h=h, meta=meta)
    csession.save(fpath, engine=engine)
    a4 = Axis('a=0..3')
    a4_01 = a3['0,1'] >> 'a01'
    e2 = ndtest((a4, 'b=b0..b2'))
    h2 = full_like(h, fill_value=10)
    TestConstrainedSession(a=a4, a01=a4_01, e=e2, h=h2).save(fpath, overwrite=False, engine=engine)
    cs = TestConstrainedSession()
    cs.load(fpath, engine=engine)
    # --- constant variables ---
    assert cs.b.equals(b)
    assert cs.b024.equals(b024)
    assert cs.anonymous.equals(anonymous)
    assert cs.ano01.equals(ano01)
    assert cs.f.equals(f)
    # --- typed variables ---
    # Array is support by all formats
    cs.e.equals(e2)
    cs.h.equals(h2)
    if engine == 'pandas_excel':
        # Session.save() via engine='pandas_excel' always overwrite the output Excel files
        # array 'g' has been dropped
        assert cs.g is NOT_LOADED
        # Axis and Group are not supported by the Excel and CSV formats
        assert cs.a is NOT_LOADED
        assert cs.a2 is NOT_LOADED
        assert cs.a01 is NOT_LOADED
    elif is_excel_or_csv:
        cs.g.equals(g)
        # Axis and Group are not supported by the Excel and CSV formats
        assert cs.a is NOT_LOADED
        assert cs.a2 is NOT_LOADED
        assert cs.a01 is NOT_LOADED
    else:
        assert list(cs.keys()) == list(csession.keys())
        assert cs.a.equals(a4)
        assert cs.a2.equals(a2)
        assert cs.a01.equals(a4_01)
    if engine != 'pandas_excel':
        assert cs.meta == meta

    # Load only some objects
    # ----------------------
    csession = TestConstrainedSession(a=a, a2=a2, a01=a01, d=d, e=e, g=g, h=h, meta=meta)
    csession.save(fpath, engine=engine)
    cs = TestConstrainedSession()
    names_to_load = ['e', 'h'] if is_excel_or_csv else ['a', 'a01', 'a2', 'e', 'h']
    cs.load(fpath, names=names_to_load, engine=engine)
    # --- keys ---
    assert list(cs.keys()) == list(csession.keys())
    # --- constant variables ---
    assert cs.b.equals(b)
    assert cs.b024.equals(b024)
    assert cs.anonymous.equals(anonymous)
    assert cs.ano01.equals(ano01)
    assert cs.f.equals(f)
    # --- typed variables ---
    # dict is not supported by any format
    assert cs.d is NOT_LOADED
    # Array is support by all formats
    assert cs.e.equals(e)
    assert cs.g is NOT_LOADED
    assert cs.h.equals(h)
    # Axis and Group are not supported by the Excel and CSV formats
    if is_excel_or_csv:
        assert cs.a is NOT_LOADED
        assert cs.a2 is NOT_LOADED
        assert cs.a01 is NOT_LOADED
    else:
        assert cs.a.equals(a)
        assert cs.a2.equals(a2)
        assert cs.a01.equals(a01)

    return fpath


@needs_pytables
def test_h5_io_cs(tmpdir, meta):
    _test_io_cs(tmpdir, meta, engine='pandas_hdf', ext='h5')


@needs_xlrd
def test_xlsx_pandas_io_cs(tmpdir, meta):
    _test_io_cs(tmpdir, meta, engine='pandas_excel', ext='xlsx')


@needs_xlwings
def test_xlsx_xlwings_io_cs(tmpdir, meta):
    _test_io_cs(tmpdir, meta, engine='xlwings_excel', ext='xlsx')


def test_csv_io_cs(tmpdir, meta):
    _test_io_cs(tmpdir, meta, engine='pandas_csv', ext='csv')


def test_pickle_io_cs(tmpdir, meta):
    _test_io_cs(tmpdir, meta, engine='pickle', ext='pkl')


def test_pickle_roundtrip_cs(constrainedsession, meta):
    cs = constrainedsession
    cs.meta = meta
    s = pickle.dumps(cs)
    res = pickle.loads(s)
    assert res.equals(cs)
    assert res.meta == meta


def test_element_equals_cs(constrainedsession):
    test_element_equals(constrainedsession)


def test_eq_cs(constrainedsession):
    test_eq(constrainedsession)


def test_ne_cs(constrainedsession):
    test_ne(constrainedsession)


def test_sub_cs(constrainedsession):
    cs = constrainedsession
    session_cls = cs.__class__

    # session - session
    other = session_cls(e=e - 1, g=zeros_like(g), f=zeros_like(f), h=ones_like(h))
    diff = cs - other
    assert isinstance(diff, session_cls)
    # --- non-array variables ---
    assert diff.b is b
    assert diff.b024 is b024
    assert diff.a is a
    assert diff.a2 is a2
    assert diff.anonymous is anonymous
    assert diff.a01 is a01
    assert diff.ano01 is ano01
    assert diff.c is c
    assert diff.d is d
    # --- array variables ---
    assert_array_nan_equal(diff.e, np.full((2, 3), 1, dtype=np.int32))
    assert_array_nan_equal(diff.g, g)
    assert_array_nan_equal(diff.f, f)
    assert_array_nan_equal(diff.h, h - ones_like(h))

    # session - scalar
    diff = cs - 2
    assert isinstance(diff, session_cls)
    # --- non-array variables ---
    assert diff.b is b
    assert diff.b024 is b024
    assert diff.a is a
    assert diff.a2 is a2
    assert diff.anonymous is anonymous
    assert diff.a01 is a01
    assert diff.ano01 is ano01
    assert diff.c is c
    assert diff.d is d
    # --- non constant arrays ---
    assert_array_nan_equal(diff.e, e - 2)
    assert_array_nan_equal(diff.g, g - 2)
    assert_array_nan_equal(diff.f, f - 2)
    assert_array_nan_equal(diff.h, h - 2)

    # session - dict(Array and scalar)
    other = {'e': ones_like(e), 'h': 1}
    diff = cs - other
    assert isinstance(diff, session_cls)
    # --- non-array variables ---
    assert diff.b is b
    assert diff.b024 is b024
    assert diff.a is a
    assert diff.a2 is a2
    assert diff.anonymous is anonymous
    assert diff.a01 is a01
    assert diff.ano01 is ano01
    assert diff.c is c
    assert diff.d is d
    # --- non constant arrays ---
    assert_array_nan_equal(diff.e, e - ones_like(e))
    assert isnan(diff.g).all()
    assert isnan(diff.f).all()
    assert_array_nan_equal(diff.h, h - 1)

    # session - array
    axes = cs.h.axes
    cs.e = ndtest(axes)
    cs.g = ones_like(cs.h)
    cs.f = full_like(cs.h, fill_value=3)
    diff = cs - ones(axes)
    assert isinstance(diff, session_cls)
    # --- non-array variables ---
    assert diff.b is b
    assert diff.b024 is b024
    assert diff.a is a
    assert diff.a2 is a2
    assert diff.anonymous is anonymous
    assert diff.a01 is a01
    assert diff.ano01 is ano01
    assert diff.c is c
    assert diff.d is d
    # --- non constant arrays ---
    assert_array_nan_equal(diff.e, cs.e - ones(axes))
    assert_array_nan_equal(diff.g, cs.g - ones(axes))
    assert_array_nan_equal(diff.f, cs.f - ones(axes))
    assert_array_nan_equal(diff.h, cs.h - ones(axes))


def test_rsub_cs(constrainedsession):
    cs = constrainedsession
    session_cls = cs.__class__

    # scalar - session
    diff = 2 - cs
    assert isinstance(diff, session_cls)
    # --- non-array variables ---
    assert diff.b is b
    assert diff.b024 is b024
    assert diff.a is a
    assert diff.a2 is a2
    assert diff.anonymous is anonymous
    assert diff.a01 is a01
    assert diff.ano01 is ano01
    assert diff.c is c
    assert diff.d is d
    # --- non constant arrays ---
    assert_array_nan_equal(diff.e, 2 - e)
    assert_array_nan_equal(diff.g, 2 - g)
    assert_array_nan_equal(diff.f, 2 - f)
    assert_array_nan_equal(diff.h, 2 - h)

    # dict(Array and scalar) - session
    other = {'e': ones_like(e), 'h': 1}
    diff = other - cs
    assert isinstance(diff, session_cls)
    # --- non-array variables ---
    assert diff.b is b
    assert diff.b024 is b024
    assert diff.a is a
    assert diff.a2 is a2
    assert diff.anonymous is anonymous
    assert diff.a01 is a01
    assert diff.ano01 is ano01
    assert diff.c is c
    assert diff.d is d
    # --- non constant arrays ---
    assert_array_nan_equal(diff.e, ones_like(e) - e)
    assert isnan(diff.g).all()
    assert isnan(diff.f).all()
    assert_array_nan_equal(diff.h, 1 - h)


if __name__ == "__main__":
    pytest.main()
