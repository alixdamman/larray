import pytest
import pickle

import numpy as np

from larray import (ConstrainedSession, ConstrainedArray, Axis, AxisCollection, Group, Array,
                    ndtest, full, full_like, zeros_like, ones, ones_like, isnan)
from larray.tests.common import (inputpath, tmp_path, assert_array_nan_equal, meta,
                                 needs_pytables, needs_xlrd, needs_xlwings)
from larray.tests.test_session import (a, a2, a3, anonymous, a01, ano01, b, b2, b024, c, d, e, f, g, h,
                                       assertObjListEqual, session, test_getitem, test_getattr, test_add,
                                       test_element_equals, test_eq, test_ne)
from larray.core.constrained import NOT_LOADED


class TestConstrainedSession(ConstrainedSession):
    b = b
    b024 = b024
    a: Axis
    a2: Axis
    anonymous = anonymous
    a01: Group
    ano01 = ano01
    c: str = c
    d = d
    e: Array
    g: Array
    f: ConstrainedArray((Axis(3), Axis(2)))
    h: ConstrainedArray((a3, b2), dtype=int)


@pytest.fixture()
def constrainedsession():
    return TestConstrainedSession(a=a, a2=a2, a01=a01, e=e, g=g, f=f, h=h)


def test_create_constrainedsession_instance(meta):
    # As of v1.0 of pydantic all fields with annotations (whether annotation-only or with a default value)
    # will precede all fields without an annotation. Within their respective groups, fields remain in the
    # order they were defined.
    # See https://pydantic-docs.helpmanual.io/usage/models/#field-ordering
    declared_variable_keys = ['a', 'a2', 'a01', 'c', 'e', 'g', 'f', 'h', 'b', 'b024', 'anonymous', 'ano01', 'd']

    # setting variables without default values
    cs = TestConstrainedSession(a, a01, a2=a2, e=e, f=f, g=g, h=h)
    assert list(cs.keys()) == declared_variable_keys
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

    # override default value
    b_alt = Axis('b=b0..b4')
    cs = TestConstrainedSession(a, a01, b=b_alt, a2=a2, e=e, f=f, g=g, h=h)
    assert cs.b is b_alt

    # test for "NOT_LOADED" variables
    not_loaded_variables = ['a', 'a2', 'a01', 'e', 'g', 'f', 'h']
    with pytest.warns(UserWarning) as caught_warnings:
        cs = TestConstrainedSession()
    assert len(caught_warnings) == len(not_loaded_variables)
    for i, var_name in enumerate(not_loaded_variables):
        assert caught_warnings[i].message.args[0] == f"No value passed for the declared variable '{var_name}'"
    assert list(cs.keys()) == declared_variable_keys
    # --- variables with default values ---
    assert cs.b.equals(b)
    assert cs.b024.equals(b024)
    assert cs.anonymous.equals(anonymous)
    assert cs.ano01.equals(ano01)
    assert cs.c == c
    assert cs.d == d
    # --- variables without default values ---
    assert cs.a is NOT_LOADED
    assert cs.a2 is NOT_LOADED
    assert cs.a01 is NOT_LOADED
    assert cs.e is NOT_LOADED
    assert cs.g is NOT_LOADED
    assert cs.f is NOT_LOADED
    assert cs.h is NOT_LOADED

    # passing a scalar to set all elements a ConstrainedArray
    cs = TestConstrainedSession(a, a01, a2=a2, e=e, f=f, g=g, h=5)
    assert cs.h.axes == AxisCollection((a3, b2))
    assert cs.h.equals(full(axes=(a3, b2), fill_value=5))

    # add the undeclared variable 'i'
    with pytest.warns(UserWarning) as caught_warnings:
        cs = TestConstrainedSession(a, a01, a2=a2, i=5, e=e, f=f, g=g, h=h)
    assert caught_warnings[0].message.args[0] == f"'i' is not declared in '{cs.__class__.__name__}'"
    assert list(cs.keys()) == declared_variable_keys + ['i']

    # test inheritance between constrained sessions
    class TestInheritance(TestConstrainedSession):
        # override variables
        b = b2
        c: int = 5
        f: ConstrainedArray((a3, b2), dtype=int)
        h: ConstrainedArray((Axis(3), Axis(2)))
        # new variables
        n0 = 'first new var'
        n1: str

    declared_variable_keys += ['n1', 'n0']
    cs = TestInheritance(a, a01, a2=a2, e=e, f=h, g=g, h=f, n1='second new var')
    assert list(cs.keys()) == declared_variable_keys
    # --- overriden variables ---
    assert cs.b.equals(b2)
    assert cs.c == 5
    assert cs.f.equals(h)
    assert cs.h.equals(f)
    # --- new variables ---
    assert cs.n0 == 'first new var'
    assert cs.n1 == 'second new var'
    # --- variables declared in the base class ---
    assert cs.b024.equals(b024)
    assert cs.a.equals(a)
    assert cs.a2.equals(a2)
    assert cs.anonymous.equals(anonymous)
    assert cs.a01.equals(a01)
    assert cs.ano01.equals(ano01)
    assert cs.d == d
    assert cs.e.equals(e)
    assert cs.g.equals(g)


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
    expected_error_msg = "Array 'h' was declared with axes {a, b} but got array with axes {a} ({b} axis is missing)"
    with pytest.raises(ValueError) as error:
        cs['h'] = ndtest(a3)
    assert str(error.value) == expected_error_msg
    # b) extra axis
    expected_error_msg = "Array 'h' was declared with axes {a, b} but got array with axes {a, b, c} " \
                         "(unexpected {c} axis)"
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
    expected_error_msg = "Array 'h' was declared with axes {a, b} but got array with axes {a} ({b} axis is missing)"
    with pytest.raises(ValueError) as error:
        cs.h = ndtest(a3)
    assert str(error.value) == expected_error_msg
    # b) extra axis
    expected_error_msg = "Array 'h' was declared with axes {a, b} but got array with axes {a, b, c} " \
                         "(unexpected {c} axis)"
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
    # As of v1.0 of pydantic all fields with annotations (whether annotation-only or with a default value)
    # will precede all fields without an annotation. Within their respective groups, fields remain in the
    # order they were defined.
    # See https://pydantic-docs.helpmanual.io/usage/models/#field-ordering
    expected = [a, a2, a01, c, e, g, f, h, b, b024, anonymous, ano01, d]
    assertObjListEqual(constrainedsession, expected)


def test_filter_cs(constrainedsession):
    # see comment in test_iter_cs() about fields ordering
    cs = constrainedsession
    cs.ax = 'ax'
    assertObjListEqual(cs.filter(), [a, a2, a01, c, e, g, f, h, b, b024, anonymous, ano01, d, 'ax'])
    assertObjListEqual(cs.filter('a*'), [a, a2, a01, anonymous, ano01, 'ax'])
    assert list(cs.filter('a*', dict)) == []
    assert list(cs.filter('a*', str)) == ['ax']
    assert list(cs.filter('a*', Axis)) == [a, a2, anonymous]
    assert list(cs.filter(kind=Axis)) == [a, a2, b, anonymous]
    assert list(cs.filter('a01', Group)) == [a01]
    assert list(cs.filter(kind=Group)) == [a01, b024, ano01]
    assertObjListEqual(cs.filter(kind=Array), [e, g, f, h])
    assert list(cs.filter(kind=dict)) == [{}]
    assert list(cs.filter(kind=(Axis, Group))) == [a, a2, a01, b, b024, anonymous, ano01]


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
    csession = TestConstrainedSession(a=a, a2=a2, a01=a01, d=d, e=e, g=g, f=f, h=h, meta=meta)
    cls_name = csession.__class__.__name__
    csession.save(fpath, engine=engine)
    cs = TestConstrainedSession()
    cs.load(fpath, engine=engine)
    # --- keys ---
    assert list(cs.keys()) == list(csession.keys())
    # --- variables with default values ---
    assert cs.b.equals(b)
    assert cs.b024.equals(b024)
    assert cs.anonymous.equals(anonymous)
    assert cs.ano01.equals(ano01)
    assert cs.d == d
    # --- typed variables ---
    # Array is support by all formats
    assert cs.e.equals(e)
    assert cs.g.equals(g)
    assert cs.f.equals(f)
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
    csession.save(fpath, engine=engine)
    cs = TestConstrainedSession()
    cs.load(fpath, engine=engine)
    # --- keys ---
    assert list(cs.keys()) == list(csession.keys())
    # --- variables with default values ---
    assert cs.b.equals(b)
    assert cs.b024.equals(b024)
    assert cs.anonymous.equals(anonymous)
    assert cs.ano01.equals(ano01)
    assert cs.d == d
    # --- typed variables ---
    # Array is support by all formats
    assert cs.e.equals(e)
    assert cs.g is NOT_LOADED
    assert cs.f is NOT_LOADED
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
    i = ndtest(6)
    j = ndtest((3, 3))
    k = ndtest((2, 2))
    csession = TestConstrainedSession(a=a, a2=a2, a01=a01, d=d, e=e, g=g, f=f, h=h, k=k, j=j, i=i, meta=meta)
    csession.save(fpath, engine=engine)
    cs = TestConstrainedSession()
    cs.load(fpath, engine=engine)
    # --- names ---
    # we do not use keys() since order of undeclared variables
    # may not be preserved (at least for the HDF format)
    assert cs.names == csession.names
    # --- extra variable ---
    assert cs.i.equals(i)
    assert cs.j.equals(j)
    assert cs.k.equals(k)

    # Update a Group + an Axis + an array (overwrite=False)
    # -----------------------------------------------------
    csession = TestConstrainedSession(a=a, a2=a2, a01=a01, d=d, e=e, g=g, f=f, h=h, meta=meta)
    csession.save(fpath, engine=engine)
    a4 = Axis('a=0..3')
    a4_01 = a3['0,1'] >> 'a01'
    e2 = ndtest((a4, 'b=b0..b2'))
    h2 = full_like(h, fill_value=10)
    TestConstrainedSession(a=a4, a01=a4_01, e=e2, h=h2).save(fpath, overwrite=False, engine=engine)
    cs = TestConstrainedSession()
    cs.load(fpath, engine=engine)
    # --- variables with default values ---
    assert cs.b.equals(b)
    assert cs.b024.equals(b024)
    assert cs.anonymous.equals(anonymous)
    assert cs.ano01.equals(ano01)
    # --- typed variables ---
    # Array is support by all formats
    assert cs.e.equals(e2)
    assert cs.h.equals(h2)
    if engine == 'pandas_excel':
        # Session.save() via engine='pandas_excel' always overwrite the output Excel files
        # arrays 'g' and 'f' have been dropped
        assert cs.g is NOT_LOADED
        assert cs.f is NOT_LOADED
        # Axis and Group are not supported by the Excel and CSV formats
        assert cs.a is NOT_LOADED
        assert cs.a2 is NOT_LOADED
        assert cs.a01 is NOT_LOADED
    elif is_excel_or_csv:
        assert cs.g.equals(g)
        assert cs.f.equals(f)
        # Axis and Group are not supported by the Excel and CSV formats
        assert cs.a is NOT_LOADED
        assert cs.a2 is NOT_LOADED
        assert cs.a01 is NOT_LOADED
    else:
        assert list(cs.keys()) == list(csession.keys())
        assert cs.a.equals(a4)
        assert cs.a2.equals(a2)
        assert cs.a01.equals(a4_01)
        assert cs.g.equals(g)
        assert cs.f.equals(f)
    if engine != 'pandas_excel':
        assert cs.meta == meta

    # Load only some objects
    # ----------------------
    csession = TestConstrainedSession(a=a, a2=a2, a01=a01, d=d, e=e, g=g, f=f, h=h, meta=meta)
    csession.save(fpath, engine=engine)
    cs = TestConstrainedSession()
    names_to_load = ['e', 'h'] if is_excel_or_csv else ['a', 'a01', 'a2', 'e', 'h']
    cs.load(fpath, names=names_to_load, engine=engine)
    # --- keys ---
    assert list(cs.keys()) == list(csession.keys())
    # --- variables with default values ---
    assert cs.b.equals(b)
    assert cs.b024.equals(b024)
    assert cs.anonymous.equals(anonymous)
    assert cs.ano01.equals(ano01)
    assert cs.d == d
    # --- typed variables ---
    # Array is support by all formats
    assert cs.e.equals(e)
    assert cs.g is NOT_LOADED
    assert cs.f is NOT_LOADED
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
    other = session_cls(a=a, a2=a2, a01=a01, e=e - 1, g=zeros_like(g), f=zeros_like(f), h=ones_like(h))
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
    assert isnan(diff.f).all()
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


def test_neg_cs(constrainedsession):
    cs = constrainedsession
    neg_cs = -cs
    # --- non-array variables ---
    assert isnan(neg_cs.b)
    assert isnan(neg_cs.b024)
    assert isnan(neg_cs.a)
    assert isnan(neg_cs.a2)
    assert isnan(neg_cs.anonymous)
    assert isnan(neg_cs.a01)
    assert isnan(neg_cs.ano01)
    assert isnan(neg_cs.c)
    assert isnan(neg_cs.d)
    # --- non constant arrays ---
    assert_array_nan_equal(neg_cs.e, -e)
    assert_array_nan_equal(neg_cs.g, -g)
    assert_array_nan_equal(neg_cs.f, -f)
    assert_array_nan_equal(neg_cs.h, -h)


if __name__ == "__main__":
    pytest.main()
