from abc import ABCMeta
from copy import deepcopy
import warnings

import numpy as np

from typing import TYPE_CHECKING, Type, Any, Dict, Set, List, no_type_check

from pydantic.fields import ModelField
from pydantic.class_validators import Validator
from pydantic.main import BaseConfig

from larray.core.metadata import Metadata
from larray.core.axis import AxisCollection
from larray.core.array import Array, empty
from larray.core.session import Session


NOT_LOADED = object()


# the implementation of the class below is inspired by the 'ConstrainedBytes' class
# from the types.py module of the 'pydantic' library
class ConstrainedArrayImpl(Array):
    expected_axes: AxisCollection
    dtype: np.dtype = np.dtype(float)

    # see https://pydantic-docs.helpmanual.io/usage/types/#classes-with-__get_validators__
    @classmethod
    def __get_validators__(cls):
        # one or more validators may be yielded which will be called in the
        # order to validate the input, each validator will receive as an input
        # the value returned from the previous validator
        yield cls.validate

    @classmethod
    def validate(cls, value, field: ModelField):
        # XXX: not sure we should accept a scalar as value but this allows to not repeat axes
        # when setting a (default) value to the associated array
        if not (isinstance(value, Array) or np.isscalar(value)):
            raise TypeError(f"Expected object of type '{Array.__name__}' or a scalar for the variable '{field.name}' "
                            f"but got object of type '{type(value).__name__}'")

        # check axes
        if isinstance(value, Array):
            error_msg = f"Array '{field.name}' was declared with axes {cls.expected_axes} but got array " \
                        f"with axes {value.axes}"
            # check for missing axes
            missing_axes = cls.expected_axes - value.axes
            if missing_axes:
                raise ValueError(f"{error_msg} ({missing_axes} {'axes are' if len(missing_axes) > 1 else 'axis is'} "
                                 f"missing)")
            # check for extra axes
            extra_axes = value.axes - cls.expected_axes
            if extra_axes:
                raise ValueError(f"{error_msg} (unexpected {extra_axes} {'axes' if len(extra_axes) > 1 else 'axis'})")
            # check compatible axes
            try:
                cls.expected_axes.check_compatible(value.axes)
            except ValueError as error:
                error_msg = str(error).replace("incompatible axes", f"Incompatible axis for array '{field.name}'")
                raise ValueError(error_msg)

        # check data-type
        dtype = value.dtype if isinstance(value, Array) else np.dtype(type(value))
        if dtype != cls.dtype:
            warnings.warn(f"Expected array or scalar of dtype {cls.dtype} for the array '{field.name}' "
                          f"but got array or scalar of dtype {dtype}")

        # transpose and cast if needed
        array = empty(axes=cls.expected_axes, dtype=cls.dtype)
        array[:] = value
        return array


# the implementation of the function below is inspired by the 'conbytes' function
# from the types.py module of the 'pydantic' library
def ConstrainedArray(axes: AxisCollection, dtype: np.dtype = float) -> Type[Array]:
    """
    Represents a constrained array.
    Its axes are assumed to be "frozen", meaning they are constant all along the execution of the program.
    A constrain on the dtype of the data can be also specified.

    Parameters
    ----------
    axes: AxisCollection
        Axes of the constrained array.
    dtype: data-type, optional
        Data-type for the constrained array. Defaults to float.

    Returns
    -------
    Array
        Constrained array.
    """
    if axes is not None and not isinstance(axes, AxisCollection):
        axes = AxisCollection(axes)
    _dtype = np.dtype(dtype)

    class ConstrainedArrayValue(ConstrainedArrayImpl):
        expected_axes = axes
        dtype = _dtype

    return ConstrainedArrayValue


class AbstractConstrainedSession:
    pass


# Simplified version of the ModelMetaclass class from pydantic:
# https://github.com/samuelcolvin/pydantic/blob/master/pydantic/main.py#L195
class ModelMetaclass(ABCMeta):
    @no_type_check  # noqa C901
    def __new__(mcs, name, bases, namespace, **kwargs):
        from pydantic.fields import Undefined
        from pydantic.class_validators import extract_validators, inherit_validators
        from pydantic.types import PyObject
        from pydantic.typing import is_classvar, resolve_annotations
        from pydantic.utils import lenient_issubclass, validate_field_name
        from pydantic.main import inherit_config, prepare_config, UNTOUCHED_TYPES

        fields: Dict[str, ModelField] = {}
        config = BaseConfig
        validators: Dict[str, List[Validator]] = {}

        for base in reversed(bases):
            if issubclass(base, AbstractConstrainedSession) and base != AbstractConstrainedSession:
                config = inherit_config(base.__config__, config)
                fields.update(deepcopy(base.__fields__))
                validators = inherit_validators(base.__validators__, validators)

        config = inherit_config(namespace.get('Config'), config)
        validators = inherit_validators(extract_validators(namespace), validators)

        # update fields inherited from base classes
        for field in fields.values():
            field.set_config(config)
            extra_validators = validators.get(field.name, [])
            if extra_validators:
                field.class_validators.update(extra_validators)
                # re-run prepare to add extra validators
                field.populate_validators()

        prepare_config(config, name)

        # extract and build fields
        class_vars = set()
        if (namespace.get('__module__'), namespace.get('__qualname__')) != \
                ('larray.core.constrained', 'ConstrainedSession'):
            untouched_types = UNTOUCHED_TYPES + config.keep_untouched

            # annotation only fields need to come first in fields
            annotations = resolve_annotations(namespace.get('__annotations__', {}), namespace.get('__module__', None))
            for ann_name, ann_type in annotations.items():
                if is_classvar(ann_type):
                    class_vars.add(ann_name)
                elif not ann_name.startswith('_'):
                    validate_field_name(bases, ann_name)
                    value = namespace.get(ann_name, Undefined)
                    if (isinstance(value, untouched_types) and ann_type != PyObject and
                            not lenient_issubclass(getattr(ann_type, '__origin__', None), Type)):
                        continue
                    fields[ann_name] = ModelField.infer(name=ann_name, value=value, annotation=ann_type,
                                                        class_validators=validators.get(ann_name, []), config=config)

            for var_name, value in namespace.items():
                # 'var_name not in annotations' because namespace.items() contains annotated fields with default values
                # 'var_name not in class_vars' to avoid to update a field if it was redeclared (by mistake)
                if (var_name not in annotations and not var_name.startswith('_') and
                        not isinstance(value, untouched_types) and var_name not in class_vars):
                    validate_field_name(bases, var_name)
                    inferred = ModelField.infer(name=var_name, value=value, annotation=annotations.get(var_name),
                                                class_validators=validators.get(var_name, []), config=config)
                    if var_name in fields and inferred.type_ != fields[var_name].type_:
                        raise TypeError(f'The type of {name}.{var_name} differs from the new default value; '
                                        f'if you wish to change the type of this field, please use a type annotation')
                    fields[var_name] = inferred

        new_namespace = {
            '__config__': config,
            '__fields__': fields,
            '__field_defaults__': {n: f.default for n, f in fields.items() if not f.required},
            '__validators__': validators,
            **{n: v for n, v in namespace.items() if n not in fields},
        }
        return super().__new__(mcs, name, bases, new_namespace, **kwargs)


class ConstrainedSession(Session, AbstractConstrainedSession, metaclass=ModelMetaclass):
    """
    This class is intended to be inherited by user defined classes in which the variables of a model are declared.
    Each declared variable is constrained by a type defined explicitly or deduced from the given default value
    (see examples below).
    All classes inheriting from `ConstrainedSession` will have access to all methods of the :py:class:`Session` class.

    The special :py:funct:`ConstantAxesArray` type represents an Array object with constant axes.
    This prevents users from modifying the dimensions and/or labels of an array by mistake and make sure that
    the definition of an array remains always valid in the model.

    By declaring variables, users will speed up the development of their models using the auto-completion
    (the feature in which development tools like PyCharm try to predict the variable or function a user intends to
    enter after only a few characters have been typed).

    As for normal Session objects, it is still possible to add undeclared variables to instances of
    classes inheriting from `ConstrainedSession` but this must be done with caution.

    Parameters
    ----------
    *args : str or dict of {str: object} or iterable of tuples (str, object)
        Path to the file containing the session to load or
        list/tuple/dictionary containing couples (name, object).
    **kwargs : dict of {str: object}

        * Objects to add written as name=object
        * meta : list of pairs or dict or OrderedDict or Metadata, optional
            Metadata (title, description, author, creation_date, ...) associated with the array.
            Keys must be strings. Values must be of type string, int, float, date, time or datetime.

    Warnings
    --------
    The :py:method:`ConstrainedSession.filter`, :py:method:`ConstrainedSession.compact`
    and :py:method:`ConstrainedSession.apply` methods return a simple Session object.
    The type of the declared variables (and the value for the declared constants) will
    no longer be checked.

    See Also
    --------
    Session, Parameters

    Examples
    --------

    Content of file 'parameters.py'

    >>> from larray import *
    >>> FIRST_YEAR = 2020
    >>> LAST_YEAR = 2030
    >>> AGE = Axis('age=0..10')
    >>> GENDER = Axis('gender=male,female')
    >>> TIME = Axis(f'time={FIRST_YEAR}..{LAST_YEAR}')

    Content of file 'model.py'

    >>> class ModelVariables(ConstrainedSession):
    ...     # --- declare variables with defined types ---
    ...     # Their values will be defined at runtime but must match the specified type.
    ...     variant_name: str
    ...     birth_rate: Array
    ...     births: Array
    ...     # --- declare variables with a default value ---
    ...     # The default value will be used to set the variable if no value is passed at instantiation (see below).
    ...     # Such variable will be constrained by the type deduced from its default value.
    ...     target_age: Group = AGE[:2] >> '0-2'
    ...     population = zeros((AGE, GENDER, TIME), dtype=int)
    ...     # --- declare constrained arrays ---
    ...     # the constrained arrays have axes assumed to be "frozen", meaning they are
    ...     # constant all along the execution of the program.
    ...     mortality_rate: ConstrainedArray((AGE, GENDER))
    ...     # for constrained arrays, the default value can be given as a scalar.
    ...     # A dtype can be also optionally specified (defaults to float).
    ...     deaths: ConstrainedArray((AGE, GENDER, TIME), dtype=int) = 0

    >>> variant_name = 'variant_1'
    >>> # instantiation --> create an instance of the ModelVariables class
    >>> # all variables declared without a default value must be set
    >>> m = ModelVariables(
    ...                   variant_name = variant_name,
    ...                   birth_rate = zeros((AGE, GENDER)),
    ...                   births = zeros((AGE, GENDER, TIME), dtype=int),
    ...                   mortality_rate = 0,
    ...                   )

    >>> # ==== model ====
    >>> # axes and dtype of arrays 'birth_rate', 'births' and 'population' are not protected,
    >>> # leading to potentially unexpected behavior of the model.
    >>> # example 1: Let's say we want to calculate the new births for the year 2025 and we assume that
    >>> # the birth rate only differ by gender.
    >>> # In the line below, we add an additional TIME axis to 'birth_rate' while it was initialized
    >>> # with the AGE and GENDER axes only
    >>> m.birth_rate = full((AGE, GENDER, TIME), fill_value=Array([0.045, 0.055], GENDER))
    >>> # here 'new_births' have the AGE, GENDER and TIME axes instead of the AGE and GENDER axes only
    >>> new_births = m.population['female', 2025] * m.birth_rate
    >>> print(new_births.info)
    11 x 2 x 11
     age [11]: 0 1 2 ... 8 9 10
     gender [2]: 'male' 'female'
     time [11]: 2020 2021 2022 ... 2028 2029 2030
    dtype: float64
    memory used: 1.89 Kb
    >>> # and the line below will crashes
    >>> m.births[2025] = new_births         # doctest: +NORMALIZE_WHITESPACE
    Traceback (most recent call last):
        ...
    ValueError: Value {time} axis is not present in target subset {age, gender}.
    A value can only have the same axes or fewer axes than the subset being targeted
    >>> # now let's try to do the same for deaths and making the same mistake as for 'birth_rate'.
    >>> # The program will crash now at the first step instead of letting you going further
    >>> m.mortality_rate = full((AGE, GENDER, TIME), fill_value=sequence(AGE, inc=0.02))# doctest: +NORMALIZE_WHITESPACE
    Traceback (most recent call last):
        ...
    ValueError: Array 'mortality_rate' was declared with axes {age, gender} but got array with axes
    {age, gender, time} (unexpected {time} axis)

    >>> # example 2: let's say we want to calculate the new births for all years.
    >>> m.birth_rate = full((AGE, GENDER, TIME), fill_value=Array([0.045, 0.055], GENDER))
    >>> new_births = m.population['female'] * m.birth_rate
    >>> # here 'new_births' has the same axes as 'births' but is a float array instead of
    >>> # an integer array as 'births'.
    >>> # The line below will make the 'births' array to become a float array while
    >>> # it was initialized as an integer array
    >>> m.births = new_births
    >>> print(m.births.info)
    11 x 11 x 2
     age [11]: 0 1 2 ... 8 9 10
     time [11]: 2020 2021 2022 ... 2028 2029 2030
     gender [2]: 'male' 'female'
    dtype: float64
    memory used: 1.89 Kb
    >>> # now let's try to do the same for deaths.
    >>> m.mortality_rate = full((AGE, GENDER), fill_value=sequence(AGE, inc=0.02))
    >>> # here the result of the multiplication of the 'population' array by the 'mortality_rate' array
    >>> # is automatically converted to an integer array (printing a warning message)
    >>> m.deaths = m.population * m.mortality_rate
    >>> print(m.deaths.info)
    11 x 2 x 11
     age [11]: 0 1 2 ... 8 9 10
     gender [2]: 'male' 'female'
     time [11]: 2020 2021 2022 ... 2028 2029 2030
    dtype: int32
    memory used: 968 bytes

    >>> # note that it still possible to add undeclared variables to a constrained session
    >>> # but this must be done with caution.
    >>> m.undeclared_var = 'undeclared_var'

    >>> # ==== output ====
    >>> # save all variables in an HDF5 file
    >>> m.save(f'{variant_name}.h5', display=True)
    dumping variant_name ... done
    dumping birth_rate ... done
    dumping births ... done
    dumping target_age ... done
    dumping mortality_rate ... done
    dumping deaths ... done
    dumping population ... done
    dumping undeclared_var ... done
    """
    if TYPE_CHECKING:
        # populated by the metaclass, defined here to help IDEs only
        __fields__: Dict[str, ModelField] = {}
        __field_defaults__: Dict[str, Any] = {}
        __validators__: Dict[str, List[Validator]] = {}
        __config__: Type[BaseConfig] = BaseConfig

    class Config:
        # whether to allow arbitrary user types for fields (they are validated simply by checking
        # if the value is an instance of the type). If False, RuntimeError will be raised on model declaration.
        # (default: False)
        arbitrary_types_allowed = True
        # whether to validate field defaults
        validate_all = True
        # whether to ignore, allow, or forbid extra attributes during model initialization (and after).
        # Accepts the string values of 'ignore', 'allow', or 'forbid', or values of the Extra enum
        # (default: Extra.ignore)
        extra = 'allow'
        # whether to perform validation on assignment to attributes
        validate_assignment = True
        # whether or not models are faux-immutable, i.e. whether __setattr__ is allowed.
        # (default: True)
        allow_mutation = True

    # Warning: order of fields is not preserved.
    # As of v1.0 of pydantic all fields with annotations (whether annotation-only or with a default value)
    # will precede all fields without an annotation. Within their respective groups, fields remain in the
    # order they were defined.
    # See https://pydantic-docs.helpmanual.io/usage/models/#field-ordering
    def __init__(self, *args, **kwargs):
        meta = kwargs.pop('meta', Metadata())
        Session.__init__(self, meta=meta)

        # create an intermediate Session object to not call the __setattr__
        # and __setitem__ overridden in the present class and in case a filepath
        # is given as only argument
        # todo: refactor Session.load() to use a private function which returns the handler directly
        # so that we can get the items out of it and avoid this
        input_data = dict(Session(*args, **kwargs))

        # --- declared variables
        for name, field in self.__fields__.items():
            value = input_data.pop(field.name, NOT_LOADED)

            if value is NOT_LOADED:
                if field.default is None:
                    warnings.warn(f"No value passed for the declared variable '{field.name}'")
                    self.__setattr__(name, value, skip_allow_mutation=True, skip_validation=True)
                else:
                    self.__setattr__(name, field.default, skip_allow_mutation=True)
            else:
                self.__setattr__(name, value, skip_allow_mutation=True)

        # --- undeclared variables
        for name, value in input_data.items():
            self.__setattr__(name, value, skip_allow_mutation=True)

    # code of the method below has been partly borrowed from pydantic.BaseModel.__setattr__()
    def _check_key_value(self, name: str, value: Any, skip_allow_mutation: bool, skip_validation: bool) -> Any:
        config = self.__config__
        if not config.extra and name not in self.__fields__:
            raise ValueError(f"Variable '{name}' is not declared in '{self.__class__.__name__}'. "
                             f"Adding undeclared variables is forbidden. "
                             f"List of declared variables is: {list(self.__fields__.keys())}.")
        if not skip_allow_mutation and not config.allow_mutation:
            raise TypeError(f"Cannot change the value of the variable '{name}' since '{self.__class__.__name__}' "
                            f"is immutable and does not support item assignment")
        known_field = self.__fields__.get(name, None)
        if known_field:
            if not skip_validation:
                value, error_ = known_field.validate(value, self.dict(exclude={name}), loc=name, cls=self.__class__)
                if error_:
                    raise error_.exc
        else:
            warnings.warn(f"'{name}' is not declared in '{self.__class__.__name__}'", stacklevel=2)
        return value

    def __setitem__(self, key, value, skip_allow_mutation=False, skip_validation=False):
        if key != 'meta':
            value = self._check_key_value(key, value, skip_allow_mutation, skip_validation)
            # we need to keep the attribute in sync
            object.__setattr__(self, key, value)
            self._objects[key] = value

    def __setattr__(self, key, value, skip_allow_mutation=False, skip_validation=False):
        if key != 'meta':
            value = self._check_key_value(key, value, skip_allow_mutation, skip_validation)
            # we need to keep the attribute in sync
            object.__setattr__(self, key, value)
        Session.__setattr__(self, key, value)

    def __getstate__(self) -> Dict[str, Any]:
        return {'__dict__': self.__dict__}

    def __setstate__(self, state: Dict[str, Any]) -> None:
        object.__setattr__(self, '__dict__', state['__dict__'])

    def dict(self, exclude: Set[str] = None):
        d = dict(self.items())
        for name in exclude:
            if name in d:
                del d[name]
        return d


class Parameters(ConstrainedSession):
    """
    Same as py:class:`ConstrainedSession` but:

        * declared variables cannot be modified after initialization
        * adding undeclared variables after initialization is forbidden.

    Parameters
    ----------
    *args : str or dict of {str: object} or iterable of tuples (str, object)
        Path to the file containing the session to load or
        list/tuple/dictionary containing couples (name, object).
    **kwargs : dict of {str: object}

        * Objects to add written as name=object
        * meta : list of pairs or dict or OrderedDict or Metadata, optional
            Metadata (title, description, author, creation_date, ...) associated with the array.
            Keys must be strings. Values must be of type string, int, float, date, time or datetime.

    See Also
    --------
    ConstrainedSession

    Examples
    --------

    Content of file 'parameters.py'

    >>> from larray import *
    >>> class ModelParameters(Parameters):
    ...     # --- declare variables with fixed values ---
    ...     # The given values can never be changed
    ...     FIRST_YEAR = 2020
    ...     LAST_YEAR = 2030
    ...     AGE = Axis('age=0..10')
    ...     GENDER = Axis('gender=male,female')
    ...     TIME = Axis(f'time={FIRST_YEAR}..{LAST_YEAR}')
    ...     # --- declare variables with defined types ---
    ...     # Their values must be defined at initialized and will be frozen after.
    ...     variant_name: str

    Content of file 'model.py'

    >>> # instantiation --> create an instance of the ModelVariables class
    >>> # all variables declared without value must be set
    >>> P = ModelParameters(variant_name='variant_1')
    >>> # once an instance is create, its variables can be accessed but not modified
    >>> P.variant_name
    'variant_1'
    >>> P.variant_name = 'new_variant'      # doctest: +NORMALIZE_WHITESPACE
    Traceback (most recent call last):
        ...
    TypeError: Cannot change the value of the variable 'variant_name' since 'ModelParameters'
    is immutable and does not support item assignment
    """
    class Config:
        # whether to ignore, allow, or forbid extra attributes during model initialization (and after).
        # Accepts the string values of 'ignore', 'allow', or 'forbid', or values of the Extra enum
        # (default: Extra.ignore)
        extra = 'forbid'
        # whether or not models are faux-immutable, i.e. whether __setattr__ is allowed.
        # (default: True)
        allow_mutation = False
