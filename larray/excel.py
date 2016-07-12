import os

import numpy as np
try:
    import xlwings as xw
    from xlwings.conversion.pandas_conv import PandasDataFrameConverter
except ImportError:
    xw = None
    PandasDataFrameConverter = object

from .core import LArray, df_aslarray, Axis
from .utils import unique, basestring

string_types = (str,)


class LArrayConverter(PandasDataFrameConverter):
    writes_types = LArray

    @classmethod
    def read_value(cls, value, options):
        df = PandasDataFrameConverter.read_value(value, options)
        return df_aslarray(df)

    @classmethod
    def write_value(cls, value, options):
        df = value.to_frame(fold_last_axis_name=True)
        return PandasDataFrameConverter.write_value(df, options)

if xw is not None:
    LArrayConverter.register(LArray)


class Workbook(object):
    def __init__(self, *args, **kwargs):
        xw_wkb = kwargs.pop('xw_wkb', None)
        if xw_wkb is None:
            xw_wkb = xw.Workbook(*args, **kwargs)
        self.xw_wkb = xw_wkb

    @classmethod
    def active(cls):
        return Workbook(xw_wkb=xw.Workbook.active())

    def _concrete_key(self, key):
        if isinstance(key, int):
            if key < 0:
                key += len(self)
            key += 1
        return key

    def __contains__(self, key):
        if isinstance(key, int):
            length = len(self)
            return -length <= key < length
        else:
            return key in self.sheet_names()

    def __getitem__(self, key):
        key = self._concrete_key(key)
        return Sheet(self, key)

    def __setitem__(self, key, value):
        if key in self:
            key = self._concrete_key(key)
            sheet = Sheet(self, key)
            sheet.clear()
        else:
            xw_sheet = xw.Sheet.add(name=key, wkb=self.xw_wkb)
            sheet = Sheet(None, None, xw_sheet=xw_sheet)
        sheet["A1"] = value

    def sheet_names(self):
        return [s.name for s in self]

    def __iter__(self):
        xw_sheets = xw.Sheet.all(self.xw_wkb)
        return iter([Sheet(None, None, xw_sheet) for xw_sheet in xw_sheets])

    def __len__(self):
        return len(xw.Sheet.all(self.xw_wkb))

    def __dir__(self):
        return list(set(dir(self.__class__)) | set(dir(self.xw_wkb)))

    def __getattr__(self, key):
        return getattr(self.xw_wkb, key)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()


def _concrete_key(key, shape):
    if not isinstance(key, tuple):
        key = (key,)

    if len(key) < len(shape):
        key = key + (slice(None),) * (len(shape) - len(key))

    return [slice(*k.indices(length)) if isinstance(k, slice) else k
            for k, length in zip(key, shape)]


class Sheet(object):
    def __init__(self, workbook, key, xw_sheet=None):
        if xw_sheet is None:
            xw_sheet = xw.Sheet(key, wkb=workbook.xw_wkb)
        self.xw_sheet = xw_sheet

    def __getitem__(self, key):
        if isinstance(key, string_types):
            return Range(self, key)

        row, col = _concrete_key(key, self.shape)
        if isinstance(row, slice) or isinstance(col, slice):
            row1, row2 = (row.start, row.stop) \
                if isinstance(row, slice) else (row, row + 1)
            col1, col2 = (col.start, col.stop) \
                if isinstance(col, slice) else (col, col + 1)
            return Range(self, (row1 + 1, col1 + 1), (row2, col2))
        else:
            return Range(self, (row + 1, col + 1))

    def __setitem__(self, key, value):
        if isinstance(value, LArray):
            value = value.dump(header=False)
        self[key].xw_range.value = value

    @property
    def shape(self):
        # include top-left empty rows/columns
        used = self.xw_sheet.xl_sheet.UsedRange
        return (used.Row + used.Rows.Count - 1,
                used.Column + used.Columns.Count - 1)

    @property
    def ndim(self):
        return 2

    def __array__(self, dtype=None):
        return np.array(self[:].value, dtype=dtype)

    def __dir__(self):
        return list(set(dir(self.__class__)) | set(dir(self.xw_sheet)))

    def __getattr__(self, key):
        return getattr(self.xw_sheet, key)

    def load(self, header=True):
        return self[:].load(header=header)

    # TODO: generalize to more than 2 dimensions or scrap it
    def array(self, data, row_labels=None, column_labels=None, names=None):
        """

        Parameters
        ----------
        data : str
            range for data
        row_labels : str, optional
            range for row labels
        column_labels : str, optional
            range for column labels
        names : list of str, optional

        Returns
        -------
        LArray
        """
        if row_labels is not None:
            row_labels = np.array(self[row_labels].value)
        if column_labels is not None:
            column_labels = np.array(self[column_labels].value)
        if names is not None:
            labels = (row_labels, column_labels)
            axes = [Axis(name, axis_labels)
                    for name, axis_labels in zip(names, labels)]
        else:
            axes = (row_labels, column_labels)
        return LArray(self[data], axes)


class Range(object):
    def __init__(self, sheet, *args):
        xw_range = xw.Range(sheet.xw_sheet, *args)

        object.__setattr__(self, 'sheet', sheet)
        object.__setattr__(self, 'xw_range', xw_range)

    def _range_key_to_sheet_key(self, key):
        # string keys does not make sense in this case
        assert not isinstance(key, string_types)
        row_offset, col_offset = self.xw_range.row1 - 1, self.xw_range.col1 - 1
        row, col = _concrete_key(key, self.xw_range.shape)
        row = slice(row.start + row_offset, row.stop + row_offset) \
            if isinstance(row, slice) else row + row_offset
        col = slice(col.start + col_offset, col.stop + col_offset) \
            if isinstance(col, slice) else col + col_offset
        return row, col

    def __getitem__(self, key):
        return self.sheet[self._range_key_to_sheet_key(key)]

    def __setitem__(self, key, value):
        self.sheet[self._range_key_to_sheet_key(key)] = value

    def __array__(self, dtype=None):
        return np.array(self.xw_range.value, dtype=dtype)

    def __larray__(self):
        return LArray(np.array(self.xw_range.value))

    def __dir__(self):
        return list(set(dir(self.__class__)) | set(dir(self.xw_range)))

    def __getattr__(self, key):
        if hasattr(LArray, key):
            return getattr(self.__larray__(), key)
        else:
            return getattr(self.xw_range, key)

    def __setattr__(self, key, value):
        return setattr(self.xw_range, key, value)

    # TODO: implement all binops
    # def __mul__(self, other):
    #     return self.__larray__() * other

    def __str__(self):
        return str(self.__larray__())

    def load(self, header=True, convert_float=True):
        list_data = self.xw_range.value
        if convert_float:
            # Excel 'numbers' are always floats
            def convert(value):
                if isinstance(value, float):
                    int_val = int(value)
                    if int_val == value:
                        return int_val
                return value

            list_data = [[convert(v) for v in line] for line in list_data]

        if header:
            # TODO: try getting values via self[1:] instead of via the
            # list so that we do not produce copies of data. Not sure which
            # would be faster
            header_line = list_data[0]
            # TODO: factor this with read_csv
            try:
                # take the first cell which contains '\'
                pos_last = next(i for i, v in enumerate(header_line)
                                if isinstance(v, basestring) and '\\' in v)
            except StopIteration:
                # if there isn't any, assume 1d array, unless "liam2 dialect"
                pos_last = 0
            axes_names = header_line[:pos_last + 1]
            # TODO: factor this with df_aslarray
            if isinstance(axes_names[-1], basestring) and \
                    '\\' in axes_names[-1]:
                last_axes = [name.strip()
                             for name in axes_names[-1].split('\\')]
                axes_names = axes_names[:-1] + last_axes

            nb_index = len(axes_names) - 1
            index_col = list(range(nb_index))

            if nb_index == 0:
                nb_index = 1
            data = np.array([line[nb_index:] for line in list_data[1:]])
            axes_labels = [list(unique([line[i] for line in list_data[1:]]))
                           for i in index_col]
            axes_labels.append(header_line[nb_index:])
            # TODO: detect anonymous axes
            axes = [Axis(name, labels)
                    for name, labels in zip(axes_names, axes_labels)]
            data = data.reshape([len(axis) for axis in axes])
            return LArray(data, axes)
        else:
            return LArray(list_data)


def open_excel(filepath=None):
    if filepath == -1:
        return Workbook.active()
    elif filepath is None:
        # creates a new/blank Workbook
        return Workbook(app_visible=True)
    else:
        basename, ext = os.path.splitext(filepath)
        if ext:
            # XXX: we might want to be more precise than .xl* because
            #      I am unsure writing anything else than .xlsx will
            #      work as intended
            if not ext.startswith('.xl'):
                raise ValueError("'%s' is not a supported file "
                                 "extension" % ext)

            # This is necessary because otherwise we cannot open/save to
            # a workbook in the current directory without giving the
            # full/absolute path. By doing this, we basically loose the
            # ability to target an already open workbook *of a saved
            # file* not in the current directory without using its
            # path. I can live with that restriction though because you
            # usually either work with relative paths or with the
            # currently active workbook.
            filepath = os.path.abspath(filepath)
            # save = True
            # close = not xw.xlplatform.is_file_open(filepath)
            # if os.path.isfile(filepath) and overwrite_file:
            #     os.remove(filepath)
        return Workbook(filepath, app_visible=None)
