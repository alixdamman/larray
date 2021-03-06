﻿New features
------------

* implemented Session.to_globals which creates global variables from variables stored in the session (closes
  :issue:`276`). Note that this should usually only be used in an interactive console and not in a script. Code editors
  are confused by this kind of manipulation and will likely consider as invalid the code using variables created in
  this way. Additionally, when using this method auto-completion, "show definition", "go to declaration" and other
  similar code editor features will probably not work for the variables created in this way and any variable derived
  from them.

    >>> s = Session(arr1=ndtest(3), arr2=ndtest((2, 2)))
    >>> s.to_globals()
    >>> arr1
    a  a0  a1  a2
        0   1   2
    >>> arr2
    a\b  b0  b1
     a0   0   1
     a1   2   3

* added new boolean argument 'overwrite' to Session.save, Session.to_hdf, Session.to_excel and Session.to_pickle
  methods (closes :issue:`293`). If overwrite=True and the target file already existed, it is deleted and replaced by a
  new one. This is the new default behavior. If overwrite=False, an existing file is updated (like it was in previous
  larray versions):

    >>> arr1, arr2, arr3 = ndtest((2, 2)), ndtest(4), ndtest((3, 2))
    >>> s = Session([('arr1', arr1), ('arr2', arr2), ('arr3', arr3)])

    >>> # save arr1, arr2 and arr3 in file output.h5
    >>> s.save('output.h5')

    >>> # replace arr1 and create arr4 + put them in an second session
    >>> arr1, arr4 = ndtest((3, 3)), ndtest((2, 3))
    >>> s2 = Session([('arr1', arr1), ('arr4', arr4)])

    >>> # replace arr1 and add arr4 in file output.h5
    >>> s2.save('output.h5', overwrite=False)

    >>> # erase content of 'output.h5' and save only arrays contained in the second session
    >>> s2.save('output.h5')


Miscellaneous improvements
--------------------------

* renamed create_sequential() to sequence() (closes :issue:`212`).

* improved auto-completion in ipython interactive consoles (e.g. the viewer console) for Axis, AxisCollection, Group
  and Workbook objects. These objects can now complete keys within [].

    >>> gender = Axis('gender=Male,Female')
    >>> gender
    Axis(['Male', 'Female'], 'gender')
    gender['Female
    >>> gender['Fe<tab>  # will be completed to `gender['Female`

    >>> arr = ndrange(gender)
    >>> arr.axes['gen<tab>  # will be completed to `arr.axes['gender`

    >>> wb = open_excel()
    >>> wb['Sh<tab>  # will be completed to `wb['Sheet1`

* added documentation for Session methods (closes :issue:`277`).

* allowed to provide explict names for arrays or sessions in compare(). Closes :issue:`307`.


Fixes
-----

* fixed title argument of `ndtest` creation function: title was not passed to the returned array.

* fixed create_sequential when arguments initial and inc are array and scalar respectively (closes :issue:`288`).

* fixed auto-completion of attributes of LArray and Group objects (closes :issue:`302`).

* fixed name of arrays/sessions in compare() not being inferred correctly (closes :issue:`306`).

* fixed indexing Excel sheets by position to always yield the requested shape even when bounds are outside the range of
  used cells. Closes :issue:`273`.

* fixed the array() method on excel.Sheet returning float labels when int labels are expected.

* fixed getting float data instead of int when converting an Excel Sheet or Range to an larray or numpy array.

* fixed some warning messages to point to the correct line in user code.

* fixed crash of Session.save method when it contained 0D arrays. They are now skipped when saving a session (closes
  :issue:`291`).

* fixed Session.save and Session.to_excel failing to create new Excel files (it only worked if the file already
  existed). Closes :issue:`313`.

* fixed Session.load(file, engine='pandas_excel') : axes were considered as anonymous.
