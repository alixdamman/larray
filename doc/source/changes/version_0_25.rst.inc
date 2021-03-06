﻿New features
------------

* viewer functions (`view`, `edit` and `compare`) have been moved to the separate `larray-editor` package,
  which needs to be installed separately, unless you are using `larrayenv`.
  Closes :issue:`332`.

* installing larray-editor (or larrayenv) from conda environment creates a new menu 'LArray' in the Windows start menu.
  It contains a link to open the documentation, a shortcut to launch the user interface in edition mode
  and a shortcut to update larrayenv. Closes :issue:`281`.

* added possibility to transpose an array in the viewer by dragging and dropping axes' names in the filter bar.

* implemented array.align(other_array) which makes two arrays compatible with each other (by making all common axes
  compatible). This is done by adding, removing or reordering labels for each common axis according to the join method
  used:

    - outer: will use a label if it is in either arrays axis (ordered like the first array). This is the default as it
      results in no information loss.
    - inner: will use a label if it is in both arrays axis (ordered like the first array)
    - left: will use the first array axis labels
    - right: will use the other array axis labels

  The fill value for missing labels defaults to nan.

    >>> arr1 = ndtest((2, 3))
    >>> arr1
    a\b  b0  b1  b2
     a0   0   1   2
     a1   3   4   5
    >>> arr2 = -ndtest((3, 2))
    >>> # reorder array to make the test more interesting
    >>> arr2 = arr2[['b1', 'b0']]
    >>> arr2
    a\\b  b1  b0
     a0  -1   0
     a1  -3  -2
     a2  -5  -4

  Align arr1 and arr2

    >>> aligned1, aligned2 = arr1.align(arr2)
    >>> aligned1
    a\b   b0   b1   b2
     a0  0.0  1.0  2.0
     a1  3.0  4.0  5.0
     a2  nan  nan  nan
    >>> aligned2
    a\b    b0    b1   b2
     a0   0.0  -1.0  nan
     a1  -2.0  -3.0  nan
     a2  -4.0  -5.0  nan

  After aligning all common axes, one can then do operations between the two arrays

    >>> aligned1 + aligned2
    a\b   b0   b1   b2
     a0  0.0  0.0  nan
     a1  1.0  1.0  nan
     a2  nan  nan  nan

  The fill value for missing labels defaults to nan but can be changed to any compatible value.

    >>> aligned1, aligned2 = arr1.align(arr2, fill_value=0)
    >>> aligned1
    a\b  b0  b1  b2
     a0   0   1   2
     a1   3   4   5
     a2   0   0   0
    >>> aligned2
    a\b  b0  b1  b2
     a0   0  -1   0
     a1  -2  -3   0
     a2  -4  -5   0
    >>> aligned1 + aligned2
    a\b  b0  b1  b2
     a0   0   0   2
     a1   1   1   5
     a2  -4  -5   0

* implemented Session.transpose(axes) to reorder axes of all arrays within a session, ignoring missing axes for each
  array. For example, let us first create a test session and a small helper function to display sessions as a short
  summary.

    >>> arr1 = ndtest((2, 2, 2))
    >>> arr2 = ndtest((2, 2))
    >>> sess = Session([('arr1', arr1), ('arr2', arr2)])
    >>> def print_summary(s):
    ...     print(s.summary("{name} -> {axes_names}"))
    >>> print_summary(sess)
    arr1 -> a, b, c
    arr2 -> a, b

  Put the 'b' axis in front of all arrays

    >>> print_summary(sess.transpose('b'))
    arr1 -> b, a, c
    arr2 -> b, a

  Axes missing on an array are ignored ('c' for arr2 in this case)

    >>> print_summary(sess.transpose('c', 'b'))
    arr1 -> c, b, a
    arr2 -> b, a

  Use ... to move axes to the end

    >>> print_summary(sess.transpose(..., 'a'))
    arr1 -> b, c, a
    arr2 -> b, a

* implemented unary operations on Session, which means one can negate all arrays in a Session or take the absolute
  value of all arrays in a Session without writing an explicit loop for that.

    >>> arr1 = ndtest(2)
    >>> arr1
    a  a0  a1
        0   1
    >>> arr2 = ndtest(4) - 1
    >>> arr2
    a  a0  a1  a2  a3
       -1   0   1   2
    >>> sess1 = Session([('arr1', arr1), ('arr2', arr2)])
    >>> sess2 = -sess1
    >>> sess2.arr1
    a  a0  a1
        0  -1
    >>> sess2.arr2
    a  a0  a1  a2  a3
        1   0  -1  -2
    >>> sess3 = abs(sess1)
    >>> sess3.arr2
    a  a0  a1  a2  a3
        1   0   1   2

* implemented stacking sessions using stack().

  Let us first create two test sessions. For example suppose we have a session storing the results of a baseline
  simulation:

    >>> arr1 = ndtest(2)
    >>> arr1
    a  a0  a1
        0   1
    >>> arr2 = ndtest(3)
    >>> arr2
    a  a0  a1  a2
        0   1   2
    >>> baseline = Session([('arr1', arr1), ('arr2', arr2)])

  and another session with a variant

    >>> arr1variant = arr1 * 2
    >>> arr1variant
    a  a0  a1
        0   2
    >>> arr2variant = 2 - arr2 / 2
    >>> arr2variant
    a   a0   a1   a2
       2.0  1.5  1.0
    >>> variant = Session([('arr1', arr1variant), ('arr2', arr2variant)])

  then we stack them together

    >>> stacked = stack([('baseline', baseline), ('variant', variant)], 'sessions')
    >>> stacked
    Session(arr1, arr2)
    >>> stacked.arr1
    a\sessions  baseline  variant
            a0         0        0
            a1         1        2
    >>> stacked.arr2
    a\sessions  baseline  variant
            a0       0.0      2.0
            a1       1.0      1.5
            a2       2.0      1.0

  Combined with the fact that we can compute some very simple expressions on sessions, this can be extremely useful to
  quickly compare all arrays of several sessions (e.g. simulation variants):

    >>> diff = variant - baseline
    >>> # compute the absolute difference and relative difference for each array of the sessions
    >>> stacked = stack([('baseline', baseline),
                         ('variant', variant),
                         ('diff', diff),
                         ('abs diff', abs(diff)),
                         ('rel diff', diff / baseline)], 'sessions')
    >>> stacked
    Session(arr1, arr2)
    >>> stacked.arr2
    a\sessions  baseline  variant  diff  abs diff  rel diff
            a0       0.0      2.0   2.0       2.0       inf
            a1       1.0      1.5   0.5       0.5       0.5
            a2       2.0      1.0  -1.0       1.0      -0.5

* implemented Axis.align(other_axis) and AxisCollection.align(other_collection) which makes two axes / axis collections
  compatible with each other, see LArray.align above.

* implemented Session.apply(function) to apply a function to all elements (arrays) of a Session and return a new
  Session.

  Let us first create a test session

    >>> arr1 = ndtest(2)
    >>> arr1
    a  a0  a1
        0   1
    >>> arr2 = ndtest(3)
    >>> arr2
    a  a0  a1  a2
        0   1   2
    >>> sess1 = Session([('arr1', arr1), ('arr2', arr2)])
    >>> sess1
    Session(arr1, arr2)

  Then define the function we want to apply to all arrays of our session

    >>> def increment(element):
    ...     return element + 1

  Apply it

    >>> sess2 = sess1.apply(increment)
    >>> sess2.arr1
    a  a0  a1
        1   2
    >>> sess2.arr2
    a  a0  a1  a2
        1   2   3

* implemented setting the value of multiple points using array.points[labels] = value

    >>> arr = ndtest((3, 4))
    >>> arr
    a\b  b0  b1  b2  b3
     a0   0   1   2   3
     a1   4   5   6   7
     a2   8   9  10  11

  Now, suppose you want to retrieve several specific combinations of labels, for example
  (a0, b1), (a0, b3), (a1, b0) and (a2, b2). You could write a loop like this:

    >>> values = []
    >>> for a, b in [('a0', 'b1'), ('a0', 'b3'), ('a1', 'b0'), ('a2', 'b2')]:
    ...     values.append(arr[a, b])
    >>> values
    [1, 3, 4, 10]

  but you could also (this already worked in previous versions) use array.points like:

    >>> arr.points[['a0', 'a0', 'a1', 'a2'], ['b1', 'b3', 'b0', 'b2']]
    a,b  a0,b1  a0,b3  a1,b0  a2,b2
             1      3      4     10

  which has the advantages of being both much faster and keep more information. Now suppose you want to *set* the value
  of those points, you could write:

    >>> for a, b in [('a0', 'b1'), ('a0', 'b3'), ('a1', 'b0'), ('a2', 'b2')]:
    ...     arr[a, b] = 42
    >>> arr
    a\b  b0  b1  b2  b3
     a0   0  42   2  42
     a1  42   5   6   7
     a2   8   9  42  11

  but now you can also use the faster alternative:

    >>> arr.points[['a0', 'a0', 'a1', 'a2'], ['b1', 'b3', 'b0', 'b2']] = 42


Miscellaneous improvements
--------------------------

* added icon to display in Windows start menu and editor windows.

* viewer keeps labels visible even when scrolling (label rows and columns are now frozen).

* added 'Getting Started' section in documentation.

* implemented axes argument to ipfp to specify on which axes the fitting procedure should be applied (closes
  :issue:`185`). For example, let us assume you have a 3D array, such as:

    >>> initial = ndrange('a=a0..a9;b=b0..b9;year=2000..2016')

  and you want to apply a 2D fitting procedure for each value of the year axis. Previously, you had to loop on that
  year axis explicitly and call ipfp within the loop, like:

    >>> result = zeros(initial.axes)
    >>> for year in initial.year:
    ...     current = initial[year]
    ...     # assume you have some targets for each year
    ...     current_targets = [current.sum(x.a) + 1, current.sum(x.b) + 1]
    ...     result[year] = ipfp(current_targets, current)

  Now you can apply the procedure on all years at once, by telling you want to do the fitting procedure on the other
  axes. This is a bit shorter to type, but this is also *much* faster.

    >>> all_targets = [initial.sum(x.a) + 1, initial.sum(x.b) + 1]
    >>> result = ipfp(all_targets, initial, axes=(x.a, x.b))

* made ipfp 10 to 20% faster (even without using the axes argument).

* implemented Session.to_globals(inplace=True) which will update the content of existing arrays instead of creating new
  variables and overwriting them. This ensures the arrays have the same axes in the session than the existing variables.

* added the ability to provide a pattern when loading several .csv files as a session. Among others, patterns can use *
  to match any number of characters and ? to match any single character.

  >>> s = Session()
  >>> # load all .csv files starting with "output" in the data directory
  >>> s.load('data/output*.csv')

* stack can be used with keyword arguments when labels are "simple strings" (i.e. no integers, no punctuation,
  no string starting with integers, etc.). This is an attractive alternative but as it only works in the usual case and
  not in all cases, it is not recommended to use it except in the interactive console.

    >>> arr1 = ones('nat=BE,FO')
    >>> arr1
    nat   BE   FO
         1.0  1.0
    >>> arr2 = zeros('nat=BE,FO')
    >>> arr2
    nat   BE   FO
         0.0  0.0
    >>> stack(M=arr1, F=arr2, axis='sex=M,F')
    nat\\sex    M    F
         BE  1.0  0.0
         FO  1.0  0.0

  Without passing an explicit order for labels like above (or an axis object), it should only be used on Python 3.6 or
  later because keyword arguments are NOT ordered on earlier Python versions.

    >>> # use this only on Python 3.6 and later
    >>> stack(M=arr1, F=arr2, axis='sex')
    nat\\sex    M    F
         BE  1.0  0.0
         FO  1.0  0.0

* binary operations between session now ignore type errors. For example, if you are comparing two sessions with many
  arrays by computing the difference between them but a few arrays contain strings, the whole operation will not fail,
  the concerned arrays will be assigned a nan instead.

* added optional argument `ignore_exceptions` to Session.load to ignore exceptions during load. This is mostly useful
  when trying to load many .csv files in a Session and some of them have an invalid format but you want to load the
  others.


Fixes
-----

* fixed disambiguating an ambiguous key by adding the axis within the string, for example
  arr['axis_name[ambiguouslabel]'] (closes :issue:`331`).

* fixed converting a string group to integer or float using int() and float() (when that makes sense).

    >>> a = Axis('a=10,20,30,total')
    >>> a
    Axis(['10', '20', '30', 'total'], 'a')
    >>> str(a.i[0])
    '10'
    >>> int(a.i[0])
    10
    >>> float(a.i[0])
    10.0
