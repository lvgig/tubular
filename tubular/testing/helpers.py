"""This module contains helper functions that are used in the tests."""

import inspect
import pandas as pd
import numpy as np
from contextlib import contextmanager

import pytest
import pytest_mock


def assert_equal_dispatch(expected, actual, msg):
    """This function is used to call specific assert functions depending on the input types.
    Often we are dealing with pandas.DataFrame or pandas.Series objects when asserting
    equality in this project and these types cannot be compared with the standard ==. This
    function allows these types to be compared appropriately as well as allowing objects
    that may contain these pandas types (e.g. list) to also be compared.

    The first assert is that actual and expected are of the same type. If this passes then
    the following types have specific assert functions;
    - pd.DataFrame
    - pd.Series
    - pd.Index
    - list
    - tuple
    - dict
    - np.float
    - np.NaN
    if on object is passed that is not one of the above types then the standard assert for
    equality is used.

    Parameters
    ----------
    actual : object
        The expected object.

    expected : object
        The actual object.

    msg : string
        A message to be used in the assert, passed onto the specific assert equality function
        that is called.

    """

    if not type(actual) == type(expected):

        raise TypeError(
            f"expected ({type(expected)}) and actual ({type(actual)}) type mismatch"
        )

    if type(expected) is pd.DataFrame:

        assert_frame_equal_msg(actual, expected, msg)

    elif type(expected) is pd.Series:

        assert_series_equal_msg(actual, expected, msg)

    elif isinstance(expected, pd.Index):

        assert_index_equal_msg(actual, expected, msg)

    elif type(expected) in [list, tuple]:

        assert_list_tuple_equal_msg(actual, expected, msg)

    elif isinstance(expected, dict):

        assert_dict_equal_msg(actual, expected, msg)

    elif (type(expected) is float or isinstance(expected, np.float)) and np.isnan(
        expected
    ):

        assert_np_nan_eqal_msg(actual, expected, msg)

    else:

        assert_equal_msg(actual, expected, msg)


def assert_equal_msg(actual, expected, msg_tag):
    """Compares actual and expected objects and simply asserts equality (==). Adds msg_tag, actual and expected
    values to AssertionException message.

    Parameters
    ----------
    actual : object
        The expected object.

    expected : object
        The actual object.

    msg_tag : string
        A tag for the AssertionException message. Use this to identify mismatching arguments in test output.

    """

    error_msg = f"{msg_tag} -\n  Expected: {expected}\n  Actual: {actual}"

    assert actual == expected, error_msg


def assert_np_nan_eqal_msg(actual, expected, msg):
    """Function to test that both values are np.NaN.

    Parameters
    ----------
    actual : object
        The expected object. Must be an numeric type for np.isnan to run.

    expected : object
        The actual object. Must be an numeric type for np.isnan to run.

    msg : string
        A tag for the AssertionException message.

    """

    assert np.isnan(actual) and np.isnan(
        expected
    ), f"Both values are not equal to np.NaN -\n  Expected: {expected}\n  Actual: {actual}"


def assert_list_tuple_equal_msg(actual, expected, msg_tag):
    """Compares two actual and expected list or tuple objects and asserts equality between the two.
    Error output will identify location of mismatch in items.

    Checks actual and expected are the same type, then equal length then loops through pariwise eleemnts
    and calls assert_equal_dispatch function.

    Parameters
    ----------
    actual : list or tuple
        The actual list or tuple to compare.

    expected : list or tuple
        The expected list or tuple to compare to actual.

    msg_tag : string
        A tag for the AssertionException message.

    """

    if not type(expected) in [list, tuple]:

        raise TypeError(
            f"expected should be of type list or tuple, but got {type(expected)}"
        )

    if not type(actual) in [list, tuple]:

        raise TypeError(
            f"actual should be of type list or tuple, but got {type(actual)}"
        )

    if not type(actual) == type(expected):

        raise TypeError(
            f"expect ({type(expected)}) and actual ({type(actual)}) type mismatch"
        )

    assert len(expected) == len(
        actual
    ), f"Unequal lengths -\n  Expected: {len(expected)}\n  Actual: {len(actual)}"

    for i, (e, a) in enumerate(zip(expected, actual)):

        assert_equal_dispatch(e, a, f"{msg_tag} index {i}")


def assert_dict_equal_msg(actual, expected, msg_tag):
    """Compares two actual and expected dict objects and asserts equality. Error output
    will identify (first) location of mismatch values.

    Checks actual and expected are both dicts, then same number of keys then loops through pariwise
    values from actual and expected and calls assert_equal_dispatch function on these pairs..

    Parameters
    ----------
    actual : dict
        The actual dict to compare.

    expected : dict
        The expected dict to compare to actual.

    msg_tag : string
        A tag for the AssertionException message.

    """

    if not isinstance(expected, dict):

        raise TypeError(f"expected should be of type dict, but got {type(expected)}")

    if not isinstance(actual, dict):

        raise TypeError(f"actual should be of type dict, but got {type(actual)}")

    assert len(expected.keys()) == len(
        actual.keys()
    ), f"Unequal number of keys -\n  Expected: {len(expected.keys())}\n  Actual: {len(actual.keys())}"

    keys_diff_e_a = set(expected.keys()) - set(actual.keys())

    keys_diff_a_e = set(actual.keys()) - set(expected.keys())

    assert (
        keys_diff_e_a == set()
    ), f"Keys in expected not in actual: {keys_diff_e_a}\nKeys in actual not in expected: {keys_diff_a_e}"

    for k in actual.keys():

        assert_equal_dispatch(expected[k], actual[k], f"{msg_tag} key {k}")


def assert_frame_equal_msg(
    actual, expected, msg_tag, print_actual_and_expected=False, **kwargs
):
    """Compares actual and expected pandas.DataFrames and asserts equality.

    Calls pd.testing.assert_frame_equal but presents msg_tag, and optionally actual and expected
    DataFrames, in addition to any other exception info.

    Parameters
    ----------
    actual : pandas DataFrame
        The expected dataframe.

    expected : pandas DataFrame
        The actual dataframe.

    msg_tag : string
        A tag for the assert error message.

    **kwargs:
        Keyword args passed to pd.testing.assert_frame_equal.

    """

    try:

        pd.testing.assert_frame_equal(expected, actual, **kwargs)

    except Exception as e:

        if print_actual_and_expected:

            error_msg = f"""{msg_tag}\nexpected:\n{expected}\nactual:\n{actual}"""

        else:

            error_msg = msg_tag

        raise AssertionError(error_msg) from e


def assert_series_equal_msg(
    actual, expected, msg_tag, print_actual_and_expected=False, **kwargs
):
    """Compares actual and expected pandas.Series and asserts equality.
    Calls pd.testing.assert_series_equal but presents msg_tag, and optionally actual and expected
    Series, in addition to any other exception info.

    Parameters
    ----------
    actual : pandas Series
        The actual Series.

    expected : pandas Series
        The expected Series.

    msg_tag : string
        A tag for the assert error message.

    print_actual_and_expected : Boolean
        print the actual and expected dataFrame along with error message tag

    **kwargs:
        Keyword args passed to pd.testing.assert_series_equal.

    """

    try:

        pd.testing.assert_series_equal(expected, actual, **kwargs)

    except Exception as e:

        if print_actual_and_expected:

            error_msg = f"""{msg_tag}\nexpected:\n{expected}\nactual:\n{actual}"""

        else:

            error_msg = msg_tag

        raise AssertionError(error_msg) from e


def assert_index_equal_msg(
    actual, expected, msg_tag, print_actual_and_expected=False, **kwargs
):
    """Compares actual and expected pandas.Index objects and asserts equality.
    Calls pd.testing.assert_index_equal but presents msg_tag, and optionally actual and expected
    Series, in addition to any other exception info.

    Parameters
    ----------
    actual : pd.Index
        The actual index.

    expected : pd.Index
        The expected index.

    msg_tag : string
        A tag for the assert error message.

    print_actual_and_expected : Boolean
        print the actual and expected valuess along with error message tag

    **kwargs:
        Keyword args passed to pd.testing.assert_index_equal.

    """

    try:

        pd.testing.assert_index_equal(expected, actual, **kwargs)

    except Exception as e:

        if print_actual_and_expected:

            error_msg = f"""{msg_tag}\nexpected:\n{expected}\nactual:\n{actual}"""

        else:

            error_msg = msg_tag

        raise AssertionError(error_msg) from e


def check_is_class(class_to_check):
    """Raises type error if class_to_check is not a class.

    Uses inspect.isclass.

    Parameters
    ----------
    class_to_check : object
        The object to be inspected.

    """

    if inspect.isclass(class_to_check) is False:

        raise TypeError(f"{class_to_check} is not a valid class")


def assert_inheritance(obj, cls):
    """Asserts whether an object inherits from a particular class.

    Uses isinstance.

    Parameters
    ----------
    obj : object
        The object to test.

    cls : Class
        Class to check obj is an instance of.

    """

    check_is_class(cls)

    assert isinstance(
        obj, cls
    ), f"Incorrect inheritance - passed obj of class {obj.__class__.__name__} is not an instance of {cls}"


def test_object_method(obj, expected_method, msg):
    """Test that a particular object has a given method and the (method) attribute is callable.

    Uses hasattr to check the method attribute exists, then callable(getattr()) to check the method
    is callable.

    Parameters
    ----------
    obj : object
        The object to test.

    expected_method : str
        Name of expected method on obj.

    """

    if not type(expected_method) is str:

        raise TypeError(
            f"expected_method should be a str but got {type(expected_method)}"
        )

    assert hasattr(
        obj, expected_method
    ), f"obj does not have attribute {expected_method}"

    assert callable(
        getattr(obj, expected_method)
    ), f"{expected_method} on obj is not callable"


def test_object_attributes(obj, expected_attributes, msg):
    """Check a particular object has given attributes.

    Function loops through key, value pairs in expected_attributes dict and checks
    there is an attribute with the name of each key is on obj then calls assert_equal_dispatch
    to check the expected value and actual value are equal.

    Parameters
    ----------
    obj : object
        Object to test attributes of.

    expected_attributes : dict
        Dict of expected attributes and their values.

    msg : str
        Message tag passed onto assert_equal_dispatch.

    """

    if not type(expected_attributes) is dict:

        raise TypeError(
            f"expected_attributes should be a dict but got {type(expected_attributes)}"
        )

    for attribute_name, expected in expected_attributes.items():

        assert hasattr(obj, attribute_name), f"obj has not attribute {attribute_name}"

        actual = getattr(obj, attribute_name)

        assert_equal_dispatch(
            expected=expected, actual=actual, msg=f"{attribute_name} {msg}"
        )


def test_function_arguments(func, expected_arguments, expected_default_values=None):
    """Test that a given function has expected arguments and default values.

    Uses inspect.getfullargspec to get the function argument information. Then loops through
    argument names and default values and uses assert_equal_msg to check actuals are equal to
    expected.

    Parameters
    ----------
    func : method
        Function to check.

    expected_arguments : list
        List of names of expected argument, in order.

    expected_default_values : tuple or None
        A tuple of lenght n of default argument values for the last n positional arguments, or None if
        there are no default values.

    """

    if not type(expected_arguments) is list:

        raise TypeError(
            f"expected_arguments should be a list but got {type(expected_arguments)}"
        )

    if expected_default_values is not None:

        if not type(expected_default_values) is tuple:

            raise TypeError(
                f"expected_default_values should be a tuple but got {type(expected_default_values)}"
            )

    arg_spec = inspect.getfullargspec(func)

    arguments = arg_spec.args

    assert len(expected_arguments) == len(
        arguments
    ), f"Incorrect number of arguments -\n  Expected: {len(expected_arguments)}\n  Actual: {len(arguments)}"

    for i, (e, a) in enumerate(zip(expected_arguments, arguments)):

        assert_equal_msg(a, e, f"Incorrect arg at index {i}")

    default_values = arg_spec.defaults

    if default_values is None:

        if expected_default_values is not None:

            raise AssertionError(
                f"Incorrect default values -\n  Expected: {expected_default_values}\n  Actual: No default values"
            )

    else:

        if expected_default_values is None:

            raise AssertionError(
                f"Incorrect default values -\n  Expected: No default values\n  Actual: {default_values}"
            )

    if (default_values is not None) and (expected_default_values is not None):

        assert len(expected_default_values) == len(
            default_values
        ), f"Incorrect number of default values -\n  Expected: {len(expected_default_values)}\n  Actual: {len(default_values)}"

        for i, (e, a) in enumerate(zip(expected_default_values, default_values)):

            assert_equal_msg(
                a, e, f"Incorrect default value at index {i} of default values"
            )


def _check_dfs_passed(df_1, df_2):
    """Function to check that two pd.DataFrames have equal indexes.

    Parameters
    ----------
    df_1 : pd.DataFrame
        First df to compare

    df_2 : pd.DataFrame
        Second df to compare

    Raises
    ------
    TypeError
        If the first or second positional arg is not a pd.DataFrame

    ValueError
        If the first and second positional args do not have the same number of rows

    ValueError
        If the first and second positonal args do not have equal indexes

    """

    if not type(df_1) is pd.DataFrame:
        raise TypeError(
            f"expecting first positional arg to be a pd.DataFrame but got {type(df_1)}"
        )

    if not type(df_2) is pd.DataFrame:
        raise TypeError(
            f"expecting second positional arg to be a pd.DataFrame but got {type(df_2)}"
        )

    if df_1.shape[0] != df_2.shape[0]:
        raise ValueError(
            f"expecting first positional arg and second positional arg to have equal number of rows but got\n  {df_1.shape[0]}\n  {df_2.shape[0]}"
        )

    if not (df_1.index == df_2.index).all():
        raise ValueError(
            f"expecting indexes for first positional arg and second positional arg to be the same but got\n  {df_1.index}\n  {df_2.index}"
        )


def row_by_row_params(df_1, df_2):
    """Helper function to split input pd.DataFrames pairs into a list of pytest.params of individual row pairs and a final
    pytest.param of the original inputs.

    This function can be used in combination with the pytest.mark.parametrize decorator to easily test that a transformer
    transform method is giving the expected outputs, when called row by row as well as multi row inputs.
    """

    _check_dfs_passed(df_1, df_2)

    params = [
        pytest.param(df_1.loc[[i]].copy(), df_2.loc[[i]].copy(), id=f"index {i}")
        for i in df_1.index
    ]

    params.append(pytest.param(df_1, df_2, id=f"all rows ({df_1.shape[0]})"))

    return params


def index_preserved_params(df_1, df_2, seed=0):
    """Helper function to create copies of input pd.DataFrames pairs in a list of pytest.params where each copy has a different
    index (random, increasing and decreasing), the last item in the list is a pytest.param of the original inputs.

    This function can be used in combination with the pytest.mark.parametrize decorator to easily test that a transformer
    transform method preserves the index of the input.
    """

    _check_dfs_passed(df_1, df_2)

    # create random, increasing and decreasing indexes to set on df args then run func with
    np.random.seed(seed)
    random_index = np.random.randint(low=-99999999, high=100000000, size=df_1.shape[0])
    start_decreasing_index = np.random.randint(low=-99999999, high=100000000, size=1)[0]
    decreasing_index = range(
        start_decreasing_index, start_decreasing_index - df_1.shape[0], -1
    )
    start_increasing_index = np.random.randint(low=-99999999, high=100000000, size=1)[0]
    increasing_index = range(
        start_increasing_index, start_increasing_index + df_1.shape[0], 1
    )

    index_names = ["random", "decreasing", "increasing"]
    indexes = [random_index, decreasing_index, increasing_index]

    params = []

    for index, index_name in zip(indexes, index_names):

        df_1_copy = df_1.copy()
        df_2_copy = df_2.copy()

        df_1_copy.index = index
        df_2_copy.index = index

        params.append(pytest.param(df_1_copy, df_2_copy, id=f"{index_name} index"))

    params.append(pytest.param(df_1, df_2, id="original index"))

    return params


@contextmanager
def assert_function_call_count(mocker, target, attribute, expected_n_calls, **kwargs):
    """Assert a function has been called a given number of times. This should be used
    as a context manager, see example below.

    Parameters
    ----------
    mocker : pytest_mock.plugin.MockerFixture
        The mocker fixture from the pytest_mock package.

    target : object
        Object with function to test is called.

    attribute : str
        Name of the method to mock on the target object

    expected_n_calls : int
        Expected number of calls to target.attribute.

    **kwargs : any
        Arbitrary keyword arguments passed on to mocker.patch.object.

    Examples
    --------
    >>> import tubular.testing.helpers as h
    >>> import tubular.testing.test_data as d
    >>> import tubular
    >>>
    >>> def test_number_calls_to_function(mocker):
    ...     df = d.create_df_1()
    ...     x = tubular.base.BaseTransformer(columns="a")
    ...     with h.assert_function_call_count(mocker, tubular.base.BaseTransformer, "columns_set_or_check", 1):
    ...         x.fit(X=df)

    """

    if type(mocker) is not pytest_mock.plugin.MockerFixture:

        raise TypeError("mocker should be the pytest_mock mocker fixture")

    mocked_method = mocker.patch.object(target, attribute, **kwargs)

    try:

        yield mocked_method

    finally:

        assert (
            mocked_method.call_count == expected_n_calls
        ), f"incorrect number of calls to {attribute}, expected {expected_n_calls} but got {mocked_method.call_count}"


@contextmanager
def assert_function_call(mocker, target, attribute, expected_calls_args, **kwargs):
    """Assert a function has been called with given arguments. This should be used
    as a context manager, see example below.

    This can be used to check multiple calls to the same function. Both the positional
    and keyword arguments must be specified for any calls to check.

    Parameters
    ----------
    mocker : pytest_mock.plugin.MockerFixture
        The mocker fixture from the pytest_mock package.

    target : object
        Object with function to test is called.

    attribute : str
        Name of the method to mock on the target object

    expected_calls_args : dict
        Expected positional and keyword arguments to specific calls to target.attribute. Keys
        of expected_calls_args must be ints (indexed from 0) indicating the call # of
        interest. The value for each key should be a dict with keys 'args' and 'kwargs'.
        The values for the keys in these sub dicts should be a tuple of expected positional
        arguments and a dict of expected keyword arguments for the given call to target.attribute.
        For example if expected_calls_args = {0: {'args':('a','b'), 'kwargs':{'c': 1}}} this
        indicates that the first call to target.attribute is expected to be called with positional
        args ('a','b') and keyword args {'c': 1}. See the example section below for more examples.

    **kwargs : any
        Arbitrary keyword arguments passed on to mocker.patch.object.

    Examples
    --------
    >>> import tubular.testing.helpers as h
    >>> import tubular
    >>>
    >>> def test_number_calls_to_function(mocker):
    ...     expected_call_arguments = {
    ...         0: {
    ...             'args': (
    ...                 'a',
    ...             ),
    ...             'kwargs': {
    ...                 'other': 1
    ...             }
    ...         },
    ...         2: {
    ...             'args': (
    ...                 ["a", "b"],
    ...             ),
    ...             'kwargs': {}
    ...         },
    ...         3: {
    ...             'args': (),
    ...             'kwargs': {
    ...                 'columns': ["a", "b"]
    ...              }
    ...         }
    ...     }
    ...     with h.assert_function_call(mocker, tubular.base.BaseTransformer, "__init__", expected_call_arguments, return_value=None):
    ...         x = tubular.base.BaseTransformer("a", other=1)
    ...         x2 = tubular.base.BaseTransformer()
    ...         x3 = tubular.base.BaseTransformer(["a", "b"])
    ...         x4 = tubular.base.BaseTransformer(columns = ["a", "b"])
    >>>

    """

    if type(mocker) is not pytest_mock.plugin.MockerFixture:
        raise TypeError("mocker should be the pytest_mock mocker fixture")

    if not type(expected_calls_args) is dict:
        raise TypeError("expected_calls_args should be a dict")

    for call_number, call_n_expected_arguments in expected_calls_args.items():

        if not type(call_number) is int:
            raise TypeError("expected_calls_args keys should be integers")

        if call_number < 0:
            raise ValueError(
                "expected_calls_args keys should be integers greater than or equal to 0"
            )

        if not type(call_n_expected_arguments) is dict:
            raise TypeError("each value in expected_calls_args should be a dict")

        if not sorted(list(call_n_expected_arguments.keys())) == ["args", "kwargs"]:
            raise ValueError(
                """keys of each sub dict in expected_calls_args should be 'args' and 'kwargs' only"""
            )

        if not type(call_n_expected_arguments["args"]) is tuple:
            raise TypeError("args in expected_calls_args should be tuples")

        if not type(call_n_expected_arguments["kwargs"]) is dict:
            raise TypeError("kwargs in expected_calls_args should be dicts")

    max_expected_call = max(expected_calls_args.keys())

    mocked_method = mocker.patch.object(target, attribute, **kwargs)

    try:

        yield mocked_method

    finally:

        assert mocked_method.call_count >= (
            max_expected_call + 1
        ), f"not enough calls to {attribute}, expected at least {max_expected_call+1} but got {mocked_method.call_count}"

        for call_number, call_n_expected_arguments in expected_calls_args.items():

            call_n_arguments = mocked_method.call_args_list[call_number]
            call_n_pos_args = call_n_arguments[0]
            call_n_kwargs = call_n_arguments[1]

            expected_call_n_pos_args = call_n_expected_arguments["args"]
            expected_call_n_kwargs = call_n_expected_arguments["kwargs"]

            assert_list_tuple_equal_msg(
                actual=call_n_pos_args,
                expected=expected_call_n_pos_args,
                msg_tag=f"positional args for call {call_number} not correct",
            )

            assert_dict_equal_msg(
                actual=call_n_kwargs,
                expected=expected_call_n_kwargs,
                msg_tag=f"kwargs for call {call_number} not correct",
            )
