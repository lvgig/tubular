import pandas as pd
import pandas
import inspect
import pytest

import tubular.testing.helpers as h


def test_arguments():
    """Test arguments for arguments of function."""

    expected_arguments = ["actual", "expected", "msg_tag", "print_actual_and_expected"]

    expected_default_values = (False,)

    expected_var_keyword_arg = "kwargs"

    arg_spec = inspect.getfullargspec(h.assert_series_equal_msg)

    arguments = arg_spec.args

    assert len(expected_arguments) == len(
        arguments
    ), f"Incorrect number of arguments -\n  Expected: {len(expected_arguments)}\n  Actual: {len(arguments)}"

    for i, (e, a) in enumerate(zip(expected_arguments, arguments)):
        assert e == a, f"Incorrect arg at index {i} -\n  Expected: {e}\n  Actual: {a}"

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
            assert (
                e == a
            ), f"Incorrect default value at index {i} of default values -\n  Expected: {e}\n  Actual: {a}"

    var_keyword_arg = arg_spec.varkw

    assert (
        var_keyword_arg == expected_var_keyword_arg
    ), f"Unexpected keyword arg variable in assert_series_equal_msg -\n  Expected: {expected_var_keyword_arg}\n  Actual: {var_keyword_arg}"


def test_pandas_assert_series_called(mocker):
    """Test the call to pandas.testing.assert_series_equal."""

    srs = pd.Series({"a": [1, 2, 3]})
    srs2 = pd.Series({"a": [1, 2, 3]})

    spy = mocker.spy(pandas.testing, "assert_series_equal")

    h.assert_series_equal_msg(expected=srs, actual=srs2, msg_tag="a", check_dtype=True)

    assert (
        spy.call_count == 1
    ), f"Unexpected number of call to pd.testing.assert_series_equal_msg -\n  Expected: 1\n  Actual: {spy.call_count}"

    call_1_args = spy.call_args_list[0]
    call_1_pos_args = call_1_args[0]
    call_1_kwargs = call_1_args[1]

    call_1_expected_kwargs = {"check_dtype": True}
    call_1_expected_pos_args = (srs, srs2)

    assert len(call_1_expected_kwargs.keys()) == len(
        call_1_kwargs.keys()
    ), f"Unexpected number of kwargs -\n  Expected: {len(call_1_expected_kwargs.keys())}\n  Actual: {len(call_1_kwargs.keys())}"

    assert (
        call_1_expected_kwargs["check_dtype"] == call_1_kwargs["check_dtype"]
    ), f"""check_dtype kwarg unexpected -\n  Expected {call_1_expected_kwargs['check_dtype']}\n  Actual: {call_1_kwargs['check_dtype']}"""

    assert len(call_1_expected_pos_args) == len(
        call_1_pos_args
    ), f"Unexpected number of kwargs -\n  Expected: {len(call_1_expected_pos_args)}\n  Actual: {len(call_1_pos_args)}"

    pd.testing.assert_series_equal(call_1_expected_pos_args[0], call_1_pos_args[0])
    pd.testing.assert_series_equal(call_1_expected_pos_args[1], call_1_pos_args[1])


def test_exception_no_print():
    """Test an assert error is raised (with correct info) in case of exception coming from assert_series_equal and
    print_actual_and_expected is False.
    """

    srs = pd.Series({"a": [1, 2, 3]})
    srs2 = pd.Series({"a": [1, 2, 4]})

    with pytest.raises(AssertionError, match="a"):
        h.assert_series_equal_msg(
            expected=srs, actual=srs2, msg_tag="a", print_actual_and_expected=False
        )


def test_exception_print():
    """Test an assert error is raised (with correct info) in case of exception coming from assert_series_equal and
    print_actual_and_expected is True.
    """

    srs = pd.Series({"a": [1, 2, 3]})
    srs2 = pd.Series({"a": [1, 2, 4]})

    with pytest.raises(AssertionError) as exc_info:
        h.assert_series_equal_msg(
            expected=srs, actual=srs2, msg_tag="a", print_actual_and_expected=True
        )

    assert exc_info.value.args[0] == "a\n" + f"expected:\n{srs}\n" + f"actual:\n{srs2}"
