import inspect
import pytest
import tubular.testing.helpers as h
import pandas as pd
from unittest import mock


def test_arguments():
    """Test arguments for arguments of tubular.testing.helpers.test_function_arguments."""

    expected_arguments = ["func", "expected_arguments", "expected_default_values"]

    expected_default_values = (None,)

    arg_spec = inspect.getfullargspec(h.test_function_arguments)

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


def test_expected_arguments_not_list_error():
    """Test that an exception is raised if expected_arguments is not a list."""

    with pytest.raises(TypeError):

        h.test_function_arguments(h.check_is_class, expected_arguments=1)


def test_expected_default_values_not_tuple_error():
    """Test that an exception is raised if expected_default_values is not a tuple (if it is not None)."""

    with pytest.raises(TypeError):

        h.test_function_arguments(
            h.check_is_class,
            expected_arguments=["class_to_check"],
            expected_default_values={},
        )


def test_getfullargspec_call_count():
    """Test the call to inspect.getfullargspec."""

    with mock.patch(
        target="inspect.getfullargspec",
        return_value=inspect.getfullargspec(h.check_is_class),
    ) as mocked_method:

        h.test_function_arguments(
            h.check_is_class, expected_arguments=["class_to_check"]
        )

        assert (
            mocked_method.call_count == 1
        ), f"Unexpected number of calls to inspect.getfullargspec -\n  Expected: 1\n  Actual: {mocked_method.call_count}"


def test_different_number_arguments_error():
    """Test that an AssertException is raised if the actual and expected positional args have different lengths."""

    incorrect_expected_args = ["actual", "expected", "msg_tag"]

    with pytest.raises(
        AssertionError,
        match=f"Incorrect number of arguments -\n  Expected: {len(incorrect_expected_args)}\n  Actual: 4",
    ):

        h.test_function_arguments(
            h.assert_frame_equal_msg,
            expected_arguments=incorrect_expected_args,
            expected_default_values=(False,),
        )


def test_assert_equal_msg_calls_for_positional_arguments():
    """Test the calls to assert_equal_msg for the positional arguments."""

    with mock.patch(target="tubular.testing.helpers.assert_equal_msg") as mocked_method:

        expected_args = ["obj", "expected_method", "msg"]

        h.test_function_arguments(
            h.test_object_method, expected_arguments=expected_args
        )

        assert mocked_method.call_count == len(
            expected_args
        ), f"Unexpected number of calls to tubular.testing.helpers.assert_equal_msg -\n  Expected: {len(expected_args)}\n  Actual: {mocked_method.call_count}"

        for call_n in range(mocked_method.call_count):

            call_n_args = mocked_method.call_args_list[call_n]
            call_n_pos_args = call_n_args[0]
            call_n_kwargs = call_n_args[1]

            call_n_expected_pos_arg = (
                expected_args[call_n],
                expected_args[call_n],
                f"Incorrect arg at index {call_n}",
            )

            assert len(call_n_pos_args) == len(
                call_n_expected_pos_arg
            ), f"Unexpected number of positional args in call {call_n} to tubular.testing.helpers.assert_equal_msg -\n  Expected: {len(call_n_expected_pos_arg)}\n  Actual:  {len(call_n_pos_args)}"

            for i, (e, a) in enumerate(zip(call_n_expected_pos_arg, call_n_pos_args)):

                assert (
                    e == a
                ), f"Unexpected positional arg at index {i} -\n  Expected: {e}\n  Actual: {a}"

            assert (
                call_n_kwargs == {}
            ), f"Unexpected keyword args in call {call_n} to tubular.testing.helpers.assert_equal_msg -\n  Expected: None\n  Actual:  {call_n_kwargs}"


def test_default_values_non_mismatch_1_error():
    """Test that an exception is raised if the actual default values are none but the expected default values
    are not.
    """

    incorrect_defualt_value = ("a",)

    with pytest.raises(
        AssertionError,
        match=r"Incorrect default values -\n  Expected: .*\n  Actual: No default values",
    ):

        h.test_function_arguments(
            h.assert_inheritance,
            expected_arguments=["obj", "cls"],
            expected_default_values=incorrect_defualt_value,
        )


def test_default_values_non_mismatch_2_error():
    """Test that an exception is raised if the actual default values are not none but the expected default values
    are.
    """

    incorrect_defualt_value = None

    with pytest.raises(
        AssertionError,
        match="Incorrect default values -\n  Expected: No default values\n  Actual: .*",
    ):

        h.test_function_arguments(
            h.assert_frame_equal_msg,
            expected_arguments=[
                "actual",
                "expected",
                "msg_tag",
                "print_actual_and_expected",
            ],
            expected_default_values=incorrect_defualt_value,
        )


def test_different_number_default_values_error():
    """Test that an AssertException is raised if the actual and expected default values have different lengths."""

    incorrect_defualt_value = ("a", "b", "c")

    with pytest.raises(
        AssertionError,
        match=f"Incorrect number of default values -\n  Expected: {len(incorrect_defualt_value)}\n  Actual: 1",
    ):

        h.test_function_arguments(
            h.assert_frame_equal_msg,
            expected_arguments=[
                "actual",
                "expected",
                "msg_tag",
                "print_actual_and_expected",
            ],
            expected_default_values=incorrect_defualt_value,
        )


def test_assert_equal_msg_calls_for_default_values():
    """Test the calls to assert_equal_msg for the keyword arguments."""

    with mock.patch(target="tubular.testing.helpers.assert_equal_msg") as mocked_method:

        expected_args = ["self", "data", "index", "columns", "dtype", "copy"]
        expected_default_values = (None, None, None, None, False)

        h.test_function_arguments(
            pd.DataFrame,
            expected_arguments=expected_args,
            expected_default_values=expected_default_values,
        )

        assert mocked_method.call_count == (
            len(expected_args) + len(expected_default_values)
        ), f"Unexpected number of calls to tubular.testing.helpers.assert_equal_msg -\n  Expected: {len(expected_args) + len(expected_default_values)}\n  Actual: {mocked_method.call_count}"

        for call_n in range(len(expected_args), mocked_method.call_count):

            call_n_args = mocked_method.call_args_list[call_n]
            call_n_pos_args = call_n_args[0]
            call_n_kwargs = call_n_args[1]

            call_n_expected_pos_arg = (
                expected_default_values[call_n - len(expected_args)],
                expected_default_values[call_n - len(expected_args)],
                f"Incorrect default value at index {call_n - len(expected_args)} of default values",
            )

            assert len(call_n_pos_args) == len(
                call_n_expected_pos_arg
            ), f"Unexpected number of positional args in call {call_n} to tubular.testing.helpers.assert_equal_msg -\n  Expected: {len(call_n_expected_pos_arg)}\n  Actual:  {len(call_n_pos_args)}"

            for i, (e, a) in enumerate(zip(call_n_expected_pos_arg, call_n_pos_args)):

                assert (
                    e == a
                ), f"Unexpected default value at index {i} -\n  Expected: {e}\n  Actual: {a}"

            assert (
                call_n_kwargs == {}
            ), f"Unexpected keyword args in call {call_n} to tubular.testing.helpers.assert_equal_msg -\n  Expected: None\n  Actual:  {call_n_kwargs}"
