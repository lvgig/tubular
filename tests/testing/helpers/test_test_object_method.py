import inspect
import pytest
import tubular.testing.helpers as h
from unittest import mock


def test_arguments():
    """Test arguments for arguments of tubular.testing.helpers.test_object_method."""

    expected_arguments = ["obj", "expected_method", "msg"]

    arg_spec = inspect.getfullargspec(h.test_object_method)

    arguments = arg_spec.args

    assert len(expected_arguments) == len(
        arguments
    ), f"Incorrect number of arguments -\n  Expected: {len(expected_arguments)}\n  Actual: {len(arguments)}"

    for i, (e, a) in enumerate(zip(expected_arguments, arguments)):

        assert e == a, f"Incorrect arg at index {i} -\n  Expected: {e}\n  Actual: {a}"

    default_values = arg_spec.defaults

    assert (
        default_values is None
    ), f"Unexpected default values -\n  Expected: None\n  Actual: {default_values}"


def test_expected_method_not_str_error():
    """Test that an exception is raised if expected_method is not passed as a str."""

    non_str_expected_method = 1

    with pytest.raises(
        TypeError,
        match=f"expected_method should be a str but got {type(non_str_expected_method)}",
    ):

        h.test_object_method(
            obj="a", expected_method=non_str_expected_method, msg="msg"
        )


def test_hasattr_call():
    """Test the call to hasattr."""

    with mock.patch(
        target="tubular.testing.helpers.hasattr", return_value=True
    ) as mocked_method:

        h.test_object_method(obj="s", expected_method="upper", msg="msg")

        assert (
            mocked_method.call_count == 1
        ), f"Unexpected number of calls to hasattr -\n  Expected: 1\n  Actual: {mocked_method.call_count}"

        call_1_args = mocked_method.call_args_list[0]
        call_1_pos_args = call_1_args[0]
        call_1_kwargs = call_1_args[1]

        expected_pos_args = ("s", "upper")

        assert (
            call_1_kwargs == {}
        ), f"Unexpected call keyword args -\n  Expected: None\n  Actual: {call_1_kwargs}"

        assert len(call_1_pos_args) == len(
            expected_pos_args
        ), f"Difference in number of positional arguments -\n  Expected: {len(expected_pos_args)}\n  Actual: {len(call_1_pos_args)}"

        for i, (e, a) in enumerate(zip(call_1_pos_args, expected_pos_args)):

            assert (
                e == a
            ), f"Difference in positional args at index {i} -\n Expected: {e}\n  Actual: {a}"


def test_callable_call():
    """Test the call to callable."""

    with mock.patch(target="tubular.testing.helpers.callable") as mocked_method:

        h.test_object_method(obj="s", expected_method="upper", msg="msg")

        assert (
            mocked_method.call_count == 1
        ), f"Unexpected number of calls to hasattr -\n  Expected: 1\n  Actual: {mocked_method.call_count}"

        call_1_args = mocked_method.call_args_list[0]
        call_1_pos_args = call_1_args[0]
        call_1_kwargs = call_1_args[1]

        expected_pos_args = (getattr("s", "upper"),)

        assert (
            call_1_kwargs == {}
        ), f"Unexpected call keyword args -\n  Expected: None\n  Actual: {call_1_kwargs}"

        assert len(call_1_pos_args) == len(
            expected_pos_args
        ), f"Difference in number of positional arguments -\n  Expected: {len(expected_pos_args)}\n  Actual: {len(call_1_pos_args)}"

        for i, (e, a) in enumerate(zip(call_1_pos_args, expected_pos_args)):

            assert (
                e == a
            ), f"Difference in positional args at index {i} -\n Expected: {e}\n  Actual: {a}"
