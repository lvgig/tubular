import pytest
import inspect
import tubular.testing.helpers as h
from unittest import mock


def test_arguments():
    """Test arguments for arguments of tubular.testing.helpers.assert_list_tuple_equal_msg."""

    expected_arguments = ["actual", "expected", "msg_tag"]

    arg_spec = inspect.getfullargspec(h.assert_list_tuple_equal_msg)

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


def test_expected_not_list_tuple_error():
    """'Test that a TypeError is raised if expected is not a tuple or list."""

    with pytest.raises(
        TypeError, match=f"expected should be of type list or tuple, but got {type(1)}"
    ):

        h.assert_list_tuple_equal_msg(expected=1, actual=[], msg_tag="test_msg")


def test_actual_not_list_tuple_error():
    """'Test that a TypeError is raised if actual is not a tuple or list."""

    with pytest.raises(
        TypeError, match=f"actual should be of type list or tuple, but got {type(1)}"
    ):

        h.assert_list_tuple_equal_msg(expected=(), actual=1, msg_tag="test_msg")


def test_expected_actual_not_same_types_error():
    """'Test that a TypeError is raised if actual and expected have different types."""

    with pytest.raises(TypeError, match="type mismatch"):

        h.assert_list_tuple_equal_msg(expected=(), actual=[], msg_tag="test_msg")


def test_different_lengths_error():
    """Test that a ValueError is raised if actual and expected have different lengths."""

    expected_value = [1, 2, 3]
    actual_value = []

    with pytest.raises(
        AssertionError,
        match=f"Unequal lengths -\n  Expected: {len(expected_value)}\n  Actual: {len(actual_value)}",
    ):

        h.assert_list_tuple_equal_msg(
            expected=expected_value, actual=actual_value, msg_tag="test_msg"
        )


def test_assert_equal_dispatch_calls():
    """Test the calls to tubular.testing.helpers.assert_equal_dispatch."""

    expected_value = [1, 2, 3]
    actual_value = [1, 2, 3]
    msg_tag_value = "test_msg"

    with mock.patch(
        target="tubular.testing.helpers.assert_equal_dispatch"
    ) as mocked_method:

        h.assert_list_tuple_equal_msg(
            expected=expected_value, actual=actual_value, msg_tag=msg_tag_value
        )

        assert mocked_method.call_count == len(
            expected_value
        ), f"Unexpeted number of calls to tubular.testing.helpers.assert_equal_dispatch -\n  Expected: {len(expected_value)}\n  Actual: {mocked_method.call_count}"

        for i, (e, a) in enumerate(zip(expected_value, actual_value)):

            call_n_args = mocked_method.call_args_list[i]
            call_n_pos_args = call_n_args[0]
            call_n_kwargs = call_n_args[1]

            expected_pos_args = (e, a, f"{msg_tag_value} index {i}")

            assert (
                call_n_kwargs == {}
            ), f"Unexpected call keyword args in call {i} to tubular.testing.helpers.assert_equal_dispatch -\n  Expected: None\n  Actual: {call_n_kwargs}"

            assert len(call_n_pos_args) == len(
                expected_pos_args
            ), f"Difference in number of positional arguments in call {i} to tubular.testing.helpers.assert_equal_dispatch -\n  Expected: {len(expected_pos_args)}\n  Actual: {len(call_n_pos_args)}"

            for j, (e, a) in enumerate(zip(call_n_pos_args, expected_pos_args)):

                assert (
                    e == a
                ), f"Difference in positional args at index {j} in call {i} to tubular.testing.helpers.assert_equal_dispatch -\n Expected: {e}\n  Actual: {a}"
