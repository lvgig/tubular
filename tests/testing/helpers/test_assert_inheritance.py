import inspect
import pytest
import tubular.testing.helpers as h
from unittest import mock


def test_arguments():
    """Test arguments for arguments of tubular.testing.helpers.assert_inheritance."""

    expected_arguments = ["obj", "cls"]

    arg_spec = inspect.getfullargspec(h.assert_inheritance)

    arguments = arg_spec.args

    assert len(expected_arguments) == len(
        arguments
    ), f"Incorrect number of arguments -\n  Expected: {len(expected_arguments)}\n  Actual: {len(arguments)}"

    for i, (e, a) in enumerate(zip(expected_arguments, arguments)):

        assert e == a, f"Incorrect arg at index {i} -\n  Expected: {e}\n  Actual: {a}"


def test_check_is_class_call():
    """Test the call to tubular.testing.helpers.check_is_class."""

    with mock.patch("tubular.testing.helpers.check_is_class") as mocked_method:

        h.assert_inheritance(1, int)

        assert (
            mocked_method.call_count == 1
        ), f"Unexpected number of call to h.check_is_class -\n  Expected: 1\n  Actual: {mocked_method.call_count}"

        call_1_args = mocked_method.call_args_list[0]
        call_1_pos_args = call_1_args[0]
        call_1_kwargs = call_1_args[1]

        call_1_expected_kwargs = {}

        call_1_expected_pos_args = (int,)

        assert (
            call_1_expected_kwargs == call_1_kwargs
        ), f"Unexpected kwargs -\n  Expected: {call_1_expected_kwargs}\n  Actual: {call_1_kwargs}"

        assert len(call_1_expected_pos_args) == len(
            call_1_pos_args
        ), f"Unexpected number of positional args -\n  Expected: {len(call_1_expected_pos_args)}\n  Actual: {len(call_1_pos_args)}"

        assert (
            call_1_expected_pos_args[0] == call_1_pos_args[0]
        ), f"Unexpected number of positional arg -\n  Expected: {call_1_expected_pos_args[0]}\n  Actual: {call_1_pos_args[0]}"


def test_isinstance_call():
    """Test the call to isinstance."""

    with mock.patch("tubular.testing.helpers.isinstance") as mocked_method:

        h.assert_inheritance(1, int)

        assert (
            mocked_method.call_count == 1
        ), f"Unexpected number of call to h.check_is_class -\n  Expected: 1\n  Actual: {mocked_method.call_count}"

        call_1_args = mocked_method.call_args_list[0]
        call_1_pos_args = call_1_args[0]
        call_1_kwargs = call_1_args[1]

        call_1_expected_kwargs = {}

        call_1_expected_pos_args = (1, int)

        assert (
            call_1_expected_kwargs == call_1_kwargs
        ), f"Unexpected kwargs -\n  Expected: {call_1_expected_kwargs}\n  Actual: {call_1_kwargs}"

        assert len(call_1_expected_pos_args) == len(
            call_1_pos_args
        ), f"Unexpected number of positional -\n  Expected: {len(call_1_expected_pos_args)}\n  Actual: {len(call_1_pos_args)}"

        assert (
            call_1_expected_pos_args[0] == call_1_pos_args[0]
        ), f"Unexpected number of positional arg in index 0 -\n  Expected: {call_1_expected_pos_args[0]}\n  Actual: {call_1_pos_args[0]}"

        assert (
            call_1_expected_pos_args[1] == call_1_pos_args[1]
        ), f"Unexpected number of positional arg in index 1 -\n  Expected: {call_1_expected_pos_args[1]}\n  Actual: {call_1_pos_args[1]}"


def test_error():
    """Test an exception with the right info is raised if obj is not an instance of cls."""

    with pytest.raises(
        AssertionError,
        match=f"Incorrect inheritance - passed obj of class {(1).__class__.__name__} is not an instance of {float}",
    ):

        h.assert_inheritance(1, float)
