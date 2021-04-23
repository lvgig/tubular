import pytest
import inspect
import tubular.testing.helpers as h


def test_arguments():
    """Test arguments for arguments of tubular.testing.helpers.assert_equal_msg."""

    expected_arguments = ["actual", "expected", "msg_tag"]

    arg_spec = inspect.getfullargspec(h.assert_equal_msg)

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


def test_error_raised_unequal():
    """Test an assertion error is raised if values are not equal."""

    expected = 1
    actual = 2
    msg_tag = "bbb"

    with pytest.raises(
        AssertionError, match=f"{msg_tag} -\n  Expected: {expected}\n  Actual: {actual}"
    ):

        h.assert_equal_msg(expected=expected, actual=actual, msg_tag=msg_tag)
