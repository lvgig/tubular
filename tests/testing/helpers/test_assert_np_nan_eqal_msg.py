import pytest
import inspect
import tubular.testing.helpers as h

import numpy as np


def test_arguments():
    """Test arguments for arguments of tubular.testing.helpers.assert_np_nan_eqal_msg."""

    expected_arguments = ["actual", "expected", "msg"]

    arg_spec = inspect.getfullargspec(h.assert_np_nan_eqal_msg)

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


@pytest.mark.parametrize("expected, actual", [(1, np.NaN), (np.NaN, 1), (1, 1)])
def test_error_raised_unequal(expected, actual):
    """Test an assertion error is raised if both values are not None."""

    msg_tag = "bbb"

    with pytest.raises(
        AssertionError,
        match=f"Both values are not equal to np.NaN -\n  Expected: {expected}\n  Actual: {actual}",
    ):

        h.assert_np_nan_eqal_msg(expected=expected, actual=actual, msg=msg_tag)
