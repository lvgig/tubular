import pytest
import tubular.testing.helpers as h
from unittest import mock


def test_inpsect_is_class_call():
    """Test the call to inspect.isclass."""

    with mock.patch("inspect.isclass") as mocked_method:

        h.check_is_class(int)

        assert (
            mocked_method.call_count == 1
        ), f"Unexpected number of call to inspect.isclass -\n  Expected: 1\n  Actual: {mocked_method.call_count}"

        call_1_args = mocked_method.call_args_list[0]
        call_1_pos_args = call_1_args[0]
        call_1_kwargs = call_1_args[1]

        h.assert_dict_equal_msg(
            actual=call_1_kwargs,
            expected={},
            msg_tag="Keyword arg assert for inspect.isclass",
        )

        assert (
            len(call_1_pos_args) == 1
        ), f"Incorrect number of positional arguments in inspect.isclass call -\n  Expected: 2\n  Actual: {len(call_1_pos_args)}"

        assert (
            call_1_pos_args[0] is int
        ), f"Incorrect first positional arg in inspect.isclass call -\n  Expected: {int}\n  Actual: {call_1_pos_args[0]}"


def test_non_class_error():
    """Test an exception is raised if check_is_class is called with non class object."""

    with pytest.raises(TypeError):

        h.check_is_class(1)
