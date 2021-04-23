import inspect
import pytest
import tubular.testing.helpers as h
from unittest import mock


class DummyClass:
    """Simple class to use in testing getattr call in test_object_attributes."""

    def __init__(self):
        self.a = 1
        self.b = 2
        self.c = 3


def test_arguments():
    """Test arguments for arguments of tubular.testing.helpers.test_object_attributes."""

    expected_arguments = ["obj", "expected_attributes", "msg"]

    arg_spec = inspect.getfullargspec(h.test_object_attributes)

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


def test_n_getattr_calls():
    """Test the number of calls to getattr."""

    x = DummyClass()

    expected_attributes = {"a": 1, "b": 2, "c": 3}

    with mock.patch(target="tubular.testing.helpers.getattr") as mocked_method:

        # mock assert_equal_dispatch (called by test_object_attributes) so it does not error from
        # getattr not returning the right things - as it is mocked
        with mock.patch(target="tubular.testing.helpers.assert_equal_dispatch"):

            h.test_object_attributes(
                obj=x, expected_attributes=expected_attributes, msg="msg"
            )

            assert mocked_method.call_count == len(
                expected_attributes.keys()
            ), f"Unexpected number of calls to hasattr -\n  Expected: {len(expected_attributes.keys())}\n  Actual: {mocked_method.call_count}"


def test_getattr_calls():
    """Test the call arguments to getattr."""

    x = DummyClass()

    expected_attributes = {"a": 1, "b": 2, "c": 3}

    call_n = 0

    with mock.patch(target="tubular.testing.helpers.getattr") as mocked_method:

        # again mock assert_equal_dispatch (called by test_object_attributes) so it does not error from
        # getattr not returning the right things - as it is mocked
        with mock.patch(target="tubular.testing.helpers.assert_equal_dispatch"):

            h.test_object_attributes(
                obj=x, expected_attributes=expected_attributes, msg="msg"
            )

            call_n = 0

            for k, v in expected_attributes.items():

                call_n_args = mocked_method.call_args_list[call_n]
                call_n_pos_args = call_n_args[0]
                call_n_kwargs = call_n_args[1]

                expected_pos_args = (x, k)

                assert (
                    call_n_kwargs == {}
                ), f"Unexpected call keyword args in call {call_n} -\n  Expected: None\n  Actual: {call_n_kwargs}"

                assert len(call_n_pos_args) == len(
                    expected_pos_args
                ), f"Difference in number of positional arguments in call {call_n} -\n  Expected: {len(expected_pos_args)}\n  Actual: {len(call_n_pos_args)}"

                for i, (e, a) in enumerate(zip(expected_pos_args, call_n_pos_args)):

                    assert (
                        e == a
                    ), f"Difference in positional args at index {i} in call {call_n} -\n Expected: {e}\n  Actual: {a}"

                call_n += 1


def test_n_assert_equal_dispatch_calls():
    """Test the number of calls to tubular.testing.helpers.assert_equal_dispatch."""

    x = DummyClass()

    expected_attributes = {"a": 1, "b": 2, "c": 3}

    with mock.patch(
        target="tubular.testing.helpers.assert_equal_dispatch"
    ) as mocked_method:

        h.test_object_attributes(
            obj=x, expected_attributes=expected_attributes, msg="msg"
        )

        assert mocked_method.call_count == len(
            expected_attributes.keys()
        ), f"Unexpected number of calls to hasattr -\n  Expected: {len(expected_attributes.keys())}\n  Actual: {mocked_method.call_count}"


def test_assert_equal_dispatch_calls():
    """Test the call arguments to assert_equal_dispatch."""

    x = DummyClass()

    expected_attributes = {"a": 1, "b": 2, "c": 3}

    call_n = 0

    with mock.patch(
        target="tubular.testing.helpers.assert_equal_dispatch"
    ) as mocked_method:

        h.test_object_attributes(
            obj=x, expected_attributes=expected_attributes, msg="test_msg"
        )

        call_n = 0

        for k, v in expected_attributes.items():

            call_n_args = mocked_method.call_args_list[call_n]
            call_n_pos_args = call_n_args[0]
            call_n_kwargs = call_n_args[1]

            expected_pos_args = ()

            expected_kwargs = {
                "expected": v,
                "actual": getattr(x, k),
                "msg": f"{k} test_msg",
            }

            assert (
                call_n_pos_args == expected_pos_args
            ), f"Unexpected call positional args in call {call_n} -\n  Expected: None\n  Actual: {call_n_pos_args}"

            assert len(call_n_kwargs.keys()) == len(
                expected_kwargs.keys()
            ), f"Difference in number of positional arguments in call {call_n} -\n  Expected: {len(call_n_kwargs.keys())}\n  Actual: {len(expected_kwargs.keys())}"

            keys_diff_e_a = set(expected_kwargs.keys()) - set(call_n_kwargs.keys())

            keys_diff_a_e = set(call_n_kwargs.keys()) - set(expected_kwargs.keys())

            assert (
                keys_diff_e_a == set()
            ), f"Keys in expected not in actual: {keys_diff_e_a}\nKeys in actual not in expected: {keys_diff_a_e}"

            for k in expected_kwargs.keys():

                e = expected_kwargs[k]
                a = call_n_kwargs[k]

                assert (
                    e == a
                ), f"Difference in keyword arg {k} -\n Expected: {e}\n  Actual: {a}"

            call_n += 1


def test_expected_attributes_non_dict_error():
    """Test that a TypeError is raised if expected_attributes passed as not a dict."""

    x = DummyClass()

    with pytest.raises(TypeError):

        h.test_object_attributes(obj=x, expected_attributes=(1, 2), msg="test_msg")


def test_expected_attribute_missing_error():
    """Test that an AssertionError is raised if an excpeted attribute is missing from object."""

    x = DummyClass()

    expected_attributes = {"a": 1, "b": 2, "c": 3, "d": 4}

    with pytest.raises(AssertionError, match="obj has not attribute d"):

        h.test_object_attributes(
            obj=x, expected_attributes=expected_attributes, msg="test_msg"
        )


def test_expected_attribute_wrong_error():
    """Test that an AssertionError is raised if an excpeted attribute value is wrong."""

    x = DummyClass()

    expected_attributes = {"a": 1, "b": 2, "c": 10}

    with pytest.raises(AssertionError, match=f"""{'c'} {'test_msg'}"""):

        h.test_object_attributes(
            obj=x, expected_attributes=expected_attributes, msg="test_msg"
        )
