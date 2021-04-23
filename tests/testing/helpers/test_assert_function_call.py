import pytest
import tubular
import tubular.testing.helpers as h
import tubular.testing.test_data as d


def test_arguments():
    """Test tubular.testing.helpers.assert_function_call has expected arguments."""

    # use of contextmanager decorator means we need to use .__wrapped__ to get back to original function
    h.test_function_arguments(
        func=h.assert_function_call.__wrapped__,
        expected_arguments=["mocker", "target", "attribute", "expected_calls_args"],
        expected_default_values=None,
    )


def test_mocker_arg_not_mocker_fixture_error():
    """Test an exception is raised if mocker is not pytest_mock.plugin.MockerFixture type."""

    with pytest.raises(
        TypeError, match="mocker should be the pytest_mock mocker fixture"
    ):

        df = d.create_df_1()

        x = tubular.base.BaseTransformer(columns="a")

        with h.assert_function_call(
            "aaaaaa",
            tubular.base.BaseTransformer,
            "columns_set_or_check",
            {0: {"args": (), "kwargs": {}}},
        ):

            x.fit(X=df)


def test_expected_calls_args_checks(mocker):
    """Test that the checks on the expected_calls_args parameter raise exceptions."""

    with pytest.raises(TypeError, match="expected_calls_args should be a dict"):

        with h.assert_function_call(
            mocker, tubular.base.BaseTransformer, "__init__", ()
        ):

            tubular.base.BaseTransformer(columns="a")

    with pytest.raises(TypeError, match="expected_calls_args keys should be integers"):

        with h.assert_function_call(
            mocker,
            tubular.base.BaseTransformer,
            "__init__",
            {"a": {"args": (), "kwargs": {"columns": "a"}}},
        ):

            tubular.base.BaseTransformer(columns="a")

    with pytest.raises(
        ValueError,
        match="expected_calls_args keys should be integers greater than or equal to 0",
    ):

        with h.assert_function_call(
            mocker,
            tubular.base.BaseTransformer,
            "__init__",
            {-1: {"args": (), "kwargs": {"columns": "a"}}},
        ):

            tubular.base.BaseTransformer(columns="a")

    with pytest.raises(
        TypeError, match="each value in expected_calls_args should be a dict"
    ):

        with h.assert_function_call(
            mocker,
            tubular.base.BaseTransformer,
            "__init__",
            {0: ((), {"columns": "a"})},
        ):

            tubular.base.BaseTransformer(columns="a")

    with pytest.raises(
        ValueError,
        match="""keys of each sub dict in expected_calls_args should be 'args' and 'kwargs' only""",
    ):

        with h.assert_function_call(
            mocker,
            tubular.base.BaseTransformer,
            "__init__",
            {0: {"argz": (), "kwargs": {"columns": "a"}}},
        ):

            tubular.base.BaseTransformer(columns="a")

    with pytest.raises(TypeError, match="args in expected_calls_args should be tuples"):

        with h.assert_function_call(
            mocker,
            tubular.base.BaseTransformer,
            "__init__",
            {0: {"args": {}, "kwargs": {"columns": "a"}}},
        ):

            tubular.base.BaseTransformer(columns="a")

    with pytest.raises(
        TypeError, match="kwargs in expected_calls_args should be dicts"
    ):

        with h.assert_function_call(
            mocker,
            tubular.base.BaseTransformer,
            "__init__",
            {0: {"args": (), "kwargs": ["a"]}},
        ):

            tubular.base.BaseTransformer(columns="a")


def test_mocker_patch_object_call(mocker):
    """Test the mocker.patch.object call."""

    mocked = mocker.spy(mocker.patch, "object")

    expected_call_arguments = {
        0: {"args": ("a",), "kwargs": {"other": 1}},
    }

    with h.assert_function_call(
        mocker,
        tubular.base.BaseTransformer,
        "__init__",
        expected_call_arguments,
        return_value=None,
    ):

        tubular.imputers.BaseImputer("a", other=1)

    assert mocked.call_count == 1, "unexpected number of calls to mocker.patch.object"

    mocker_patch_object_call = mocked.call_args_list[0]
    call_pos_args = mocker_patch_object_call[0]
    call_kwargs = mocker_patch_object_call[1]

    assert call_pos_args == (
        tubular.base.BaseTransformer,
        "__init__",
    ), "unexpected positional args in mocker.patch.object call"

    assert call_kwargs == {
        "return_value": None
    }, "unexpected kwargs in mocker.patch.object call"


def test_successful_function_call(mocker):
    """Test a successful function call with the correct argumnets specified."""

    expected_call_arguments = {
        0: {"args": ("a",), "kwargs": {"other": 1}},
        2: {"args": (["a", "b"],), "kwargs": {}},
        3: {"args": (), "kwargs": {"columns": ["a", "b"]}},
    }

    with h.assert_function_call(
        mocker,
        tubular.base.BaseTransformer,
        "__init__",
        expected_call_arguments,
        return_value=None,
    ):

        tubular.imputers.BaseImputer("a", other=1)
        tubular.imputers.ArbitraryImputer(1, columns=["a"])
        tubular.imputers.BaseTransformer(["a", "b"])
        tubular.imputers.BaseImputer(columns=["a", "b"])


def test_not_enough_function_calls_exception(mocker):
    """Test an exception is raised if the mocked function is not called enough time for the number of expected_call_arguments items."""

    expected_call_arguments = {
        2: {"args": (["a", "b"],), "kwargs": {}},
        5: {"args": (), "kwargs": {"columns": ["a", "b"]}},
    }

    with pytest.raises(
        AssertionError,
        match="not enough calls to __init__, expected at least 6 but got 4",
    ):

        with h.assert_function_call(
            mocker,
            tubular.base.BaseTransformer,
            "__init__",
            expected_call_arguments,
            return_value=None,
        ):

            tubular.imputers.BaseImputer("a", other=1)
            tubular.imputers.ArbitraryImputer(1, columns=["a"])
            tubular.base.BaseTransformer(["a", "b"])
            tubular.imputers.BaseImputer(columns=["a", "b"])


@pytest.mark.parametrize(
    "expected_args",
    [
        {
            1: {"args": (1,), "kwargs": {}},
            3: {"args": (), "kwargs": {"columns": ["a", "b"]}},
        },
        {
            1: {"args": (), "kwargs": {"columns": ["a"]}},
            3: {"args": (), "kwargs": {"columns": ["a", "b"]}},
        },
        {
            1: {"args": (), "kwargs": {}},
            3: {"args": (1,), "kwargs": {"columns": ["a", "b"]}},
        },
        {
            1: {"args": (), "kwargs": {}},
            3: {"args": (1,), "kwargs": {"columns": ["a"]}},
        },
        {2: {"args": (), "kwargs": {}}},
        {0: {"args": ("a",), "kwargs": {"other": 2}}},
    ],
)
def test_incorrect_call_args_exception(mocker, expected_args):
    """Test an exception is raised if the mocked function is not called with expected arguments."""

    with pytest.raises(AssertionError):

        with h.assert_function_call(
            mocker,
            tubular.base.BaseTransformer,
            "__init__",
            expected_args,
            return_value=None,
        ):

            tubular.base.BaseTransformer("a", other=1)
            tubular.base.BaseTransformer()
            tubular.base.BaseTransformer(["a", "b"])
            tubular.base.BaseTransformer(columns=["a", "b"])


def test_assert_dict_equal_msg_call(mocker):
    """Test the calls to assert_dict_equal_msg."""

    # this is patched so it will not cause errors below when expected_args do not match
    mocked_dict_assert = mocker.patch.object(
        tubular.testing.helpers, "assert_dict_equal_msg"
    )

    expected_args = {
        0: {"args": ("a",), "kwargs": {"other": 1}},
        1: {"args": (), "kwargs": {}},
        2: {"args": (["a", "b"],), "kwargs": {}},
        3: {"args": (), "kwargs": {"columns": ["a", "c"]}},
    }

    with h.assert_function_call(
        mocker,
        tubular.base.BaseTransformer,
        "__init__",
        expected_args,
        return_value=None,
    ):

        tubular.base.BaseTransformer("a", other=1)
        tubular.base.BaseTransformer()
        tubular.base.BaseTransformer(["a", "b"])
        tubular.base.BaseTransformer(columns=["a", "b"])

    assert (
        mocked_dict_assert.call_count == 4
    ), "unexpected number of calls to tubular.testing.helpers.assert_dict_equal_msg"

    call_n_args = mocked_dict_assert.call_args_list[0]
    call_n_pos_args = call_n_args[0]
    call_n_kwargs = call_n_args[1]

    assert call_n_pos_args == (), "unexpected pos args in assert_dict_equal_msg call 1"

    assert call_n_kwargs == {
        "actual": {"other": 1},
        "expected": {"other": 1},
        "msg_tag": "kwargs for call 0 not correct",
    }, "unexpected kwargs in assert_dict_equal_msg call 1"

    call_n_args = mocked_dict_assert.call_args_list[1]
    call_n_pos_args = call_n_args[0]
    call_n_kwargs = call_n_args[1]

    assert call_n_pos_args == (), "unexpected pos args in assert_dict_equal_msg call 2"

    assert call_n_kwargs == {
        "actual": {},
        "expected": {},
        "msg_tag": "kwargs for call 1 not correct",
    }, "unexpected kwargs in assert_dict_equal_msg call 2"

    call_n_args = mocked_dict_assert.call_args_list[2]
    call_n_pos_args = call_n_args[0]
    call_n_kwargs = call_n_args[1]

    assert call_n_pos_args == (), "unexpected pos args in assert_dict_equal_msg call 3"

    assert call_n_kwargs == {
        "actual": {},
        "expected": {},
        "msg_tag": "kwargs for call 2 not correct",
    }, "unexpected kwargs in assert_dict_equal_msg call 3"

    call_n_args = mocked_dict_assert.call_args_list[3]
    call_n_pos_args = call_n_args[0]
    call_n_kwargs = call_n_args[1]

    assert call_n_pos_args == (), "unexpected pos args in assert_dict_equal_msg call 4"

    assert call_n_kwargs == {
        "actual": {"columns": ["a", "b"]},
        "expected": expected_args[3]["kwargs"],
        "msg_tag": "kwargs for call 3 not correct",
    }, "unexpected kwargs in assert_dict_equal_msg call 4"


def test_assert_list_tuple_equal_msg_call(mocker):
    """Test the calls to assert_list_tuple_equal_msg."""

    # this is patched so it will not cause errors below when expected_args do not match
    mocked_dict_assert = mocker.patch.object(
        tubular.testing.helpers, "assert_list_tuple_equal_msg"
    )

    expected_args = {
        0: {"args": ("a",), "kwargs": {"other": 1}},
        1: {"args": (), "kwargs": {}},
        2: {"args": (["a", "c"],), "kwargs": {}},
        3: {"args": (), "kwargs": {"columns": "a"}},
    }

    with h.assert_function_call(
        mocker,
        tubular.base.BaseTransformer,
        "__init__",
        expected_args,
        return_value=None,
    ):

        tubular.base.BaseTransformer("a", other=1)
        tubular.base.BaseTransformer()
        tubular.base.BaseTransformer(["a", "b"])
        tubular.base.BaseTransformer(columns="a")

    assert (
        mocked_dict_assert.call_count == 4
    ), "unexpected number of calls to tubular.testing.helpers.assert_list_tuple_equal_msg"

    call_n_args = mocked_dict_assert.call_args_list[0]
    call_n_pos_args = call_n_args[0]
    call_n_kwargs = call_n_args[1]

    assert (
        call_n_pos_args == ()
    ), "unexpected pos args in assert_list_tuple_equal_msg call 1"

    assert call_n_kwargs == {
        "actual": ("a",),
        "expected": ("a",),
        "msg_tag": "positional args for call 0 not correct",
    }, "unexpected pos args in assert_list_tuple_equal_msg call 1"

    call_n_args = mocked_dict_assert.call_args_list[1]
    call_n_pos_args = call_n_args[0]
    call_n_kwargs = call_n_args[1]

    assert (
        call_n_pos_args == ()
    ), "unexpected pos args in assert_list_tuple_equal_msg call 2"

    assert call_n_kwargs == {
        "actual": (),
        "expected": (),
        "msg_tag": "positional args for call 1 not correct",
    }, "unexpected pos args in assert_list_tuple_equal_msg call 2"

    call_n_args = mocked_dict_assert.call_args_list[2]
    call_n_pos_args = call_n_args[0]
    call_n_kwargs = call_n_args[1]

    assert (
        call_n_pos_args == ()
    ), "unexpected pos args in assert_list_tuple_equal_msg call 3"

    assert call_n_kwargs == {
        "actual": (["a", "b"],),
        "expected": expected_args[2]["args"],
        "msg_tag": "positional args for call 2 not correct",
    }, "unexpected pos args in assert_list_tuple_equal_msg call 3"

    call_n_args = mocked_dict_assert.call_args_list[3]
    call_n_pos_args = call_n_args[0]
    call_n_kwargs = call_n_args[1]

    assert (
        call_n_pos_args == ()
    ), "unexpected pos args in assert_list_tuple_equal_msg call 4"

    assert call_n_kwargs == {
        "actual": (),
        "expected": (),
        "msg_tag": "positional args for call 3 not correct",
    }, "unexpected pos args in assert_list_tuple_equal_msg call 4"
