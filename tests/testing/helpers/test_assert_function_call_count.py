import pytest
import tubular
import tubular.testing.helpers as h
import tubular.testing.test_data as d


def test_arguments():
    """Test tubular.testing.helpers.assert_function_call_count has expected arguments."""

    # use of contextmanager decorator means we need to use .__wrapped__ to get back to original function
    h.test_function_arguments(
        func=h.assert_function_call_count.__wrapped__,
        expected_arguments=["mocker", "target", "attribute", "expected_n_calls"],
        expected_default_values=None,
    )


def test_mocker_arg_not_mocker_fixture_error():
    """Test an exception is raised if mocker is not pytest_mock.plugin.MockerFixture type."""

    with pytest.raises(
        TypeError, match="mocker should be the pytest_mock mocker fixture"
    ):

        df = d.create_df_1()

        x = tubular.base.BaseTransformer(columns="a")

        with h.assert_function_call_count(
            "aaaaaa", tubular.base.BaseTransformer, "columns_set_or_check", 1
        ):

            x.fit(X=df)


def test_mocker_patch_object_call(mocker):
    """Test the mocker.patch.object call."""

    mocked = mocker.spy(mocker.patch, "object")

    with h.assert_function_call_count(
        mocker,
        tubular.base.BaseTransformer,
        "__init__",
        1,
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


def test_successful_usage(mocker):
    """Test an example of successful run of h.assert_function_call_count."""

    df = d.create_df_1()

    x = tubular.base.BaseTransformer(columns="a")

    with h.assert_function_call_count(
        mocker, tubular.base.BaseTransformer, "columns_set_or_check", 1
    ):

        x.fit(X=df)


def test_exception_raised_more_calls_expected(mocker):
    """Test an exception is raised in the case more calls to a function are expected than happen."""

    with pytest.raises(
        AssertionError,
        match="incorrect number of calls to columns_set_or_check, expected 2 but got 1",
    ):

        df = d.create_df_1()

        x = tubular.base.BaseTransformer(columns="a")

        with h.assert_function_call_count(
            mocker, tubular.base.BaseTransformer, "columns_set_or_check", 2
        ):

            x.fit(X=df)


def test_exception_raised_more_calls_expected2(mocker):
    """Test an exception is raised in the case more calls to a function are expected than happen."""

    with pytest.raises(
        AssertionError,
        match="incorrect number of calls to __init__, expected 4 but got 0",
    ):

        df = d.create_df_1()

        x = tubular.base.BaseTransformer(columns="a")

        with h.assert_function_call_count(
            mocker, tubular.base.BaseTransformer, "__init__", 4
        ):

            x.fit(X=df)


def test_exception_raised_less_calls_expected(mocker):
    """Test an exception is raised in the case less calls to a function are expected than happen."""

    with pytest.raises(
        AssertionError,
        match="incorrect number of calls to columns_set_or_check, expected 1 but got 2",
    ):

        df = d.create_df_1()

        x = tubular.base.BaseTransformer(columns="a")

        with h.assert_function_call_count(
            mocker, tubular.base.BaseTransformer, "columns_set_or_check", 1
        ):

            x.fit(X=df)
            x.fit(X=df)


def test_exception_raised_less_calls_expected2(mocker):
    """Test an exception is raised in the case less calls to a function are expected than happen."""

    with pytest.raises(
        AssertionError,
        match="incorrect number of calls to columns_set_or_check, expected 0 but got 1",
    ):

        df = d.create_df_1()

        x = tubular.base.BaseTransformer(columns="a")

        with h.assert_function_call_count(
            mocker, tubular.base.BaseTransformer, "columns_set_or_check", 0
        ):

            x.fit(X=df)
