import inspect
import tubular.testing.helpers as h
import tubular
import pandas as pd
import numpy as np
from unittest import mock
from _pytest.mark.structures import ParameterSet


def test_arguments():
    """Test arguments for arguments of tubular.testing.helpers.index_preserved_params."""

    expected_arguments = ["df_1", "df_2", "seed"]

    arg_spec = inspect.getfullargspec(h.index_preserved_params)

    arguments = arg_spec.args

    assert len(expected_arguments) == len(
        arguments
    ), f"Incorrect number of arguments -\n  Expected: {len(expected_arguments)}\n  Actual: {len(arguments)}"

    for i, (e, a) in enumerate(zip(expected_arguments, arguments)):

        assert e == a, f"Incorrect arg at index {i} -\n  Expected: {e}\n  Actual: {a}"

    default_values = arg_spec.defaults

    assert default_values == (
        0,
    ), f"Unexpected default values -\n  Expected: {(0, )}\n  Actual: {default_values}"


def test__check_dfs_passed_call():
    """Test the call to _check_dfs_passed."""

    df1 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}, index=[7, 8, 9])
    df2 = pd.DataFrame({"a": [2, 3, 4], "b": [5, 6, 7]}, index=[7, 8, 9])

    with mock.patch.object(tubular.testing.helpers, "_check_dfs_passed") as mocked:

        h.index_preserved_params(df1, df2, seed=1)

    assert mocked.call_count == 1, "unexpected number of calls to _check_dfs_passed"

    call_args = mocked.call_args_list[0]

    assert call_args[1] == {}, "unexpected kwargs in _check_dfs_passed call"

    assert call_args[0] == (
        df1,
        df2,
    ), "unexpected positional args in _check_dfs_passed call"


def test_returned_object():
    """Test the function returns the expected output."""

    df1_1 = pd.DataFrame({"a": [1], "b": [4]}, index=[7])
    df1_2 = pd.DataFrame({"a": [2], "b": [5]}, index=[8])
    df1_3 = pd.DataFrame({"a": [3], "b": [6]}, index=[9])

    df2_1 = pd.DataFrame({"c": [10], "d": [13]}, index=[7])
    df2_2 = pd.DataFrame({"c": [11], "d": [14]}, index=[8])
    df2_3 = pd.DataFrame({"c": [12], "d": [15]}, index=[9])

    df1 = pd.concat([df1_1, df1_2, df1_3], axis=0)
    df2 = pd.concat([df2_1, df2_2, df2_3], axis=0)

    seed_value = 111

    np.random.seed(seed_value)
    random_index = np.random.randint(low=-99999999, high=100000000, size=df1.shape[0])
    start_decreasing_index = np.random.randint(low=-99999999, high=100000000, size=1)[0]
    decreasing_index = range(
        start_decreasing_index, start_decreasing_index - df1.shape[0], -1
    )
    start_increasing_index = np.random.randint(low=-99999999, high=100000000, size=1)[0]
    increasing_index = range(
        start_increasing_index, start_increasing_index + df1.shape[0], 1
    )

    df1_copy = df1.copy()
    df2_copy = df2.copy()

    df1_copy.index = random_index
    df2_copy.index = random_index

    expected_df_pairs = [(df1_copy, df2_copy)]

    df1_copy = df1.copy()
    df2_copy = df2.copy()

    df1_copy.index = decreasing_index
    df2_copy.index = decreasing_index

    expected_df_pairs.append((df1_copy, df2_copy))

    df1_copy = df1.copy()
    df2_copy = df2.copy()

    df1_copy.index = increasing_index
    df2_copy.index = increasing_index

    expected_df_pairs.append((df1_copy, df2_copy))

    expected_df_pairs.append((df1, df2))

    expected_ids = [
        "random index",
        "decreasing index",
        "increasing index",
        "original index",
    ]

    results = h.index_preserved_params(df1, df2, seed=seed_value)

    assert (
        type(results) is list
    ), "unexpected type for object returned from index_preserved_params"
    assert len(results) == len(
        expected_df_pairs
    ), "unexpected len of object returned from index_preserved_params"

    for i in range(len(expected_df_pairs)):

        assert (
            type(results[i]) is ParameterSet
        ), f"unexpected type for {i}th item in returned list"

        h.assert_equal_dispatch(
            expected_df_pairs[i],
            results[i].values,
            f"unexpected values for {i}th item in returned list",
        )

        assert (
            results[i].marks == ()
        ), f"unexpected marks for {i}th item in returned list"
        assert (
            results[i].id == expected_ids[i]
        ), f"unexpected id for {i}th item in returned list"
