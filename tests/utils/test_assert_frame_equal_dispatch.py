from copy import deepcopy

import pandas as pd
import polars as pl
import pytest

from tests.utils import assert_frame_equal_dispatch


@pytest.mark.parametrize("library", ["pandas", "polars"])
def test_successful_outcome(library):
    "test outcome of function"

    dataframe_init = None
    if library == "pandas":
        dataframe_init = pd.DataFrame

    elif library == "polars":
        dataframe_init = pl.DataFrame

    df1 = dataframe_init({"a": [1, 2, 3], "b": [4, 5, 6]})

    df2 = deepcopy(df1)

    assert_frame_equal_dispatch(df1, df2)


@pytest.mark.parametrize("library", ["pandas", "polars"])
def test_failed_outcome(library):
    "test outcome of function"

    dataframe_init = None
    failed_assert_error = None
    if library == "pandas":
        dataframe_init = pd.DataFrame
        failed_assert_error = "DataFrame.columns values are different"

    elif library == "polars":
        dataframe_init = pl.DataFrame
        failed_assert_error = "DataFrames are different"

    df1 = dataframe_init({"a": [1, 2, 3], "b": [4, 5, 6]})

    df2 = dataframe_init({"b": [1, 2, 3], "a": [4, 5, 6]})

    with pytest.raises(
        AssertionError,
        match=failed_assert_error,
    ):
        assert_frame_equal_dispatch(df1, df2)
