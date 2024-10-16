import pandas as pd
import polars as pl
from narwhals.typing import FrameT
from pandas.testing import assert_frame_equal as assert_pandas_frame_equal
from polars.testing import assert_frame_equal as assert_polars_frame_equal


def assert_frame_equal_dispatch(df1: FrameT, df2: FrameT) -> None:
    """fixture to return correct pandas/polars assert_frame_equal method

    Parameters
    ----------
    df1 : pd.DataFrame
        first df for comparison

    df1 : pd.DataFrame
        second df for comparison

    """

    if isinstance(df1, pd.DataFrame) and isinstance(df2, pd.DataFrame):
        return assert_pandas_frame_equal(df1, df2)

    if isinstance(df1, pl.DataFrame) and isinstance(df2, pl.DataFrame):
        return assert_polars_frame_equal(df1, df2)

    invalid_request_error = "tubular is setup to handle only pandas or polars inputs, and dfs input to this function should be from same library"
    raise ValueError(invalid_request_error)
