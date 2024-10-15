import pandas as pd
import polars as pl
from narwhals.typing import FrameT
from pandas.testing import assert_frame_equal as assert_pandas_frame_equal
from polars.testing import assert_frame_equal as assert_polars_frame_equal


def get_assert_frame_equal(df: FrameT):
    """fixture to return correct pandas/polars assert_frame_equal method

    Parameters
    ----------
    df : pd.DataFrame
        A dataframe with required columns to be capped.

    Returns
    ----------
    Callable: pandas or polars assert_frame_equal function
    """

    if isinstance(df, pd.DataFrame):
        return assert_pandas_frame_equal

    if isinstance(df, pl.DataFrame):
        return assert_polars_frame_equal

    invalid_request_error = "tubular is setup to handle only pandas or polars inputs"
    raise ValueError(invalid_request_error)
