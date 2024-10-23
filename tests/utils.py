import numpy as np
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


def convert_values(values, library):
    if library == "pandas":
        return [np.nan if v is None else v for v in values]
    return values


def dataframe_init_dispatch(
    dataframe_dict: dict,
    library: str,
) -> pl.DataFrame | pd.DataFrame:
    """
    Initialize a DataFrame using either Pandas or Polars library based on the specified library name.

    Parameters:
    dataframe_dict (dict): A dictionary where keys are column names and values are lists of column data.
    library (str): The name of the library to use for DataFrame creation. Should be either "pandas" or "polars".

    Returns:
    pl.DataFrame | pd.DataFrame: A DataFrame object from the specified library.

    Raises:
    ValueError: If the `library` parameter is not "pandas" or "polars".
    """
    if library not in ["pandas", "polars"]:
        library_error_message = (
            "The library parameter should be either 'pandas' or 'polars'."
        )
        raise ValueError(library_error_message)

    converted_dict = {k: convert_values(v, library) for k, v in dataframe_dict.items()}

    if library == "pandas":
        return pd.DataFrame(converted_dict)
    return pl.DataFrame(converted_dict, strict=False)
