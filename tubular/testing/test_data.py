"""This module contains functions that create simple datasets that are used in the tests."""

import datetime
import pandas as pd
import numpy as np
from pandas.api.types import CategoricalDtype
from sklearn.datasets import load_boston


def prepare_boston_df():

    # load dataset from sklean
    boston = load_boston()

    # add missings
    np.random.seed(555)
    missing_loc = np.random.randint(10, size=(boston["data"].shape))
    missing_loc[:, 2] = 1
    missing_loc[:, 3] = 1
    boston["data"][missing_loc == 0] = np.NaN

    # add columns and response
    boston_df = pd.DataFrame(boston["data"], columns=boston["feature_names"])
    boston_df["target"] = boston["target"]

    # add categorical variables (note cannot have nulls when converting to categorical when specifying levels)
    boston_df["ZN_cat"] = boston_df["ZN"]
    ZN_levels = boston_df["ZN_cat"].unique()
    ZN_levels.sort()
    ZN_levels = ZN_levels[np.logical_not(np.isnan(ZN_levels))]
    boston_df["ZN_cat"] = boston_df["ZN_cat"].astype(
        CategoricalDtype(categories=ZN_levels, ordered=True)
    )
    boston_df["ZN"] = boston_df["ZN"].astype(str)
    boston_df.loc[boston_df["ZN"] == "nan", "ZN"] = np.NaN
    boston_df["CHAS_cat"] = boston_df["CHAS"]
    boston_df["CHAS_cat"] = boston_df["CHAS_cat"].astype("category")
    boston_df["CHAS"] = boston_df["CHAS"].astype(str)
    boston_df.loc[boston_df["CHAS"] == "nan", "CHAS"] = np.NaN
    boston_df["RAD_cat"] = boston_df["RAD"]
    RAD_levels = boston_df["RAD_cat"].unique()
    RAD_levels.sort()
    RAD_levels = RAD_levels[np.logical_not(np.isnan(RAD_levels))]
    boston_df["RAD_cat"] = boston_df["RAD_cat"].astype(
        CategoricalDtype(categories=RAD_levels, ordered=True)
    )
    boston_df["RAD"] = boston_df["RAD"].astype(str)
    boston_df.loc[boston_df["RAD"] == "nan", "RAD"] = np.NaN

    return boston_df


def create_series_1(n=6):
    """Create simple series of [0:n-1]."""

    df = pd.Series(np.arange(n))

    return df


def create_1_int_column_df(n=6):
    """Create single column DataFrame of [0:n-1]."""

    df = pd.DataFrame({"a": np.arange(n)})

    return df


def create_zeros_array(shape=(10, 3)):
    """Create simple 2d numpy array of zeros of given shape."""

    arr = np.zeros(shape)

    return arr


def create_df_1():
    """Create simple DataFrame with the following...

    6 rows
    2 columns;
    - a integer 1:6
    - b object a:f
    no nulls
    """

    df = pd.DataFrame({"a": [1, 2, 3, 4, 5, 6], "b": ["a", "b", "c", "d", "e", "f"]})

    return df


def create_df_2():
    """Create simple DataFrame to use in other tests"""

    df = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5, 6, np.NaN],
            "b": ["a", "b", "c", "d", "e", "f", np.NaN],
            "c": ["a", "b", "c", "d", "e", "f", np.NaN],
        }
    )

    df["c"] = df["c"].astype("category")

    return df


def create_df_4():
    """Create simple DataFrame to use in other tests"""

    df = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5, 6, 7, np.NaN],
            "b": ["a", "b", "c", "d", "e", "f", "g", np.NaN],
            "c": ["a", "b", "c", "d", "e", "f", "g", np.NaN],
        }
    )

    df["c"] = df["c"].astype("category")

    return df


def create_df_3():
    """Create simple DataFrame to use in other tests"""

    df = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5, 6, np.NaN],
            "b": [1, 2, 3, np.NaN, 7, 8, 9],
            "c": [np.NaN, 1, 2, 3, -4, -5, -6],
        }
    )

    return df


def create_df_5():
    """Create simple DataFrame to use in other tests"""

    df = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5, 6, 7, 8, 9, np.NaN],
            "b": ["a", "a", "a", "d", "e", "f", "g", np.NaN, np.NaN, np.NaN],
            "c": ["a", "a", "c", "c", "e", "e", "f", "g", "h", np.NaN],
        }
    )

    df["c"] = df["c"].astype("category")

    return df


def create_df_6():
    """Nulls in different positions to check summing weights by col with nulls"""

    df = pd.DataFrame(
        {
            "a": [2, 2, 2, 2, np.NaN, 2, 2, 2, 3, 3],
            "b": ["a", "a", "a", "d", "e", "f", "g", np.NaN, np.NaN, np.NaN],
            "c": ["a", "b", "c", "d", "f", "f", "f", "g", "g", np.NaN],
        }
    )

    df["c"] = df["c"].astype("category")

    return df


def create_df_7():
    """Create simple DataFrame to use in other tests"""

    df = pd.DataFrame(
        {
            "a": [4, 2, 2, 1, 3],
            "b": ["x", "z", "y", "x", "x"],
            "c": ["c", "a", "a", "c", "b"],
        }
    )

    df["c"] = df["c"].astype("category")

    return df


def create_df_8():
    """Create simple DataFrame to use in other tests"""

    df = pd.DataFrame(
        {
            "a": [1, 5, 2, 3, 3],
            "b": ["w", "w", "z", "y", "x"],
            "c": ["a", "a", "c", "b", "a"],
        },
        index=[10, 15, 200, 251, 59],
    )

    df["c"] = df["c"].astype("category")

    return df


def create_df_9():
    """Create simple DataFrame to use in other tests"""

    df = pd.DataFrame(
        {
            "a": [1, 2, np.nan, 4, np.nan, 6],
            "b": [np.nan, 5, 4, 3, 2, 1],
            "c": [3, 2, 1, 4, 5, 6],
        }
    )

    return df


def create_df_10():
    """Create simple DataFrame to use in other tests"""

    df = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5, 6, 7, 8, 9, np.NaN],
            "b": ["a", "a", "a", "d", "e", "f", "g", np.NaN, np.NaN, np.NaN],
            "c": [1, 1, 3, 4, 5, 6, 5, 8, 9, 50],
        }
    )

    return df


def create_large_null_df(n_col=1000):
    """Create large single row df with all null values.

    Parameters
    ----------
    n_col : int
        Number of columns to add to output dataframe. Columns are named 'col_0', 'col_1', etc..

    Returns
    -------
    data_df : pd.DataFrame
        Single row dataframe with n_col columns; entirely populated with null values.

    """

    data_dict = {}

    for i in range(n_col):

        data_dict["col_" + str(i)] = np.NaN

    data_df = pd.DataFrame(data_dict, index=[0])

    return data_df


def create_large_half_null_df(n_col=1000):
    """Create large 2 row df with all null values in 1 row and all 1s in the other.

    Parameters
    ----------
    n_col : int
        Number of columns to add to output dataframe. Columns are named 'col_0', 'col_1', etc..

    Returns
    -------
    data_df : pd.DataFrame
        Two row dataframe with n_col columns; the first row taking entirely values of 1 and the
        second taking entirely null values.

    """

    data_dict = {}

    col_values = [1.0, np.NaN]

    for i in range(n_col):

        data_dict["col_" + str(i)] = col_values

    data_df = pd.DataFrame(data_dict, index=[0, 1])

    return data_df


def create_MeanResponseTransformer_test_df():
    """Create DataFrame to use MeanResponseTransformer tests that correct values are

    DataFrame column a is the response, the other columns are categorical columns
    of types; object, category, int, float, bool.

    """

    df = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5, 6],
            "b": ["a", "b", "c", "d", "e", "f"],
            "c": ["a", "b", "c", "d", "e", "f"],
            "d": [1, 2, 3, 4, 5, 6],
            "e": [1, 2, 3, 4, 5, 6.0],
            "f": [False, False, False, True, True, True],
        }
    )

    df["c"] = df["c"].astype("category")

    return df


def create_OrdinalEncoderTransformer_test_df():
    """Create DataFrame to use OrdinalEncoderTransformer tests that correct values are

    DataFrame column a is the response, the other columns are categorical columns
    of types; object, category, int, float, bool.

    """

    df = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5, 6],
            "b": ["a", "b", "c", "d", "e", "f"],
            "c": ["a", "b", "c", "d", "e", "f"],
            "d": [1, 2, 3, 4, 5, 6],
            "e": [3, 4, 5, 6, 7, 8.0],
            "f": [False, False, False, True, True, True],
        }
    )

    df["c"] = df["c"].astype("category")

    return df


def create_NearestMeanResponseImputer_test_df():
    """Create DataFrame to use in NearestMeanResponseImputer tests.

    DataFrame column c is the response, the other columns are numerical columns containing null entries.

    """

    df = pd.DataFrame(
        {
            "a": [1, 1, 2, 3, 3, np.nan],
            "b": [np.nan, np.nan, 1, 3, 3, 4],
            "c": [2, 3, 2, 1, 4, 1],
        }
    )

    return df


def create_values_map(df):

    value_map = {}

    for i in df.columns:

        value_map[i] = {}

        for j in df[i].unique():

            value_map[i][j] = j

    return value_map


def create_date_test_df():
    """Create DataFrame for DateDiffLeapYearTransformer tests."""

    df = pd.DataFrame(
        {
            "a": [
                datetime.date(1993, 9, 27),
                datetime.date(2000, 3, 19),
                datetime.date(2018, 11, 10),
                datetime.date(2018, 10, 10),
                datetime.date(2018, 10, 10),
                datetime.date(2018, 10, 10),
                datetime.date(2018, 12, 10),
                datetime.date(1985, 7, 23),
            ],
            "b": [
                datetime.date(2020, 5, 1),
                datetime.date(2019, 12, 25),
                datetime.date(2018, 11, 10),
                datetime.date(2018, 11, 10),
                datetime.date(2018, 9, 10),
                datetime.date(2015, 11, 10),
                datetime.date(2015, 11, 10),
                datetime.date(2015, 7, 23),
            ],
        }
    )

    return df


def create_date_test_nulls_df():
    """Create DataFrame with nulls only for DateDiffLeapYearTransformer, DateDifferenceTransformer tests."""

    df = pd.DataFrame(
        {
            "a": [
                np.NaN,
            ],
            "b": [
                np.NaN,
            ],
        },
        index=[0],
    )

    return df


def create_datediff_test_df():
    """Create DataFrame for DateDifferenceTransformer tests."""

    df = pd.DataFrame(
        {
            "a": [
                datetime.datetime(1993, 9, 27, 11, 58, 58),
                datetime.datetime(2000, 3, 19, 12, 59, 59),
                datetime.datetime(2018, 11, 10, 11, 59, 59),
                datetime.datetime(2018, 10, 10, 11, 59, 59),
                datetime.datetime(2018, 10, 10, 11, 59, 59),
                datetime.datetime(2018, 10, 10, 10, 59, 59),
                datetime.datetime(2018, 12, 10, 11, 59, 59),
                datetime.datetime(1985, 7, 23, 11, 59, 59),
            ],
            "b": [
                datetime.datetime(2020, 5, 1, 12, 59, 59),
                datetime.datetime(2019, 12, 25, 11, 58, 58),
                datetime.datetime(2018, 11, 10, 11, 59, 59),
                datetime.datetime(2018, 11, 10, 11, 59, 59),
                datetime.datetime(2018, 9, 10, 9, 59, 59),
                datetime.datetime(2015, 11, 10, 11, 59, 59),
                datetime.datetime(2015, 11, 10, 12, 59, 59),
                datetime.datetime(2015, 7, 23, 11, 59, 59),
            ],
        }
    )

    return df


def create_datediff_test_nulls_df():
    """Create DataFrame with nulls only for DateDifferenceTransformer tests."""

    df = pd.DataFrame(
        {
            "a": [
                datetime.datetime(1993, 9, 27, 11, 58, 58),
                np.NaN,
            ],
            "b": [
                np.NaN,
                datetime.datetime(2019, 12, 25, 11, 58, 58),
            ],
        },
        index=[0, 1],
    )

    return df


def create_to_datetime_test_df():
    """Create DataFrame to be used in the ToDatetimeTransformer tests."""

    df = pd.DataFrame(
        {"a": [1950, 1960, 2000, 2001, np.NaN, 2010], "b": [1, 2, 3, 4, 5, np.NaN]}
    )

    return df


def create_is_between_dates_df_1():
    """Create df to use in IsBetweenDates tests. Contains 3 columns of 2 datatime values."""

    df = pd.DataFrame(
        {
            "a": pd.date_range(start="1/1/2016", end="27/02/2017", periods=2),
            "b": pd.date_range(start="1/2/2016", end="27/09/2017", periods=2),
            "c": pd.date_range(start="1/3/2016", end="27/04/2017", periods=2),
        }
    )

    return df


def create_is_between_dates_df_2():
    """Create df to use in IsBetweenDates tests. Contains 3 columns of 5 datatime values, covers edge cases."""

    df = pd.DataFrame(
        {
            "a": [
                datetime.datetime(1990, 2, 1),
                datetime.datetime(1990, 2, 1),
                datetime.datetime(1990, 2, 1),
                datetime.datetime(1990, 2, 1),
                datetime.datetime(1990, 2, 1),
                datetime.datetime(1990, 2, 1),
            ],
            "b": [
                datetime.datetime(1990, 1, 20),
                datetime.datetime(1990, 2, 1),
                datetime.datetime(1990, 2, 2),
                datetime.datetime(1990, 2, 6),
                datetime.datetime(1990, 3, 1),
                datetime.datetime(1990, 3, 2),
            ],
            "c": [
                datetime.datetime(1990, 3, 1),
                datetime.datetime(1990, 3, 1),
                datetime.datetime(1990, 3, 1),
                datetime.datetime(1990, 3, 1),
                datetime.datetime(1990, 3, 1),
                datetime.datetime(1990, 3, 1),
            ],
        }
    )

    return df
