"""This module contains functions that create simple datasets that are used in the tests."""

import datetime

import numpy as np
import pandas as pd


def create_series_1(n=6):
    """Create simple series of [0:n-1]."""
    return pd.Series(np.arange(n))


def create_1_int_column_df(n=6):
    """Create single column DataFrame of [0:n-1]."""
    return pd.DataFrame({"a": np.arange(n)})


def create_zeros_array(shape=(10, 3)):
    """Create simple 2d numpy array of zeros of given shape."""
    return np.zeros(shape)


def create_numeric_df_1():
    """Example with numeric dataframe."""
    return pd.DataFrame(
        {
            "a": [34.48, 21.71, 32.83, 1.08, 32.93, 4.74, 2.76, 75.7, 14.08, 61.31],
            "b": [12.03, 20.32, 24.12, 24.18, 68.99, 0.0, 0.0, 59.46, 11.02, 60.68],
            "c": [17.06, 12.25, 19.15, 29.73, 1.98, 8.23, 15.22, 20.59, 3.82, 39.73],
            "d": [25.94, 70.22, 72.94, 64.55, 0.41, 13.62, 30.22, 4.6, 67.13, 10.38],
            "e": [94.3, 4.18, 51.7, 16.63, 2.6, 16.57, 3.51, 30.79, 66.19, 25.44],
        },
    )


def create_df_1():
    """Create simple DataFrame with the following...

    6 rows
    2 columns;
    - a integer 1:6
    - b object a:f
    no nulls
    """
    return pd.DataFrame({"a": [1, 2, 3, 4, 5, 6], "b": ["a", "b", "c", "d", "e", "f"]})


def create_df_2():
    """Create simple DataFrame to use in other tests."""
    df = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5, 6, np.NaN],
            "b": ["a", "b", "c", "d", "e", "f", np.NaN],
            "c": ["a", "b", "c", "d", "e", "f", np.NaN],
        },
    )

    df["c"] = df["c"].astype("category")

    return df


def create_df_4():
    """Create simple DataFrame to use in other tests."""
    df = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5, 6, 7, np.NaN],
            "b": ["a", "b", "c", "d", "e", "f", "g", np.NaN],
            "c": ["a", "b", "c", "d", "e", "f", "g", np.NaN],
        },
    )

    df["c"] = df["c"].astype("category")

    return df


def create_df_3():
    """Create simple DataFrame to use in other tests."""
    return pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5, 6, np.NaN],
            "b": [1, 2, 3, np.NaN, 7, 8, 9],
            "c": [np.NaN, 1, 2, 3, -4, -5, -6],
        },
    )


def create_df_5():
    """Create simple DataFrame to use in other tests."""
    df = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5, 6, 7, 8, 9, np.NaN],
            "b": ["a", "a", "a", "d", "e", "f", "g", np.NaN, np.NaN, np.NaN],
            "c": ["a", "a", "c", "c", "e", "e", "f", "g", "h", np.NaN],
        },
    )

    df["c"] = df["c"].astype("category")

    return df


def create_df_6():
    """Nulls in different positions to check summing weights by col with nulls."""
    df = pd.DataFrame(
        {
            "a": [2, 2, 2, 2, np.NaN, 2, 2, 2, 3, 3],
            "b": ["a", "a", "a", "d", "e", "f", "g", np.NaN, np.NaN, np.NaN],
            "c": ["a", "b", "c", "d", "f", "f", "f", "g", "g", np.NaN],
        },
    )

    df["c"] = df["c"].astype("category")

    return df


def create_df_7():
    """Create simple DataFrame to use in other tests."""
    df = pd.DataFrame(
        {
            "a": [4, 2, 2, 1, 3],
            "b": ["x", "z", "y", "x", "x"],
            "c": ["c", "a", "a", "c", "b"],
        },
    )

    df["c"] = df["c"].astype("category")

    return df


def create_df_8():
    """Create simple DataFrame to use in other tests."""
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
    """Create simple DataFrame to use in other tests."""
    return pd.DataFrame(
        {
            "a": [1, 2, np.nan, 4, np.nan, 6],
            "b": [np.nan, 5, 4, 3, 2, 1],
            "c": [3, 2, 1, 4, 5, 6],
        },
    )


def create_df_10():
    """Create simple DataFrame to use in other tests."""
    return pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5, 6, 7, 8, 9, np.NaN],
            "b": ["a", "a", "a", "d", "e", "f", "g", np.NaN, np.NaN, np.NaN],
            "c": [1, 1, 3, 4, 5, 6, 5, 8, 9, 50],
        },
    )


def create_df_11():
    """Create simple DataFrame to use in other tests."""
    return pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [4, 5, 6],
        },
    )


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

    return pd.DataFrame(data_dict, index=[0])


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

    return pd.DataFrame(data_dict, index=[0, 1])


def create_weighted_imputers_test_df():
    """Create DataFrame to use imputer tests that correct values are imputed for weighted dataframes.

    weight column contains the weights between 0 and 1
    """
    return pd.DataFrame(
        {
            "a": [1.0, 1.0, 1.0, 3.0, 5.0, 5.0],
            "b": ["a", "a", "a", "d", "e", "e"],
            "c": ["a", "a", np.nan, np.nan, np.nan, "f"],
            "d": [1.0, 5.0, 3.0, np.nan, np.nan, 1.0],
            "response": [0, 1, 0, 1, 1, 1],
            "weight": [0.1, 0.1, 0.8, 0.5, 0.9, 0.8],
        },
    )


def create_MeanResponseTransformer_test_df():
    """Create DataFrame to use MeanResponseTransformer tests that correct values are.

    DataFrame column a is the response, the other columns are categorical columns
    of types; object, category, int, float, bool.

    """
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "b": ["a", "b", "c", "d", "e", "f"],
            "c": ["a", "b", "c", "d", "e", "f"],
            "d": [1, 2, 3, 4, 5, 6],
            "e": [1, 2, 3, 4, 5, 6.0],
            "f": [False, False, False, True, True, True],
            "multi_level_response": [
                "blue",
                "blue",
                "yellow",
                "yellow",
                "green",
                "green",
            ],
        },
    )

    df["c"] = df["c"].astype("category")

    return df


def create_MeanResponseTransformer_test_df_unseen_levels():
    """Create DataFrame to use in MeanResponseTransformer tests that check correct values are
    generated when using transform method on data with unseen levels.
    DataFrame column a is the response, the other columns are categorical columns
    of types; object, category, int, float, bool.

    """
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 3.0],
            "b": ["a", "b", "c", "d", "e", "f", "g", "h"],
            "c": ["a", "b", "c", "d", "e", "f", "g", "h"],
            "d": [1, 2, 3, 4, 5, 6, 7, 8],
            "e": [1, 2, 3, 4, 5, 6.0, 7, 8],
            "f": [False, False, False, True, True, True, True, False],
            "multi_level_response": [
                "blue",
                "blue",
                "yellow",
                "yellow",
                "green",
                "green",
                "yellow",
                "blue",
            ],
        },
    )

    df["c"] = df["c"].astype("category")

    return df


def create_OrdinalEncoderTransformer_test_df():
    """Create DataFrame to use OrdinalEncoderTransformer tests that correct values are.

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
        },
    )

    df["c"] = df["c"].astype("category")

    return df


def create_NearestMeanResponseImputer_test_df():
    """Create DataFrame to use in NearestMeanResponseImputer tests.

    DataFrame column c is the response, the other columns are numerical columns containing null entries.

    """
    return pd.DataFrame(
        {
            "a": [1, 1, 2, 3, 3, np.nan],
            "b": [np.nan, np.nan, 1, 3, 3, 4],
            "c": [2, 3, 2, 1, 4, 1],
        },
    )


def create_values_map(df):
    value_map = {}

    for i in df.columns:
        value_map[i] = {}

        for j in df[i].unique():
            value_map[i][j] = j

    return value_map


def create_date_test_df():
    """Create DataFrame for DateDiffLeapYearTransformer tests."""
    return pd.DataFrame(
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
        },
    )


def create_date_test_nulls_df():
    """Create DataFrame with nulls only for DateDiffLeapYearTransformer, DateDifferenceTransformer tests."""
    return pd.DataFrame(
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


def create_datediff_test_df():
    """Create DataFrame for DateDifferenceTransformer tests."""
    return pd.DataFrame(
        {
            "a": [
                datetime.datetime(
                    1993,
                    9,
                    27,
                    11,
                    58,
                    58,
                    tzinfo=datetime.timezone.utc,
                ),
                datetime.datetime(
                    2000,
                    3,
                    19,
                    12,
                    59,
                    59,
                    tzinfo=datetime.timezone.utc,
                ),
                datetime.datetime(
                    2018,
                    11,
                    10,
                    11,
                    59,
                    59,
                    tzinfo=datetime.timezone.utc,
                ),
                datetime.datetime(
                    2018,
                    10,
                    10,
                    11,
                    59,
                    59,
                    tzinfo=datetime.timezone.utc,
                ),
                datetime.datetime(
                    2018,
                    10,
                    10,
                    11,
                    59,
                    59,
                    tzinfo=datetime.timezone.utc,
                ),
                datetime.datetime(
                    2018,
                    10,
                    10,
                    10,
                    59,
                    59,
                    tzinfo=datetime.timezone.utc,
                ),
                datetime.datetime(
                    2018,
                    12,
                    10,
                    11,
                    59,
                    59,
                    tzinfo=datetime.timezone.utc,
                ),
                datetime.datetime(
                    1985,
                    7,
                    23,
                    11,
                    59,
                    59,
                    tzinfo=datetime.timezone.utc,
                ),
            ],
            "b": [
                datetime.datetime(2020, 5, 1, 12, 59, 59, tzinfo=datetime.timezone.utc),
                datetime.datetime(
                    2019,
                    12,
                    25,
                    11,
                    58,
                    58,
                    tzinfo=datetime.timezone.utc,
                ),
                datetime.datetime(
                    2018,
                    11,
                    10,
                    11,
                    59,
                    59,
                    tzinfo=datetime.timezone.utc,
                ),
                datetime.datetime(
                    2018,
                    11,
                    10,
                    11,
                    59,
                    59,
                    tzinfo=datetime.timezone.utc,
                ),
                datetime.datetime(2018, 9, 10, 9, 59, 59, tzinfo=datetime.timezone.utc),
                datetime.datetime(
                    2015,
                    11,
                    10,
                    11,
                    59,
                    59,
                    tzinfo=datetime.timezone.utc,
                ),
                datetime.datetime(
                    2015,
                    11,
                    10,
                    12,
                    59,
                    59,
                    tzinfo=datetime.timezone.utc,
                ),
                datetime.datetime(
                    2015,
                    7,
                    23,
                    11,
                    59,
                    59,
                    tzinfo=datetime.timezone.utc,
                ),
            ],
        },
    )


def create_datediff_test_nulls_df():
    """Create DataFrame with nulls only for DateDifferenceTransformer tests."""
    return pd.DataFrame(
        {
            "a": [
                datetime.datetime(
                    1993,
                    9,
                    27,
                    11,
                    58,
                    58,
                    tzinfo=datetime.timezone.utc,
                ),
                np.NaN,
            ],
            "b": [
                np.NaN,
                datetime.datetime(
                    2019,
                    12,
                    25,
                    11,
                    58,
                    58,
                    tzinfo=datetime.timezone.utc,
                ),
            ],
        },
        index=[0, 1],
    )


def create_to_datetime_test_df():
    """Create DataFrame to be used in the ToDatetimeTransformer tests."""
    return pd.DataFrame(
        {"a": [1950, 1960, 2000, 2001, np.NaN, 2010], "b": [1, 2, 3, 4, 5, np.NaN]},
    )


def create_is_between_dates_df_1():
    """Create df to use in IsBetweenDates tests. Contains 3 columns of 2 datatime values."""
    return pd.DataFrame(
        {
            "a": pd.date_range(start="1/1/2016", end="27/02/2017", periods=2),
            "b": pd.date_range(start="1/2/2016", end="27/09/2017", periods=2),
            "c": pd.date_range(start="1/3/2016", end="27/04/2017", periods=2),
        },
    )


def create_is_between_dates_df_2():
    """Create df to use in IsBetweenDates tests. Contains 3 columns of 5 datatime values, covers edge cases."""
    return pd.DataFrame(
        {
            "a": [
                datetime.datetime(1990, 2, 1, tzinfo=datetime.timezone.utc),
                datetime.datetime(1990, 2, 1, tzinfo=datetime.timezone.utc),
                datetime.datetime(1990, 2, 1, tzinfo=datetime.timezone.utc),
                datetime.datetime(1990, 2, 1, tzinfo=datetime.timezone.utc),
                datetime.datetime(1990, 2, 1, tzinfo=datetime.timezone.utc),
                datetime.datetime(1990, 2, 1, tzinfo=datetime.timezone.utc),
            ],
            "b": [
                datetime.datetime(1990, 1, 20, tzinfo=datetime.timezone.utc),
                datetime.datetime(1990, 2, 1, tzinfo=datetime.timezone.utc),
                datetime.datetime(1990, 2, 2, tzinfo=datetime.timezone.utc),
                datetime.datetime(1990, 2, 6, tzinfo=datetime.timezone.utc),
                datetime.datetime(1990, 3, 1, tzinfo=datetime.timezone.utc),
                datetime.datetime(1990, 3, 2, tzinfo=datetime.timezone.utc),
            ],
            "c": [
                datetime.datetime(1990, 3, 1, tzinfo=datetime.timezone.utc),
                datetime.datetime(1990, 3, 1, tzinfo=datetime.timezone.utc),
                datetime.datetime(1990, 3, 1, tzinfo=datetime.timezone.utc),
                datetime.datetime(1990, 3, 1, tzinfo=datetime.timezone.utc),
                datetime.datetime(1990, 3, 1, tzinfo=datetime.timezone.utc),
                datetime.datetime(1990, 3, 1, tzinfo=datetime.timezone.utc),
            ],
        },
    )


def create_is_between_dates_df_3():
    """Create df to use in IsBetweenDates tests. Contains 3 columns of 5 datatime values, covers edge cases, has mixed date data types."""
    return pd.DataFrame(
        {
            "a_date": [
                datetime.datetime(1990, 2, 1, tzinfo=datetime.timezone.utc),
                datetime.datetime(1990, 2, 1, tzinfo=datetime.timezone.utc),
                datetime.datetime(1990, 2, 1, tzinfo=datetime.timezone.utc),
                datetime.datetime(1990, 2, 1, tzinfo=datetime.timezone.utc),
                datetime.datetime(1990, 2, 1, tzinfo=datetime.timezone.utc),
                datetime.datetime(1990, 2, 1, tzinfo=datetime.timezone.utc),
            ],
            "b_date": [
                datetime.datetime(1990, 1, 20, tzinfo=datetime.timezone.utc),
                datetime.datetime(1990, 2, 1, tzinfo=datetime.timezone.utc),
                datetime.datetime(1990, 2, 2, tzinfo=datetime.timezone.utc),
                datetime.datetime(1990, 2, 6, tzinfo=datetime.timezone.utc),
                datetime.datetime(1990, 3, 1, tzinfo=datetime.timezone.utc),
                datetime.datetime(1990, 3, 2, tzinfo=datetime.timezone.utc),
            ],
            "c_date": [
                datetime.datetime(1990, 3, 1, tzinfo=datetime.timezone.utc),
                datetime.datetime(1990, 3, 1, tzinfo=datetime.timezone.utc),
                datetime.datetime(1990, 3, 1, tzinfo=datetime.timezone.utc),
                datetime.datetime(1990, 3, 1, tzinfo=datetime.timezone.utc),
                datetime.datetime(1990, 3, 1, tzinfo=datetime.timezone.utc),
                datetime.datetime(1990, 3, 1, tzinfo=datetime.timezone.utc),
            ],
            "a_datetime": [
                datetime.datetime(1990, 2, 1, tzinfo=datetime.timezone.utc),
                datetime.datetime(1990, 2, 1, tzinfo=datetime.timezone.utc),
                datetime.datetime(1990, 2, 1, tzinfo=datetime.timezone.utc),
                datetime.datetime(1990, 2, 1, tzinfo=datetime.timezone.utc),
                datetime.datetime(1990, 2, 1, tzinfo=datetime.timezone.utc),
                datetime.datetime(1990, 2, 1, tzinfo=datetime.timezone.utc),
            ],
            "b_datetime": [
                datetime.datetime(1990, 1, 20, tzinfo=datetime.timezone.utc),
                datetime.datetime(1990, 2, 1, tzinfo=datetime.timezone.utc),
                datetime.datetime(1990, 2, 2, tzinfo=datetime.timezone.utc),
                datetime.datetime(1990, 2, 6, tzinfo=datetime.timezone.utc),
                datetime.datetime(1990, 3, 1, tzinfo=datetime.timezone.utc),
                datetime.datetime(1990, 3, 2, tzinfo=datetime.timezone.utc),
            ],
            "c_datetime": [
                datetime.datetime(1990, 3, 1, tzinfo=datetime.timezone.utc),
                datetime.datetime(1990, 3, 1, tzinfo=datetime.timezone.utc),
                datetime.datetime(1990, 3, 1, tzinfo=datetime.timezone.utc),
                datetime.datetime(1990, 3, 1, tzinfo=datetime.timezone.utc),
                datetime.datetime(1990, 3, 1, tzinfo=datetime.timezone.utc),
                datetime.datetime(1990, 3, 1, tzinfo=datetime.timezone.utc),
            ],
        },
    )


# Example DataFrame for downcasting dtypes tests
def create_downcast_df():
    """Create a dataframe with mixed dtypes to use in downcasting tests."""
    return pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "b": [1.0, 2.0, 3.0, 4.0, 5.0],
        },
    )


def create_date_diff_different_dtypes():
    """Dataframe with different datetime formats"""
    return pd.DataFrame(
        {
            "date_col_1": [
                datetime.date(1993, 9, 27),
                datetime.date(2000, 3, 19),
                datetime.date(2018, 11, 10),
                datetime.date(2018, 10, 10),
                datetime.date(2018, 10, 10),
                datetime.date(2018, 10, 10),
                datetime.date(2018, 12, 10),
                datetime.date(
                    1985,
                    7,
                    23,
                ),
            ],
            "date_col_2": [
                datetime.date(2020, 5, 1),
                datetime.date(2019, 12, 25),
                datetime.date(2018, 11, 10),
                datetime.date(2018, 11, 10),
                datetime.date(2018, 9, 10),
                datetime.date(2015, 11, 10),
                datetime.date(2015, 11, 10),
                datetime.date(2015, 7, 23),
            ],
            "datetime_col_1": [
                datetime.datetime(1993, 9, 27, tzinfo=datetime.timezone.utc),
                datetime.datetime(2000, 3, 19, tzinfo=datetime.timezone.utc),
                datetime.datetime(2018, 11, 10, tzinfo=datetime.timezone.utc),
                datetime.datetime(2018, 10, 10, tzinfo=datetime.timezone.utc),
                datetime.datetime(2018, 10, 10, tzinfo=datetime.timezone.utc),
                datetime.datetime(2018, 10, 10, tzinfo=datetime.timezone.utc),
                datetime.datetime(2018, 12, 10, tzinfo=datetime.timezone.utc),
                datetime.datetime(
                    1985,
                    7,
                    23,
                    tzinfo=datetime.timezone.utc,
                ),
            ],
            "datetime_col_2": [
                datetime.datetime(2020, 5, 1, tzinfo=datetime.timezone.utc),
                datetime.datetime(2019, 12, 25, tzinfo=datetime.timezone.utc),
                datetime.datetime(2018, 11, 10, tzinfo=datetime.timezone.utc),
                datetime.datetime(2018, 11, 10, tzinfo=datetime.timezone.utc),
                datetime.datetime(2018, 9, 10, tzinfo=datetime.timezone.utc),
                datetime.datetime(2015, 11, 10, tzinfo=datetime.timezone.utc),
                datetime.datetime(2015, 11, 10, tzinfo=datetime.timezone.utc),
                datetime.datetime(2015, 7, 23, tzinfo=datetime.timezone.utc),
            ],
        },
    )


def create_date_diff_different_dtypes_and_nans():
    """Dataframe with different datetime formats with nans in the data"""
    return pd.DataFrame(
        {
            "date_col_1": [
                np.nan,
                datetime.date(2000, 3, 19),
                datetime.date(2018, 11, 10),
                datetime.date(2018, 10, 10),
                datetime.date(2018, 10, 10),
                datetime.date(2018, 10, 10),
                datetime.date(2018, 12, 10),
                datetime.date(
                    1985,
                    7,
                    23,
                ),
            ],
            "date_col_2": [
                datetime.date(2020, 5, 1),
                datetime.date(2019, 12, 25),
                datetime.date(2018, 11, 10),
                datetime.date(2018, 11, 10),
                datetime.date(2018, 9, 10),
                datetime.date(2015, 11, 10),
                datetime.date(2015, 11, 10),
                datetime.date(2015, 7, 23),
            ],
            "datetime_col_1": [
                datetime.datetime(1993, 9, 27, tzinfo=datetime.timezone.utc),
                datetime.datetime(2000, 3, 19, tzinfo=datetime.timezone.utc),
                datetime.datetime(2018, 11, 10, tzinfo=datetime.timezone.utc),
                datetime.datetime(2018, 10, 10, tzinfo=datetime.timezone.utc),
                datetime.datetime(2018, 10, 10, tzinfo=datetime.timezone.utc),
                datetime.datetime(2018, 10, 10, tzinfo=datetime.timezone.utc),
                datetime.datetime(2018, 12, 10, tzinfo=datetime.timezone.utc),
                datetime.datetime(
                    1985,
                    7,
                    23,
                    tzinfo=datetime.timezone.utc,
                ),
            ],
            "datetime_col_2": [
                np.nan,
                datetime.datetime(2019, 12, 25, tzinfo=datetime.timezone.utc),
                datetime.datetime(2018, 11, 10, tzinfo=datetime.timezone.utc),
                datetime.datetime(2018, 11, 10, tzinfo=datetime.timezone.utc),
                datetime.datetime(2018, 9, 10, tzinfo=datetime.timezone.utc),
                datetime.datetime(2015, 11, 10, tzinfo=datetime.timezone.utc),
                datetime.datetime(2015, 11, 10, tzinfo=datetime.timezone.utc),
                datetime.datetime(2015, 7, 23, tzinfo=datetime.timezone.utc),
            ],
        },
    )


def expected_date_diff_df_1():
    """Expected output for test_expected_output_drop_cols_true."""
    return pd.DataFrame(
        {
            "c": [
                26,
                19,
                0,
                0,
                0,
                -2,
                -3,
                30,
            ],
        },
    )


def expected_date_diff_df_2():
    """Expected output for test_expected_output_drop_cols_true."""
    return pd.DataFrame(
        {
            "c": [
                np.nan,
                19,
                0,
                0,
                0,
                -2,
                -3,
                30,
            ],
        },
    )


def create_date_diff_incorrect_dtypes():
    """Dataframe with different datetime formats"""
    return pd.DataFrame(
        {
            "date_col": [
                datetime.date(1993, 9, 27),
                datetime.date(2000, 3, 19),
                datetime.date(2018, 11, 10),
                datetime.date(2018, 10, 10),
                datetime.date(2018, 10, 10),
                datetime.date(2018, 10, 10),
                datetime.date(2018, 12, 10),
                datetime.date(
                    1985,
                    7,
                    23,
                ),
            ],
            "numeric_col": [
                2,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
            ],
            "string_col": [
                "wowza",
                "wowza",
                "wowza",
                "wowza",
                "wowza",
                "wowza",
                "wowza",
                "wowza",
            ],
            "bool_col": [
                True,
                True,
                True,
                True,
                True,
                False,
                False,
                False,
            ],
            "empty_col": [
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ],
        },
    )


def create_date_diff_different_dtypes_2():
    """Expected output for test_expected_output_units_D."""
    return pd.DataFrame(
        {
            "date_col_1": [
                datetime.date(
                    1993,
                    9,
                    27,
                ),
                datetime.date(
                    2000,
                    3,
                    19,
                ),
                datetime.date(
                    2018,
                    11,
                    10,
                ),
                datetime.date(
                    2018,
                    10,
                    10,
                ),
                datetime.date(
                    2018,
                    10,
                    10,
                ),
                datetime.date(
                    2018,
                    10,
                    10,
                ),
                datetime.date(
                    2018,
                    12,
                    10,
                ),
                datetime.date(
                    1985,
                    7,
                    23,
                ),
            ],
            "date_col_2": [
                datetime.date(
                    2020,
                    5,
                    1,
                ),
                datetime.date(
                    2019,
                    12,
                    25,
                ),
                datetime.date(
                    2018,
                    11,
                    10,
                ),
                datetime.date(
                    2018,
                    11,
                    10,
                ),
                datetime.date(
                    2018,
                    9,
                    10,
                ),
                datetime.date(
                    2015,
                    11,
                    10,
                ),
                datetime.date(
                    2015,
                    11,
                    10,
                ),
                datetime.date(
                    2015,
                    7,
                    23,
                ),
            ],
            "datetime_col_1": [
                datetime.datetime(
                    1993,
                    9,
                    27,
                    11,
                    58,
                    58,
                    tzinfo=datetime.timezone.utc,
                ),
                datetime.datetime(
                    2000,
                    3,
                    19,
                    12,
                    59,
                    59,
                    tzinfo=datetime.timezone.utc,
                ),
                datetime.datetime(
                    2018,
                    11,
                    10,
                    11,
                    59,
                    59,
                    tzinfo=datetime.timezone.utc,
                ),
                datetime.datetime(
                    2018,
                    10,
                    10,
                    11,
                    59,
                    59,
                    tzinfo=datetime.timezone.utc,
                ),
                datetime.datetime(
                    2018,
                    10,
                    10,
                    11,
                    59,
                    59,
                    tzinfo=datetime.timezone.utc,
                ),
                datetime.datetime(
                    2018,
                    10,
                    10,
                    10,
                    59,
                    59,
                    tzinfo=datetime.timezone.utc,
                ),
                datetime.datetime(
                    2018,
                    12,
                    10,
                    11,
                    59,
                    59,
                    tzinfo=datetime.timezone.utc,
                ),
                datetime.datetime(
                    1985,
                    7,
                    23,
                    11,
                    59,
                    59,
                    tzinfo=datetime.timezone.utc,
                ),
            ],
            "datetime_col_2": [
                datetime.datetime(
                    2020,
                    5,
                    1,
                    12,
                    59,
                    59,
                    tzinfo=datetime.timezone.utc,
                ),
                datetime.datetime(
                    2019,
                    12,
                    25,
                    11,
                    58,
                    58,
                    tzinfo=datetime.timezone.utc,
                ),
                datetime.datetime(
                    2018,
                    11,
                    10,
                    11,
                    59,
                    59,
                    tzinfo=datetime.timezone.utc,
                ),
                datetime.datetime(
                    2018,
                    11,
                    10,
                    11,
                    59,
                    59,
                    tzinfo=datetime.timezone.utc,
                ),
                datetime.datetime(
                    2018,
                    9,
                    10,
                    9,
                    59,
                    59,
                    tzinfo=datetime.timezone.utc,
                ),
                datetime.datetime(
                    2015,
                    11,
                    10,
                    11,
                    59,
                    59,
                    tzinfo=datetime.timezone.utc,
                ),
                datetime.datetime(
                    2015,
                    11,
                    10,
                    12,
                    59,
                    59,
                    tzinfo=datetime.timezone.utc,
                ),
                datetime.datetime(
                    2015,
                    7,
                    23,
                    11,
                    59,
                    59,
                    tzinfo=datetime.timezone.utc,
                ),
            ],
        },
    )


def expected_date_diff_df_3():
    return pd.DataFrame(
        {
            "datetime output": [
                9713.042372685186,
                7219.957627314815,
                0.0,
                31.0,
                -30.083333333333332,
                -1064.9583333333333,
                -1125.9583333333333,
                10957.0,
            ],
            "dates output": [
                9713.0,
                7220.0,
                0.0,
                31.0,
                -30.0,
                -1065.0,
                -1126.0,
                10957.0,
            ],
        },
    )
