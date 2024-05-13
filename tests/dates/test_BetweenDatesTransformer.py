import datetime

import pandas as pd
import pytest
import test_aide as ta

import tests.test_data as d
from tubular.dates import BetweenDatesTransformer


class TestInit:
    "tests for BetweenDatesTransformer.__init__."

    @pytest.mark.parametrize("column_index", [0, 1, 2])
    def test_columns_non_list_of_str_error(self, column_index):
        """Test that an exception is raised if columns not list of str."""

        columns = ["a", "b", "c"]
        columns[column_index] = False
        with pytest.raises(
            TypeError,
            match=r"BetweenDatesTransformer: each element of columns should be a single \(string\) column name",
        ):
            BetweenDatesTransformer(
                columns=columns,
                new_column_name="a",
            )

    def test_upper_inclusive_non_bool_error(self):
        """Test that an exception is raised if upper_inclusive not a bool."""
        with pytest.raises(
            TypeError,
            match="BetweenDatesTransformer: upper_inclusive should be a bool",
        ):
            BetweenDatesTransformer(
                columns=["a", "b", "c"],
                new_column_name="d",
                upper_inclusive=1,
            )


class TestTransform:
    """Tests for BetweenDatesTransformer.transform."""

    def expected_df_1():
        """Expected output from transform in test_output."""
        df = d.create_is_between_dates_df_1()

        df["d"] = [True, False]

        return df

    def expected_df_2():
        """Expected output from transform in test_output_both_exclusive."""
        df = d.create_is_between_dates_df_2()

        df["e"] = [False, False, True, True, False, False]

        return df

    def expected_df_3():
        """Expected output from transform in test_output_lower_exclusive."""
        df = d.create_is_between_dates_df_2()

        df["e"] = [False, False, True, True, True, False]

        return df

    def expected_df_4():
        """Expected output from transform in test_output_upper_exclusive."""
        df = d.create_is_between_dates_df_2()

        df["e"] = [False, True, True, True, False, False]

        return df

    def expected_df_5():
        """Expected output from transform in test_output_both_inclusive."""
        df = d.create_is_between_dates_df_2()

        df["e"] = [False, True, True, True, True, False]

        return df

    @pytest.mark.parametrize(
        ("columns, bad_col"),
        [
            (["date_col", "numeric_col", "date_col2"], 1),
            (["date_col", "string_col", "date_col2"], 1),
            (["date_col", "bool_col", "date_col2"], 1),
            (["date_col", "empty_col", "date_col2"], 1),
            (["numeric_col", "date_col", "date_col2"], 0),
            (["string_col", "date_col", "date_col2"], 0),
            (["bool_col", "date_col", "date_col2"], 0),
            (["empty_col", "date_col", "date_col2"], 0),
            (["date_col", "date_col2", "numeric_col"], 2),
            (["date_col", "date_col2", "string_col"], 2),
            (["date_col", "date_col2", "bool_col"], 2),
            (["date_col", "date_col2", "empty_col"], 2),
        ],
    )
    def test_input_data_check_column_errors(self, columns, bad_col):
        """Check that errors are raised on a variety of different non date datatypes"""
        x = BetweenDatesTransformer(
            columns=columns,
            new_column_name="d",
        )
        df = d.create_date_diff_incorrect_dtypes()
        # types don't seem to come out of the above function as expected, hard enforce
        df["date_col"] = pd.to_datetime(df["date_col"])
        df["date_col2"] = df["date_col"].copy()
        df = df[columns]

        msg = rf"{x.classname()}: {columns[bad_col]} type should be in \['datetime64', 'date'\] but got {df[columns[bad_col]].dtype}"

        with pytest.raises(TypeError, match=msg):
            x.transform(df)

    @pytest.mark.parametrize(
        ("df", "expected"),
        ta.pandas.adjusted_dataframe_params(
            d.create_is_between_dates_df_1(),
            expected_df_1(),
        ),
    )
    def test_output(self, df, expected):
        """Test the output of transform is as expected."""
        x = BetweenDatesTransformer(
            columns=["a", "b", "c"],
            new_column_name="d",
            lower_inclusive=False,
            upper_inclusive=False,
        )

        df_transformed = x.transform(df)

        ta.equality.assert_equal_dispatch(
            expected=expected,
            actual=df_transformed,
            msg="BetweenDatesTransformer.transform results not as expected",
        )

    @pytest.mark.parametrize(
        ("df", "expected"),
        ta.pandas.adjusted_dataframe_params(
            d.create_is_between_dates_df_2(),
            expected_df_2(),
        ),
    )
    def test_output_both_exclusive(self, df, expected):
        """Test the output of transform is as expected if both limits are exclusive."""
        x = BetweenDatesTransformer(
            columns=["a", "b", "c"],
            new_column_name="e",
            lower_inclusive=False,
            upper_inclusive=False,
        )

        df_transformed = x.transform(df)

        ta.equality.assert_equal_dispatch(
            expected=expected,
            actual=df_transformed,
            msg="BetweenDatesTransformer.transform results not as expected",
        )

    @pytest.mark.parametrize(
        ("df", "expected"),
        ta.pandas.adjusted_dataframe_params(
            d.create_is_between_dates_df_2(),
            expected_df_3(),
        ),
    )
    def test_output_lower_exclusive(self, df, expected):
        """Test the output of transform is as expected if the lower limits are exclusive only."""
        x = BetweenDatesTransformer(
            columns=["a", "b", "c"],
            new_column_name="e",
            lower_inclusive=False,
            upper_inclusive=True,
        )

        df_transformed = x.transform(df)

        ta.equality.assert_equal_dispatch(
            expected=expected,
            actual=df_transformed,
            msg="BetweenDatesTransformer.transform results not as expected",
        )

    @pytest.mark.parametrize(
        ("df", "expected"),
        ta.pandas.adjusted_dataframe_params(
            d.create_is_between_dates_df_2(),
            expected_df_4(),
        ),
    )
    def test_output_upper_exclusive(self, df, expected):
        """Test the output of transform is as expected if the upper limits are exclusive only."""
        x = BetweenDatesTransformer(
            columns=["a", "b", "c"],
            new_column_name="e",
            lower_inclusive=True,
            upper_inclusive=False,
        )

        df_transformed = x.transform(df)

        ta.equality.assert_equal_dispatch(
            expected=expected,
            actual=df_transformed,
            msg="BetweenDatesTransformer.transform results not as expected",
        )

    @pytest.mark.parametrize(
        ("df", "expected"),
        ta.pandas.adjusted_dataframe_params(
            d.create_is_between_dates_df_2(),
            expected_df_5(),
        ),
    )
    def test_output_both_inclusive(self, df, expected):
        """Test the output of transform is as expected if the both limits are inclusive."""
        x = BetweenDatesTransformer(
            columns=["a", "b", "c"],
            new_column_name="e",
            lower_inclusive=True,
            upper_inclusive=True,
        )

        df_transformed = x.transform(df)

        ta.equality.assert_equal_dispatch(
            expected=expected,
            actual=df_transformed,
            msg="BetweenDatesTransformer.transform results not as expected",
        )

    def test_warning_message(self):
        """Test a warning is generated if not all the values in column_upper are greater than or equal to column_lower."""
        x = BetweenDatesTransformer(
            columns=["a", "b", "c"],
            new_column_name="e",
            lower_inclusive=True,
            upper_inclusive=True,
        )

        df = d.create_is_between_dates_df_2()

        df.loc[0, "c"] = datetime.datetime(1989, 3, 1, tzinfo=datetime.timezone.utc)

        with pytest.warns(Warning, match="not all c are greater than or equal to a"):
            x.transform(df)

    @pytest.mark.parametrize(
        ("columns"),
        [
            ["a_date", "b_date", "c_date"],
            ["a_date", "b_date", "c_datetime"],
            ["a_date", "b_datetime", "c_datetime"],
            ["a_datetime", "b_date", "c_date"],
            ["a_datetime", "b_date", "c_datetime"],
            ["a_datetime", "b_datetime", "c_date"],
        ],
    )
    def test_output_different_date_dtypes(self, columns):
        """Test the output of transform is as expected if both limits are exclusive."""
        x = BetweenDatesTransformer(
            columns=columns,
            new_column_name="e",
            lower_inclusive=False,
            upper_inclusive=False,
        )

        df = d.create_is_between_dates_df_3()
        output = [False, False, True, True, False, False]
        expected = df.copy()
        expected["e"] = output

        df_transformed = x.transform(df)

        ta.equality.assert_equal_dispatch(
            expected=expected,
            actual=df_transformed,
            msg="BetweenDatesTransformer.transform results not as expected",
        )
