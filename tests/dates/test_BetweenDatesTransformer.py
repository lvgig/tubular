import datetime

import pytest
import test_aide as ta

import tests.test_data as d
import tubular
from tubular.dates import BetweenDatesTransformer


class TestInit:
    "tests for BetweenDatesTransformer.__init__."

    def test_super_init_called(self, mocker):
        """Test that super.__init__ called."""
        expected_call_args = {
            0: {
                "args": (),
                "kwargs": {"columns": ["a", "b", "c"], "verbose": False, "copy": True},
            },
        }

        with ta.functions.assert_function_call(
            mocker,
            tubular.base.BaseTransformer,
            "__init__",
            expected_call_args,
        ):
            BetweenDatesTransformer(
                column_lower="a",
                column_between="b",
                column_upper="c",
                new_column_name="d",
                verbose=False,
                copy=True,
            )

    def test_first_non_str_error(self):
        """Test that an exception is raised if column_lower not str."""
        with pytest.raises(
            TypeError,
            match="BetweenDatesTransformer: column_lower should be str",
        ):
            BetweenDatesTransformer(
                column_lower=False,
                column_between="b",
                column_upper="c",
                new_column_name="a",
            )

    def test_column_between_non_str_error(self):
        """Test that an exception is raised if column_between not str."""
        with pytest.raises(
            TypeError,
            match="BetweenDatesTransformer: column_between should be str",
        ):
            BetweenDatesTransformer(
                column_lower="a",
                column_between=1,
                column_upper="c",
                new_column_name="c",
            )

    def test_column_upper_non_str_error(self):
        """Test that an exception is raised if column_upper not str."""
        with pytest.raises(
            TypeError,
            match="BetweenDatesTransformer: column_upper should be str",
        ):
            BetweenDatesTransformer(
                column_lower="a",
                column_between="b",
                column_upper=1.2,
                new_column_name="c",
            )

    def test_new_column_name_non_str_error(self):
        """Test that an exception is raised if new_column_name not str."""
        with pytest.raises(
            TypeError,
            match="BetweenDatesTransformer: new_column_name should be str",
        ):
            BetweenDatesTransformer(
                column_lower="a",
                column_between="b",
                column_upper="c",
                new_column_name=(),
            )

    def test_lower_inclusive_non_bool_error(self):
        """Test that an exception is raised if lower_inclusive not a bool."""
        with pytest.raises(
            TypeError,
            match="BetweenDatesTransformer: lower_inclusive should be a bool",
        ):
            BetweenDatesTransformer(
                column_lower="a",
                column_between="b",
                column_upper="c",
                new_column_name="d",
                lower_inclusive=1,
            )

    def test_upper_inclusive_non_bool_error(self):
        """Test that an exception is raised if upper_inclusive not a bool."""
        with pytest.raises(
            TypeError,
            match="BetweenDatesTransformer: upper_inclusive should be a bool",
        ):
            BetweenDatesTransformer(
                column_lower="a",
                column_between="b",
                column_upper="c",
                new_column_name="d",
                upper_inclusive=1,
            )

    def test_values_passed_in_init_set_to_attribute(self):
        """Test that attributes are set by init."""
        x = BetweenDatesTransformer(
            column_lower="a",
            column_between="b",
            column_upper="c",
            new_column_name="d",
            lower_inclusive=False,
            upper_inclusive=False,
        )

        ta.classes.test_object_attributes(
            obj=x,
            expected_attributes={
                "column_lower": "a",
                "column_between": "b",
                "column_upper": "c",
                "columns": ["a", "b", "c"],
                "new_column_name": "d",
                "lower_inclusive": False,
                "upper_inclusive": False,
            },
            msg="Attributes for BetweenDatesTransformer set in init",
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

    def test_super_transform_call(self, mocker):
        """Test that call the BaseTransformer.transform() is as expected."""
        df = d.create_is_between_dates_df_1()

        x = BetweenDatesTransformer(
            column_lower="a",
            column_between="b",
            column_upper="c",
            new_column_name="d",
        )

        expected_call_args = {
            0: {"args": (d.create_is_between_dates_df_1(),), "kwargs": {}},
        }

        with ta.functions.assert_function_call(
            mocker,
            tubular.base.BaseTransformer,
            "transform",
            expected_call_args,
            return_value=d.create_is_between_dates_df_1(),
        ):
            x.transform(df)

    @pytest.mark.parametrize(
        ("columns, bad_col"),
        [
            (["date_col", "numeric_col", "date_col"], 1),
            (["date_col", "string_col", "date_col"], 1),
            (["date_col", "bool_col", "date_col"], 1),
            (["date_col", "empty_col", "date_col"], 1),
            (["numeric_col", "date_col", "date_col"], 0),
            (["string_col", "date_col", "date_col"], 0),
            (["bool_col", "date_col", "date_col"], 0),
            (["empty_col", "date_col", "date_col"], 0),
            (["date_col", "date_col", "numeric_col"], 2),
            (["date_col", "date_col", "string_col"], 2),
            (["date_col", "date_col", "bool_col"], 2),
            (["date_col", "date_col", "empty_col"], 2),
        ],
    )
    def test_input_data_check_column_errors(self, columns, bad_col):
        """Check that errors are raised on a variety of different non date datatypes"""
        x = BetweenDatesTransformer(
            column_lower=columns[0],
            column_between=columns[1],
            column_upper=columns[2],
            new_column_name="d",
        )
        df = d.create_date_diff_incorrect_dtypes()

        msg = f"{x.classname()}: {columns[bad_col]} should be datetime64 or date type but got {df[columns[bad_col]].dtype}"

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
            column_lower="a",
            column_between="b",
            column_upper="c",
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
            column_lower="a",
            column_between="b",
            column_upper="c",
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
            column_lower="a",
            column_between="b",
            column_upper="c",
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
            column_lower="a",
            column_between="b",
            column_upper="c",
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
            column_lower="a",
            column_between="b",
            column_upper="c",
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
            column_lower="a",
            column_between="b",
            column_upper="c",
            new_column_name="e",
            lower_inclusive=True,
            upper_inclusive=True,
        )

        df = d.create_is_between_dates_df_2()

        df["c"][0] = datetime.datetime(1989, 3, 1, tzinfo=datetime.timezone.utc)

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
            column_lower=columns[0],
            column_between=columns[1],
            column_upper=columns[2],
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
