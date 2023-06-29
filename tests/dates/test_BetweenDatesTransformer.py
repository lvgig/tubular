import pytest
import test_aide as ta
import tests.test_data as d
import pandas as pd
import datetime

import tubular
from tubular.base import BaseTransformer
from tubular.dates import BetweenDatesTransformer


class TestInit(object):
    "tests for BetweenDatesTransformer.__init__"

    def test_arguments(self):
        """Test that init has expected arguments."""

        ta.functions.test_function_arguments(
            func=BetweenDatesTransformer.__init__,
            expected_arguments=[
                "self",
                "column_lower",
                "column_between",
                "column_upper",
                "new_column_name",
                "lower_inclusive",
                "upper_inclusive",
            ],
            expected_default_values=(True, True),
        )

    def test_inheritance(self):
        """Test that BetweenDatesTransformer inherits from BaseTransformer."""

        x = BetweenDatesTransformer(
            column_lower="a", column_between="b", column_upper="c", new_column_name="d"
        )

        ta.classes.assert_inheritance(x, BaseTransformer)

    def test_super_init_called(self, mocker):
        """Test that super.__init__ called."""

        expected_call_args = {
            0: {
                "args": (),
                "kwargs": {"columns": ["a", "b", "c"], "verbose": False, "copy": True},
            }
        }

        with ta.functions.assert_function_call(
            mocker, tubular.base.BaseTransformer, "__init__", expected_call_args
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
            TypeError, match="BetweenDatesTransformer: column_lower should be str"
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
            TypeError, match="BetweenDatesTransformer: column_between should be str"
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
            TypeError, match="BetweenDatesTransformer: column_upper should be str"
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
            TypeError, match="BetweenDatesTransformer: new_column_name should be str"
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
            TypeError, match="BetweenDatesTransformer: lower_inclusive should be a bool"
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
            TypeError, match="BetweenDatesTransformer: upper_inclusive should be a bool"
        ):
            BetweenDatesTransformer(
                column_lower="a",
                column_between="b",
                column_upper="c",
                new_column_name="d",
                upper_inclusive=1,
            )

    def test_class_methods(self):
        """Test that BetweenDatesTransformer has transform method."""

        x = BetweenDatesTransformer(
            column_lower="a", column_between="b", column_upper="c", new_column_name="d"
        )

        ta.classes.test_object_method(
            obj=x, expected_method="transform", msg="transform method not present"
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


class TestTransform(object):
    """Tests for BetweenDatesTransformer.transform"""

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

    def test_arguments(self):
        """Test that fit has expected arguments."""

        ta.functions.test_function_arguments(
            func=BetweenDatesTransformer.transform,
            expected_arguments=["self", "X"],
            expected_default_values=None,
        )

    def test_super_transform_call(self, mocker):
        """Test that call the BaseTransformer.transform() is as expected."""

        df = d.create_is_between_dates_df_1()

        x = BetweenDatesTransformer(
            column_lower="a", column_between="b", column_upper="c", new_column_name="d"
        )

        expected_call_args = {
            0: {"args": (d.create_is_between_dates_df_1(),), "kwargs": {}}
        }

        with ta.functions.assert_function_call(
            mocker,
            tubular.base.BaseTransformer,
            "transform",
            expected_call_args,
            return_value=d.create_is_between_dates_df_1(),
        ):
            x.transform(df)

    def test_cols_not_datetime(self):
        """Test that an exception is raised if cols not datetime."""

        df = pd.DataFrame(
            {
                "a": [2, 1],
                "b": pd.date_range(start="1/3/2016", end="27/09/2017", periods=2),
                "c": pd.date_range(start="1/2/2016", end="27/04/2017", periods=2),
            }
        )

        x = BetweenDatesTransformer(
            column_lower="a", column_between="b", column_upper="c", new_column_name="d"
        )

        with pytest.raises(
            TypeError,
            match=r"BetweenDatesTransformer: a should be datetime64\[ns\] type but got int64",
        ):
            x.transform(df)

    @pytest.mark.parametrize(
        "df, expected",
        ta.pandas.adjusted_dataframe_params(
            d.create_is_between_dates_df_1(), expected_df_1()
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
        "df, expected",
        ta.pandas.adjusted_dataframe_params(
            d.create_is_between_dates_df_2(), expected_df_2()
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
        "df, expected",
        ta.pandas.adjusted_dataframe_params(
            d.create_is_between_dates_df_2(), expected_df_3()
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
        "df, expected",
        ta.pandas.adjusted_dataframe_params(
            d.create_is_between_dates_df_2(), expected_df_4()
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
        "df, expected",
        ta.pandas.adjusted_dataframe_params(
            d.create_is_between_dates_df_2(), expected_df_5()
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

        df["c"][0] = datetime.datetime(1989, 3, 1)

        with pytest.warns(Warning, match="not all c are greater than or equal to a"):
            x.transform(df)
