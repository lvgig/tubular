import datetime

import pandas as pd
import pytest
import test_aide as ta

import tests.test_data as d
from tests.base_tests import (
    ColumnStrListInitTests,
    DropOriginalInitMixinTests,
    DropOriginalTransformMixinTests,
    GenericTransformTests,
    NewColumnNameInitMixintests,
    OtherBaseBehaviourTests,
)
from tests.dates.test_BaseGenericDateTransformer import (
    GenericDatesMixinTransformTests,
    create_date_diff_different_dtypes,
)
from tubular.dates import BetweenDatesTransformer


class TestInit(
    ColumnStrListInitTests,
    NewColumnNameInitMixintests,
    DropOriginalInitMixinTests,
):
    "tests for BetweenDatesTransformer.__init__."

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BetweenDatesTransformer"

    @pytest.mark.parametrize(
        ("param", "value"),
        [
            ("upper_inclusive", 1000),
            ("lower_inclusive", "hi"),
        ],
    )
    def test_inclusive_args_non_bool_error(self, param, value):
        """Test that an exception is raised if upper_inclusive not a bool."""

        param_dict = {param: value}
        with pytest.raises(
            TypeError,
            match=f"BetweenDatesTransformer: {param} should be a bool",
        ):
            BetweenDatesTransformer(
                columns=["a", "b", "c"],
                new_column_name="d",
                **param_dict,
            )

    @pytest.mark.parametrize(
        "columns",
        [
            ["a", "b"],
            ["a", "b", "c", "d"],
        ],
    )
    def test_wrong_col_count_error(self, columns):
        """Test that an exception is raised if too many/too few columns."""

        with pytest.raises(
            ValueError,
            match="BetweenDatesTransformer: This transformer works with three columns only",
        ):
            BetweenDatesTransformer(
                columns=columns,
                new_column_name="d",
            )


class TestTransform(
    GenericTransformTests,
    GenericDatesMixinTransformTests,
    DropOriginalTransformMixinTests,
):
    """Tests for BetweenDatesTransformer.transform."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BetweenDatesTransformer"

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

    # overloading below test as column count is different for this one
    @pytest.mark.parametrize(
        ("columns, datetime_col, date_col"),
        [
            (["date_col_1", "datetime_col_2", "date_col_2"], 1, 0),
            (["datetime_col_1", "date_col_2", "datetime_col_2"], 0, 1),
        ],
    )
    def test_mismatched_datetypes_error(
        self,
        columns,
        datetime_col,
        date_col,
        uninitialized_transformers,
    ):
        "Test that transform raises an error if one column is a date and one is datetime"

        x = uninitialized_transformers[self.transformer_name](
            columns=columns,
            new_column_name="c",
        )

        df = create_date_diff_different_dtypes()
        # types don't seem to come out of the above function as expected, hard enforce
        for col in ["date_col_1", "date_col_2"]:
            df[col] = pd.to_datetime(df[col]).dt.date

        for col in ["datetime_col_1", "datetime_col_2"]:
            df[col] = pd.to_datetime(df[col])

        present_types = (
            {"datetime64", "date"} if datetime_col == 0 else {"date", "datetime64"}
        )
        msg = rf"Columns fed to datetime transformers should be \['datetime64', 'date'\] and have consistent types, but found {present_types}. Please use ToDatetimeTransformer to standardise"
        with pytest.raises(
            TypeError,
            match=msg,
        ):
            x.transform(df)


class TestOtherBaseBehaviour(OtherBaseBehaviourTests):
    """
    Class to run tests for BaseTransformerBehaviour outside the three standard methods.

    May need to overwite specific tests in this class if the tested transformer modifies this behaviour.
    """

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BetweenDatesTransformer"
