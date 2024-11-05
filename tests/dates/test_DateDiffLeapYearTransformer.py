import datetime

import numpy as np
import pandas as pd
import pytest
import test_aide as ta

import tests.test_data as d
from tests.base_tests import (
    DropOriginalInitMixinTests,
    DropOriginalTransformMixinTests,
    GenericTransformTests,
    NewColumnNameInitMixintests,
    OtherBaseBehaviourTests,
    TwoColumnListInitTests,
)
from tests.dates.test_BaseGenericDateTransformer import (
    GenericDatesMixinTransformTests,
)
from tubular.dates import DateDiffLeapYearTransformer


class TestCalculateAge:
    """Tests for the calculate_age function in dates.py."""

    def test_row_type_error(self):
        """Test that an exception is raised if row is not a pd.Series."""
        row = "dummy_row"
        date_transformer = DateDiffLeapYearTransformer(
            columns=["a", "b"],
            new_column_name="c",
            drop_original=True,
        )

        with pytest.raises(
            TypeError,
            match="DateDiffLeapYearTransformer: row should be a pd.Series",
        ):
            date_transformer.calculate_age(row=row)

    def test_null_replacement(self):
        """Test correct value is replaced using null_replacement."""
        row = pd.Series({"a": np.nan, "b": np.nan})
        date_transformer = DateDiffLeapYearTransformer(
            columns=["a", "b"],
            new_column_name="c",
            drop_original=True,
            missing_replacement="missing_replacement",
        )

        val = date_transformer.calculate_age(row=row)

        assert val == "missing_replacement"


class TestInit(
    NewColumnNameInitMixintests,
    DropOriginalInitMixinTests,
    TwoColumnListInitTests,
):
    """Tests for DateDiffLeapYearTransformer.init()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "DateDiffLeapYearTransformer"

    def test_missing_replacement_type_error(self):
        """Test that an exception is raised if missing_replacement is not the correct type."""
        with pytest.raises(
            TypeError,
            match="DateDiffLeapYearTransformer: if not None, missing_replacement should be an int, float or string",
        ):
            DateDiffLeapYearTransformer(
                columns=["dummy_1", "dummy_2"],
                new_column_name="dummy_3",
                drop_original=False,
                missing_replacement=[1, 2, 3],
            )


class TestTransform(
    DropOriginalTransformMixinTests,
    GenericTransformTests,
    GenericDatesMixinTransformTests,
):
    """Tests for DateDiffLeapYearTransformer.transform()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "DateDiffLeapYearTransformer"

    def expected_df_1():
        """Expected output for test_expected_output_drop_original_true."""
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

    def expected_df_2():
        """Expected output for test_expected_output_drop_original_false."""
        return pd.DataFrame(
            {
                "a": [
                    datetime.date(1993, 9, 27),  # day/month greater than
                    datetime.date(2000, 3, 19),  # day/month less than
                    datetime.date(2018, 11, 10),  # same day
                    datetime.date(2018, 10, 10),  # same year day/month greater than
                    datetime.date(2018, 10, 10),  # same year day/month less than
                    datetime.date(2018, 10, 10),  # negative day/month less than
                    datetime.date(2018, 12, 10),  # negative day/month greater than
                    datetime.date(
                        1985,
                        7,
                        23,
                    ),  # large gap, this is incorrect with timedelta64 solutions
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

    def expected_df_3():
        """Expected output for test_expected_output_nulls."""
        return pd.DataFrame(
            {
                "a": [
                    np.nan,
                ],
                "b": [
                    np.nan,
                ],
                "c": [None],
            },
        )

    @pytest.mark.parametrize(
        ("df", "expected"),
        ta.pandas.adjusted_dataframe_params(d.create_date_test_df(), expected_df_1()),
    )
    def test_expected_output_drop_original_true(self, df, expected):
        """Test that the output is expected from transform, when drop_original is True.

        This tests positive year gaps, negative year gaps, and missing values.

        """
        x = DateDiffLeapYearTransformer(
            columns=["a", "b"],
            new_column_name="c",
            drop_original=True,
        )

        df_transformed = x.transform(df)

        ta.equality.assert_frame_equal_msg(
            actual=df_transformed,
            expected=expected,
            msg_tag="Unexpected values in DateDiffLeapYearTransformer.transform (with drop_original)",
        )

    @pytest.mark.parametrize(
        ("df", "expected"),
        ta.pandas.adjusted_dataframe_params(d.create_date_test_df(), expected_df_2()),
    )
    def test_expected_output_drop_original_false(self, df, expected):
        """Test that the output is expected from transform, when drop_original is False.

        This tests positive year gaps , negative year gaps, and missing values.

        """
        x = DateDiffLeapYearTransformer(
            columns=["a", "b"],
            new_column_name="c",
            drop_original=False,
        )

        df_transformed = x.transform(df)

        ta.equality.assert_frame_equal_msg(
            actual=df_transformed,
            expected=expected,
            msg_tag="Unexpected values in DateDiffLeapYearTransformer.transform (with drop_original=False)",
        )

    @pytest.mark.parametrize(
        ("columns"),
        [
            ["date_col_1", "date_col_2"],
            ["datetime_col_1", "datetime_col_2"],
        ],
    )
    def test_expected_output_nans_in_data(self, columns):
        "Test that transform works for different date datatype combinations with nans in data"
        x = DateDiffLeapYearTransformer(
            columns=columns,
            new_column_name="c",
            drop_original=True,
        )

        expected = d.expected_date_diff_df_2()

        df = d.create_date_diff_different_dtypes_and_nans()

        df_transformed = x.transform(df[columns])

        ta.equality.assert_frame_equal_msg(
            actual=df_transformed,
            expected=expected,
            msg_tag=f"Unexpected values in DateDiffLeapYearTransformer.transform between {columns[0]} and {columns[1]}",
        )


class TestOtherBaseBehaviour(OtherBaseBehaviourTests):
    """
    Class to run tests for BaseTransformerBehaviour outside the three standard methods.

    May need to overwite specific tests in this class if the tested transformer modifies this behaviour.
    """

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "DateDiffLeapYearTransformer"
