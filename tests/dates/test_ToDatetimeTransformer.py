import datetime

import narwhals as nw
import numpy as np
import pandas as pd
import polars as pl
import pytest
from pandas.testing import assert_frame_equal

from tests.base_tests import (
    ColumnStrListInitTests,
    DropOriginalInitMixinTests,
    GenericTransformTests,
    NewColumnNameInitMixintests,
    OtherBaseBehaviourTests,
)
from tubular.dates import ToDatetimeTransformer


class TestInit(
    NewColumnNameInitMixintests,
    DropOriginalInitMixinTests,
    ColumnStrListInitTests,
):
    """Generic tests for transformer.init()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseDatetimeTransformer"

    def test_to_datetime_kwargs_type_error(self):
        """Test that an exception is raised if to_datetime_kwargs is not a dict."""
        with pytest.raises(
            TypeError,
            match=r"""ToDatetimeTransformer: to_datetime_kwargs should be a dict but got \<class 'int'\>""",
        ):
            ToDatetimeTransformer(column="b", new_column_name="a", to_datetime_kwargs=1)

    def test_to_datetime_kwargs_key_type_error(self):
        """Test that an exception is raised if to_datetime_kwargs has keys which are not str."""
        with pytest.raises(
            TypeError,
            match=r"""ToDatetimeTransformer: unexpected type <class 'int'> for to_datetime_kwargs key, must be str""",
        ):
            ToDatetimeTransformer(
                new_column_name="a",
                column="b",
                to_datetime_kwargs={"a": 1, 2: "b"},
            )


class TestTransform(GenericTransformTests):
    """Tests for ToDatetimeTransformer.transform."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseDatetimeTransformer"

    def expected_df_1():
        """Expected output for test_expected_output."""
        return pd.DataFrame(
            {
                "a": [1950, 1960, 2000, 2001, np.nan, 2010],
                "b": [1, 2, 3, 4, 5, np.nan],
                "a_Y": [
                    datetime.datetime(1950, 1, 1, tzinfo=datetime.timezone.utc),
                    datetime.datetime(1960, 1, 1, tzinfo=datetime.timezone.utc),
                    datetime.datetime(2000, 1, 1, tzinfo=datetime.timezone.utc),
                    datetime.datetime(2001, 1, 1, tzinfo=datetime.timezone.utc),
                    pd.NaT,
                    datetime.datetime(2010, 1, 1, tzinfo=datetime.timezone.utc),
                ],
                "b_m": [
                    datetime.datetime(1900, 1, 1, tzinfo=datetime.timezone.utc),
                    datetime.datetime(1900, 2, 1, tzinfo=datetime.timezone.utc),
                    datetime.datetime(1900, 3, 1, tzinfo=datetime.timezone.utc),
                    datetime.datetime(1900, 4, 1, tzinfo=datetime.timezone.utc),
                    datetime.datetime(1900, 5, 1, tzinfo=datetime.timezone.utc),
                    pd.NaT,
                ],
            },
        ).astype({"a": "float64", "b": "float64"})

    def create_to_datetime_test_df_pandas():
        """Create Pandas DataFrame to be used in the ToDatetimeTransformer tests."""
        return pd.DataFrame(
            {"a": [1950, 1960, 2000, 2001, np.nan, 2010], "b": [1, 2, 3, 4, 5, np.nan]},
        )

    def create_to_datetime_test_df_polars():
        """Create Polars DataFrame to be used in the ToDatetimeTransformer tests."""
        return pl.DataFrame(
            {"a": [1950, 1960, 2000, 2001, None, 2010], "b": [1, 2, 3, 4, 5, None]},
        )

    @pytest.mark.parametrize(
        "df",
        [create_to_datetime_test_df_pandas(), create_to_datetime_test_df_polars()],
    )
    def test_expected_output(self, df):
        """Test input data is transformed as expected for both Pandas and Polars."""

        df = nw.from_native(df)

        df = df.with_columns(df["a"].cast(nw.String).alias("a"))

        to_dt_1 = ToDatetimeTransformer(
            column="a",
            new_column_name="a_Y",
            to_datetime_kwargs={"format": "%Y", "utc": datetime.timezone.utc},
        )
        to_dt_2 = ToDatetimeTransformer(
            column="b",
            new_column_name="b_m",
            to_datetime_kwargs={"format": "%m", "utc": datetime.timezone.utc},
        )

        df_transformed = to_dt_1.transform(df)
        df_transformed = to_dt_2.transform(df_transformed)

        df_transformed_native = (
            df_transformed.to_native()
            if hasattr(df_transformed, "to_native")
            else df_transformed
        )
        expected_native = TestTransform.expected_df_1()

        df_transformed_native = df_transformed_native.astype(
            {"a": "float64", "b": "float64"},
        )

        assert_frame_equal(df_transformed_native, expected_native)


class TestOtherBaseBehaviour(OtherBaseBehaviourTests):
    """
    Class to run tests for BaseTransformerBehaviour outside the three standard methods.

    May need to overwrite specific tests in this class if the tested transformer modifies this behaviour.
    """

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseDatetimeTransformer"
