import numpy as np
import pandas as pd
import pytest
import test_aide as ta

import tests.test_data as d
from tests.base_tests import (
    ColumnStrListInitTests,
    GenericTransformTests,
    OtherBaseBehaviourTests,
)
from tubular.imputers import NullIndicator


class TestInit(ColumnStrListInitTests):
    """Tests for NullIndicator.init()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "NullIndicator"


class TestTransform(GenericTransformTests):
    """Tests for NullIndicator.transform()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "NullIndicator"

    def expected_df_1():
        """Expected output for test_null_indicator_columns_correct."""
        return pd.DataFrame(
            {
                "a": [1, 2, np.nan, 4, np.nan, 6],
                "b": [np.nan, 5, 4, 3, 2, 1],
                "c": [3, 2, 1, 4, 5, 6],
                "b_nulls": [1, 0, 0, 0, 0, 0],
                "c_nulls": [0, 0, 0, 0, 0, 0],
            },
        )

    @pytest.mark.parametrize(
        ("df", "expected"),
        ta.pandas.adjusted_dataframe_params(d.create_df_9(), expected_df_1()),
    )
    def test_null_indicator_columns_correct(self, df, expected):
        """Test that the created indicator column is correct - and unrelated columns are unchanged."""
        columns = ["b", "c"]
        x = NullIndicator(columns=columns)

        df_transformed = x.transform(df)

        for col in [column + "_nulls" for column in columns]:
            expected[col] = expected[col].astype(np.int8)

        ta.equality.assert_equal_dispatch(
            expected=expected,
            actual=df_transformed,
            msg="Check null indicator columns created correctly in transform.",
        )


class TestOtherBaseBehaviour(OtherBaseBehaviourTests):
    """
    Class to run tests for BaseTransformerBehaviour outside the three standard methods.

    May need to overwite specific tests in this class if the tested transformer modifies this behaviour.
    """

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "NullIndicator"
