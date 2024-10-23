import numpy as np
import pandas as pd
import polars as pl
import pytest

import tests.test_data as d
from tests.base_tests import (
    ColumnStrListInitTests,
    GenericTransformTests,
    OtherBaseBehaviourTests,
)
from tests.utils import assert_frame_equal_dispatch
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

    def expected_df_1(self, library="pandas"):
        """Expected output for test_null_indicator_columns_correct."""

        df = pd.DataFrame(
            {
                "a": [1, 2, np.nan, 4, np.nan, 6],
                "b": [np.nan, 5, 4, 3, 2, 1],
                "c": [3, 2, 1, 4, 5, 6],
                "b_nulls": [1, 0, 0, 0, 0, 0],
                "c_nulls": [0, 0, 0, 0, 0, 0],
            },
        )

        if library == "polars":
            df = pl.from_pandas(df)

        return df

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_null_indicator_columns_correct(self, library):
        """Test that the created indicator column is correct - and unrelated columns are unchanged."""
        columns = ["b", "c"]
        transformer = NullIndicator(columns=columns)

        df = d.create_df_9(library=library)
        expected = self.expected_df_1(library=library)

        df_transformed = transformer.transform(df)

        for col in [column + "_nulls" for column in columns]:
            if library == "pandas":
                expected[col] = expected[col].astype(np.int8)
            else:
                expected = expected.with_columns(expected[col].cast(pl.Int8))

        assert_frame_equal_dispatch(df_transformed, expected)


class TestOtherBaseBehaviour(OtherBaseBehaviourTests):
    """
    Class to run tests for BaseTransformerBehaviour outside the three standard methods.

    May need to overwite specific tests in this class if the tested transformer modifies this behaviour.
    """

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "NullIndicator"
