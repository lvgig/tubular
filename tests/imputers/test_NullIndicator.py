import numpy as np
import pandas as pd
import pytest
import test_aide as ta

import tests.test_data as d
import tubular
from tubular.imputers import NullIndicator


class TestInit:
    """Tests for NullIndicator.init()."""

    def test_super_init_called(self, mocker):
        """Test that init calls BaseTransformer.init."""
        expected_call_args = {
            0: {"args": (), "kwargs": {"columns": None, "verbose": True}},
        }

        with ta.functions.assert_function_call(
            mocker,
            tubular.base.BaseTransformer,
            "__init__",
            expected_call_args,
        ):
            NullIndicator(columns=None, verbose=True)


class TestTransform:
    """Tests for NullIndicator.transform()."""

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

    def test_super_transform_called(self, mocker):
        """Test that BaseTransformer.transform called."""
        df = d.create_df_1()

        x = NullIndicator(columns="a")

        expected_call_args = {0: {"args": (d.create_df_1(),), "kwargs": {}}}

        with ta.functions.assert_function_call(
            mocker,
            tubular.base.BaseTransformer,
            "transform",
            expected_call_args,
        ):
            x.transform(df)

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
