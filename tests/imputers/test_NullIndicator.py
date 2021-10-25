import pytest
import test_aide as ta
import tests.test_data as d
import pandas as pd
import numpy as np

import tubular
from tubular.imputers import NullIndicator


class TestInit(object):
    """Tests for NullIndicator.init()"""

    def test_arguments(self):
        """Test that init has expected arguments."""

        ta.function_helpers.test_function_arguments(
            func=NullIndicator.__init__,
            expected_arguments=["self", "columns"],
            expected_default_values=(None,),
        )

    def test_class_methods(self):
        """Test that NullIndicator has transform method."""

        x = NullIndicator()

        ta.class_helpers.test_object_method(
            obj=x, expected_method="transform", msg="transform"
        )

    def test_inheritance(self):
        """Test that NullIndicator inherits from BaseTransformer."""

        x = NullIndicator()

        ta.class_helpers.assert_inheritance(x, tubular.base.BaseTransformer)

    def test_super_init_called(self, mocker):
        """Test that init calls BaseTransformer.init."""

        expected_call_args = {
            0: {"args": (), "kwargs": {"columns": None, "verbose": True, "copy": True}}
        }

        with ta.function_helpers.assert_function_call(
            mocker, tubular.base.BaseTransformer, "__init__", expected_call_args
        ):

            NullIndicator(columns=None, verbose=True, copy=True)


class TestTransform(object):
    """Tests for NullIndicator.transform()"""

    def expected_df_1():
        """Expected output for test_null_indicator_columns_correct."""

        df = pd.DataFrame(
            {
                "a": [1, 2, np.nan, 4, np.nan, 6],
                "b": [np.nan, 5, 4, 3, 2, 1],
                "c": [3, 2, 1, 4, 5, 6],
                "b_nulls": [1, 0, 0, 0, 0, 0],
                "c_nulls": [0, 0, 0, 0, 0, 0],
            }
        )

        return df

    def test_arguments(self):
        """Test that transform has expected arguments."""

        ta.function_helpers.test_function_arguments(
            func=NullIndicator.transform, expected_arguments=["self", "X"]
        )

    def test_super_transform_called(self, mocker):
        """Test that BaseTransformer.transform called."""

        df = d.create_df_1()

        x = NullIndicator(columns="a")

        expected_call_args = {0: {"args": (d.create_df_1(),), "kwargs": {}}}

        with ta.function_helpers.assert_function_call(
            mocker, tubular.base.BaseTransformer, "transform", expected_call_args
        ):

            x.transform(df)

    @pytest.mark.parametrize(
        "df, expected",
        ta.pandas_helpers.row_by_row_params(d.create_df_9(), expected_df_1())
        + ta.pandas_helpers.index_preserved_params(
            d.create_df_9(), expected_df_1()
        ),
    )
    def test_null_indicator_columns_correct(self, df, expected):
        """Test that the created indicator column is correct - and unrelated columns are unchanged"""

        x = NullIndicator(columns=["b", "c"])

        df_transformed = x.transform(df)

        ta.equality_helpers.assert_equal_dispatch(
            expected=expected,
            actual=df_transformed,
            msg="Check null indicator columns created correctly in transform.",
        )
