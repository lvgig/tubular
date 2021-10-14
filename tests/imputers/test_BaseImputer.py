import pytest
import test_aide as ta
import pandas as pd
import numpy as np

import tubular
from tubular.imputers import BaseImputer


class TestInit:
    """Tests for BaseImputer.init."""

    def test_class_methods(self):
        """Test that BaseImputer has transform method."""

        x = BaseImputer()

        ta.class_helpers.test_object_method(
            obj=x, expected_method="transform", msg="transform"
        )

    def test_inheritance(self):
        """Test that BaseImputer inherits from BaseTransformer."""

        x = BaseImputer()

        ta.class_helpers.assert_inheritance(x, tubular.base.BaseTransformer)


class TestTransform:
    """Tests for BaseImputer.transform."""

    def expected_df_1():
        """Expected output of test_expected_output_1."""

        df = pd.DataFrame(
            {
                "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                "b": ["a", "b", "c", "d", "e", "f", np.NaN],
                "c": ["a", "b", "c", "d", "e", "f", np.NaN],
            }
        )

        df["c"] = df["c"].astype("category")

        return df

    def expected_df_2():
        """Expected output of test_expected_output_2."""

        df2 = pd.DataFrame(
            {
                "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, np.NaN],
                "b": ["a", "b", "c", "d", "e", "f", "g"],
                "c": ["a", "b", "c", "d", "e", "f", np.NaN],
            }
        )

        df2["c"] = df2["c"].astype("category")

        return df2

    def expected_df_3():
        """Expected output of test_expected_output_3."""

        df3 = pd.DataFrame(
            {
                "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, np.NaN],
                "b": ["a", "b", "c", "d", "e", "f", "g"],
                "c": ["a", "b", "c", "d", "e", "f", "f"],
            }
        )

        df3["c"] = df3["c"].astype("category")

        return df3

    def test_arguments(self):
        """Test that transform has expected arguments."""

        ta.function_helpers.test_function_arguments(
            func=BaseImputer.transform, expected_arguments=["self", "X"]
        )

    @pytest.mark.parametrize(
        "df, expected",
        ta.pandas_helpers.row_by_row_params(ta.test_data.create_df_2(), expected_df_1())
        + ta.pandas_helpers.index_preserved_params(
            ta.test_data.create_df_2(), expected_df_1()
        ),
    )
    def test_expected_output_1(self, df, expected):
        """Test that transform is giving the expected output when applied to float column."""

        x1 = BaseImputer()
        x1.columns = ["a"]
        x1.impute_values_ = {"a": 7}

        df_transformed = x1.transform(df)

        ta.equality_helpers.assert_equal_dispatch(
            expected=expected,
            actual=df_transformed,
            msg="ArbitraryImputer transform col a",
        )

    @pytest.mark.parametrize(
        "df, expected",
        ta.pandas_helpers.row_by_row_params(ta.test_data.create_df_2(), expected_df_2())
        + ta.pandas_helpers.index_preserved_params(
            ta.test_data.create_df_2(), expected_df_2()
        ),
    )
    def test_expected_output_2(self, df, expected):
        """Test that transform is giving the expected output when applied to object column."""

        x1 = BaseImputer()
        x1.columns = ["b"]
        x1.impute_values_ = {"b": "g"}

        df_transformed = x1.transform(df)

        ta.equality_helpers.assert_equal_dispatch(
            expected=expected,
            actual=df_transformed,
            msg="ArbitraryImputer transform col b",
        )

    @pytest.mark.parametrize(
        "df, expected",
        ta.pandas_helpers.row_by_row_params(ta.test_data.create_df_2(), expected_df_3())
        + ta.pandas_helpers.index_preserved_params(
            ta.test_data.create_df_2(), expected_df_3()
        ),
    )
    def test_expected_output_3(self, df, expected):
        """Test that transform is giving the expected output when applied to object and categorical columns."""

        x1 = BaseImputer()
        x1.columns = ["b", "c"]
        x1.impute_values_ = {"b": "g", "c": "f"}

        df_transformed = x1.transform(df)

        ta.equality_helpers.assert_equal_dispatch(
            expected=expected,
            actual=df_transformed,
            msg="ArbitraryImputer transform col b, c",
        )

    def test_check_is_fitted_called(self, mocker):
        """Test that BaseTransformer check_is_fitted called."""

        df = ta.test_data.create_df_1()

        x = BaseImputer()
        x.columns = []

        expected_call_args = {0: {"args": (["impute_values_"],), "kwargs": {}}}

        with ta.function_helpers.assert_function_call(
            mocker, tubular.base.BaseTransformer, "check_is_fitted", expected_call_args
        ):

            x.transform(df)

    def test_super_transform_called(self, mocker):
        """Test that BaseImputer.transform called."""

        df = ta.test_data.create_df_2()

        x = BaseImputer()
        x.columns = []
        x.impute_values_ = {}

        expected_call_args = {0: {"args": (ta.test_data.create_df_2(),), "kwargs": {}}}

        with ta.function_helpers.assert_function_call(
            mocker, tubular.base.BaseTransformer, "transform", expected_call_args
        ):

            x.transform(df)
