import numpy as np
import pandas as pd
import pytest
import test_aide as ta

import tests.test_data as d
import tubular
from tubular.imputers import ModeImputer


class TestInit:
    """Tests for ModeImputer.init()."""

    def test_super_init_called(self, mocker):
        """Test that init calls BaseTransformer.init."""
        expected_call_args = {
            0: {"args": (), "kwargs": {"columns": None, "verbose": True, "copy": True}},
        }

        with ta.functions.assert_function_call(
            mocker,
            tubular.base.BaseTransformer,
            "__init__",
            expected_call_args,
        ):
            ModeImputer(columns=None, verbose=True, copy=True)

    @pytest.mark.parametrize("weight", (0, ["a"], {"a": 10}))
    def test_weight_arg_errors(self, weight):
        """Test that appropriate errors are thrown for bad weight arg."""
        with pytest.raises(
            ValueError,
            match="ModeImputer: weight should be a string or None",
        ):
            ModeImputer(columns=None, weight=weight)


class TestFit:
    """Tests for ModeImputer.fit()."""

    def test_super_fit_called(self, mocker):
        """Test that fit calls BaseTransformer.fit."""
        df = d.create_df_3()

        x = ModeImputer(columns=["a", "b", "c"])

        expected_call_args = {0: {"args": (d.create_df_3(), None), "kwargs": {}}}

        with ta.functions.assert_function_call(
            mocker,
            tubular.base.BaseTransformer,
            "fit",
            expected_call_args,
        ):
            x.fit(df)

    def test_check_weights_column_called(self, mocker):
        """Test that fit calls BaseTransformer.check_weights_column - when weights are used."""
        df = d.create_df_9()

        x = ModeImputer(columns=["a", "b"], weight="c")

        expected_call_args = {0: {"args": (d.create_df_9(), "c"), "kwargs": {}}}

        with ta.functions.assert_function_call(
            mocker,
            tubular.base.BaseTransformer,
            "check_weights_column",
            expected_call_args,
        ):
            x.fit(df)

    def test_learnt_values(self):
        """Test that the impute values learnt during fit are expected."""
        df = d.create_df_3()

        x = ModeImputer(columns=["a", "b", "c"])

        x.fit(df)

        ta.classes.test_object_attributes(
            obj=x,
            expected_attributes={
                "impute_values_": {
                    "a": df["a"].mode()[0],
                    "b": df["b"].mode()[0],
                    "c": df["c"].mode()[0],
                },
            },
            msg="impute_values_ attribute",
        )

    def test_learnt_values_weighted_df(self):
        """Test that the impute values learnt during fit are expected when df is weighted."""
        df = d.create_weighted_imputers_test_df()

        x = ModeImputer(columns=["a", "b", "c", "d"], weight="weight")

        x.fit(df)

        ta.classes.test_object_attributes(
            obj=x,
            expected_attributes={
                "impute_values_": {
                    "a": np.float64(5.0),
                    "b": "e",
                    "c": "f",
                    "d": np.float64(1.0),
                },
            },
            msg="impute_values_ attribute",
        )

    def test_fit_returns_self(self):
        """Test fit returns self?."""
        df = d.create_df_1()

        x = ModeImputer(columns="a")

        x_fitted = x.fit(df)

        assert x_fitted is x, "Returned value from ModeImputer.fit not as expected."

    def test_fit_returns_self_weighted(self):
        """Test fit returns self?."""
        df = d.create_df_9()

        x = ModeImputer(columns="a", weight="c")

        x_fitted = x.fit(df)

        assert x_fitted is x, "Returned value from ModeImputer.fit not as expected."

    def test_fit_not_changing_data(self):
        """Test fit does not change X."""
        df = d.create_df_1()

        x = ModeImputer(columns="a")

        x.fit(df)

        ta.equality.assert_equal_dispatch(
            expected=d.create_df_1(),
            actual=df,
            msg="Check X not changing during fit",
        )

    def test_fit_not_changing_data_weighted(self):
        """Test fit does not change X - when weights are used."""
        df = d.create_df_9()

        x = ModeImputer(columns="a", weight="c")

        x.fit(df)

        ta.equality.assert_equal_dispatch(
            expected=d.create_df_9(),
            actual=df,
            msg="Check X not changing during fit",
        )

    def expected_df_nan():
        return pd.DataFrame({"a": ["NaN", "NaN", "NaN"], "b": [None, None, None]})

    @pytest.mark.parametrize(
        ("df", "expected"),
        ta.pandas.row_by_row_params(
            pd.DataFrame({"a": [np.nan, np.nan, np.nan], "b": [None, None, None]}),
            expected_df_nan(),
        )
        + ta.pandas.index_preserved_params(
            pd.DataFrame({"a": [np.nan, np.nan, np.nan], "b": [None, None, None]}),
            expected_df_nan(),
        ),
    )
    def test_warning_mode_is_nan(self, df, expected):
        """Test that warning is raised when mode is NaN."""
        x = ModeImputer(columns=["a", "b"])

        with pytest.warns(Warning, match="ModeImputer: The Mode of column a is NaN."):
            x.fit(df)

        with pytest.warns(Warning, match="ModeImputer: The Mode of column b is NaN."):
            x.fit(df)


class TestTransform:
    """Tests for ModeImputer.transform()."""

    def expected_df_1():
        """Expected output for test_nulls_imputed_correctly."""
        df = pd.DataFrame(
            {
                "a": [1, 2, 3, 4, 5, 6, np.nan],
                "b": [1, 2, 3, np.nan, 7, 8, 9],
                "c": [np.nan, 1, 2, 3, -4, -5, -6],
            },
        )

        for col in ["a", "b", "c"]:
            df[col].loc[df[col].isna()] = df[col].mode()[0]

        return df

    def expected_df_2():
        """Expected output for test_nulls_imputed_correctly_2."""
        df = pd.DataFrame(
            {
                "a": [1, 2, 3, 4, 5, 6, np.nan],
                "b": [1, 2, 3, np.nan, 7, 8, 9],
                "c": [np.nan, 1, 2, 3, -4, -5, -6],
            },
        )

        for col in ["a"]:
            df[col].loc[df[col].isna()] = df[col].mode()[0]

        return df

    def expected_df_3():
        """Expected output for test_nulls_imputed_correctly_3."""
        df = d.create_df_9()

        for col in ["a"]:
            df[col].loc[df[col].isna()] = 6

        return df

    def test_check_is_fitted_called(self, mocker):
        """Test that BaseTransformer check_is_fitted called."""
        df = d.create_df_1()

        x = ModeImputer(columns="a")

        x.fit(df)

        expected_call_args = {0: {"args": (["impute_values_"],), "kwargs": {}}}

        with ta.functions.assert_function_call(
            mocker,
            tubular.base.BaseTransformer,
            "check_is_fitted",
            expected_call_args,
        ):
            x.transform(df)

    def test_super_transform_called(self, mocker):
        """Test that BaseTransformer.transform called."""
        df = d.create_df_1()

        x = ModeImputer(columns="a")

        x.fit(df)

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
        ta.pandas.adjusted_dataframe_params(d.create_df_3(), expected_df_1()),
    )
    def test_nulls_imputed_correctly(self, df, expected):
        """Test missing values are filled with the correct values."""
        x = ModeImputer(columns=["a", "b", "c"])

        # set the impute values dict directly rather than fitting x on df so test works with helpers
        x.impute_values_ = {"a": 1.0, "b": 1.0, "c": -6.0}

        df_transformed = x.transform(df)

        ta.equality.assert_equal_dispatch(
            expected=expected,
            actual=df_transformed,
            msg="Check nulls filled correctly in transform",
        )

    @pytest.mark.parametrize(
        ("df", "expected"),
        ta.pandas.adjusted_dataframe_params(d.create_df_3(), expected_df_2()),
    )
    def test_nulls_imputed_correctly_2(self, df, expected):
        """Test missing values are filled with the correct values - and unrelated columns are not changed."""
        x = ModeImputer(columns=["a"])

        # set the impute values dict directly rather than fitting x on df so test works with helpers
        x.impute_values_ = {"a": 1.0}

        df_transformed = x.transform(df)

        ta.equality.assert_equal_dispatch(
            expected=expected,
            actual=df_transformed,
            msg="Check nulls filled correctly in transform",
        )

    @pytest.mark.parametrize(
        ("df", "expected"),
        ta.pandas.row_by_row_params(d.create_df_9(), expected_df_3())
        + ta.pandas.index_preserved_params(d.create_df_9(), expected_df_3()),
    )
    def test_nulls_imputed_correctly_3(self, df, expected):
        """Test missing values are filled with the correct values - and unrelated columns are not changed
        (when weight is used).
        """
        x = ModeImputer(columns=["a"], weight="c")

        # set the impute values dict directly rather than fitting x on df so test works with helpers
        x.impute_values_ = {"a": 6}

        df_transformed = x.transform(df)

        ta.equality.assert_equal_dispatch(
            expected=expected,
            actual=df_transformed,
            msg="Check nulls filled correctly in transform",
        )

    def test_learnt_values_not_modified(self):
        """Test that the impute_values_ from fit are not changed in transform."""
        df = d.create_df_3()

        x = ModeImputer(columns=["a", "b", "c"])

        x.fit(df)

        x2 = ModeImputer(columns=["a", "b", "c"])

        x2.fit_transform(df)

        ta.equality.assert_equal_dispatch(
            expected=x.impute_values_,
            actual=x2.impute_values_,
            msg="Impute values not changed in transform",
        )

    def test_learnt_values_not_modified_weights(self):
        """Test that the impute_values_ from fit are not changed in transform - when using weights."""
        df = d.create_df_9()

        x = ModeImputer(columns=["a", "b"], weight="c")

        x.fit(df)

        x2 = ModeImputer(columns=["a", "b"], weight="c")

        x2.fit_transform(df)

        ta.equality.assert_equal_dispatch(
            expected=x.impute_values_,
            actual=x2.impute_values_,
            msg="Impute values not changed in transform",
        )
