import pytest
import test_aide as ta
import tests.test_data as d
import pandas as pd
import numpy as np

import tubular
from tubular.imputers import NearestMeanResponseImputer


class TestInit(object):
    """Tests for NearestMeanResponseImputer.__init__"""

    def test_arguments(self):
        """Test that init has expected arguments."""

        ta.function_helpers.test_function_arguments(
            func=NearestMeanResponseImputer.__init__,
            expected_arguments=["self", "response_column", "columns"],
            expected_default_values=(None,),
        )

    def test_class_methods(self):
        """Test that NearestMeanResponseImputer has fit and transform methods."""

        x = NearestMeanResponseImputer(response_column="c", columns=None)

        ta.class_helpers.test_object_method(obj=x, expected_method="fit", msg="fit")

        ta.class_helpers.test_object_method(
            obj=x, expected_method="transform", msg="transform"
        )

    def test_inheritance(self):
        """Test that NearestMeanResponseImputer inherits from BaseImputer."""

        x = NearestMeanResponseImputer(response_column="c", columns=None)

        ta.class_helpers.assert_inheritance(x, tubular.imputers.BaseImputer)

    def test_super_init_called(self, mocker):
        """Test that init calls BaseTransformer.init."""

        expected_call_args = {
            0: {"args": (), "kwargs": {"columns": None, "verbose": True, "copy": True}}
        }

        with ta.function_helpers.assert_function_call(
            mocker, tubular.base.BaseTransformer, "__init__", expected_call_args
        ):

            NearestMeanResponseImputer(
                response_column="c",
                columns=None,
                verbose=True,
                copy=True,
            )

    def test_response_column_not_str_error(self):
        """Test that an exception is raised if response_column is not str"""

        with pytest.raises(TypeError, match="response_column must be a str"):

            NearestMeanResponseImputer(response_column=0)

    def test_values_passed_in_init_set_to_attribute(self):
        """Test that the values passed in init are saved in an attribute of the same name."""

        x = NearestMeanResponseImputer(
            response_column="c",
            columns="a",
        )

        ta.class_helpers.test_object_attributes(
            obj=x,
            expected_attributes={"response_column": "c"},
            msg="Attributes for NearestMeanResponseImputer set in init",
        )


class TestFit(object):
    """Tests for NearestMeanResponseImputer.fit"""

    def test_arguments(self):
        """Test that fit has expected arguments."""

        ta.function_helpers.test_function_arguments(
            func=NearestMeanResponseImputer.fit,
            expected_arguments=["self", "X", "y"],
            expected_default_values=(None,),
        )

    def test_super_fit_called(self, mocker):
        """Test that fit calls BaseTransformer.fit."""

        df = d.create_NearestMeanResponseImputer_test_df()

        x = NearestMeanResponseImputer(response_column="c", columns=["a", "b"])

        expected_call_args = {
            0: {
                "args": (
                    d.create_NearestMeanResponseImputer_test_df(),
                    None,
                ),
                "kwargs": {},
            }
        }

        with ta.function_helpers.assert_function_call(
            mocker, tubular.base.BaseTransformer, "fit", expected_call_args
        ):

            x.fit(df)

    def test_non_numeric_response_column_error(self):
        """Test an error is raised if response_column is non-numeric"""

        df = pd.DataFrame(
            {"a": [1, 2, 3, 4, 5], "b": [5, 4, 3, 2, 1], "c": ["a", "b", "c", "d", "e"]}
        )

        x = NearestMeanResponseImputer(response_column="c", columns=["a", "b"])

        with pytest.raises(
            ValueError, match="dtypes in response_column must be numeric."
        ):

            x.fit(df)

    def test_null_values_in_response_error(self):
        """Test an error is raised if the response column contains null entries."""

        df = d.create_df_3()

        x = NearestMeanResponseImputer(response_column="c", columns=["a", "b"])

        with pytest.raises(ValueError, match=r"Response column \(c\) has null values."):

            x.fit(df)

    def test_columns_with_no_nulls_error(self):
        """Test an error is raised if a non-response column contains no nulls."""

        df = pd.DataFrame(
            {"a": [1, 2, 3, 4, 5], "b": [5, 4, 3, 2, 1], "c": [3, 2, 1, 4, 5]}
        )

        x = NearestMeanResponseImputer(response_column="c", columns=["a", "b"])

        with pytest.raises(
            ValueError,
            match="Column a has no missing values, cannot use this transformer.",
        ):

            x.fit(df)

    def test_fit_returns_self(self):
        """Test fit returns self?"""

        df = d.create_NearestMeanResponseImputer_test_df()

        x = NearestMeanResponseImputer(response_column="c", columns=["a", "b"])

        x_fitted = x.fit(df)

        assert (
            x_fitted is x
        ), "Returned value from NearestMeanResponseImputer.fit not as expected."

    def test_fit_not_changing_data(self):
        """Test fit does not change X."""

        df = d.create_NearestMeanResponseImputer_test_df()

        x = NearestMeanResponseImputer(response_column="c", columns=["a", "b"])

        x.fit(df)

        ta.equality_helpers.assert_equal_dispatch(
            expected=d.create_NearestMeanResponseImputer_test_df(),
            actual=df,
            msg="Check X not changing during fit",
        )

    def test_learnt_values(self):
        """Test that the nearest response values learnt during fit are expected."""

        df = d.create_NearestMeanResponseImputer_test_df()

        x = NearestMeanResponseImputer(response_column="c", columns=["a", "b"])

        x.fit(df)

        ta.class_helpers.test_object_attributes(
            obj=x,
            expected_attributes={
                "impute_values_": {"a": np.float64(2), "b": np.float64(3)}
            },
            msg="impute_values_ attribute",
        )

    def test_learnt_values2(self):
        """Test that the nearest mean response values learnt during fit are expected"""

        df = pd.DataFrame(
            {
                "a": [1, 1, np.nan, np.nan, 3, 5],
                "b": [np.nan, np.nan, 1, 3, 3, 4],
                "c": [2, 3, 2, 1, 4, 1],
            }
        )

        x = NearestMeanResponseImputer(response_column="c", columns=["a", "b"])

        x.fit(df)

        ta.class_helpers.test_object_attributes(
            obj=x,
            expected_attributes={
                "impute_values_": {"a": np.float64(5), "b": np.float64(3)}
            },
            msg="impute_values_ attribute",
        )


class TestTransform(object):
    """Tests for NearestMeanResponseImputer.transform"""

    def expected_df_1():
        """Expected output for ."""

        df = pd.DataFrame(
            {"a": [1, 1, 2, 3, 3, 2], "b": [3, 3, 1, 3, 3, 4], "c": [2, 3, 2, 1, 4, 1]}
        )

        df[["a", "b"]] = df[["a", "b"]].astype("float64")

        return df

    def expected_df_2():
        """Expected output for ."""

        df = pd.DataFrame(
            {
                "a": [1, 1, 2, 3, 3, 2],
                "b": [np.nan, np.nan, 1, 3, 3, 4],
                "c": [2, 3, 2, 1, 4, 1],
            }
        )

        df["a"] = df["a"].astype("float64")

        return df

    def expected_df_3():
        """Expected output for ."""

        df = pd.DataFrame({"a": [2, 3, 4, 1, 4, 2]})

        df["a"] = df["a"].astype("float64")

        return df

    def test_arguments(self):
        """Test that transform has expected arguments."""

        ta.function_helpers.test_function_arguments(
            func=NearestMeanResponseImputer.transform, expected_arguments=["self", "X"]
        )

    def test_check_is_fitted_called(self, mocker):
        """Test that BaseTransformer check_is_fitted called."""

        df = d.create_NearestMeanResponseImputer_test_df()

        x = NearestMeanResponseImputer(response_column="c", columns=["a", "b"])

        x.fit(df)

        expected_call_args = {0: {"args": (["impute_values_"],), "kwargs": {}}}

        with ta.function_helpers.assert_function_call(
            mocker, tubular.base.BaseTransformer, "check_is_fitted", expected_call_args
        ):

            x.transform(df)

    def test_super_transform_called(self, mocker):
        """Test that BaseTransformer.transform called."""

        df = d.create_NearestMeanResponseImputer_test_df()

        x = NearestMeanResponseImputer(response_column="c", columns=["a", "b"])

        x.fit(df)

        expected_call_args = {
            0: {
                "args": (d.create_NearestMeanResponseImputer_test_df(),),
                "kwargs": {},
            }
        }

        with ta.function_helpers.assert_function_call(
            mocker, tubular.base.BaseTransformer, "transform", expected_call_args
        ):

            x.transform(df)

    @pytest.mark.parametrize(
        "df, expected",
        ta.pandas_helpers.adjusted_dataframe_params(
            d.create_NearestMeanResponseImputer_test_df(), expected_df_1()
        ),
    )
    def test_nulls_imputed_correctly(self, df, expected):
        """Test missing values are filled with the correct values."""

        x = NearestMeanResponseImputer(response_column="c", columns=["a", "b"])

        # set the impute values dict directly rather than fitting x on df so test works with helpers
        x.impute_values_ = {"a": 2.0, "b": 3.0}

        df_transformed = x.transform(df)

        ta.equality_helpers.assert_equal_dispatch(
            expected=expected,
            actual=df_transformed,
            msg="Check nulls filled correctly in transform",
        )

    @pytest.mark.parametrize(
        "df, expected",
        ta.pandas_helpers.adjusted_dataframe_params(
            d.create_NearestMeanResponseImputer_test_df(), expected_df_2()
        ),
    )
    def test_nulls_imputed_correctly2(self, df, expected):
        """Test missing values are filled with the correct values - and unrelated columns are unchanged."""

        x = NearestMeanResponseImputer(response_column="c", columns="a")

        # set the impute values dict directly rather than fitting x on df so test works with helpers
        x.impute_values_ = {"a": 2.0}

        df_transformed = x.transform(df)

        ta.equality_helpers.assert_equal_dispatch(
            expected=expected,
            actual=df_transformed,
            msg="Check nulls filled correctly in transform",
        )

    @pytest.mark.parametrize(
        "df, expected",
        ta.pandas_helpers.adjusted_dataframe_params(
            pd.DataFrame({"a": [np.nan, 3, 4, 1, 4, np.nan]}), expected_df_3()
        ),
    )
    def test_nulls_imputed_correctly3(self, df, expected):
        """Test missing values are filled with the correct values - with median value from separate dataframe."""

        x = NearestMeanResponseImputer(response_column="c", columns="a")

        # set the impute values dict directly rather than fitting x on df so test works with helpers
        x.impute_values_ = {"a": 2.0}

        df_transformed = x.transform(df)

        ta.equality_helpers.assert_equal_dispatch(
            expected=expected,
            actual=df_transformed,
            msg="Check nulls filled correctly in transform",
        )

    def test_learnt_values_not_modified(self):
        """Test that the impute_values_ from fit are not changed in transform."""

        df = d.create_NearestMeanResponseImputer_test_df()

        x = NearestMeanResponseImputer(response_column="c", columns=["a", "b"])

        x.fit(df)

        x2 = NearestMeanResponseImputer(response_column="c", columns=["a", "b"])

        x2.fit(df)

        x2.transform(df)

        ta.equality_helpers.assert_equal_dispatch(
            expected=x.impute_values_,
            actual=x2.impute_values_,
            msg="Impute values not changed in transform",
        )
