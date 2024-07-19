import numpy as np
import pandas as pd

# import pytest
# import test_aide as ta
# import tests.test_data as d
from tests.base_tests import (
    ColumnStrListInitTests,
)


# Dataframe used exclusively in this testing script
def create_NearestMeanResponseImputer_test_df():
    """Create DataFrame to use in NearestMeanResponseImputer tests.

    DataFrame column c is the response, the other columns are numerical columns containing null entries.

    """
    return pd.DataFrame(
        {
            "a": [1, 1, 2, 3, 3, np.nan],
            "b": [np.nan, np.nan, 1, 3, 3, 4],
            "c": [2, 3, 2, 1, 4, 1],
        },
    )


class TestInit(ColumnStrListInitTests):
    """Generic tests for transformer.init()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "NearestMeanResponseImputer"


# class TestFit:
#     """Tests for NearestMeanResponseImputer.fit."""

#     def test_super_fit_called(self, mocker):
#         """Test that fit calls BaseTransformer.fit."""
#         df = create_NearestMeanResponseImputer_test_df()

#         x = NearestMeanResponseImputer(columns=["a", "b"])

#         expected_call_args = {
#             0: {
#                 "args": (
#                     create_NearestMeanResponseImputer_test_df(),
#                     create_NearestMeanResponseImputer_test_df()["c"],
#                 ),
#                 "kwargs": {},
#             },
#         }

#         with ta.functions.assert_function_call(
#             mocker,
#             tubular.base.BaseTransformer,
#             "fit",
#             expected_call_args,
#         ):
#             x.fit(df, df["c"])

#     def test_null_values_in_response_error(self):
#         """Test an error is raised if the response column contains null entries."""
#         df = d.create_df_3()

#         x = NearestMeanResponseImputer(columns=["a", "b"])

#         with pytest.raises(
#             ValueError,
#             match="NearestMeanResponseImputer: y has 1 null values",
#         ):
#             x.fit(df, df["c"])

#     def test_columns_with_no_nulls_error(self):
#         """Test an error is raised if a non-response column contains no nulls."""
#         df = pd.DataFrame(
#             {"a": [1, 2, 3, 4, 5], "b": [5, 4, 3, 2, 1], "c": [3, 2, 1, 4, 5]},
#         )

#         x = NearestMeanResponseImputer(columns=["a", "b"])

#         with pytest.raises(
#             ValueError,
#             match="NearestMeanResponseImputer: Column a has no missing values, cannot use this transformer.",
#         ):
#             x.fit(df, df["c"])

#     def test_fit_returns_self(self):
#         """Test fit returns self?."""
#         df = create_NearestMeanResponseImputer_test_df()

#         x = NearestMeanResponseImputer(columns=["a", "b"])

#         x_fitted = x.fit(df, df["c"])

#         assert (
#             x_fitted is x
#         ), "Returned value from NearestMeanResponseImputer.fit not as expected."

#     def test_fit_not_changing_data(self):
#         """Test fit does not change X."""
#         df = create_NearestMeanResponseImputer_test_df()

#         x = NearestMeanResponseImputer(columns=["a", "b"])

#         x.fit(df, df["c"])

#         ta.equality.assert_equal_dispatch(
#             expected=create_NearestMeanResponseImputer_test_df(),
#             actual=df,
#             msg="Check X not changing during fit",
#         )

#     def test_learnt_values(self):
#         """Test that the nearest response values learnt during fit are expected."""
#         df = create_NearestMeanResponseImputer_test_df()

#         x = NearestMeanResponseImputer(columns=["a", "b"])

#         x.fit(df, df["c"])

#         ta.classes.test_object_attributes(
#             obj=x,
#             expected_attributes={
#                 "impute_values_": {"a": np.float64(2), "b": np.float64(3)},
#             },
#             msg="impute_values_ attribute",
#         )

#     def test_learnt_values2(self):
#         """Test that the nearest mean response values learnt during fit are expected."""
#         df = pd.DataFrame(
#             {
#                 "a": [1, 1, np.nan, np.nan, 3, 5],
#                 "b": [np.nan, np.nan, 1, 3, 3, 4],
#                 "c": [2, 3, 2, 1, 4, 1],
#             },
#         )

#         x = NearestMeanResponseImputer(columns=["a", "b"])

#         x.fit(df, df["c"])

#         ta.classes.test_object_attributes(
#             obj=x,
#             expected_attributes={
#                 "impute_values_": {"a": np.float64(5), "b": np.float64(3)},
#             },
#             msg="impute_values_ attribute",
#         )


# class TestTransform:
#     """Tests for NearestMeanResponseImputer.transform."""

#     def expected_df_1():
#         """Expected output for ."""
#         df = pd.DataFrame(
#             {"a": [1, 1, 2, 3, 3, 2], "b": [3, 3, 1, 3, 3, 4], "c": [2, 3, 2, 1, 4, 1]},
#         )

#         df[["a", "b"]] = df[["a", "b"]].astype("float64")

#         return df

#     def expected_df_2():
#         """Expected output for ."""
#         df = pd.DataFrame(
#             {
#                 "a": [1, 1, 2, 3, 3, 2],
#                 "b": [np.nan, np.nan, 1, 3, 3, 4],
#                 "c": [2, 3, 2, 1, 4, 1],
#             },
#         )

#         df["a"] = df["a"].astype("float64")

#         return df

#     def expected_df_3():
#         """Expected output for ."""
#         df = pd.DataFrame({"a": [2, 3, 4, 1, 4, 2]})

#         df["a"] = df["a"].astype("float64")

#         return df

#     def test_check_is_fitted_called(self, mocker):
#         """Test that BaseTransformer check_is_fitted called."""
#         df = create_NearestMeanResponseImputer_test_df()

#         x = NearestMeanResponseImputer(columns=["a", "b"])

#         x.fit(df, df["c"])

#         expected_call_args = {0: {"args": (["impute_values_"],), "kwargs": {}}}

#         with ta.functions.assert_function_call(
#             mocker,
#             tubular.base.BaseTransformer,
#             "check_is_fitted",
#             expected_call_args,
#         ):
#             x.transform(df)

#     def test_super_transform_called(self, mocker):
#         """Test that BaseTransformer.transform called."""
#         df = create_NearestMeanResponseImputer_test_df()

#         x = NearestMeanResponseImputer(columns=["a", "b"])

#         x.fit(df, df["c"])

#         expected_call_args = {
#             0: {"args": (create_NearestMeanResponseImputer_test_df(),), "kwargs": {}},
#         }

#         with ta.functions.assert_function_call(
#             mocker,
#             tubular.base.BaseTransformer,
#             "transform",
#             expected_call_args,
#         ):
#             x.transform(df)

#     @pytest.mark.parametrize(
#         ("df", "expected"),
#         ta.pandas.adjusted_dataframe_params(
#             create_NearestMeanResponseImputer_test_df(),
#             expected_df_1(),
#         ),
#     )
#     def test_nulls_imputed_correctly(self, df, expected):
#         """Test missing values are filled with the correct values."""
#         x = NearestMeanResponseImputer(columns=["a", "b"])

#         # set the impute values dict directly rather than fitting x on df so test works with helpers
#         x.impute_values_ = {"a": 2.0, "b": 3.0}

#         df_transformed = x.transform(df)

#         ta.equality.assert_equal_dispatch(
#             expected=expected,
#             actual=df_transformed,
#             msg="Check nulls filled correctly in transform",
#         )

#     @pytest.mark.parametrize(
#         ("df", "expected"),
#         ta.pandas.adjusted_dataframe_params(
#             create_NearestMeanResponseImputer_test_df(),
#             expected_df_2(),
#         ),
#     )
#     def test_nulls_imputed_correctly2(self, df, expected):
#         """Test missing values are filled with the correct values - and unrelated columns are unchanged."""
#         x = NearestMeanResponseImputer(columns="a")

#         # set the impute values dict directly rather than fitting x on df so test works with helpers
#         x.impute_values_ = {"a": 2.0}

#         df_transformed = x.transform(df)

#         ta.equality.assert_equal_dispatch(
#             expected=expected,
#             actual=df_transformed,
#             msg="Check nulls filled correctly in transform",
#         )

#     @pytest.mark.parametrize(
#         ("df", "expected"),
#         ta.pandas.adjusted_dataframe_params(
#             pd.DataFrame({"a": [np.nan, 3, 4, 1, 4, np.nan]}),
#             expected_df_3(),
#         ),
#     )
#     def test_nulls_imputed_correctly3(self, df, expected):
#         """Test missing values are filled with the correct values - with median value from separate dataframe."""
#         x = NearestMeanResponseImputer(columns="a")

#         # set the impute values dict directly rather than fitting x on df so test works with helpers
#         x.impute_values_ = {"a": 2.0}

#         df_transformed = x.transform(df)

#         ta.equality.assert_equal_dispatch(
#             expected=expected,
#             actual=df_transformed,
#             msg="Check nulls filled correctly in transform",
#         )

#     def test_learnt_values_not_modified(self):
#         """Test that the impute_values_ from fit are not changed in transform."""
#         df = create_NearestMeanResponseImputer_test_df()

#         x = NearestMeanResponseImputer(columns=["a", "b"])

#         x.fit(df, df["c"])

#         x2 = NearestMeanResponseImputer(columns=["a", "b"])

#         x2.fit(df, df["c"])

#         x2.transform(df)

#         ta.equality.assert_equal_dispatch(
#             expected=x.impute_values_,
#             actual=x2.impute_values_,
#             msg="Impute values not changed in transform",
#         )
