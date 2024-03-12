# tests to apply to all columns str or list transformers
import re

import numpy as np
import pandas as pd
import pytest
import sklearn.base as b

import tests.test_data as d


class GenericInitTests:
    """
    Generic tests for transformer.init(). This test class does not contain tests for the behaviours
    associated with the "columns" argument because the structure of this argument varies between
    transformers. In this file are other test classes that inherit from this one which are specific
    to the different "columns" argument structures. Please choose the appropriate one of them to inherit
    when writing tests unless the transformer is a special case and needs unique tests written for
    "columns", in which case inherit this class.

    Note this class name deliberately avoids starting with "Tests" so that the tests are not run on import.
    """

    def test_print(self, initialized_transformers):
        """Test that transformer can be printed.
        If an error is raised in this test it will not prevent the transformer from working correctly,
        but will stop other unit tests passing.
        """

        print(initialized_transformers[self.transformer_name])

    def test_clone(self, initialized_transformers):
        """Test that transformer can be used in sklearn.base.clone function."""

        b.clone(initialized_transformers[self.transformer_name])

    @pytest.mark.parametrize("non_bool", [1, "True", {"a": 1}, [1, 2], None])
    def test_verbose_non_bool_error(
        self,
        non_bool,
        minimal_attribute_dict,
        uninitialized_transformers,
    ):
        """Test an error is raised if verbose is not specified as a bool."""

        with pytest.raises(
            TypeError,
            match=f"{self.transformer_name}: verbose must be a bool",
        ):
            uninitialized_transformers[self.transformer_name](
                verbose=non_bool,
                **minimal_attribute_dict[self.transformer_name],
            )


class ColumnStrListInitTests(GenericInitTests):
    """
    Tests for BaseTransformer.init() behaviour specific to when a transformer takes columns as string or list.
    Note this deliberately avoids starting with "Tests" so that the tests are not run on import.
    """

    def test_columns_empty_list_error(
        self,
        minimal_attribute_dict,
        uninitialized_transformers,
    ):
        """Test an error is raised if columns is specified as an empty list."""

        args = minimal_attribute_dict[self.transformer_name].copy()
        args["columns"] = []

        with pytest.raises(ValueError):
            uninitialized_transformers[self.transformer_name](**args)

    @pytest.mark.parametrize("non_string", [1, True, {"a": 1}, [1, 2], None])
    def test_columns_list_element_error(
        self,
        non_string,
        minimal_attribute_dict,
        uninitialized_transformers,
    ):
        """Test an error is raised if columns list contains non-string elements."""

        args = minimal_attribute_dict[self.transformer_name].copy()
        args["columns"] = [non_string, non_string]

        with pytest.raises(
            TypeError,
            match=re.escape(
                f"{self.transformer_name}: each element of columns should be a single (string) column name",
            ),
        ):
            uninitialized_transformers[self.transformer_name](**args)

    @pytest.mark.parametrize("non_string_or_list", [1, True, {"a": 1}, None])
    def test_columns_non_string_or_list_error(
        self,
        non_string_or_list,
        minimal_attribute_dict,
        uninitialized_transformers,
    ):
        """Test an error is raised if columns is not passed as a string or list."""

        args = minimal_attribute_dict[self.transformer_name].copy()
        args["columns"] = non_string_or_list

        with pytest.raises(
            TypeError,
            match=re.escape(
                f"{self.transformer_name}: columns must be a string or list with the columns to be pre-processed (if specified)",
            ),
        ):
            uninitialized_transformers[self.transformer_name](**args)


class ColumnsFromDictInitTests(GenericInitTests):
    """
    Tests for BaseTransformer.init() behaviour specific to when a transformer reads columns from a dict.
    Note this deliberately avoids starting with "Tests" so that the tests are not run on import.
    """

    @pytest.mark.parametrize("non_string", [1, True, None])
    def test_columns_list_element_error(
        self,
        non_string,
        minimal_attribute_dict,
        uninitialized_transformers,
    ):
        """Test an error is raised if columns list contains non-string elements."""

        args = minimal_attribute_dict[self.transformer_name].copy()
        args["mappings"][non_string] = {1: 2, 3: 4}

        with pytest.raises(
            TypeError,
            match=re.escape(
                f"{self.transformer_name}: each element of columns should be a single (string) column name",
            ),
        ):
            uninitialized_transformers[self.transformer_name](**args)


class GenericFitTests:
    """
    Generic tests for transformer.fit().
    Note this deliberately avoids starting with "Tests" so that the tests are not run on import.
    """

    def test_fit_returns_self(
        self,
        initialized_transformers,
    ):
        """Test fit returns self?."""

        df = d.create_numeric_df_1()

        x = initialized_transformers[self.transformer_name]

        x_fitted = x.fit(df, df["c"])

        assert (
            x_fitted is x
        ), f"Returned value from {self.transformer_name}.fit not as expected."

    @pytest.mark.parametrize("non_df", [1, True, "a", [1, 2], {"a": 1}, None])
    def test_X_non_df_error(
        self,
        initialized_transformers,
        non_df,
    ):
        """Test an error is raised if X is not passed as a pd.DataFrame."""

        df = d.create_numeric_df_1()

        x = initialized_transformers[self.transformer_name]

        with pytest.raises(
            TypeError,
            match=f"{self.transformer_name}: X should be a pd.DataFrame",
        ):
            x.fit(non_df, df["a"])

    @pytest.mark.parametrize("non_series", [1, True, "a", [1, 2], {"a": 1}])
    def test_non_pd_type_error(
        self,
        non_series,
        initialized_transformers,
    ):
        """Test an error is raised if y is not passed as a pd.Series."""

        df = d.create_numeric_df_1()

        x = initialized_transformers[self.transformer_name]

        with pytest.raises(
            TypeError,
            match=f"{self.transformer_name}: unexpected type for y, should be a pd.Series",
        ):
            x.fit(X=df, y=non_series)

    def test_X_no_rows_error(
        self,
        initialized_transformers,
    ):
        """Test an error is raised if X has no rows."""

        x = initialized_transformers[self.transformer_name]

        df = pd.DataFrame(columns=["a", "b", "c"])

        with pytest.raises(
            ValueError,
            match=re.escape(f"{self.transformer_name}: X has no rows; (0, 3)"),
        ):
            x.fit(df, df["a"])

    def test_Y_no_rows_error(
        self,
        initialized_transformers,
    ):
        """Test an error is raised if Y has no rows."""

        x = initialized_transformers[self.transformer_name]

        df = pd.DataFrame({"a": 1, "b": "wow", "c": np.nan}, index=[0])

        with pytest.raises(
            ValueError,
            match=re.escape(f"{self.transformer_name}: y is empty; (0,)"),
        ):
            x.fit(X=df, y=pd.Series(name="d", dtype=object))

    def test_unexpected_kwarg_error(
        self,
        minimal_attribute_dict,
        uninitialized_transformers,
    ):
        with pytest.raises(
            TypeError,
            match=re.escape(
                "__init__() got an unexpected keyword argument 'unexpected_kwarg'",
            ),
        ):
            uninitialized_transformers[self.transformer_name](
                unexpected_kwarg="spanish inquisition",
                **minimal_attribute_dict[self.transformer_name],
            )


class GenericTransformTests:
    """
    Generic tests for transformer.transform().
    Note this deliberately avoids starting with "Tests" so that the tests are not run on import.
    """

    @pytest.mark.parametrize("non_df", [1, True, "a", [1, 2], {"a": 1}, None])
    def test_non_pd_type_error(
        self,
        non_df,
        initialized_transformers,
    ):
        """Test that an error is raised in transform is X is not a pd.DataFrame."""

        df = d.create_df_10()

        x = initialized_transformers[self.transformer_name]

        x_fitted = x.fit(df, df["c"])

        with pytest.raises(
            TypeError,
            match=f"{self.transformer_name}: X should be a pd.DataFrame",
        ):
            x_fitted.transform(X=non_df)

    def test_copy_returned(self, minimal_attribute_dict, uninitialized_transformers):
        """Test check that a copy is returned if copy is set to True"""
        df = d.create_df_10()

        x = uninitialized_transformers[self.transformer_name](
            copy=True,
            **minimal_attribute_dict[self.transformer_name],
        )

        x = x.fit(df, df["c"])

        df_transformed = x.transform(df)

        assert df_transformed is not df

    def test_no_rows_error(self, initialized_transformers):
        """Test an error is raised if X has no rows."""
        df = d.create_df_10()

        x = initialized_transformers[self.transformer_name]

        x = x.fit(df, df["c"])

        df = pd.DataFrame(columns=["a", "b", "c"])

        with pytest.raises(
            ValueError,
            match=re.escape(f"{self.transformer_name}: X has no rows; (0, 3)"),
        ):
            x.transform(df)


class ColumnsCheckTests:
    """
    Tests for columns_check method.
    Note this deliberately avoids starting with "Tests" so that the tests are not run on import.
    """

    def test_non_pd_df_error(
        self,
        initialized_transformers,
    ):
        """Test an error is raised if X is not passed as a pd.DataFrame."""

        x = initialized_transformers[self.transformer_name]

        with pytest.raises(
            TypeError,
            match=f"{self.transformer_name}: X should be a pd.DataFrame",
        ):
            x.columns_check(X=[1, 2, 3, 4, 5, 6])

    @pytest.mark.parametrize("non_list", [1, True, {"a": 1}, None, "True"])
    def test_columns_not_list_error(
        self,
        non_list,
        initialized_transformers,
    ):
        """Test an error is raised if self.columns is not a list."""
        df = d.create_df_1()

        x = initialized_transformers[self.transformer_name]

        x.columns = non_list

        with pytest.raises(
            TypeError,
            match=f"{self.transformer_name}: self.columns should be a list",
        ):
            x.columns_check(X=df)

    def test_columns_not_in_X_error(
        self,
        initialized_transformers,
    ):
        """Test an error is raised if self.columns contains a value not in X."""
        df = d.create_df_1()

        x = initialized_transformers[self.transformer_name]

        x.columns = ["a", "z"]

        with pytest.raises(ValueError):
            x.columns_check(X=df)


class CombineXYTests:
    """
    Tests for the BaseTransformer._combine_X_y method.
    Note this deliberately avoids starting with "Tests" so that the tests are not run on import.
    """

    @pytest.mark.parametrize("non_df", [1, True, "a", [1, 2], {"a": 1}, None])
    def test_X_not_DataFrame_error(
        self,
        non_df,
        initialized_transformers,
    ):
        """Test an exception is raised if X is not a pd.DataFrame."""

        x = initialized_transformers[self.transformer_name]

        with pytest.raises(
            TypeError,
            match=f"{self.transformer_name}: X should be a pd.DataFrame",
        ):
            x._combine_X_y(X=non_df, y=pd.Series([1, 2]))

    @pytest.mark.parametrize("non_series", [1, True, "a", [1, 2], {"a": 1}])
    def test_y_not_Series_error(
        self,
        non_series,
        initialized_transformers,
    ):
        """Test an exception is raised if y is not a pd.Series."""

        x = initialized_transformers[self.transformer_name]

        with pytest.raises(
            TypeError,
            match=f"{self.transformer_name}: y should be a pd.Series",
        ):
            x._combine_X_y(X=pd.DataFrame({"a": [1, 2]}), y=non_series)

    def test_X_and_y_different_number_of_rows_error(
        self,
        initialized_transformers,
    ):
        """Test an exception is raised if X and y have different numbers of rows."""

        x = initialized_transformers[self.transformer_name]

        with pytest.raises(
            ValueError,
            match=re.escape(
                f"{self.transformer_name}: X and y have different numbers of rows (2 vs 1)",
            ),
        ):
            x._combine_X_y(X=pd.DataFrame({"a": [1, 2]}), y=pd.Series([2]))

    def test_X_and_y_different_indexes_warning(
        self,
        initialized_transformers,
    ):
        """Test a warning is raised if X and y have different indexes, but the output is still X and y."""

        x = initialized_transformers[self.transformer_name]

        with pytest.warns(
            UserWarning,
            match=f"{self.transformer_name}: X and y do not have equal indexes",
        ):
            x._combine_X_y(
                X=pd.DataFrame({"a": [1, 2]}, index=[1, 2]),
                y=pd.Series([2, 4]),
            )

    def test_output_same_indexes(
        self,
        initialized_transformers,
    ):
        """Test output is correct if X and y have the same index."""
        x = initialized_transformers[self.transformer_name]

        result = x._combine_X_y(
            X=pd.DataFrame({"a": [1, 2]}, index=[1, 2]),
            y=pd.Series([2, 4], index=[1, 2]),
        )

        expected_output = pd.DataFrame(
            {"a": [1, 2], "_temporary_response": [2, 4]},
            index=[1, 2],
        )

        pd.testing.assert_frame_equal(result, expected_output)


class CheckWeightsColumnTests:
    """
    tests for check_weights_column method.
    Note this deliberately avoids starting with "Tests" so that the tests are not run on import.
    """

    def get_df_error_combos():
        return [
            (
                pd.DataFrame({"a": [1, 2], "b": [3, 4]}),
                r"weight col \(c\) is not present in columns of data",
                "c",
            ),
            (
                pd.DataFrame({"a": [1, 2], "b": ["a", "b"]}),
                r"weight column must be numeric.",
                "b",
            ),
            (
                pd.DataFrame({"a": [1, 2], "b": [-1, 0]}),
                r"weight column must be positive",
                "b",
            ),
            (
                pd.DataFrame({"a": [1, 2], "b": [np.nan, 0]}),
                r"weight column must be non-null",
                "b",
            ),
        ]

    @pytest.mark.parametrize("df, error, col", get_df_error_combos())
    def test_weight_not_in_X_error(self, df, error, col, uninitialized_transformers):
        """Test an error is raised if weight is not in X."""

        with pytest.raises(
            ValueError,
            match=error,
        ):
            uninitialized_transformers[self.transformer_name].check_weights_column(
                df,
                col,
            )


class OtherBaseBehaviourTests(
    ColumnsCheckTests,
    CheckWeightsColumnTests,
    CombineXYTests,
):
    """
    Class to collect and hold tests for BaseTransformerBehaviour outside the three standard methods.
    Note this deliberately avoids starting with "Tests" so that the tests are not run on import.
    """
