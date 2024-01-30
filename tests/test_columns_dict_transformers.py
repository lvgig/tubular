# tests to apply to all transformers where columns is read from a dictionary
import re

import numpy as np
import pandas as pd
import pytest
import sklearn.base as b

import tests.test_data as d
from tests.test_transformers import get_all_classes

columns_dict_classes_to_test = [
    "CappingTransformer",
]


def get_classes_to_test():

    all_classes = get_all_classes()
    classes_to_test = []

    for name_transformer_pair in all_classes:
        if name_transformer_pair[0] in columns_dict_classes_to_test:
            classes_to_test.append(name_transformer_pair)

    return classes_to_test


@pytest.fixture()
def instantiated_transformers(minimal_attribute_dict):
    classes_to_test = get_classes_to_test()
    return {x[0]: x[1](**minimal_attribute_dict[x[0]]) for x in classes_to_test}


class TestInit:
    """Generic tests for transformer.init()."""

    @pytest.mark.parametrize("name_transformer_pair", get_classes_to_test())
    def test_print(self, name_transformer_pair, instantiated_transformers):
        """Test that transformer can be printed.
        If an error is raised in this test it will not prevent the transformer from working correctly,
        but will stop other unit tests passing.
        """
        print(instantiated_transformers[name_transformer_pair[0]])

    @pytest.mark.parametrize("name_transformer_pair", get_classes_to_test())
    def test_clone(self, name_transformer_pair, instantiated_transformers):
        """Test that transformer can be used in sklearn.base.clone function."""
        b.clone(instantiated_transformers[name_transformer_pair[0]])

    @pytest.mark.parametrize("non_bool", [1, "True", {"a": 1}, [1, 2], None])
    @pytest.mark.parametrize("transformer_name, transformer", get_classes_to_test())
    def test_verbose_non_bool_error(
        self,
        transformer_name,
        transformer,
        non_bool,
        minimal_attribute_dict,
    ):
        """Test an error is raised if verbose is not specified as a bool."""

        with pytest.raises(
            TypeError,
            match=f"{transformer_name}: verbose must be a bool",
        ):
            transformer(
                verbose=non_bool,
                **minimal_attribute_dict[transformer_name],
            )

    @pytest.mark.parametrize("non_bool", [1, "True", {"a": 1}, [1, 2], None])
    @pytest.mark.parametrize("transformer_name, transformer", get_classes_to_test())
    def test_copy_non_bool_error(
        self,
        transformer_name,
        transformer,
        non_bool,
        minimal_attribute_dict,
    ):
        """Test an error is raised if copy is not specified as a bool."""

        with pytest.raises(TypeError, match=f"{transformer_name}: copy must be a bool"):
            transformer(copy=non_bool, **minimal_attribute_dict[transformer_name])

    @pytest.mark.parametrize("transformer_name, transformer", get_classes_to_test())
    def test_columns_empty_list_error(
        self,
        transformer_name,
        transformer,
        minimal_attribute_dict,
    ):
        """Test an error is raised if columns is specified as an empty list."""

        args = minimal_attribute_dict[transformer_name].copy()
        args["columns"] = []

        with pytest.raises(ValueError):
            transformer(**args)

    @pytest.mark.parametrize("non_string", [1, True, {"a": 1}, [1, 2], None])
    @pytest.mark.parametrize("transformer_name, transformer", get_classes_to_test())
    def test_columns_list_element_error(
        self,
        transformer_name,
        transformer,
        non_string,
        minimal_attribute_dict,
    ):
        """Test an error is raised if columns list contains non-string elements."""

        args = minimal_attribute_dict[transformer_name].copy()
        args["columns"] = [non_string, non_string]

        with pytest.raises(
            TypeError,
            match=re.escape(
                f"{transformer_name}: each element of columns should be a single (string) column name",
            ),
        ):
            transformer(**args)

    @pytest.mark.parametrize("non_string_or_list", [1, True, {"a": 1}, None])
    @pytest.mark.parametrize("transformer_name, transformer", get_classes_to_test())
    def test_columns_non_string_error(
        self,
        transformer_name,
        transformer,
        non_string_or_list,
        minimal_attribute_dict,
    ):
        """Test an error is raised if columns is not passed as a string or list."""

        args = minimal_attribute_dict[transformer_name].copy()
        args["columns"] = non_string_or_list

        with pytest.raises(
            TypeError,
            match=re.escape(
                f"{transformer_name}: columns must be a string or list with the columns to be pre-processed (if specified)",
            ),
        ):
            transformer(**args)


class TestFit:
    """Generic tests for transfromer.fit()"""

    @pytest.mark.parametrize("transformer_name, transformer", get_classes_to_test())
    def test_fit_returns_self(
        self,
        transformer_name,
        transformer,
        minimal_attribute_dict,
    ):
        """Test fit returns self?."""

        df = d.create_numeric_df_1()

        x = transformer(**minimal_attribute_dict[transformer_name])

        x_fitted = x.fit(df, df["c"])

        assert (
            x_fitted is x
        ), f"Returned value from {transformer_name}.fit not as expected."

    @pytest.mark.parametrize("non_df", [1, True, "a", [1, 2], {"a": 1}, None])
    @pytest.mark.parametrize("transformer_name, transformer", get_classes_to_test())
    def test_X_non_df_error(
        self,
        transformer_name,
        transformer,
        non_df,
        minimal_attribute_dict,
    ):
        """Test an error is raised if X is not passed as a pd.DataFrame."""

        df = d.create_numeric_df_1()

        x = transformer(**minimal_attribute_dict[transformer_name])

        with pytest.raises(
            TypeError,
            match=f"{transformer_name}: X should be a pd.DataFrame",
        ):
            x.fit(non_df, df["a"])

    @pytest.mark.parametrize("non_series", [1, True, "a", [1, 2], {"a": 1}])
    @pytest.mark.parametrize("transformer_name, transformer", get_classes_to_test())
    def test_non_pd_type_error(
        self,
        transformer_name,
        transformer,
        non_series,
        minimal_attribute_dict,
    ):
        """Test an error is raised if y is not passed as a pd.Series."""

        df = d.create_numeric_df_1()

        x = transformer(**minimal_attribute_dict[transformer_name])

        with pytest.raises(
            TypeError,
            match=f"{transformer_name}: unexpected type for y, should be a pd.Series",
        ):
            x.fit(X=df, y=non_series)

    @pytest.mark.parametrize("transformer_name, transformer", get_classes_to_test())
    def test_X_no_rows_error(
        self,
        transformer_name,
        transformer,
        minimal_attribute_dict,
    ):
        """Test an error is raised if X has no rows."""

        x = transformer(**minimal_attribute_dict[transformer_name])

        df = pd.DataFrame(columns=["a", "b", "c"])

        with pytest.raises(
            ValueError,
            match=re.escape(f"{transformer_name}: X has no rows; (0, 3)"),
        ):
            x.fit(df, df["a"])

    @pytest.mark.parametrize("transformer_name, transformer", get_classes_to_test())
    def test_Y_no_rows_error(
        self,
        transformer_name,
        transformer,
        minimal_attribute_dict,
    ):
        """Test an error is raised if Y has no rows."""

        x = transformer(**minimal_attribute_dict[transformer_name])

        df = pd.DataFrame({"a": 1, "b": "wow", "c": np.nan}, index=[0])

        with pytest.raises(
            ValueError,
            match=re.escape(f"{transformer_name}: y is empty; (0,)"),
        ):
            x.fit(X=df, y=pd.Series(name="d", dtype=object))

    @pytest.mark.parametrize("transformer_name, transformer", get_classes_to_test())
    def test_unexpected_kwarg_error(
        self,
        transformer_name,
        transformer,
        minimal_attribute_dict,
    ):
        with pytest.raises(
            TypeError,
            match=re.escape(
                "__init__() got an unexpected keyword argument 'unexpected_kwarg'",
            ),
        ):
            transformer(
                unexpected_kwarg="spanish inquisition",
                **minimal_attribute_dict[transformer_name],
            )


class TestTransform:
    """Generic tests for transformer.transform()."""

    @pytest.mark.parametrize("non_df", [1, True, "a", [1, 2], {"a": 1}, None])
    @pytest.mark.parametrize("transformer_name, transformer", get_classes_to_test())
    def test_non_pd_type_error(
        self,
        transformer_name,
        transformer,
        non_df,
        minimal_attribute_dict,
    ):
        """Test that an error is raised in transform is X is not a pd.DataFrame."""

        df = d.create_df_10()

        x = transformer(**minimal_attribute_dict[transformer_name])

        x_fitted = x.fit(df, df["c"])

        with pytest.raises(
            TypeError,
            match=f"{transformer_name}: X should be a pd.DataFrame",
        ):
            x_fitted.transform(X=non_df)

    @pytest.mark.parametrize("transformer_name, transformer", get_classes_to_test())
    def test_copy_returned(self, transformer_name, transformer, minimal_attribute_dict):
        """Test check that a copy is returned if copy is set to True"""
        df = d.create_df_10()

        x = transformer(copy=True, **minimal_attribute_dict[transformer_name])

        x = x.fit(df, df["c"])

        df_transformed = x.transform(df)

        assert df_transformed is not df

    @pytest.mark.parametrize("transformer_name, transformer", get_classes_to_test())
    def test_no_rows_error(self, transformer_name, transformer, minimal_attribute_dict):
        """Test an error is raised if X has no rows."""
        df = d.create_df_10()

        x = transformer(**minimal_attribute_dict[transformer_name])

        x = x.fit(df, df["c"])

        df = pd.DataFrame(columns=["a", "b", "c"])

        with pytest.raises(
            ValueError,
            match=re.escape(f"{transformer_name}: X has no rows; (0, 3)"),
        ):
            x.transform(df)
