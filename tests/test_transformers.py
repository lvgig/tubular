# tests to apply to all transformers
import inspect
import pkgutil
import re
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import sklearn.base as b

import tests.test_data as d
import tubular.base as base

str_lst_classes_to_test = [
    "BaseTransformer",
    "DataFrameMethodTransformer",
]


def get_all_classes():
    root = str(Path(__file__).parent.parent)

    all_classes = []
    modules_to_ignore = [
        "tests",
        "conftest",
        "setup",
    ]

    for _importer, modname, _ispkg in pkgutil.walk_packages(
        path=[root],
    ):
        mod_parts = modname.split(".")
        if any(part in modules_to_ignore for part in mod_parts) or "_" in modname:
            continue
        module = import_module(modname)
        classes = inspect.getmembers(module, inspect.isclass)

        classes = [
            (name, transformer)
            for name, transformer in classes
            if issubclass(transformer, base.BaseTransformer)
        ]

        all_classes.extend(classes)

    return set(all_classes)


# replace this function with get_all_classes later
def get_classes_to_test():

    all_classes = get_all_classes()
    classes_to_test = []

    for name_transformer_pair in all_classes:
        if name_transformer_pair[0] in str_lst_classes_to_test:
            classes_to_test.append(name_transformer_pair)

    return classes_to_test


@pytest.fixture(scope="module")
def minimal_attribute_dict():
    return {
        "BaseTransformer": {
            "columns": ["a"],
        },
        "DataFrameMethodTransformer": {
            "columns": ["a", "c"],
            "new_column_names": "f",
            "pd_method_name": "sum",
        },
        "CappingTransformer": {
            "capping_values": {"a": [0.1, 0.2]},
        },
        "OutOfRangeNullTransformer": {
            "capping_values": {"a": [0.1, 0.2]},
        },
        "EqualityChecker": {
            "columns": ["a", "b"],
            "new_col_name": "c",
        },
        "DateDiffLeapYearTransformer": {
            "column_lower": "a",
            "column_upper": "b",
            "new_column_name": "c",
            "drop_cols": False,
        },
        "DateDifferenceTransformer": {
            "column_lower": "a",
            "column_upper": "b",
        },
        "ToDatetimeTransformer": {
            "new_column_name": "b",
            "column": "a",
        },
        "DatetimeInfoExtractor": {
            "columns": ["a"],
        },
        "SeriesDtMethodTransformer": {
            "new_column_name": "a",
            "pd_method_name": "month",
            "column": "b",
        },
        "BetweenDatesTransformer": {
            "new_column_name": "c",
            "column_lower": "a",
            "column_upper": "b",
            "column_between": "c",
        },
        "DatetimeSinusoidCalculator": {
            "columns": ["a"],
            "method": ["sin"],
            "units": "month",
        },
        "BaseImputer": {
            "columns": None,
        },
        "ArbitraryImputer": {
            "columns": ["a"],
            "impute_value": 1,
        },
        "MedianImputer": {
            "columns": ["a"],
        },
        "MeanImputer": {
            "columns": ["a"],
        },
        "ModeImputer": {
            "columns": ["a"],
        },
        "NearestMeanResponseImputer": {
            "columns": ["a"],
        },
        "NullIndicator": {
            "columns": ["a"],
        },
        "BaseMappingTransformer": {
            "mappings": {"a": {1: 2, 3: 4}},
        },
        "BaseMappingTransformMixin": {
            "columns": None,
        },
        "MappingTransformer": {
            "mappings": {"a": {1: 2, 3: 4}},
        },
        "CrossColumnMappingTransformer": {
            "mappings": {"a": {1: 2, 3: 4}},
            "adjust_column": "b",
        },
        "CrossColumnMultiplyTransformer": {
            "mappings": {"a": {1: 2, 3: 4}},
            "adjust_column": "b",
        },
        "CrossColumnAddTransformer": {
            "mappings": {"a": {1: 2, 3: 4}},
            "adjust_column": "b",
        },
        "SetValueTransformer": {
            "value": 1,
            "columns": ["a"],
        },
        "SetColumnDtype": {
            "columns": ["a"],
            "dtype": str,
        },
        "BaseNominalTransformer": {
            "columns": None,
        },
        "NominalToIntegerTransformer": {
            "columns": ["b"],
        },
        "GroupRareLevelsTransformer": {
            "columns": ["b"],
        },
        "MeanResponseTransformer": {
            "columns": ["b"],
        },
        "OrdinalEncoderTransformer": {
            "columns": ["b"],
        },
        "OneHotEncodingTransformer": {
            "columns": ["c"],
        },
        "LogTransformer": {
            "columns": ["a"],
        },
        "CutTransformer": {
            "new_column_name": "b",
            "column": "a",
            "cut_kwargs": {
                "bins": 3,
            },
        },
        "TwoColumnOperatorTransformer": {
            "columns": ["a", "b"],
            "new_column_name": "c",
            "pd_method_name": "add",
        },
        "ScalingTransformer": {
            "scaler_type": "standard",
            "columns": ["a"],
        },
        "PCATransformer": {
            "columns": ["a", "c"],
        },
        "InteractionTransformer": {
            "columns": ["a", "b"],
        },
        "SeriesStrMethodTransformer": {
            "columns": ["b"],
            "new_column_name": "a",
            "pd_method_name": "split",
        },
        "StringConcatenator": {
            "columns": ["a", "b"],
            "new_column": "c",
        },
    }


@pytest.fixture(scope="module")
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
        # TODO remove once arguments are made more consistent across the package
        if "verbose" in inspect.getfullargspec(transformer).args:
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


class TestColumnsCheck:
    """Tests for columns_check method."""

    @pytest.mark.parametrize("transformer_name, transformer", get_classes_to_test())
    def test_non_pd_df_error(
        self,
        transformer_name,
        transformer,
        minimal_attribute_dict,
    ):
        """Test an error is raised if X is not passed as a pd.DataFrame."""

        x = transformer(**minimal_attribute_dict[transformer_name])

        with pytest.raises(
            TypeError,
            match=f"{transformer_name}: X should be a pd.DataFrame",
        ):
            x.columns_check(X=[1, 2, 3, 4, 5, 6])

    @pytest.mark.parametrize("non_list", [1, True, {"a": 1}, None, "True"])
    @pytest.mark.parametrize("transformer_name, transformer", get_classes_to_test())
    def test_columns_not_list_error(
        self,
        transformer_name,
        transformer,
        non_list,
        minimal_attribute_dict,
    ):
        """Test an error is raised if self.columns is not a list."""
        df = d.create_df_1()

        x = transformer(**minimal_attribute_dict[transformer_name])

        x.columns = non_list

        with pytest.raises(
            TypeError,
            match=f"{transformer_name}: self.columns should be a list",
        ):
            x.columns_check(X=df)

    @pytest.mark.parametrize("transformer_name, transformer", get_classes_to_test())
    def test_columns_not_in_X_error(
        self,
        transformer_name,
        transformer,
        minimal_attribute_dict,
    ):
        """Test an error is raised if self.columns contains a value not in X."""
        df = d.create_df_1()

        args = minimal_attribute_dict[transformer_name]
        args["columns"] = ["a", "z"]

        x = transformer(**args)

        with pytest.raises(ValueError):
            x.columns_check(X=df)


class TestCombineXy:
    """Tests for the BaseTransformer._combine_X_y method."""

    @pytest.mark.parametrize("non_df", [1, True, "a", [1, 2], {"a": 1}, None])
    @pytest.mark.parametrize("transformer_name, transformer", get_classes_to_test())
    def test_X_not_DataFrame_error(
        self,
        transformer_name,
        transformer,
        non_df,
        minimal_attribute_dict,
    ):
        """Test an exception is raised if X is not a pd.DataFrame."""

        x = transformer(**minimal_attribute_dict[transformer_name])

        with pytest.raises(
            TypeError,
            match=f"{transformer_name}: X should be a pd.DataFrame",
        ):
            x._combine_X_y(X=non_df, y=pd.Series([1, 2]))

    @pytest.mark.parametrize("non_series", [1, True, "a", [1, 2], {"a": 1}])
    @pytest.mark.parametrize("transformer_name, transformer", get_classes_to_test())
    def test_y_not_Series_error(
        self,
        transformer_name,
        transformer,
        non_series,
        minimal_attribute_dict,
    ):
        """Test an exception is raised if y is not a pd.Series."""

        x = transformer(**minimal_attribute_dict[transformer_name])

        with pytest.raises(
            TypeError,
            match=f"{transformer_name}: y should be a pd.Series",
        ):
            x._combine_X_y(X=pd.DataFrame({"a": [1, 2]}), y=non_series)

    @pytest.mark.parametrize("transformer_name, transformer", get_classes_to_test())
    def test_X_and_y_different_number_of_rows_error(
        self,
        transformer_name,
        transformer,
        minimal_attribute_dict,
    ):
        """Test an exception is raised if X and y have different numbers of rows."""

        x = transformer(**minimal_attribute_dict[transformer_name])

        with pytest.raises(
            ValueError,
            match=re.escape(
                f"{transformer_name}: X and y have different numbers of rows (2 vs 1)",
            ),
        ):
            x._combine_X_y(X=pd.DataFrame({"a": [1, 2]}), y=pd.Series([2]))

    @pytest.mark.parametrize("transformer_name, transformer", get_classes_to_test())
    def test_X_and_y_different_indexes_warning(
        self,
        transformer_name,
        transformer,
        minimal_attribute_dict,
    ):
        """Test a warning is raised if X and y have different indexes, but the output is still X and y."""

        x = transformer(**minimal_attribute_dict[transformer_name])

        with pytest.warns(
            UserWarning,
            match=f"{transformer_name}: X and y do not have equal indexes",
        ):
            x._combine_X_y(
                X=pd.DataFrame({"a": [1, 2]}, index=[1, 2]),
                y=pd.Series([2, 4]),
            )

    @pytest.mark.parametrize("transformer_name, transformer", get_classes_to_test())
    def test_output_same_indexes(
        self,
        transformer_name,
        transformer,
        minimal_attribute_dict,
    ):
        """Test output is correct if X and y have the same index."""
        x = transformer(**minimal_attribute_dict[transformer_name])

        result = x._combine_X_y(
            X=pd.DataFrame({"a": [1, 2]}, index=[1, 2]),
            y=pd.Series([2, 4], index=[1, 2]),
        )

        expected_output = pd.DataFrame(
            {"a": [1, 2], "_temporary_response": [2, 4]},
            index=[1, 2],
        )

        pd.testing.assert_frame_equal(result, expected_output)


class TestCheckWeightsColumn:
    "tests for check_weights_column method."

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
                pd.DataFrame({"a": [1, 2], "b": [np.NaN, 0]}),
                r"weight column must be non-null",
                "b",
            ),
        ]

    @pytest.mark.parametrize("df, error, col", get_df_error_combos())
    @pytest.mark.parametrize("name_transformer_pair", get_classes_to_test())
    def test_weight_not_in_X_error(self, df, error, col, name_transformer_pair):
        """Test an error is raised if weight is not in X."""

        with pytest.raises(
            ValueError,
            match=error,
        ):
            name_transformer_pair[1].check_weights_column(df, col)
