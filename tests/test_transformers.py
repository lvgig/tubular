# tests to apply to all transformers
import pytest
from unittest import mock
import sklearn.base as b
import pkgutil
from pathlib import Path
from importlib import import_module
import inspect
import test_aide as ta
import re
import pandas as pd
import numpy as np
import tubular.base as base
import tests.test_data as d


def get_all_classes():
    root = str(Path(__file__).parent.parent)

    all_classes = []
    modules_to_ignore = [
        "tests",
        "conftest",
        "setup",
    ]

    for importer, modname, ispkg in pkgutil.walk_packages(
        path = [root], #prefix = "tubular.tubular."
        ):
        mod_parts = modname.split(".")
        if any(part in modules_to_ignore for part in mod_parts) or "_" in modname:
            continue
        module = import_module(modname)
        classes = inspect.getmembers(module, inspect.isclass)
        classes = [
            (name, transformer) for name, transformer in classes if issubclass(transformer, base.BaseTransformer)
        ] 

        all_classes.extend(classes)

    all_classes = set(all_classes)
    
    return all_classes

@pytest.fixture(scope = 'module')
def minimal_attribute_dict():
    return {
    'BaseTransformer': {
        'columns': ['a']
        },
    'DataFrameMethodTransformer': {
        'columns': ['b'],
        'new_column_name': 'a',
        'pd_method_name': 'sum',
        },
    'CappingTransformer': {
        'capping_values':{"a": [0.1, 0.2]}
        },
    'OutOfRangeNullTransformer': {
        'capping_values':{"a": [0.1, 0.2]}
        },
    'EqualityChecker': {
        'columns': ['a', 'b'],
        'new_col_name': 'c',
        },
    'DateDiffLeapYearTransformer': {
        'column_lower': 'a',
        'column_upper': 'b',
        'new_column_name' : 'c',
        'drop_cols' : False,
        },
    'DateDifferenceTransformer': {
        'column_lower': 'a',
        'column_upper': 'b'
        },
    'ToDatetimeTransformer': {
        'new_column_name': 'b',
        'column': 'a',
        },
    'DatetimeInfoExtractor': {
        'columns': ['a'],
        },
    'SeriesDtMethodTransformer': {
        'new_column_name': 'a',
        'pd_method_name': 'month',
        'column': 'b'
        },
    'BetweenDatesTransformer': {
        'new_column_name': 'c',
        'column_lower': 'a',
        'column_upper': 'b',
        'column_between': 'c'
        },
    'DatetimeSinusoidCalculator': {
        'columns': ['a'],
        'method': ['sin'],
        'units': 'month',
        },
    'BaseImputer': {
        'columns' : None,
        },
    'ArbitraryImputer': {
        'columns': ['a'],
        'impute_value': 1
        },
    'MedianImputer': {
        'columns': ['a'],
        },
    'MeanImputer': {'columns': ['a'],
                    },
    'ModeImputer': {'columns': ['a'],
                    },
    'NearestMeanResponseImputer': {'columns': ['a']
                                   },
    'NullIndicator': {'columns': ['a']
                      },
    'BaseMappingTransformer': {
        'mappings': {'a': {1: 2, 3: 4}}, 
        },
    'BaseMappingTransformMixin': {
        'columns' : None,
        },
    'MappingTransformer': {
        'mappings': {'a': {1: 2, 3: 4}}, 
        },
    'CrossColumnMappingTransformer': {
        'mappings': {'a': {1: 2, 3: 4}},
        'adjust_column': 'b'
        },
    'CrossColumnMultiplyTransformer': {
        'mappings': {'a': {1: 2, 3: 4}},
        'adjust_column': 'b'
        },
    'CrossColumnAddTransformer': {
        'mappings': {'a': {1: 2, 3: 4}},
        'adjust_column': 'b'
        },
    'SetValueTransformer': {
        'value': 1, 
        'columns': ['a']
        },
    'SetColumnDtype': {
        'columns': ['a'], 
        'dtype': str
        },
    'BaseNominalTransformer': {
        'columns' : None,
        },
    'NominalToIntegerTransformer': {
        'columns': ['b'], 
        },
    'GroupRareLevelsTransformer': {
        'columns': ['b'],
        },
    'MeanResponseTransformer': {
        'columns': ['b']
        },
    'OrdinalEncoderTransformer': {
        'columns': ['b']
        },
    'OneHotEncodingTransformer': {
        'columns': ['c'],
        },
    'LogTransformer': {
        'columns': ['a'],
        },
    #TODO possible to instantiate this with a combinatio pd_method_name and pd_method_kwargs that will fail later
    #e.g. no kwargs
    'CutTransformer': {
        'new_column_name': 'b',
        'column': 'a',
        'cut_kwargs': {
            'bins' : 3
        }
        },
    'TwoColumnOperatorTransformer': {
        'columns': ['a', 'b'],
        'new_column_name': 'c',
        'pd_method_name': 'add',
        },
    'ScalingTransformer': {
        'scaler_type': 'standard',
        'columns': ['a']
        },
    'PCATransformer': {
        'columns': ['a', 'c'],
        },
    'InteractionTransformer': {
        'columns': ['a', 'b'],
        },
    #TODO possible to instantiate this with a combinatio pd_method_name and pd_method_kwargs that will fail later
    #e.g. find and none
    'SeriesStrMethodTransformer': {
        'columns': ['b'],
        'new_column_name': 'a',
        'pd_method_name': 'split',
        },
    'StringConcatenator': {
        'columns': ['a', 'b'],
        'new_column': 'c',
        }
    }


@pytest.fixture(scope = 'module')
def instantiated_transformers(minimal_attribute_dict):
    all_classes = get_all_classes()
    return { x[0] : x[1](**minimal_attribute_dict[x[0]]) for x in all_classes} 

@pytest.fixture(scope = 'module')
def columns_only_as_list_transformers():
    "Fixture containing transformers that will only accept the columns parameter as a list rather than list or string"
    return [
        'EqualityChecker',
        'InteractionTransformer',
        'TwoColumnOperatorTransformer',
    ]
@pytest.fixture(scope = 'module')
def date_data_transformers():
    "Fixture containing transformers that need date data for the transform method to work"
    return [
        'DateDiffLeapYearTransformer',
        'DateDifferenceTransformer',
        'DatetimeInfoExtractor',
        'SeriesDtMethodTransformer',
        'BetweenDatesTransformer',
        'DatetimeSinusoidCalculator',
    ]

@pytest.fixture(scope = 'module')
def numeric_data_transformers():
    "Fixture containing transformers that need numeric data for the transform method to work"
    return [
        'TwoColumnOperatorTransformer',
        'CrossColumnMultiplyTransformer',
        'CrossColumnAddTransformer',
    ]

@pytest.fixture(scope = 'module')
def nominal_encoding_transformers():
    "Fixture containing problem nominal transformers"
    return [
        'OrdinalEncoderTransformer',
        'NominalToIntegerTransformer',
        'MeanResponseTransformer',
    ]


class TestInit:
    """Tests for transformer.init()."""

    @pytest.mark.parametrize("name_transformer_pair", get_all_classes())   
    def test_print(self, name_transformer_pair, instantiated_transformers):
        """Test that transformer can be printed.
        If an error is raised in this test it will not prevent the transformer from working correctly,
        but will stop other unit tests passing.
        """
        print(instantiated_transformers[name_transformer_pair[0]])

    @pytest.mark.parametrize("name_transformer_pair", get_all_classes())   
    def test_clone(self, name_transformer_pair, instantiated_transformers):
        """Test that transformer can be used in sklearn.base.clone function."""
        b.clone(instantiated_transformers[name_transformer_pair[0]])

    # @pytest.mark.parametrize("transformer", ListOfTransformers())
    # def test_unexpected_kwarg(self, transformer):
    #     """Test that transformer can be used in sklearn.base.clone function."""
    #     b.clone(transformer)
    

    # EXAMPLE TEST ATTRIBUTES TEST -- CAN'T BE AS THOROUGH
    @pytest.mark.parametrize("transformer_name, transformer", get_all_classes())   
    def test_attributes_set_from_passed_values(self, transformer_name, transformer, minimal_attribute_dict):
        """Test attributes set from values passed in init have the correct values."""
        
        expected_attributes = minimal_attribute_dict[transformer_name]

        x = transformer(**expected_attributes)

        ta.classes.test_object_attributes(
            obj=x,
            expected_attributes=expected_attributes,
            msg="Attributes set in init from passed values",
        )

    @pytest.mark.parametrize("transformer_name, transformer", get_all_classes())   
    def test_columns_str_to_list(self, transformer_name, transformer, minimal_attribute_dict, columns_only_as_list_transformers):
        """Test columns is converted to list if passed as string. Transformers that can only take columns as a list (rather
        than list or string) should be added to the columns_only_as_list_transformers fixture so that they are skipped by this test"""
        
        # Some transformers generate columns of their own accord, and do not take it as an argument
        # Some transformers can only take columns as a list, as they always operate on multiple columns
        # We want to exclued both of these
        #TODO remove once arguments are made more consistent across the package
        if 'columns' in inspect.getfullargspec(transformer).args and transformer_name not in columns_only_as_list_transformers:
            minimal_attributes = minimal_attribute_dict[transformer_name].copy()
            expected_attributes = minimal_attributes


            minimal_attributes['columns'] = "a"
            if 'column' in minimal_attributes:
                del minimal_attributes['column']

            expected_attributes["columns"] = ["a"]
            
            x = transformer(**minimal_attributes)

            ta.classes.test_object_attributes(
                obj=x,
                expected_attributes=expected_attributes,
                msg="String put in list for columns",
            )

    @pytest.mark.parametrize("transformer_name, transformer", get_all_classes())  
    def test_verbose_non_bool_error(self, transformer_name, transformer, minimal_attribute_dict):
        """Test an error is raised if verbose is not specified as a bool."""
        #TODO remove once arguments are made more consistent across the package
        if 'verbose' in inspect.getfullargspec(transformer).args:
            with pytest.raises(TypeError, match=f"{transformer_name}: verbose must be a bool"):
                transformer(verbose=1, **minimal_attribute_dict[transformer_name])

    @pytest.mark.parametrize("transformer_name, transformer", get_all_classes())  
    def test_copy_non_bool_error(self, transformer_name, transformer, minimal_attribute_dict):
        """Test an error is raised if copy is not specified as a bool."""
        #TODO remove once arguments are made more consistent across the package
        if 'copy' in inspect.getfullargspec(transformer).args:
            with pytest.raises(TypeError, match=f"{transformer_name}: copy must be a bool"):
                transformer(copy=1, **minimal_attribute_dict[transformer_name])

    @pytest.mark.xfail(reason = "bug in TwoColumnOperatorTransformer lets empty columns list through") #TODO fix this bug
    @pytest.mark.parametrize("transformer_name, transformer", get_all_classes())  
    def test_columns_empty_list_error(self, transformer_name, transformer, minimal_attribute_dict):
        """Test an error is raised if columns is specified as an empty list."""

        # Some transformers generate columns of their own accord, and do not take it as an argument
        if 'columns' in inspect.getfullargspec(transformer).args:
            args = minimal_attribute_dict[transformer_name].copy()
            args['columns'] = []
            if 'column' in args:
                del args['column']

            with pytest.raises(ValueError):
                transformer(**args)

    @pytest.mark.parametrize("transformer_name, transformer", get_all_classes())  
    def test_columns_list_element_error(self, transformer_name, transformer, minimal_attribute_dict):
        """Test an error is raised if columns list contains non-string elements."""

        # Some transformers generate columns of their own accord, and do not take it as an argument
        # Series str method takes columns but only length 1
        if 'columns' in inspect.getfullargspec(transformer).args and transformer_name != 'SeriesStrMethodTransformer':
            
            args = minimal_attribute_dict[transformer_name].copy()
            args['columns'] = [[], "a"]
            if 'column' in args:
                del args['column']

            with pytest.raises(
                TypeError,
                match=re.escape(
                    f"{transformer_name}: each element of columns should be a single (string) column name",
                ),
            ):
                transformer(**args)

    @pytest.mark.xfail(reason = "bug in TwoColumnOperatorTransformer input checking") #TODO fix this bug
    @pytest.mark.parametrize("transformer_name, transformer", get_all_classes())  
    def test_columns_non_string_error(self, transformer_name, transformer, minimal_attribute_dict):
        """Test an error is raised if columns is not passed as a string."""

        # Some transformers generate columns of their own accord, and do not take it as an argument
        if 'columns' in inspect.getfullargspec(transformer).args:
            
            args = minimal_attribute_dict[transformer_name].copy()
            args['columns'] = 1
            if 'column' in args:
                del args['column']

            with pytest.raises(
                TypeError,
                match=re.escape(
                    f"{transformer_name}: columns must be a string or list with the columns to be pre-processed (if specified)",
                ),
            ):
                transformer(**args)

class TestFit:
    """Tests for BaseTransformer.fit()."""

    
    @pytest.mark.parametrize("transformer_name, transformer", get_all_classes())
    def test_fit_returns_self(self, transformer_name, transformer, minimal_attribute_dict):
        """Test fit returns self?."""
        
        #PCA can't take nans when fitting, other transformers will throw an error in the absence of nans
        if transformer_name == 'PCATransformer':
        
            df = d.create_numeric_df_1()

        else:

            df = d.create_df_10()

        x = transformer(**minimal_attribute_dict[transformer_name])

        x_fitted = x.fit(df, df["c"])

        
        
        assert x_fitted is x, f"Returned value from {transformer_name}.fit not as expected."            

    @pytest.mark.xfail(reason = "bug in BaseNominalTransformer: not raising this error") #TODO fix this bug
    @pytest.mark.parametrize("transformer_name, transformer", get_all_classes())
    def test_X_non_df_error(self, transformer_name, transformer, minimal_attribute_dict):
        """Test an error is raised if X is not passed as a pd.DataFrame."""
        
        df = d.create_numeric_df_1()

        x = transformer(**minimal_attribute_dict[transformer_name])

        with pytest.raises(
            TypeError,
            match=f"{transformer_name}: X should be a pd.DataFrame",
        ):
            x.fit("a", df["a"])

    @pytest.mark.parametrize("transformer_name, transformer", get_all_classes())
    def test_non_pd_type_error(self, transformer_name, transformer, minimal_attribute_dict):
        """Test an error is raised if y is not passed as a pd.Series."""
        
        df = d.create_df_12()

        x = transformer(**minimal_attribute_dict[transformer_name])

        with pytest.raises(
            TypeError,
            match=f"{transformer_name}: unexpected type for y, should be a pd.Series",
        ):
            x.fit(X=df, y=[1, 2, 3, 4, 5, 6])

    @pytest.mark.parametrize("transformer_name, transformer", get_all_classes())
    def test_columns_set_or_check_called(self, mocker, transformer_name, transformer, minimal_attribute_dict):
        """Test that self.columns_set_or_check is called during fit."""

        #PCA can't take nans when fitting, other transformers will throw an error in the absence of nans
        if transformer_name == 'PCATransformer':
        
            df = d.create_numeric_df_1()

        else:

            df = d.create_df_10()

        x = transformer(**minimal_attribute_dict[transformer_name])

        expected_call_args = {0: {"args": (df,), "kwargs": {}}}

        with ta.functions.assert_function_call(
            mocker,
            transformer,
            "columns_set_or_check",
            expected_call_args,
        ):
            x.fit(df, df["c"])

    @pytest.mark.parametrize("transformer_name, transformer", get_all_classes())
    def test_X_no_rows_error(self, transformer_name, transformer, minimal_attribute_dict):
        """Test an error is raised if X has no rows."""
        
        x = transformer(**minimal_attribute_dict[transformer_name])

        df = pd.DataFrame(columns=["a", "b", "c"])

        with pytest.raises(
            ValueError,
            match=re.escape(f"{transformer_name}: X has no rows; (0, 3)"),
        ):
            x.fit(df, df['a'])

    @pytest.mark.parametrize("transformer_name, transformer", get_all_classes())
    def test_Y_no_rows_error(self, transformer_name, transformer, minimal_attribute_dict):   
        """Test an error is raised if Y has no rows."""

        x = transformer(**minimal_attribute_dict[transformer_name])

        df = pd.DataFrame({"a": 1, "b" : "wow", "c" : np.nan}, index=[0])

        with pytest.raises(
            ValueError,
            match=re.escape(f"{transformer_name}: y is empty; (0,)"),
        ):
            x.fit(X=df, y=pd.Series(name="d", dtype=object))

    @pytest.mark.parametrize("transformer_name, transformer", get_all_classes())
    def test_unexpected_kwarg_error(self, transformer_name, transformer, minimal_attribute_dict):
        with pytest.raises(
            TypeError,
            match=re.escape(
                "__init__() got an unexpected keyword argument 'unexpected_kwarg'",
            ),
        ):
            transformer(unexpected_kwarg="spanish inquisition", **minimal_attribute_dict[transformer_name])

class TestTransform:
    """Tests for BaseTransformer.transform()."""

    #TODO fix this bug
    @pytest.mark.xfail(
            reason = """"
                       DateDiffLeapYearTransformer raises incorrect type error on datetime.datetime objects.
                       
                       The check on upper column values yields pandas._libs.tslibs.timestamps.Timestamp dtype, rather than datetime.
                       """
                       )
    @pytest.mark.parametrize("transformer_name, transformer", get_all_classes())
    def test_columns_check_called(self, mocker, transformer_name, transformer, minimal_attribute_dict, date_data_transformers, numeric_data_transformers, nominal_encoding_transformers):
        """Test that self.columns_check is called during transform."""
        
        # These base classes only have transform methods, so the init and fit methods come directly from BaseTransformer
        # Can't figure it out exactly but this means __sklearn_is_fitted__ != True
        if transformer_name not in ['BaseMappingTransformMixin', 'BaseImputer']:

            #PCA can't take nans when fitting, other transformers will throw an error in the absence of nans
            if transformer_name == 'PCATransformer' or transformer_name in numeric_data_transformers:
            
                df = d.create_numeric_df_1()

            elif transformer_name in date_data_transformers:

                df = d.create_is_between_dates_df_2()  

            elif transformer_name in nominal_encoding_transformers:     

                df = d.create_df_2()
                df['c'] = df['a'].fillna(2)
                # TODO this step highlights a bug in check_mappable_rows
                # An incorrect value error will be raised if nulls ever make it into the data to be mapped
                df['b'] = df['b'].fillna('a')

            else:

                df = d.create_df_10()

            x = transformer(**minimal_attribute_dict[transformer_name])

            df_for_fitting = df.copy()
            x_fitted = x.fit(df_for_fitting, df['c'])

            expected_call_args = {0: {"args": (df,), "kwargs": {}}}

            with ta.functions.assert_function_call(
                mocker,
                transformer,
                "columns_check",
                expected_call_args,
            ):
                x_fitted.transform(X=df)
    
    #TODO fix these bugs
    @pytest.mark.xfail(
            reason = """
                    This behaviour is not present in the following four transformers and needs fixing. All of them do 
                    something expecting a pd.DataFrame before checking to see if one is present.

                    MeanResponseTransformer, NominalToIntegerTransformer, OrdinalEncoderTransformer:
                        - happens when X gets to check_mappable_rows() in BaseNominalTransformer

                    MappingTransformer:
                        - happens when transform tries to extract original dtypes
                       """
                       )
    @pytest.mark.parametrize("transformer_name, transformer", get_all_classes())
    def test_non_pd_type_error(self, transformer_name, transformer, minimal_attribute_dict, date_data_transformers, numeric_data_transformers, nominal_encoding_transformers):
        """Test that self.columns_check is called during transform."""
        
        # These base classes only have transform methods, so the init and fit methods come directly from BaseTransformer
        # Can't figure it out exactly but this means __sklearn_is_fitted__ != True
        if transformer_name not in ['BaseMappingTransformMixin', 'BaseImputer']:

            #PCA can't take nans when fitting, other transformers will throw an error in the absence of nans
            if transformer_name == 'PCATransformer' or transformer_name in numeric_data_transformers:
            
                df = d.create_numeric_df_1()

            elif transformer_name in date_data_transformers:

                df = d.create_is_between_dates_df_2()  

            elif transformer_name in nominal_encoding_transformers:     

                df = d.create_df_2()
                df['c'] = df['a'].fillna(2)
                # TODO this step highlights a bug in check_mappable_rows
                # An incorrect value error will be raised if nulls ever make it into the data to be mapped
                df['b'] = df['b'].fillna('a')

            else:

                df = d.create_df_10()

            x = transformer(**minimal_attribute_dict[transformer_name])

            df_for_fitting = df.copy()
            x_fitted = x.fit(df_for_fitting, df['c'])

            with pytest.raises(
                TypeError,
                match=f"{transformer_name}: X should be a pd.DataFrame",
            ):
                x_fitted.transform(X=[1, 2, 3, 4, 5, 6])
