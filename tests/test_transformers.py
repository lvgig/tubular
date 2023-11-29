# tests to apply to all transformers
import pytest
import sklearn.base as b
import pkgutil
from pathlib import Path
from importlib import import_module
import inspect
import test_aide as ta

import tubular.base as base
# import tubular.capping as capping
# import tubular.comparison as comparison
# import tubular.dates as dates
# import tubular.imputers as imputers
# import tubular.mapping as mapping
# import tubular.misc as misc
# import tubular.nominal as nominal
# import tubular.numeric as numeric
# import tubular.strings as strings

#move all of this to conftest.py
root = str(Path(__file__).parent.parent)

print()

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

minimal_attribute_dict = {
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
        'columns': ['a'], 
        },
    'GroupRareLevelsTransformer': {
        'columns': ['a'],
        },
    'MeanResponseTransformer': {
        'columns': ['a']
        },
    'OrdinalEncoderTransformer': {
        'columns': ['a']
        },
    'OneHotEncodingTransformer': {
        'columns': ['a'],
        },
    'LogTransformer': {
        'columns': ['a'],
        },
    'CutTransformer': {
        'new_column_name': 'b',
        'column': 'a',
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
        'columns': ['a'],
        },
    'InteractionTransformer': {
        'columns': ['a', 'b'],
        },
    'SeriesStrMethodTransformer': {
        'columns': ['b'],
        'new_column_name': 'a',
        'pd_method_name': 'find',
        },
    'StringConcatenator': {
        'columns': ['a', 'b'],
        'new_column': 'c',
        }
    }

instantiated_transformers = { x[0] : x[1](**minimal_attribute_dict[x[0]]) for x in all_classes} 


class TestInit:
    """Tests for transformer.init()."""

    # def ListOfTransformers():
    #     """List of transformers in tubular to be used in subsequent tests."""
    #     return [
    #         base.BaseTransformer(columns=["a"]),
    #         base.DataFrameMethodTransformer(
    #             new_column_name="a",
    #             pd_method_name="sum",
    #             columns="b",
    #         ),
    #         capping.CappingTransformer(capping_values={"a": [0.1, 0.2]}),
    #         capping.OutOfRangeNullTransformer(capping_values={"a": [0.1, 0.2]}),
    #         comparison.EqualityChecker(columns=["a", "b"], new_col_name="c"),
    #         dates.DateDiffLeapYearTransformer(
    #             column_lower="a",
    #             column_upper="b",
    #             new_column_name="c",
    #             drop_cols=True,
    #         ),
    #         dates.DateDifferenceTransformer(
    #             column_lower="a",
    #             column_upper="b",
    #             new_column_name="c",
    #             units="D",
    #         ),
    #         dates.ToDatetimeTransformer(column="a", new_column_name="b"),
    #         dates.DatetimeInfoExtractor(columns="a"),
    #         dates.SeriesDtMethodTransformer(
    #             new_column_name="a",
    #             pd_method_name="month",
    #             column="b",
    #         ),
    #         dates.BetweenDatesTransformer(
    #             column_lower="a",
    #             column_upper="b",
    #             column_between="c",
    #             new_column_name="c",
    #         ),
    #         dates.DatetimeSinusoidCalculator(
    #             "a",
    #             "sin",
    #             "month",
    #             12,
    #         ),
    #         imputers.BaseImputer(),
    #         imputers.ArbitraryImputer(impute_value=1, columns="a"),
    #         imputers.MedianImputer(columns="a"),
    #         imputers.MeanImputer(columns="a"),
    #         imputers.ModeImputer(columns="a"),
    #         imputers.NearestMeanResponseImputer(columns="a"),
    #         imputers.NullIndicator(columns="a"),
    #         mapping.BaseMappingTransformer(mappings={"a": {1: 2, 3: 4}}),
    #         mapping.BaseMappingTransformMixin(),
    #         mapping.MappingTransformer(mappings={"a": {1: 2, 3: 4}}),
    #         mapping.CrossColumnMappingTransformer(
    #             adjust_column="b",
    #             mappings={"a": {1: 2, 3: 4}},
    #         ),
    #         mapping.CrossColumnMultiplyTransformer(
    #             adjust_column="b",
    #             mappings={"a": {1: 2, 3: 4}},
    #         ),
    #         mapping.CrossColumnAddTransformer(
    #             adjust_column="b",
    #             mappings={"a": {1: 2, 3: 4}},
    #         ),
    #         misc.SetValueTransformer(columns="a", value=1),
    #         misc.SetColumnDtype(columns="a", dtype=str),
    #         nominal.BaseNominalTransformer(),
    #         nominal.NominalToIntegerTransformer(columns="a"),
    #         nominal.GroupRareLevelsTransformer(columns="a"),
    #         nominal.MeanResponseTransformer(columns="a"),
    #         nominal.OrdinalEncoderTransformer(columns="a"),
    #         nominal.OneHotEncodingTransformer(columns="a"),
    #         numeric.LogTransformer(columns="a"),
    #         numeric.CutTransformer(column="a", new_column_name="b"),
    #         numeric.TwoColumnOperatorTransformer(
    #             pd_method_name="add",
    #             columns=["a", "b"],
    #             new_column_name="c",
    #         ),
    #         numeric.ScalingTransformer(columns="a", scaler_type="standard"),
    #         numeric.PCATransformer(columns="a"),
    #         numeric.InteractionTransformer(columns=["a","b"]),
    #         strings.SeriesStrMethodTransformer(
    #             new_column_name="a",
    #             pd_method_name="find",
    #             columns="b",
    #             pd_method_kwargs={"sub": "a"},
    #         ),
    #         strings.StringConcatenator(columns=["a", "b"], new_column="c"),
    #     ]

    @pytest.mark.parametrize("transformer", instantiated_transformers)
    def test_print(self, transformer):
        """Test that transformer can be printed.
        If an error is raised in this test it will not prevent the transformer from working correctly,
        but will stop other unit tests passing.
        """
        print(transformer)

    @pytest.mark.parametrize("transformer", instantiated_transformers)
    def test_clone(self, transformer):
        """Test that transformer can be used in sklearn.base.clone function."""
        b.clone(transformer)

    # @pytest.mark.parametrize("transformer", ListOfTransformers())
    # def test_unexpected_kwarg(self, transformer):
    #     """Test that transformer can be used in sklearn.base.clone function."""
    #     b.clone(transformer)
    

    # EXAMPLE TEST ATTRIBUTES TEST -- CAN'T BE AS THOROUGH
    @pytest.mark.parametrize("name_transformer_pair", all_classes)   
    def test_attributes_set_from_passed_values(self, name_transformer_pair):
        """Test attributes set from values passed in init have the correct values."""
        
        transformer_name = name_transformer_pair[0]
        transformer = name_transformer_pair[1]
        
        expected_attributes = minimal_attribute_dict[transformer_name]
        # expected_attributes = minimal_attributes.copy()
        # expected_attributes["verbose"] = True

        # if "copy" in vars(instantiated_transformers[transformer_name]).keys():
        #     expected_attributes["copy"] = False

        x = transformer(**expected_attributes)

        ta.classes.test_object_attributes(
            obj=x,
            expected_attributes=expected_attributes,
            msg="Attributes set in init from passed values",
        )

    @pytest.mark.parametrize("name_transformer_pair", all_classes)   
    def test_columns_str_to_list(self, name_transformer_pair):
        """Test columns is converted to list if passed as string."""

        transformer_name = name_transformer_pair[0]
        transformer = name_transformer_pair[1]

        columns_only_as_list_transformers = [
            'EqualityChecker',
            'InteractionTransformer',
            'TwoColumnOperatorTransformer',
        ]
        
        # Some transformers generate columns of their own accord, and do not take it as an argument
        # Some transformers can only take columns as a list, as they always operate on multiple columns
        # We want to exclued both of these
        if 'columns' in inspect.getfullargspec(transformer).args and transformer_name not in columns_only_as_list_transformers:
            minimal_attributes = minimal_attribute_dict[transformer_name]
            expected_attributes = minimal_attributes.copy()


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