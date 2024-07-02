import inspect
import pkgutil
from importlib import import_module
from pathlib import Path

import pytest

import tubular.base as base

"""
How To Use This Testing Framework
-----------------------------------

Fixtures in this file are used in tests for shared behaviour which can be run on multiple transformers through inheritance.
setup_class is defined in each test class to define cls.transformer_name as the name of the transformer being tested.

Test methods for shared behaviour are defined in base_tests.py or child test classes which are themselves inherited.  The
fixtures in this file can be used in these tests to access initiated/ uninitiated transformers and minimum attributes for
intiatialization.

Test classes which can import tests for shared behaviour inherit from an appropriate parent and define cls.transformer_name
in setup_class so that these fixtures will retrieve the appropriate transformer instance. setup_class is called by pytest
at a class level before its test methods are run https://docs.pytest.org/en/8.0.x/how-to/xunit_setup.html.

New transformers will need to be added to minimal_attribute_dict.
 """


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


@pytest.fixture()
def minimal_attribute_dict():
    """defines minmal attributes (values) needed to initiate each transformer named (key).
    New transformers need to be added here"""
    return {
        "ArbitraryImputer": {
            "columns": ["a"],
            "impute_value": 1,
        },
        "BaseCappingTransformer": {
            "capping_values": {"a": [0.1, 0.2]},
        },
        "BaseCrossColumnMappingTransformer": {
            "mappings": {"a": {1: 2, 3: 4}},
            "adjust_column": "b",
        },
        "BaseCrossColumnNumericTransformer": {
            "mappings": {"a": {1: 2, 3: 4}},
            "adjust_column": "b",
        },
        "BaseDateTransformer": {
            "columns": ["a"],
            "new_column_name": "bla",
        },
        "BaseDateTwoColumnTransformer": {
            "columns": ["a", "b"],
            "new_column_name": "bla",
        },
        "BaseImputer": {
            "columns": ["a"],
        },
        "BaseMappingTransformer": {
            "mappings": {"a": {1: 2, 3: 4}},
        },
        "BaseMappingTransformMixin": {
            "columns": ["a"],
        },
        "BaseNominalTransformer": {
            "columns": ["b"],
        },
        "BaseTransformer": {
            "columns": ["a"],
        },
        "BaseTwoColumnTransformer": {
            "columns": ["a", "b"],
            "new_col_name": "c",
        },
        "BetweenDatesTransformer": {
            "new_column_name": "c",
            "columns": ["a", "c", "b"],
        },
        "CappingTransformer": {
            "capping_values": {"a": [0.1, 0.2]},
        },
        "ColumnDtypeSetter": {
            "columns": ["a"],
            "dtype": str,
        },
        "CrossColumnAddTransformer": {
            "mappings": {"a": {1: 2, 3: 4}},
            "adjust_column": "b",
        },
        "CrossColumnMappingTransformer": {
            "mappings": {"a": {1: 2, 3: 4}},
            "adjust_column": "b",
        },
        "CrossColumnMultiplyTransformer": {
            "mappings": {"a": {1: 2, 3: 4}},
            "adjust_column": "b",
        },
        "CutTransformer": {
            "new_column_name": "b",
            "column": "a",
            "cut_kwargs": {
                "bins": 3,
            },
        },
        "DataFrameMethodTransformer": {
            "columns": ["a", "b"],
            "new_column_names": "f",
            "pd_method_name": "sum",
            "drop_original": True,
        },
        "DateDifferenceTransformer": {
            "columns": ["a", "b"],
            "new_column_name": "c",
        },
        "DateDiffLeapYearTransformer": {
            "columns": ["a", "b"],
            "new_column_name": "c",
        },
        "DatetimeInfoExtractor": {
            "columns": ["a"],
        },
        "DatetimeSinusoidCalculator": {
            "columns": ["a"],
            "method": ["sin"],
            "units": "month",
        },
        "EqualityChecker": {
            "columns": ["a", "b"],
            "new_col_name": "c",
            "drop_original": True,
        },
        "GroupRareLevelsTransformer": {
            "columns": ["b"],
        },
        "InteractionTransformer": {
            "columns": ["a", "b"],
        },
        "LogTransformer": {
            "columns": ["a"],
        },
        "MappingTransformer": {
            "mappings": {"a": {1: 2, 3: 4}},
        },
        "MeanImputer": {
            "columns": ["a"],
        },
        "MeanResponseTransformer": {
            "columns": ["b"],
        },
        "MedianImputer": {
            "columns": ["a"],
        },
        "ModeImputer": {
            "columns": ["a"],
        },
        "NearestMeanResponseImputer": {
            "columns": ["a"],
        },
        "NominalToIntegerTransformer": {
            "columns": ["b"],
        },
        "NullIndicator": {
            "columns": ["a"],
        },
        "OneHotEncodingTransformer": {
            "columns": ["a", "b"],
            "drop_original": True,
            "separator": "-",
        },
        "OrdinalEncoderTransformer": {
            "columns": ["b"],
        },
        "OutOfRangeNullTransformer": {
            "capping_values": {"a": [0.1, 0.2]},
        },
        "PCATransformer": {
            "columns": ["a", "c"],
        },
        "ScalingTransformer": {
            "scaler_type": "standard",
            "columns": ["a"],
        },
        "SeriesDtMethodTransformer": {
            "new_column_name": "a",
            "pd_method_name": "month",
            "column": "b",
        },
        "SeriesStrMethodTransformer": {
            "columns": ["b"],
            "new_column_name": "a",
            "pd_method_name": "split",
        },
        "SetValueTransformer": {
            "value": 1,
            "columns": ["a"],
        },
        "StringConcatenator": {
            "columns": ["a", "b"],
            "new_column": "c",
        },
        "ToDatetimeTransformer": {
            "new_column_name": "b",
            "column": "a",
        },
        "TwoColumnOperatorTransformer": {
            "columns": ["a", "b"],
            "new_column_name": "c",
            "pd_method_name": "add",
        },
    }


@pytest.fixture()
def initialized_transformers(minimal_attribute_dict):
    """dictionary of {transformer name : initiated transformer} pairs for all transformers"""
    return {x[0]: x[1](**minimal_attribute_dict[x[0]]) for x in get_all_classes()}


@pytest.fixture()
def uninitialized_transformers():
    """dictionary of {transformer name : uninitiated transformer} pairs for all transformers"""
    return {x[0]: x[1] for x in get_all_classes()}
