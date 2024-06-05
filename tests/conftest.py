from __future__ import annotations

import inspect
import pkgutil
from importlib import import_module
from pathlib import Path

import pytest

import tubular.base as base
from tests.test_data import (
    create_is_between_dates_df_1,
    create_numeric_df_1,
    create_object_df,
)

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


def get_all_classes(
    wanted_module: str | None = None,
) -> dict[str, base.BaseTransformer]:
    """Method to call the weighted_quantile method and prepare the outputs.

    If there are no None values in the supplied quantiles then the outputs from weighted_quantile
    are returned as is. If there are then prepare_quantiles removes the None values before
    calling weighted_quantile and adds them back into the output, in the same position, after
    calling.

    Parameters
    ----------
    wanted_module : str or None
        str indicating which module to load classes from (e.g. 'tubular.dates'). If none, loads from all modules.

    Returns
    -------
    all_classes : dict[str, BaseTransformer]
        Dictionary containing classes in format {transformer_name:transformer}

    """
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
        if wanted_module and modname != wanted_module:
            continue
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
        "BaseTransformer": {
            "columns": ["a"],
        },
        "BaseTwoColumnTransformer": {
            "columns": ["a", "b"],
            "new_col_name": "c",
        },
        "DataFrameMethodTransformer": {
            "columns": ["a", "c"],
            "new_column_names": "f",
            "pd_method_name": "sum",
            "drop_original": True,
        },
        "BaseDateTransformer": {
            "columns": ["a"],
            "new_column_name": "bla",
        },
        "BaseDateTwoColumnTransformer": {
            "columns": ["a", "b"],
            "new_column_name": "bla",
        },
        "BaseCappingTransformer": {
            "capping_values": {"a": [0.1, 0.2]},
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
            "drop_original": True,
        },
        "DateDiffLeapYearTransformer": {
            "columns": ["a", "b"],
            "new_column_name": "c",
        },
        "DateDifferenceTransformer": {
            "columns": ["a", "b"],
            "new_column_name": "c",
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
            "columns": ["a", "c", "b"],
        },
        "DatetimeSinusoidCalculator": {
            "columns": ["a"],
            "method": ["sin"],
            "units": "month",
        },
        "BaseImputer": {
            "columns": ["a"],
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
            "columns": ["a"],
        },
        "MappingTransformer": {
            "mappings": {"a": {1: 2, 3: 4}},
        },
        "BaseCrossColumnMappingTransformer": {
            "mappings": {"a": {1: 2, 3: 4}},
            "adjust_column": "b",
        },
        "BaseCrossColumnNumericTransformer": {
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
        "CrossColumnAddTransformer": {
            "mappings": {"a": {1: 2, 3: 4}},
            "adjust_column": "b",
        },
        "SetValueTransformer": {
            "value": 1,
            "columns": ["a"],
        },
        "ColumnDtypeSetter": {
            "columns": ["a"],
            "dtype": str,
        },
        "BaseNominalTransformer": {
            "columns": ["b"],
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
            "drop_original": True,
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


@pytest.fixture()
def minimal_dataframe_lookup():
    """links transformers to minimal dataframes needed to successfully run transformer.
    New transformers need to be added here"""

    num_df = create_numeric_df_1()
    object_df = create_object_df()
    date_df = create_is_between_dates_df_1()

    # generally most transformers will work with num_df
    min_df_dict = {x[0]: num_df for x in get_all_classes()}

    # where they do not, will need to override
    date_transformers = [x[0] for x in get_all_classes(wanted_module="tubular.dates")]
    for transformer in date_transformers:
        min_df_dict[transformer] = date_df

    object_modules = ["tubular.mapping", "tubular.nominal", "tubular.strings"]
    object_transformers = []
    for module in object_modules:
        object_transformers = object_transformers + [
            x[0] for x in get_all_classes(wanted_module=module)
        ]

    for transformer in object_transformers:
        min_df_dict[transformer] = object_df

    # Some may require further manual overwrites
    other_num_transformers = [
        "CrossColumnMultiplyTransformer",
        "CrossColumnAddTransformer",
        "BaseCrossColumnNumericTransformer",
    ]
    for transformer in other_num_transformers:
        min_df_dict[transformer] = num_df

    return min_df_dict


@pytest.fixture()
def initialized_transformers(minimal_attribute_dict):
    """dictionary of {transformer name : initiated transformer} pairs for all transformers"""
    return {x[0]: x[1](**minimal_attribute_dict[x[0]]) for x in get_all_classes()}


@pytest.fixture()
def uninitialized_transformers():
    """dictionary of {transformer name : uninitiated transformer} pairs for all transformers"""
    return {x[0]: x[1] for x in get_all_classes()}
