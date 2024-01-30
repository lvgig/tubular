import inspect
import pkgutil
from importlib import import_module
from pathlib import Path

import pytest

import tubular.base as base


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
    return {
        "BaseTransformer": {
            "columns": ["a"],
        },
        "DataFrameMethodTransformer": {
            "columns": ["a", "c"],
            "new_column_names": "f",
            "pd_method_name": "sum",
        },
        "BaseDateTransformer": {
            "columns": ["a"],
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
            "columns": ["a"],
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


@pytest.fixture()
def instantiated_transformers(minimal_attribute_dict):
    return {x[0]: x[1](**minimal_attribute_dict[x[0]]) for x in get_all_classes()}


@pytest.fixture()
def uninstantiated_transformers():
    return {x[0]: x[1] for x in get_all_classes()}
