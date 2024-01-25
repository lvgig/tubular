import pytest

pytest.fixture()


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
