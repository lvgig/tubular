# tests to apply to all transformers
import pytest
import sklearn.base as b

import tubular.base as base
import tubular.capping as capping
import tubular.comparison as comparison
import tubular.dates as dates
import tubular.imputers as imputers
import tubular.mapping as mapping
import tubular.misc as misc
import tubular.nominal as nominal
import tubular.numeric as numeric
import tubular.strings as strings


class TestInit:
    """Tests for transformer.init()."""

    def ListOfTransformers():
        """List of transformers in tubular to be used in subsequent tests."""
        return [
            base.BaseTransformer(columns=["a"]),
            base.DataFrameMethodTransformer(
                new_column_name="a",
                pd_method_name="sum",
                columns="b",
            ),
            capping.CappingTransformer(capping_values={"a": [0.1, 0.2]}),
            capping.OutOfRangeNullTransformer(capping_values={"a": [0.1, 0.2]}),
            comparison.EqualityChecker(columns=["a", "b"], new_col_name="c"),
            dates.DateDiffLeapYearTransformer(
                column_lower="a",
                column_upper="b",
                new_column_name="c",
                drop_cols=True,
            ),
            dates.DateDifferenceTransformer(
                column_lower="a",
                column_upper="b",
                new_column_name="c",
                units="D",
            ),
            dates.ToDatetimeTransformer(column="a", new_column_name="b"),
            dates.DatetimeInfoExtractor(columns="a"),
            dates.SeriesDtMethodTransformer(
                new_column_name="a",
                pd_method_name="month",
                column="b",
            ),
            dates.BetweenDatesTransformer(
                column_lower="a",
                column_upper="b",
                column_between="c",
                new_column_name="c",
            ),
            dates.DatetimeSinusoidCalculator(
                "a",
                "sin",
                "month",
                12,
            ),
            imputers.BaseImputer(),
            imputers.ArbitraryImputer(impute_value=1, columns="a"),
            imputers.MedianImputer(columns="a"),
            imputers.MeanImputer(columns="a"),
            imputers.ModeImputer(columns="a"),
            imputers.NearestMeanResponseImputer(columns="a"),
            imputers.NullIndicator(columns="a"),
            mapping.BaseMappingTransformer(mappings={"a": {1: 2, 3: 4}}),
            mapping.BaseMappingTransformMixin(),
            mapping.MappingTransformer(mappings={"a": {1: 2, 3: 4}}),
            mapping.CrossColumnMappingTransformer(
                adjust_column="b",
                mappings={"a": {1: 2, 3: 4}},
            ),
            mapping.CrossColumnMultiplyTransformer(
                adjust_column="b",
                mappings={"a": {1: 2, 3: 4}},
            ),
            mapping.CrossColumnAddTransformer(
                adjust_column="b",
                mappings={"a": {1: 2, 3: 4}},
            ),
            misc.SetValueTransformer(columns="a", value=1),
            misc.SetColumnDtype(columns="a", dtype=str),
            nominal.BaseNominalTransformer(),
            nominal.NominalToIntegerTransformer(columns="a"),
            nominal.GroupRareLevelsTransformer(columns="a"),
            nominal.MeanResponseTransformer(columns="a"),
            nominal.OrdinalEncoderTransformer(columns="a"),
            nominal.OneHotEncodingTransformer(columns="a"),
            numeric.LogTransformer(columns="a"),
            numeric.CutTransformer(column="a", new_column_name="b"),
            numeric.TwoColumnOperatorTransformer(
                pd_method_name="add",
                columns=["a", "b"],
                new_column_name="c",
            ),
            numeric.ScalingTransformer(columns="a", scaler_type="standard"),
            strings.SeriesStrMethodTransformer(
                new_column_name="a",
                pd_method_name="find",
                columns="b",
                pd_method_kwargs={"sub": "a"},
            ),
            strings.StringConcatenator(columns=["a", "b"], new_column="c"),
        ]

    @pytest.mark.parametrize("transformer", ListOfTransformers())
    def test_print(self, transformer):
        """Test that transformer can be printed.
        If an error is raised in this test it will not prevent the transformer from working correctly,
        but will stop other unit tests passing.
        """
        print(transformer)

    @pytest.mark.parametrize("transformer", ListOfTransformers())
    def test_clone(self, transformer):
        """Test that transformer can be used in sklearn.base.clone function."""
        b.clone(transformer)

    @pytest.mark.parametrize("transformer", ListOfTransformers())
    def test_unexpected_kwarg(self, transformer):
        """Test that transformer can be used in sklearn.base.clone function."""
        b.clone(transformer)
