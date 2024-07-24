import numpy as np
import pytest
import test_aide as ta

import tests.test_data as d
from tests.base_tests import (
    ColumnStrListInitTests,
    GenericFitTests,
    GenericTransformTests,
)
from tests.imputers.test_BaseImputer import (
    GenericImputerTransformTests,
)
from tubular.imputers import NearestMeanResponseImputer


class TestInit(ColumnStrListInitTests):
    """Generic tests for transformer.init()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "NearestMeanResponseImputer"


class TestFit(GenericFitTests):
    """Generic tests for transformer.fit()"""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "NearestMeanResponseImputer"

    def test_null_values_in_response_error(self):
        """Test an error is raised if the response column contains null entries."""
        df = d.create_df_3()

        x = NearestMeanResponseImputer(columns=["b"])

        with pytest.raises(
            ValueError,
            match="NearestMeanResponseImputer: y has 1 null values",
        ):
            x.fit(df, df["c"])

    def test_columns_with_no_nulls_error(self):
        """Test an error is raised if a non-response column contains no nulls."""
        df = d.create_numeric_df_1()

        x = NearestMeanResponseImputer(columns=["a", "b"])

        with pytest.raises(
            ValueError,
            match="NearestMeanResponseImputer: Column a has no missing values, cannot use this transformer.",
        ):
            x.fit(df, df["c"])

    def test_learnt_values(self):
        """Test that the nearest response values learnt during fit are expected."""
        df = d.create_numeric_df_2()

        x = NearestMeanResponseImputer(columns=["a", "b"])

        x.fit(df, df["c"])

        ta.classes.test_object_attributes(
            obj=x,
            expected_attributes={
                "impute_values_": {"a": np.float64(2), "b": np.float64(3)},
            },
            msg="impute_values_ attribute",
        )


class TestTransform(
    GenericTransformTests,
    GenericImputerTransformTests,
):
    """Tests for transformer.transform."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "NearestMeanResponseImputer"
