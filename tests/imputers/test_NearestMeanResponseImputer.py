import numpy as np
import pytest

import tests.test_data as d
from tests.base_tests import (
    ColumnStrListInitTests,
    GenericFitTests,
    GenericTransformTests,
    OtherBaseBehaviourTests,
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

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_null_values_in_response_error(self, library):
        """Test an error is raised if the response column contains null entries."""
        df = d.create_df_3(library=library)

        transformer = NearestMeanResponseImputer(columns=["b"])

        with pytest.raises(
            ValueError,
            match="NearestMeanResponseImputer: y has 1 null values",
        ):
            transformer.fit(df, df["a"])

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_columns_with_no_nulls_error(self, library):
        """Test an error is raised if a non-response column contains no nulls."""
        df = d.create_numeric_df_1(library=library)

        transformer = NearestMeanResponseImputer(columns=["b", "c"])

        with pytest.raises(
            ValueError,
            match="NearestMeanResponseImputer: Column b has no missing values, cannot use this transformer.",
        ):
            transformer.fit(df, df["c"])

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_learnt_values(self, library):
        """Test that the nearest response values learnt during fit are expected."""
        df = d.create_numeric_df_2(library=library)

        transformer = NearestMeanResponseImputer(columns=["b", "c"])

        transformer.fit(df, df["a"])

        assert transformer.impute_values_ == {
            "b": np.float64(3),
            "c": np.float64(2),
        }, "impute_values_ attribute"


class TestTransform(
    GenericTransformTests,
    GenericImputerTransformTests,
):
    """Tests for transformer.transform."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "NearestMeanResponseImputer"


class TestOtherBaseBehaviour(OtherBaseBehaviourTests):
    """
    Class to run tests for BaseTransformerBehaviour outside the three standard methods.

    May need to overwite specific tests in this class if the tested transformer modifies this behaviour.
    """

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "NearestMeanResponseImputer"
