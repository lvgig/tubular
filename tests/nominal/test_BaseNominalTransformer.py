import re

import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

import tests.test_data as d
from tests.base_tests import (
    ColumnStrListInitTests,
    GenericFitTests,
    GenericTransformTests,
    OtherBaseBehaviourTests,
)


# The first part of this file builds out the tests for BaseNominalTransformer so that they can be
# imported into other test files (by not starting the class name with Test)
# The second part actually calls these tests (along with all other require tests) for the BaseNominalTransformer
class GenericBaseNominalTransformerTests:
    """
    Tests for BaseNominalTransformer.transform().
    Note this deliberately avoids starting with "Tests" so that the tests are not run on import.
    """

    def test_not_fitted_error_raised(self, initialized_transformers):
        if initialized_transformers[self.transformer_name].FITS:
            df = d.create_df_1()

            with pytest.raises(NotFittedError):
                initialized_transformers[self.transformer_name].transform(df)

    def test_non_mappable_rows_exception_raised(self, initialized_transformers):
        """Test an exception is raised if non-mappable rows are present in X."""
        df = d.create_df_1()

        x = initialized_transformers[self.transformer_name]

        x.fit(df)

        x.mappings = {
            "a": {1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7},
            "b": {"a": 1, "c": 2, "d": 3, "e": 4, "f": 5},
        }

        with pytest.raises(
            ValueError,
            match=f"{self.transformer_name}: nulls would be introduced into column b from levels not present in mapping",
        ):
            x.transform(df)

    def test_original_df_not_updated(self, initialized_transformers):
        """Test that the original dataframe is not transformed when transform method used."""

        df = d.create_df_1()

        x = initialized_transformers[self.transformer_name]

        x = x.fit(df)

        x.mappings = {"b": {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6}}

        _ = x.transform(df)

        pd.testing.assert_frame_equal(df, d.create_df_1())

    def test_no_rows_error(self, initialized_transformers):
        """Test an error is raised if X has no rows."""
        df = d.create_df_1()

        x = initialized_transformers[self.transformer_name]

        x = x.fit(df)

        x.mappings = {"b": {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6}}

        df = pd.DataFrame(columns=["a", "b", "c"])

        with pytest.raises(
            ValueError,
            match=re.escape(f"{self.transformer_name}: X has no rows; (0, 3)"),
        ):
            x.transform(df)


class TestInit(ColumnStrListInitTests):
    """Generic tests for transformer.init()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseNominalTransformer"


class TestFit(GenericFitTests):
    """Generic tests for transformer.fit()"""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseNominalTransformer"


class TestTransform(GenericBaseNominalTransformerTests, GenericTransformTests):
    """Tests for BaseImputer.transform."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseNominalTransformer"


class TestOtherBaseBehaviour(OtherBaseBehaviourTests):
    """
    Class to run tests for BaseTransformerBehaviour outside the three standard methods.

    May need to overwite specific tests in this class if the tested transformer modifies this behaviour.
    """

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseNominalTransformer"
