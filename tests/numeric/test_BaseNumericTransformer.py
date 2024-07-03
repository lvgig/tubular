import re

import pytest

import tests.test_data as d
from tests.base_tests import GenericFitTests, GenericInitTests, GenericTransformTests


class BaseNumericTransformerInitTests(GenericInitTests):
    """
    Tests for BaseNumericTransformer.init().
    Note this deliberately avoids starting with "Tests" so that the tests are not run on import.
    """


class BaseNumericTransformerFitTests(GenericFitTests):
    """
    Tests for BaseNumericTransformer.fit().
    Note this deliberately avoids starting with "Tests" so that the tests are not run on import.
    """

    def test_non_numeric_exception_raised(self, initialized_transformers):
        """Test an exception is raised if self.columns are non-numeric in X."""
        df = d.create_df_2()

        x = initialized_transformers[self.transformer_name]

        with pytest.raises(
            TypeError,
            match=rf"{self.transformer_name}: The following columns are not numeric in X; \['b'\]",
        ):
            x.fit(df, df["c"])


class BaseNumericTransformerTransformTests(
    GenericTransformTests,
):
    """
    Tests for the transform method on BaseNumericTransformer.
    Note this deliberately avoids starting with "Tests" so that the tests are not run on import.
    """

    def test_non_numeric_exception_raised(self, initialized_transformers):
        """Test an exception is raised if self.columns are non-numeric in X."""
        df = d.create_df_2()
        # make df all non-numeric
        df["a"] = df["b"]

        x = initialized_transformers[self.transformer_name]

        with pytest.raises(
            TypeError,
            match=re.escape(
                rf"{self.transformer_name}: The following columns are not numeric in X; {x.columns}",
            ),
        ):
            x.transform(df)


class TestInit(BaseNumericTransformerInitTests):
    """Tests for BaseNumericTransformer.init()"""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseNumericTransformer"


class TestFit(BaseNumericTransformerFitTests):
    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseNumericTransformer"


class TestTransform(BaseNumericTransformerTransformTests):
    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseNumericTransformer"
