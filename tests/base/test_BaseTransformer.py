import narwhals as nw
import polars as pl
import pytest

from tests.base_tests import (
    ColumnStrListInitTests,
    GenericFitTests,
    GenericTransformTests,
    OtherBaseBehaviourTests,
)
from tests.utils import assert_frame_equal_dispatch


class TestInit(ColumnStrListInitTests):
    """Generic tests for transformer.init()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseTransformer"


class TestFit(GenericFitTests):
    """Generic tests for transformer.fit()"""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseTransformer"


class TestTransform(GenericTransformTests):
    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseTransformer"

    @pytest.mark.parametrize(
        "minimal_dataframe_lookup",
        ["pandas", "polars"],
        indirect=True,
    )
    def test_X_returned(self, minimal_dataframe_lookup, initialized_transformers):
        """Test that X is returned from transform."""
        df = minimal_dataframe_lookup[self.transformer_name]
        x = initialized_transformers[self.transformer_name]

        # if transformer is not polars compatible, skip polars test
        if not x.polars_compatible and isinstance(df, pl.DataFrame):
            return

        df = nw.from_native(df)
        expected = df.clone()

        df = nw.to_native(df)
        expected = nw.to_native(expected)

        df_transformed = x.transform(X=df)

        assert_frame_equal_dispatch(expected, df_transformed)


class TestOtherBaseBehaviour(OtherBaseBehaviourTests):
    """
    Class to run tests for BaseTransformerBehaviour outside the three standard methods.

    May need to overwite specific tests in this class if the tested transformer modifies this behaviour.
    """

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseTransformer"
