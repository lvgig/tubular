import copy
import re

import pandas as pd
import pytest
import test_aide as ta

import tests.test_data as d
from tests.base_tests import (
    ColumnStrListInitTests,
    GenericFitTests,
    GenericTransformTests,
    OtherBaseBehaviourTests,
)
from tests.utils import assert_frame_equal_dispatch
from tubular.mapping import BaseMappingTransformMixin

# Note there are no tests that need inheriting from this file as the only difference is an expected transform output


@pytest.fixture()
def mapping():
    return {
        "a": {1: "a", 2: "b", 3: "c", 4: "d", 5: "e", 6: "f"},
        "b": {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6},
    }


class TestInit(ColumnStrListInitTests):
    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseMappingTransformMixin"


class TestFit(GenericFitTests):
    """Generic tests for transformer.fit()"""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseMappingTransformMixin"


class TestTransform(GenericTransformTests):
    """
    Tests for BaseMappingTransformMixin.transform().

    Because this is a Mixin transformer it is not always appropriate to inherit the generic transform tests. A number of the tests below overwrite the tests in GenericTransformTests.
    """

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseMappingTransformMixin"

    def test_expected_output(self, mapping):
        """Test that X is returned from transform."""

        df = d.create_df_1()

        expected = pd.DataFrame(
            {
                "a": ["a", "b", "c", "d", "e", "f"],
                "b": [1, 2, 3, 4, 5, 6],
            },
        )

        x = BaseMappingTransformMixin(columns=["a", "b"])

        x.mappings = mapping

        df_transformed = x.transform(df)

        ta.equality.assert_equal_dispatch(
            expected=expected,
            actual=df_transformed,
            msg="BaseMappingTransformMixin from transform",
        )

    def test_mappings_unchanged(self, mapping):
        """Test that mappings is unchanged in transform."""
        df = d.create_df_1()

        x = BaseMappingTransformMixin(columns=["a", "b"])

        x.mappings = mapping

        x.transform(df)

        ta.equality.assert_equal_dispatch(
            expected=mapping,
            actual=x.mappings,
            msg="BaseMappingTransformer.transform has changed self.mappings unexpectedly",
        )

    @pytest.mark.parametrize("non_df", [1, True, "a", [1, 2], {"a": 1}, None])
    def test_non_pd_type_error(
        self,
        non_df,
        mapping,
    ):
        """Test that an error is raised in transform is X is not a pd.DataFrame."""

        df = d.create_df_10()

        x = BaseMappingTransformMixin(columns=["a"])

        x.mappings = mapping

        x_fitted = x.fit(df, df["c"])

        with pytest.raises(
            TypeError,
            match="BaseMappingTransformMixin: X should be a polars or pandas DataFrame/LazyFrame",
        ):
            x_fitted.transform(X=non_df)

    def test_no_rows_error(self, mapping):
        """Test an error is raised if X has no rows."""
        df = d.create_df_10()

        x = BaseMappingTransformMixin(columns=["a"])

        x.mappings = mapping

        x = x.fit(df, df["c"])

        df = pd.DataFrame(columns=["a", "b", "c"])

        with pytest.raises(
            ValueError,
            match=re.escape("BaseMappingTransformMixin: X has no rows; (0, 3)"),
        ):
            x.transform(df)

    def test_original_df_not_updated(self, mapping):
        """Test that the original dataframe is not transformed when transform method used."""

        df = d.create_df_10()

        x = BaseMappingTransformMixin(columns=["a"])

        x.mappings = mapping

        x = x.fit(df, df["c"])

        _ = x.transform(df)

        pd.testing.assert_frame_equal(df, d.create_df_10())

    @pytest.mark.parametrize(
        "minimal_dataframe_lookup",
        ["pandas"],
        indirect=True,
    )
    def test_pandas_index_not_updated(
        self,
        initialized_transformers,
        minimal_dataframe_lookup,
        mapping,
    ):
        """Test that the original (pandas) dataframe index is not transformed when transform method used."""

        df = minimal_dataframe_lookup[self.transformer_name]
        x = initialized_transformers[self.transformer_name]
        x.mappings = mapping

        # update to abnormal index
        df.index = [2 * i for i in df.index]

        original_df = copy.deepcopy(df)

        x = x.fit(df, df["a"])

        _ = x.transform(df)

        assert_frame_equal_dispatch(df, original_df)


class TestOtherBaseBehaviour(OtherBaseBehaviourTests):
    """
    Class to run tests for BaseTransformerBehaviour outside the three standard methods.

    May need to overwite specific tests in this class if the tested transformer modifies this behaviour.
    """

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseMappingTransformMixin"
