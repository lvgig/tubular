import pytest
import test_aide as ta

import tests.test_data as d
from tests.base_tests import (
    ColumnStrListInitTests,
    GenericFitTests,
    GenericTransformTests,
    OtherBaseBehaviourTests,
)
from tubular.misc import SetValueTransformer


class TestInit(ColumnStrListInitTests):
    """Generic tests for SetValueTransformer.init()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "SetValueTransformer"


class TestFit(GenericFitTests):
    """Generic tests for SetValueTransformer.fit()"""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "SetValueTransformer"


class TestTransform(GenericTransformTests):
    """Tests for SetValueTransformer.transform."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "SetValueTransformer"

    def expected_df_1():
        """Expected output of test_value_set_in_transform."""
        df = d.create_df_2()

        df["a"] = "a"
        df["b"] = "a"

        return df

    @pytest.mark.parametrize(
        ("df", "expected"),
        ta.pandas.adjusted_dataframe_params(d.create_df_2(), expected_df_1()),
    )
    def test_value_set_in_transform(self, df, expected):
        """Test that transform sets the value as expected."""
        x = SetValueTransformer(columns=["a", "b"], value="a")

        df_transformed = x.transform(df)

        ta.equality.assert_equal_dispatch(
            actual=df_transformed,
            expected=expected,
            msg="incorrect value after SetValueTransformer transform",
        )


class TestOtherBaseBehaviour(OtherBaseBehaviourTests):
    """
    Class to run tests for SetValueTransformer behaviour outside the three standard methods.

    May need to overwite specific tests in this class if the tested transformer modifies this behaviour.
    """

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "SetValueTransformer"
