import numpy as np
import test_aide as ta

import tests.test_data as d
from tests.base_tests import (
    ColumnStrListInitTests,
    GenericFitTests,
    GenericTransformTests,
    OtherBaseBehaviourTests,
    WeightColumnFitMixinTests,
    WeightColumnInitMixinTests,
)
from tests.imputers.test_BaseImputer import (
    GenericImputerTransformTests,
    GenericImputerTransformTestsWeight,
)
from tubular.imputers import MeanImputer


class TestInit(WeightColumnInitMixinTests, ColumnStrListInitTests):
    """Generic tests for transformer.init()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "MeanImputer"


class TestFit(WeightColumnFitMixinTests, GenericFitTests):
    """Generic tests for transformer.fit()"""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "MeanImputer"

    def test_learnt_values(self):
        """Test that the impute values learnt during fit are expected."""
        df = d.create_df_3()

        x = MeanImputer(columns=["a", "b", "c"])

        x.fit(df)

        ta.classes.test_object_attributes(
            obj=x,
            expected_attributes={
                "impute_values_": {
                    "a": df["a"].mean(),
                    "b": df["b"].mean(),
                    "c": df["c"].mean(),
                },
            },
            msg="impute_values_ attribute",
        )

    def test_learnt_values_weighted(self):
        """Test that the impute values learnt during fit are expected - when weights are used."""
        df = d.create_df_9()

        x = MeanImputer(columns=["a", "b"], weights_column="c")

        x.fit(df)

        ta.classes.test_object_attributes(
            obj=x,
            expected_attributes={
                "impute_values_": {
                    "a": np.float64((3 + 4 + 16 + 36) / (3 + 2 + 4 + 6)),
                    "b": np.float64((10 + 4 + 12 + 10 + 6) / (2 + 1 + 4 + 5 + 6)),
                },
            },
            msg="impute_values_ attribute",
        )


class TestTransform(
    GenericTransformTests,
    GenericImputerTransformTestsWeight,
    GenericImputerTransformTests,
):
    """Tests for transformer.transform."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "MeanImputer"


class TestOtherBaseBehaviour(OtherBaseBehaviourTests):
    """
    Class to run tests for BaseTransformerBehaviour behaviour outside the three standard methods.
    May need to overwite specific tests in this class if the tested transformer modifies this behaviour.
    """

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "MeanImputer"
