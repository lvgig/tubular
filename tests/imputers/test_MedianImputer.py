import numpy as np
import pandas as pd
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
from tubular.imputers import MedianImputer


class TestInit(ColumnStrListInitTests, WeightColumnInitMixinTests):
    """Generic tests for transformer.init()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "MedianImputer"


class TestFit(WeightColumnFitMixinTests, GenericFitTests):
    """Generic tests for transformer.fit()"""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "MedianImputer"

    def test_learnt_values(self):
        """Test that the impute values learnt during fit are expected."""
        df = d.create_df_3()
        df["d"] = np.nan

        x = MedianImputer(columns=["a", "b", "c", "d"])

        x.fit(df)

        ta.classes.test_object_attributes(
            obj=x,
            expected_attributes={
                "impute_values_": {
                    "a": df["a"].median(),
                    "b": df["b"].median(),
                    "c": df["c"].median(),
                    "d": np.float64(np.nan),
                },
            },
            msg="impute_values_ attribute",
        )

    def test_learnt_values_weighted(self):
        """Test that the impute values learnt during fit are expected - when using weights."""
        df = d.create_df_9()
        df["d"] = np.nan

        df = pd.DataFrame(
            {
                "a": [1, 2, 4, 6],
                "c": [3, 2, 4, 6],
                "d": np.nan,
            },
        )

        x = MedianImputer(columns=["a", "d"], weights_column="c")

        x.fit(df)

        ta.classes.test_object_attributes(
            obj=x,
            expected_attributes={
                "impute_values_": {
                    "a": np.int64(4),
                    "d": np.nan,
                },
            },
            msg="impute_values_ attribute",
        )

    def test_fit_not_changing_data(self):
        """Test fit does not change X."""
        df = d.create_df_1()

        x = MedianImputer(columns="a")

        x.fit(df)

        ta.equality.assert_equal_dispatch(
            expected=d.create_df_1(),
            actual=df,
            msg="Check X not changing during fit",
        )


class TestTransform(
    GenericImputerTransformTests,
    GenericImputerTransformTestsWeight,
    GenericTransformTests,
):
    """Tests for transformer.transform."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "MedianImputer"


class TestOtherBaseBehaviour(OtherBaseBehaviourTests):
    """
    Class to run tests for BaseTransformerBehaviour behaviour outside the three standard methods.
    May need to overwite specific tests in this class if the tested transformer modifies this behaviour.
    """

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "MedianImputer"
