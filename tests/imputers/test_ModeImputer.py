import numpy as np
import pandas as pd
import pytest
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
from tubular.imputers import ModeImputer


class TestInit(ColumnStrListInitTests, WeightColumnInitMixinTests):
    """Generic tests for transformer.init()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "ModeImputer"


class TestFit(WeightColumnFitMixinTests, GenericFitTests):
    """Generic tests for transformer.fit()"""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "ModeImputer"

    def test_learnt_values(self):
        """Test that the impute values learnt during fit are expected."""
        df = d.create_df_3()

        x = ModeImputer(columns=["a", "b", "c"])

        x.fit(df)

        ta.classes.test_object_attributes(
            obj=x,
            expected_attributes={
                "impute_values_": {
                    "a": df["a"].mode()[0],
                    "b": df["b"].mode()[0],
                    "c": df["c"].mode()[0],
                },
            },
            msg="impute_values_ attribute",
        )

    def expected_df_nan():
        return pd.DataFrame({"a": ["NaN", "NaN", "NaN"], "b": [None, None, None]})

    @pytest.mark.parametrize(
        ("df", "expected"),
        ta.pandas.row_by_row_params(
            pd.DataFrame({"a": [np.nan, np.nan, np.nan], "b": [None, None, None]}),
            expected_df_nan(),
        )
        + ta.pandas.index_preserved_params(
            pd.DataFrame({"a": [np.nan, np.nan, np.nan], "b": [None, None, None]}),
            expected_df_nan(),
        ),
    )
    def test_warning_mode_is_nan(self, df, expected):
        """Test that warning is raised when mode is NaN."""
        x = ModeImputer(columns=["a", "b"])

        with pytest.warns(Warning, match="ModeImputer: The Mode of column a is NaN."):
            x.fit(df)

        with pytest.warns(Warning, match="ModeImputer: The Mode of column b is NaN."):
            x.fit(df)

    def test_learnt_values_weighted_df(self):
        """Test that the impute values learnt during fit are expected when df is weighted."""
        df = d.create_weighted_imputers_test_df()

        x = ModeImputer(columns=["a", "b", "c", "d"], weights_column="weights_column")

        x.fit(df)

        ta.classes.test_object_attributes(
            obj=x,
            expected_attributes={
                "impute_values_": {
                    "a": np.float64(5.0),
                    "b": "e",
                    "c": "f",
                    "d": np.float64(1.0),
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
        cls.transformer_name = "ModeImputer"


class TestOtherBaseBehaviour(OtherBaseBehaviourTests):
    """
    Class to run tests for BaseTransformerBehaviour outside the three standard methods.

    May need to overwite specific tests in this class if the tested transformer modifies this behaviour.
    """

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "ModeImputer"
