import copy

import numpy as np
import pandas as pd
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


# Dataframes used exclusively in this testing script
def create_NearestMeanResponseImputer_test_df():
    """Create DataFrame to use in NearestMeanResponseImputer tests.

    DataFrame column c is the response, the other columns are numerical columns containing null entries.

    """
    return pd.DataFrame(
        {
            "a": [1, 1, 2, 3, 3, np.nan],
            "b": [np.nan, np.nan, 1, 3, 3, 4],
            "c": [2, 3, 2, 1, 4, 1],
        },
    )


def create_NearestMeanResponseImputer_test_df_2():
    """Create second DataFrame to use in NearestMeanResponseImputer tests.

    DataFrame column c is the response, the other columns are numerical columns containing null entries.

    """
    return pd.DataFrame(
        {
            "a": [1, 1, np.nan, np.nan, 3, 5],
            "b": [np.nan, np.nan, 1, 3, 3, 4],
            "c": [2, 3, 2, 1, 4, 1],
        },
    )


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

    def test_fit_passed_series(
        self,
        initialized_transformers,
        minimal_dataframe_lookup,
    ):
        """Test fit is passed a series as y argument."""

        df = minimal_dataframe_lookup[self.transformer_name]

        x = initialized_transformers[self.transformer_name]

        with pytest.raises(
            TypeError,
            match="unexpected type for y, should be a pd.Series",
        ):
            x.fit(df)(columns=["c"], separator=333)

    def test_fit_not_changing_data(
        self,
        initialized_transformers,
        minimal_dataframe_lookup,
    ):
        """Test fit does not change X."""

        df = minimal_dataframe_lookup[self.transformer_name]
        original_df = copy.deepcopy(df)

        x = initialized_transformers[self.transformer_name]

        x.fit(df, df["c"])

        ta.equality.assert_equal_dispatch(
            expected=original_df,
            actual=df,
            msg="Check X not changing during fit",
        )

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
        df = create_NearestMeanResponseImputer_test_df()

        x = NearestMeanResponseImputer(columns=["a", "b"])

        x.fit(df, df["c"])

        ta.classes.test_object_attributes(
            obj=x,
            expected_attributes={
                "impute_values_": {"a": np.float64(2), "b": np.float64(3)},
            },
            msg="impute_values_ attribute",
        )

    def test_learnt_values2(self):
        """Test that the nearest mean response values learnt during fit are expected."""

        df = create_NearestMeanResponseImputer_test_df_2()

        x = NearestMeanResponseImputer(columns=["a", "b"])

        x.fit(df, df["c"])

        ta.classes.test_object_attributes(
            obj=x,
            expected_attributes={
                "impute_values_": {"a": np.float64(5), "b": np.float64(3)},
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
