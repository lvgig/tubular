import numpy as np
import pandas as pd
import pytest
import test_aide as ta
from sklearn.exceptions import NotFittedError

import tests.test_data as d
from tests.base_tests import (
    ColumnStrListInitTests,
    GenericFitTests,
    GenericTransformTests,
    OtherBaseBehaviourTests,
)
from tubular.imputers import BaseImputer


class BaseImputerTransformTests(GenericTransformTests):
    def test_not_fitted_error_raised(self):
        df = pd.DataFrame(
            {
                "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                "b": ["a", "b", "c", "d", "e", "f", np.NaN],
                "c": ["a", "b", "c", "d", "e", "f", np.NaN],
            },
        )

        x = BaseImputer(columns=["b", "c"])

        with pytest.raises(NotFittedError):
            x.transform(df)


class TestInit(ColumnStrListInitTests):
    """Generic tests for transformer.init()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseImputer"


class TestFit(GenericFitTests):
    """Generic tests for transformer.fit()"""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseTransformer"


class TestTransform(BaseImputerTransformTests):
    """Tests for BaseImputer.transform."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseTransformer"

    def expected_df_1():
        """Expected output of test_expected_output_1."""
        df = pd.DataFrame(
            {
                "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                "b": ["a", "b", "c", "d", "e", "f", np.nan],
                "c": ["a", "b", "c", "d", "e", "f", np.nan],
            },
        )

        df["c"] = df["c"].astype("category")

        return df

    def expected_df_2():
        """Expected output of test_expected_output_2."""
        df2 = pd.DataFrame(
            {
                "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, np.nan],
                "b": ["a", "b", "c", "d", "e", "f", "g"],
                "c": ["a", "b", "c", "d", "e", "f", np.nan],
            },
        )

        df2["c"] = df2["c"].astype("category")

        return df2

    def expected_df_3():
        """Expected output of test_expected_output_3."""
        df3 = pd.DataFrame(
            {
                "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, np.nan],
                "b": ["a", "b", "c", "d", "e", "f", "g"],
                "c": ["a", "b", "c", "d", "e", "f", "f"],
            },
        )

        df3["c"] = df3["c"].astype("category")

        return df3

    @pytest.mark.parametrize(
        ("df", "expected"),
        ta.pandas.adjusted_dataframe_params(d.create_df_2(), expected_df_1()),
    )
    def test_expected_output_1(self, df, expected):
        """Test that transform is giving the expected output when applied to float column."""
        x1 = BaseImputer(columns="a")
        x1.impute_values_ = {"a": 7}

        df_transformed = x1.transform(df)

        ta.equality.assert_equal_dispatch(
            expected=expected,
            actual=df_transformed,
            msg="ArbitraryImputer transform col a",
        )

    @pytest.mark.parametrize(
        ("df", "expected"),
        ta.pandas.adjusted_dataframe_params(d.create_df_2(), expected_df_2()),
    )
    def test_expected_output_2(self, df, expected):
        """Test that transform is giving the expected output when applied to object column."""
        x1 = BaseImputer(columns=["b"])

        x1.impute_values_ = {"b": "g"}

        df_transformed = x1.transform(df)

        ta.equality.assert_equal_dispatch(
            expected=expected,
            actual=df_transformed,
            msg="ArbitraryImputer transform col b",
        )

    @pytest.mark.parametrize(
        ("df", "expected"),
        ta.pandas.adjusted_dataframe_params(d.create_df_2(), expected_df_3()),
    )
    def test_expected_output_3(self, df, expected):
        """Test that transform is giving the expected output when applied to object and categorical columns."""
        x1 = BaseImputer(columns=["b", "c"])

        x1.impute_values_ = {"b": "g", "c": "f"}

        df_transformed = x1.transform(df)

        ta.equality.assert_equal_dispatch(
            expected=expected,
            actual=df_transformed,
            msg="ArbitraryImputer transform col b, c",
        )


class TestOtherBaseBehaviour(OtherBaseBehaviourTests):
    """
    Class to run tests for BaseTransformerBehaviour outside the three standard methods.

    May need to overwite specific tests in this class if the tested transformer modifies this behaviour.
    """

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseImputer"
