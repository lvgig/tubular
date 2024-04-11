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
)
from tests.imputers.test_BaseImputer import GenericImputerTransformTests
from tubular.imputers import ModeImputer


class TestInit(ColumnStrListInitTests):
    """Generic tests for transformer.init()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "ModeImputer"

    def test_weight_value_type_error(
        self,
        uninitialized_transformers,
        minimal_attribute_dict,
    ):
        """Test that an exception is raised if weight is not a str."""

        args = minimal_attribute_dict[self.transformer_name].copy()
        args["weight"] = 1

        with pytest.raises(
            ValueError,
            match="weight should be a string or None",
        ):
            uninitialized_transformers[self.transformer_name](**args)


class TestFit(GenericFitTests):
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

        x = ModeImputer(columns=["a", "b", "c", "d"], weight="weight")

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

    def test_fit_returns_self_weighted(self):
        """Test fit returns self?."""
        df = d.create_df_9()

        x = ModeImputer(columns="a", weight="c")

        x_fitted = x.fit(df)

        assert x_fitted is x, "Returned value from ModeImputer.fit not as expected."

    def test_fit_not_changing_data_weighted(self):
        """Test fit does not change X - when weights are used."""
        df = d.create_df_9()

        x = ModeImputer(columns="a", weight="c")

        x.fit(df)

        ta.equality.assert_equal_dispatch(
            expected=d.create_df_9(),
            actual=df,
            msg="Check X not changing during fit",
        )


class TestTransform(GenericTransformTests, GenericImputerTransformTests):
    """Tests for transformer.transform."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "ModeImputer"

    def expected_df_9():
        """Expected output for test_nulls_imputed_correctly_weighted."""
        df = d.create_df_9()

        for col in ["a"]:
            df.loc[df[col].isna(), col] = 6

        return df

    @pytest.mark.parametrize(
        ("df", "expected"),
        ta.pandas.row_by_row_params(
            d.create_df_9(),
            expected_df_9(),
        )
        + ta.pandas.index_preserved_params(
            d.create_df_9(),
            expected_df_9(),
        ),
    )
    def test_nulls_imputed_correctly_weighted(self, df, expected):
        """Test missing values are filled with the correct values - and unrelated columns are not changed
        (when weight is used).
        """
        x = ModeImputer(columns=["a"], weight="c")

        # set the impute values dict directly rather than fitting x on df so test works with helpers
        x.impute_values_ = {"a": 6}

        df_transformed = x.transform(df)

        ta.equality.assert_equal_dispatch(
            expected=expected,
            actual=df_transformed,
            msg="Check nulls filled correctly in transform",
        )

    def test_learnt_values_not_modified_weights(self):
        """Test that the impute_values_ from fit are not changed in transform - when using weights."""
        df = d.create_df_9()

        x = ModeImputer(columns=["a", "b"], weight="c")

        x.fit(df)

        x2 = ModeImputer(columns=["a", "b"], weight="c")

        x2.fit_transform(df)

        ta.equality.assert_equal_dispatch(
            expected=x.impute_values_,
            actual=x2.impute_values_,
            msg="Impute values not changed in transform",
        )


class TestOtherBaseBehaviour(OtherBaseBehaviourTests):
    """
    Class to run tests for BaseTransformerBehaviour outside the three standard methods.

    May need to overwite specific tests in this class if the tested transformer modifies this behaviour.
    """

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "ModeImputer"
