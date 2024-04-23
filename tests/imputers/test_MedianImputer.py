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
from tubular.imputers import MedianImputer


class TestInit(ColumnStrListInitTests):
    """Generic tests for transformer.init()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "MedianImputer"

    @pytest.mark.parametrize("weight", (0, ["a"], {"a": 10}))
    def test_weight_arg_errors(
        self,
        uninitialized_transformers,
        minimal_attribute_dict,
        weight,
    ):
        """Test that appropriate errors are throw for bad weight arg."""
        args = minimal_attribute_dict[self.transformer_name].copy()
        args["weight"] = weight

        with pytest.raises(
            TypeError,
            match="weight should be str or None",
        ):
            uninitialized_transformers[self.transformer_name](**args)


class TestFit(GenericFitTests):
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

        x = MedianImputer(columns=["a", "d"], weight="c")

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

    def test_fit_returns_self_weighted(self):
        """Test fit returns self?."""
        df = d.create_df_9()

        x = MedianImputer(columns="a", weight="c")

        x_fitted = x.fit(df)

        assert x_fitted is x, "Returned value from MedianImputer.fit not as expected."

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

    def test_fit_not_changing_data_weighted(self):
        """Test fit does not change X."""
        df = d.create_df_9()

        x = MedianImputer(columns="a", weight="c")

        x.fit(df)

        ta.equality.assert_equal_dispatch(
            expected=d.create_df_9(),
            actual=df,
            msg="Check X not changing during fit",
        )


class TestTransform(GenericImputerTransformTests, GenericTransformTests):
    """Tests for transformer.transform."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "MedianImputer"

    def expected_df_weights():
        """Expected output for test_nulls_imputed_correctly_weights."""
        df = d.create_df_9()

        for col in ["a"]:
            df.loc[df[col].isna(), col] = 4

        return df

    @pytest.mark.parametrize(
        ("df", "expected"),
        ta.pandas.row_by_row_params(d.create_df_9(), expected_df_weights())
        + ta.pandas.index_preserved_params(d.create_df_9(), expected_df_weights()),
    )
    def test_nulls_imputed_correctly_weights(self, df, expected):
        """Test missing values are filled with the correct values - and unrelated columns are not changed
        (when weight is used).
        """
        x = MedianImputer(columns=["a"], weight="c")

        # set the impute values dict directly rather than fitting x on df so test works with helpers
        x.impute_values_ = {"a": 4}

        df_transformed = x.transform(df)

        ta.equality.assert_equal_dispatch(
            expected=expected,
            actual=df_transformed,
            msg="Check nulls filled correctly in transform",
        )

    def test_learnt_values_not_modified_weights(self):
        """Test that the impute_values_ from fit are not changed in transform - when using weights."""
        df = d.create_df_9()

        x = MedianImputer(columns=["a", "b"], weight="c")

        x.fit(df)

        x2 = MedianImputer(columns=["a", "b"], weight="c")

        x2.fit_transform(df)

        ta.equality.assert_equal_dispatch(
            expected=x.impute_values_,
            actual=x2.impute_values_,
            msg="Impute values not changed in transform",
        )


class TestOtherBaseBehaviour(OtherBaseBehaviourTests):
    """
    Class to run tests for BaseTransformerBehaviour behaviour outside the three standard methods.
    May need to overwite specific tests in this class if the tested transformer modifies this behaviour.
    """

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "MedianImputer"
