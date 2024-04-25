from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
import test_aide as ta
from sklearn.exceptions import NotFittedError

import tests.test_data as d
from tests.base_tests import (
    ColumnStrListInitTests,
    GenericFitTests,
    OtherBaseBehaviourTests,
)


class GenericImputerFitTestsWeight:
    def test_fit_returns_self_weighted(
        self,
        minimal_attribute_dict,
        uninitialized_transformers,
    ):
        """Test fit returns self?."""
        df = d.create_df_9()
        args = minimal_attribute_dict[self.transformer_name].copy()
        args["weight"] = "c"

        transformer = uninitialized_transformers[self.transformer_name](**args)

        x_fitted = transformer.fit(df)

        assert (
            x_fitted is transformer
        ), f"Returned value from {self.transformer_name}.fit not as expected."

    def test_fit_not_changing_data_weighted(
        self,
        minimal_attribute_dict,
        uninitialized_transformers,
    ):
        """Test fit does not change X - when weights are used."""
        df = d.create_df_9()

        args = minimal_attribute_dict[self.transformer_name].copy()
        args["weight"] = "c"

        transformer = uninitialized_transformers[self.transformer_name](**args)

        transformer.fit(df)
        ta.equality.assert_equal_dispatch(
            expected=d.create_df_9(),
            actual=df,
            msg=f"X changed during fit for {self.transformer_name}",
        )


class GenericImputerTransformTests:
    def test_not_fitted_error_raised(self, initialized_transformers):
        if initialized_transformers[self.transformer_name].FITS:
            df = pd.DataFrame(
                {
                    "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                    "b": ["a", "b", "c", "d", "e", "f", np.nan],
                    "c": ["a", "b", "c", "d", "e", "f", np.nan],
                },
            )

            with pytest.raises(NotFittedError):
                initialized_transformers[self.transformer_name].transform(df)

    def test_impute_value_unchanged(self, initialized_transformers):
        """Test that self.impute_value is unchanged after transform."""
        df = d.create_df_1()

        transformer = initialized_transformers[self.transformer_name]
        transformer.impute_values_ = {"a": 1}

        impute_values = deepcopy(transformer.impute_values_)

        transformer.transform(df)

        ta.classes.test_object_attributes(
            obj=transformer,
            expected_attributes={"impute_values_": impute_values},
            msg="impute_values_ changed in transform",
        )

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
    def test_expected_output_1(self, df, expected, initialized_transformers):
        """Test that transform is giving the expected output when applied to float column."""
        x1 = initialized_transformers[self.transformer_name]
        x1.impute_values_ = {"a": 7}
        x1.columns = ["a"]

        df_transformed = x1.transform(df)

        ta.equality.assert_equal_dispatch(
            expected=expected,
            actual=df_transformed,
            msg=f"Error from {self.transformer_name} transform col a",
        )

    @pytest.mark.parametrize(
        ("df", "expected"),
        ta.pandas.adjusted_dataframe_params(d.create_df_2(), expected_df_2()),
    )
    def test_expected_output_2(self, df, expected, initialized_transformers):
        """Test that transform is giving the expected output when applied to object column."""
        x1 = initialized_transformers[self.transformer_name]

        x1.impute_values_ = {"b": "g"}
        x1.columns = ["b"]

        df_transformed = x1.transform(df)

        ta.equality.assert_equal_dispatch(
            expected=expected,
            actual=df_transformed,
            msg=f"Error from {self.transformer_name} transform col b",
        )

    @pytest.mark.parametrize(
        ("df", "expected"),
        ta.pandas.adjusted_dataframe_params(d.create_df_2(), expected_df_3()),
    )
    def test_expected_output_3(self, df, expected, initialized_transformers):
        """Test that transform is giving the expected output when applied to object and categorical columns."""
        x1 = initialized_transformers[self.transformer_name]

        x1.impute_values_ = {"b": "g", "c": "f"}
        x1.columns = ["b", "c"]

        df_transformed = x1.transform(df)

        # arb imputer will add a new categorical level to cat columns,
        # make sure expected takes this into account
        if self.transformer_name == "ArbitraryImputer":
            expected["c"] = expected["c"].cat.add_categories(
                x1.impute_value,
            )

        ta.equality.assert_equal_dispatch(
            expected=expected,
            actual=df_transformed,
            msg=f"Error from {self.transformer_name} transform col b, c",
        )


class GenericImputerTransformTestsWeight:
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
    def test_nulls_imputed_correctly_weights(
        self,
        df,
        expected,
        minimal_attribute_dict,
        uninitialized_transformers,
    ):
        """Test missing values are filled with the correct values - and unrelated columns are not changed
        (when weight is used).
        """
        args = minimal_attribute_dict[self.transformer_name].copy()
        args["weight"] = "c"

        transformer = uninitialized_transformers[self.transformer_name](**args)

        # set the impute values dict directly rather than fitting x on df so test works with helpers
        transformer.impute_values_ = {"a": 4}

        df_transformed = transformer.transform(df)

        ta.equality.assert_equal_dispatch(
            expected=expected,
            actual=df_transformed,
            msg=f"Check nulls filled correctly in transform for {self.transformer_name}",
        )

    def test_learnt_values_not_modified_weights(
        self,
        minimal_attribute_dict,
        uninitialized_transformers,
    ):
        """Test that the impute_values_ from fit are not changed in transform - when using weights."""
        df = d.create_df_9()

        args = minimal_attribute_dict[self.transformer_name].copy()
        args["columns"] = ["a", "b"]
        args["weight"] = "c"

        transformer1 = uninitialized_transformers[self.transformer_name](**args)

        transformer1.fit(df)

        transformer2 = uninitialized_transformers[self.transformer_name](**args)

        transformer2.fit_transform(df)

        ta.equality.assert_equal_dispatch(
            expected=transformer1.impute_values_,
            actual=transformer2.impute_values_,
            msg=f"Impute values changed in transform for {self.transformer_name}",
        )


class TestInit(ColumnStrListInitTests):
    """Generic tests for transformer.init()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseImputer"


class TestFit(GenericFitTests):
    """Generic tests for transformer.fit()"""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseImputer"


class TestTransform(GenericImputerTransformTests):
    """Tests for BaseImputer.transform."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseImputer"


class TestOtherBaseBehaviour(OtherBaseBehaviourTests):
    """
    Class to run tests for BaseTransformerBehaviour outside the three standard methods.

    May need to overwite specific tests in this class if the tested transformer modifies this behaviour.
    """

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseImputer"
