import pandas as pd
import pytest

from tests.base_tests import (
    ColumnStrListInitTests,
    GenericFitTests,
    GenericTransformTests,
    OtherBaseBehaviourTests,
)
from tests.imputers.test_BaseImputer import GenericImputerTransformTests
from tubular.imputers import ArbitraryImputer


# Dataframe used exclusively in this testing script
def create_downcast_df():
    """Create a dataframe with mixed dtypes to use in downcasting tests."""
    return pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "b": [1.0, 2.0, 3.0, 4.0, 5.0],
        },
    )


class TestInit(ColumnStrListInitTests):
    """Generic tests for transformer.init()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "ArbitraryImputer"

    def test_impute_value_type_error(
        self,
        uninitialized_transformers,
        minimal_attribute_dict,
    ):
        """Test that an exception is raised if impute_value is not an int, float or str."""

        args = minimal_attribute_dict[self.transformer_name].copy()
        args["impute_value"] = [1, 2]

        with pytest.raises(
            ValueError,
            match="ArbitraryImputer: impute_value should be a single value .*",
        ):
            uninitialized_transformers[self.transformer_name](**args)


class TestFit(GenericFitTests):
    """Generic tests for transformer.fit()"""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "ArbitraryImputer"


class TestTransform(GenericImputerTransformTests, GenericTransformTests):
    """Tests for transformer.transform."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "ArbitraryImputer"

    # Unit testing to check if downcast datatypes of columns is preserved after imputation is done
    def test_impute_value_preserve_dtype(self):
        """Testing downcast dtypes of columns are preserved after imputation using the create_downcast_df dataframe.

        Explicitly setting the dtype of "a" to int8 and "b" to float16 and check if the dtype of the columns are preserved after imputation.
        """
        df = (
            create_downcast_df()
        )  # By default the dtype of "a" and "b" are int64 and float64 respectively

        # Imputing the dataframe
        x = ArbitraryImputer(impute_value=1, columns=["a", "b"])

        # Setting the dtype of "a" to int8 and "b" to float16
        df["a"] = df["a"].astype("int8")
        df["b"] = df["b"].astype("float16")

        # Checking if the dtype of "a" and "b" are int8 and float16 respectively
        assert df["a"].dtype == "int8"
        assert df["b"].dtype == "float16"

        # Impute the dataframe
        df = x.transform(df)

        # Checking if the dtype of "a" and "b" are int8 and float16 respectively after imputation
        assert df["a"].dtype == "int8"
        assert df["b"].dtype == "float16"


class TestOtherBaseBehaviour(OtherBaseBehaviourTests):
    """
    Class to run tests for BaseTransformerBehaviour outside the three standard methods.

    May need to overwite specific tests in this class if the tested transformer modifies this behaviour.
    """

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "ArbitraryImputer"
