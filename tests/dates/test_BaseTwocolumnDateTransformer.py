import pytest

from tests.dates.test_BaseDateTransformer import (
    ColumnStrListInitTests,
    GenericDatesTransformTests,
    GenericFitTests,
    OtherBaseBehaviourTests,
)


class GenericTwoColumnDatesInitTests:
    """Generic tests for Init Dates Transformers which take two columns"""

    @pytest.mark.parametrize("columns", [["a"], ["a", "b", "c"]])
    def test_not_two_columns_error(
        self,
        uninitialized_transformers,
        minimal_attribute_dict,
        columns,
    ):
        """ "test that two correct error raised when passed more or less than two columns"""

        init_args = minimal_attribute_dict[self.transformer_name]
        init_args[columns] = columns

        msg = f"{self.transformer_name}: This transformer works with two columns only"

        with pytest.raises(
            ValueError,
            match=msg,
        ):
            uninitialized_transformers[self.transformer_name](init_args)


class TestInit(ColumnStrListInitTests, GenericTwoColumnDatesInitTests):
    """Generic tests for transformer.init()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseDateTwoColumnTransformer"


class TestFit(GenericFitTests):
    """Generic tests for transformer.fit()"""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseDateTwoColumnTransformer"


class TestTransform(GenericDatesTransformTests):
    """Tests for BaseTwoColumnDateTransformer.transform."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseDateTwoColumnTransformer"


class TestOtherBaseBehaviour(OtherBaseBehaviourTests):
    """
    Class to run tests for BaseTransformerBehaviour outside the three standard methods.

    May need to overwite specific tests in this class if the tested transformer modifies this behaviour.
    """

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseDateTwoColumnTransformer"
