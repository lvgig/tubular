from tests.base_tests import (
    ColumnStrListInitTests,
    GenericFitTests,
    GenericTransformTests,
    OtherBaseBehaviourTests,
)


class GenericTwoColumnDatesTransformTests:
    """Generic tests for Dates Transformers"""


class TestInit(ColumnStrListInitTests):
    """Generic tests for transformer.init()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseDateTwoColumnTransformer"


class TestFit(GenericFitTests):
    """Generic tests for transformer.fit()"""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseDateTwoColumnTransformer"


class TestTransform(GenericTransformTests, GenericTwoColumnDatesTransformTests):
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
