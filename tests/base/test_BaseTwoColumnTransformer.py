from tests.base_tests import (
    GenericFitTests,
    GenericTransformTests,
    OtherBaseBehaviourTests,
    TwoColumnListInitTests,
)


class TestInit(TwoColumnListInitTests):
    """Generic tests for transformer.init()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseTwoColumnTransformer"


class TestFit(GenericFitTests):
    """Generic tests for transformer.fit()"""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseTwoColumnTransformer"


class TestTransform(GenericTransformTests):
    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseTwoColumnTransformer"


class TestOtherBaseBehaviour(OtherBaseBehaviourTests):
    """
    Class to run tests for BaseTransformerBehaviour outside the three standard methods.

    May need to overwite specific tests in this class if the tested transformer modifies this behaviour.
    """

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseTwoColumnTransformer"
