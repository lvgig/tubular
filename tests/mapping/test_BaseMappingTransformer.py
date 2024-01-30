import pytest
import test_aide as ta

import tests.test_data as d
from tests.base_tests import GenericFitTests, OtherBaseBehaviourTests
from tests.specific_column_type_tests import ColumnsFromDictInitTests


# The first part of this file builds out the tests for BaseMappingTransformer so that they can be
# imported into other test files (by not starting the class name with Test)
# The second part actually calls these tests (along with all other require tests) for the BaseMappingTransformer
class BaseMappingTransformerInitTests(ColumnsFromDictInitTests):
    """
    Tests for BaseMappingTransformer.init().
    Note this deliberately avoids starting with "Tests" so that the tests are not run on import.
    """

    def test_no_keys_dict_error(self, uninstantiated_transformers):
        """Test that an exception is raised if mappings is a dict but with no keys."""
        with pytest.raises(
            ValueError,
            match="BaseMappingTransformer: mappings has no values",
        ):
            uninstantiated_transformers[self.transformer_name](mappings={})

    def test_mappings_contains_non_dict_items_error(self, uninstantiated_transformers):
        """Test that an exception is raised if mappings contains non-dict items."""
        with pytest.raises(
            ValueError,
            match="BaseMappingTransformer: values in mappings dictionary should be dictionaries",
        ):
            uninstantiated_transformers[self.transformer_name](
                mappings={"a": {"a": 1}, "b": 1},
            )

    def test_mappings_not_dict_error(self, uninstantiated_transformers):
        """Test that an exception is raised if mappings is not a dict."""
        with pytest.raises(
            ValueError,
            match="BaseMappingTransformer: mappings must be a dictionary",
        ):
            uninstantiated_transformers[self.transformer_name](mappings=())

    def test_mappings_set_to_attribute(self, uninstantiated_transformers):
        """Test that the value passed for mappings is saved in an attribute of the same name."""
        value = {"a": {"a": 1}, "b": {"a": 1}}

        x = uninstantiated_transformers[self.transformer_name](mappings=value)

        ta.classes.test_object_attributes(
            obj=x,
            expected_attributes={"mappings": value},
            msg="Attributes for BaseMappingTransformer set in init",
        )


class BaseMappingTransformerTransformerTests:
    """
    Tests for the transform method on MappingTransformer.
    Note this deliberately avoids starting with "Tests" so that the tests are not run on import.
    """

    @pytest.mark.parametrize(
        ("df", "expected"),
        ta.pandas.adjusted_dataframe_params(d.create_df_1(), d.create_df_1()),
    )
    def test_X_returned(self, df, expected, uninstantiated_transformers):
        """Test that X is returned from transform."""
        mapping = {
            "a": {1: "a", 2: "b", 3: "c", 4: "d", 5: "e", 6: "f"},
            "b": {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6},
        }

        x = uninstantiated_transformers[self.transformer_name](mappings=mapping)

        df_transformed = x.transform(df)

        ta.equality.assert_equal_dispatch(
            expected=expected,
            actual=df_transformed,
            msg="Check X returned from transform",
        )

    def test_mappings_unchanged(self, uninstantiated_transformers):
        """Test that mappings is unchanged in transform."""
        df = d.create_df_1()

        mapping = {
            "a": {1: "a", 2: "b", 3: "c", 4: "d", 5: "e", 6: "f"},
            "b": {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6},
        }

        x = uninstantiated_transformers[self.transformer_name](mappings=mapping)

        x.transform(df)

        ta.equality.assert_equal_dispatch(
            expected=mapping,
            actual=x.mappings,
            msg="BaseMappingTransformer.transform has changed self.mappings unexpectedly",
        )


### Running the BaseMappingTransformerTestSuite


class TestInit(BaseMappingTransformerInitTests):
    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseMappingTransformer"


class TestFit(GenericFitTests):
    """Generic tests for transformer.fit()"""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseMappingTransformer"


class TestTransform(BaseMappingTransformerTransformerTests):
    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseMappingTransformer"


class TestOtherBaseBehaviour(OtherBaseBehaviourTests):
    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseMappingTransformer"
