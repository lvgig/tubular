import pytest
import test_aide as ta
import tests.test_data as d

import tubular
from tubular.mapping import BaseMappingTransformer


class TestInit(object):
    """Tests for BaseMappingTransformer.init()."""

    def test_arguments(self):
        """Test that init has expected arguments."""

        ta.function_helpers.test_function_arguments(
            func=BaseMappingTransformer.__init__,
            expected_arguments=["self", "mappings"],
            expected_default_values=None,
        )

    def test_class_methods(self):
        """Test that BaseMappingTransformer has transform method."""

        x = BaseMappingTransformer(mappings={"a": {"a": 1}})

        ta.class_helpers.test_object_method(
            obj=x, expected_method="transform", msg="transform"
        )

    def test_inheritance(self):
        """Test that BaseMappingTransformer inherits from BaseTransformer."""

        x = BaseMappingTransformer(mappings={"a": {"a": 1}})

        ta.class_helpers.assert_inheritance(x, tubular.base.BaseTransformer)

    def test_super_init_called(self, mocker):
        """Test that init calls BaseTransformer.init."""

        expected_call_args = {
            0: {"args": (), "kwargs": {"columns": ["a"], "verbose": True, "copy": True}}
        }

        with ta.function_helpers.assert_function_call(
            mocker, tubular.base.BaseTransformer, "__init__", expected_call_args
        ):

            BaseMappingTransformer(mappings={"a": {"a": 1}}, verbose=True, copy=True)

    def test_no_keys_dict_error(self):
        """Test that an exception is raised if mappings is a dict but with no keys."""

        with pytest.raises(ValueError, match="mappings has no values"):

            BaseMappingTransformer(mappings={})

    def test_mappings_contains_non_dict_items_error(self):
        """Test that an exception is raised if mappings contains non-dict items."""

        with pytest.raises(
            ValueError, match="values in mappings dictionary should be dictionaries"
        ):

            BaseMappingTransformer(mappings={"a": {"a": 1}, "b": 1})

    def test_mappings_not_dict_error(self):
        """Test that an exception is raised if mappings is not a dict."""

        with pytest.raises(ValueError, match="mappings must be a dictionary"):

            BaseMappingTransformer(mappings=())

    def test_mappings_set_to_attribute(self):
        """Test that the value passed for mappings is saved in an attribute of the same name."""

        value = {"a": {"a": 1}, "b": {"a": 1}}

        x = BaseMappingTransformer(mappings=value)

        ta.class_helpers.test_object_attributes(
            obj=x,
            expected_attributes={"mappings": value},
            msg="Attributes for BaseMappingTransformer set in init",
        )


class TestTransform(object):
    """Tests for the transform method on MappingTransformer."""

    def test_arguments(self):
        """Test that transform has expected arguments."""

        ta.function_helpers.test_function_arguments(
            func=BaseMappingTransformer.transform,
            expected_arguments=["self", "X"],
            expected_default_values=None,
        )

    def test_check_is_fitted_call(self, mocker):
        """Test the call to check_is_fitted."""

        df = d.create_df_1()

        mapping = {
            "a": {1: "a", 2: "b", 3: "c", 4: "d", 5: "e", 6: "f"},
            "b": {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6},
        }

        x = BaseMappingTransformer(mappings=mapping)

        expected_call_args = {0: {"args": (["mappings"],), "kwargs": {}}}

        with ta.function_helpers.assert_function_call(
            mocker, tubular.base.BaseTransformer, "check_is_fitted", expected_call_args
        ):

            x.transform(df)

    def test_super_transform_call(self, mocker):
        """Test the call to BaseTransformer.transform."""

        df = d.create_df_1()

        mapping = {
            "a": {1: "a", 2: "b", 3: "c", 4: "d", 5: "e", 6: "f"},
            "b": {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6},
        }

        x = BaseMappingTransformer(mappings=mapping)

        expected_call_args = {0: {"args": (d.create_df_1(),), "kwargs": {}}}

        with ta.function_helpers.assert_function_call(
            mocker, tubular.base.BaseTransformer, "transform", expected_call_args
        ):

            x.transform(df)

    @pytest.mark.parametrize(
        "df, expected",
        ta.pandas_helpers.row_by_row_params(
            d.create_df_1(), d.create_df_1()
        )
        + ta.pandas_helpers.index_preserved_params(
            d.create_df_1(), d.create_df_1()
        ),
    )
    def test_X_returned(self, df, expected):
        """Test that X is returned from transform."""

        mapping = {
            "a": {1: "a", 2: "b", 3: "c", 4: "d", 5: "e", 6: "f"},
            "b": {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6},
        }

        x = BaseMappingTransformer(mappings=mapping)

        df_transformed = x.transform(df)

        ta.equality_helpers.assert_equal_dispatch(
            expected=expected,
            actual=df_transformed,
            msg="Check X returned from transform",
        )

    def test_mappings_unchanged(self):
        """Test that mappings is unchanged in transform."""

        df = d.create_df_1()

        mapping = {
            "a": {1: "a", 2: "b", 3: "c", 4: "d", 5: "e", 6: "f"},
            "b": {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6},
        }

        x = BaseMappingTransformer(mappings=mapping)

        x.transform(df)

        ta.equality_helpers.assert_equal_dispatch(
            expected=mapping,
            actual=x.mappings,
            msg="BaseMappingTransformer.transform has changed self.mappings unexpectedly",
        )
