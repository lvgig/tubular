import pytest
import test_aide as ta
import tests.test_data as d
import pandas as pd

import tubular
from tubular.mapping import MappingTransformer
from tubular.base import ReturnKeyDict


class TestInit(object):
    """Tests for MappingTransformer.init()."""

    def test_arguments(self):
        """Test that init has expected arguments."""

        ta.function_helpers.test_function_arguments(
            func=MappingTransformer.__init__,
            expected_arguments=["self", "mappings"],
            expected_default_values=None,
        )

    def test_class_methods(self):
        """Test that MappingTransformer has transform method."""

        x = MappingTransformer(mappings={"a": {"a": 1}})

        ta.class_helpers.test_object_method(
            obj=x, expected_method="transform", msg="transform"
        )

    def test_inheritance(self):
        """Test that MappingTransformer inherits from BaseMappingTransformer and BaseMappingTransformMixin."""

        x = MappingTransformer(mappings={"a": {"a": 1}})

        ta.class_helpers.assert_inheritance(x, tubular.mapping.BaseMappingTransformer)
        ta.class_helpers.assert_inheritance(
            x, tubular.mapping.BaseMappingTransformMixin
        )

    def test_super_init_called(self, mocker):
        """Test that init calls BaseMappingTransformer.init."""

        spy = mocker.spy(tubular.mapping.BaseMappingTransformer, "__init__")

        x = MappingTransformer(mappings={"a": {"a": 1}}, verbose=True, copy=True)

        assert (
            spy.call_count == 1
        ), "unexpected number of calls to BaseMappingTransformer.__init__"

        call_args = spy.call_args_list[0]
        call_pos_args = call_args[0]
        call_kwargs = call_args[1]

        expected_kwargs = {
            "mappings": {"a": ReturnKeyDict({"a": 1})},
            "verbose": True,
            "copy": True,
        }

        assert (
            call_kwargs == expected_kwargs
        ), "unexpected kwargs in BaseMappingTransformer.__init__ call"

        expected_pos_args = (x,)

        assert (
            expected_pos_args == call_pos_args
        ), "unexpected positional args in BaseMappingTransformer.__init__ call"

    def test_mapping_arg_conversion(self):
        """Test that sub dict in mappings arg are converted into ReturnKeyDict objects."""

        mappings = {
            "a": {"a": 1, "b": 2},
            "b": {1: 4.5, 2: 3.1},
            "c": {False: None, True: 1},
        }

        expected_mappings = {
            "a": ReturnKeyDict(mappings["a"]),
            "b": ReturnKeyDict(mappings["b"]),
            "c": ReturnKeyDict(mappings["c"]),
        }

        x = MappingTransformer(mappings=mappings)

        ta.equality_helpers.assert_equal_dispatch(
            expected_mappings,
            x.mappings,
            "mappings attribute not correctly converted sub dicts to ReturnKeyDict",
        )

    def test_mapping_non_dict_item_error(self):
        """Test an exception is raised if mappings contains non-dict values."""

        mappings = {"a": {"a": 1, "b": 2}, "b": {1: 4.5, 2: 3.1}, "c": 1}

        with pytest.raises(
            TypeError,
            match=f"each item in mappings should be a dict but got type {type(1)} for key c",
        ):

            MappingTransformer(mappings=mappings)


class TestTransform(object):
    """Tests for the transform method on MappingTransformer."""

    def expected_df_1():
        """Expected output for test_expected_output."""

        df = pd.DataFrame(
            {"a": ["a", "b", "c", "d", "e", "f"], "b": [1, 2, 3, 4, 5, 6]}
        )

        return df

    def expected_df_2():
        """Expected output for test_non_specified_values_unchanged."""

        df = pd.DataFrame(
            {"a": [5, 6, 7, 4, 5, 6], "b": ["z", "y", "x", "d", "e", "f"]}
        )

        return df

    def test_arguments(self):
        """Test that transform has expected arguments."""

        ta.function_helpers.test_function_arguments(
            func=MappingTransformer.transform,
            expected_arguments=["self", "X"],
            expected_default_values=None,
        )

    def test_super_transform_call(self, mocker):
        """Test the call to BaseMappingTransformMixin.transform."""

        df = d.create_df_1()

        mapping = {
            "a": {1: "a", 2: "b", 3: "c", 4: "d", 5: "e", 6: "f"},
            "b": {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6},
        }

        x = MappingTransformer(mappings=mapping)

        spy = mocker.spy(tubular.mapping.BaseMappingTransformMixin, "transform")

        x.transform(df)

        assert (
            spy.call_count == 1
        ), "unexpected number of calls to BaseMappingTransformMixin.transform"

        call_args = spy.call_args_list[0]
        call_pos_args = call_args[0]
        call_kwargs = call_args[1]

        expected_kwargs = {}

        assert (
            call_kwargs == expected_kwargs
        ), "unexpected kwargs in BaseMappingTransformMixin.transform call"

        expected_pos_args = (x, d.create_df_1())

        assert (
            expected_pos_args[0] == call_pos_args[0]
        ), "unexpected 1st positional arg in BaseMappingTransformMixin.transform call"

        ta.equality_helpers.assert_equal_dispatch(
            expected_pos_args[1],
            call_pos_args[1],
            "unexpected 2ns positional arg in BaseMappingTransformMixin.transform call",
        )

    @pytest.mark.parametrize(
        "df, expected",
        ta.pandas_helpers.row_by_row_params(d.create_df_1(), expected_df_1())
        + ta.pandas_helpers.index_preserved_params(
            d.create_df_1(), expected_df_1()
        ),
    )
    def test_expected_output(self, df, expected):
        """Test that transform is giving the expected output."""

        mapping = {
            "a": {1: "a", 2: "b", 3: "c", 4: "d", 5: "e", 6: "f"},
            "b": {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6},
        }

        x = MappingTransformer(mappings=mapping)

        df_transformed = x.transform(df)

        ta.equality_helpers.assert_frame_equal_msg(
            actual=df_transformed,
            expected=expected,
            msg_tag="expected output from mapping transformer",
        )

    @pytest.mark.parametrize(
        "df, expected",
        ta.pandas_helpers.row_by_row_params(d.create_df_1(), expected_df_2())
        + ta.pandas_helpers.index_preserved_params(
            d.create_df_1(), expected_df_2()
        ),
    )
    def test_non_specified_values_unchanged(self, df, expected):
        """Test that values not specified in mappings are left unchanged in transform."""

        mapping = {"a": {1: 5, 2: 6, 3: 7}, "b": {"a": "z", "b": "y", "c": "x"}}

        x = MappingTransformer(mappings=mapping)

        df_transformed = x.transform(df)

        ta.equality_helpers.assert_frame_equal_msg(
            actual=df_transformed,
            expected=expected,
            msg_tag="expected output from mapping transformer",
        )

    def test_mappings_unchanged(self):
        """Test that mappings is unchanged in transform."""

        df = d.create_df_1()

        mapping = {
            "a": {1: "a", 2: "b", 3: "c", 4: "d", 5: "e", 6: "f"},
            "b": {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6},
        }

        preserve_original_value_mapping = {
            "a": ReturnKeyDict(mapping["a"]),
            "b": ReturnKeyDict(mapping["b"]),
        }

        x = MappingTransformer(mappings=mapping)

        x.transform(df)

        ta.equality_helpers.assert_equal_dispatch(
            actual=x.mappings,
            expected=preserve_original_value_mapping,
            msg="MappingTransformer.transform has changed self.mappings unexpectedly",
        )
