import pandas as pd
import pytest
import test_aide as ta
from pandas.api.types import (
    is_bool_dtype,
    is_categorical_dtype,
    is_float_dtype,
    is_integer_dtype,
    is_object_dtype,
)

import tests.test_data as d
import tubular
from tubular.mapping import MappingTransformer


class TestInit:
    """Tests for MappingTransformer.init()."""

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
            "mappings": {"a": {"a": 1}},
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

    def test_mapping_non_dict_item_error(self):
        """Test an exception is raised if mappings contains non-dict values."""
        mappings = {"a": {"a": 1, "b": 2}, "b": {1: 4.5, 2: 3.1}, "c": 1}

        with pytest.raises(
            TypeError,
            match=f"MappingTransformer: each item in mappings should be a dict but got type {int} for key c",
        ):
            MappingTransformer(mappings=mappings)


class TestTransform:
    """Tests for the transform method on MappingTransformer."""

    def expected_df_1():
        """Expected output for test_expected_output."""
        return pd.DataFrame(
            {"a": ["a", "b", "c", "d", "e", "f"], "b": [1, 2, 3, 4, 5, 6]},
        )

    def expected_df_2():
        """Expected output for test_non_specified_values_unchanged."""
        return pd.DataFrame(
            {"a": [5, 6, 7, 4, 5, 6], "b": ["z", "y", "x", "d", "e", "f"]},
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

        ta.equality.assert_equal_dispatch(
            expected_pos_args[1],
            call_pos_args[1],
            "unexpected 2ns positional arg in BaseMappingTransformMixin.transform call",
        )

    @pytest.mark.parametrize(
        ("df", "expected"),
        ta.pandas.adjusted_dataframe_params(d.create_df_1(), expected_df_1()),
    )
    def test_expected_output(self, df, expected):
        """Test that transform is giving the expected output."""
        mapping = {
            "a": {1: "a", 2: "b", 3: "c", 4: "d", 5: "e", 6: "f"},
            "b": {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6},
        }

        x = MappingTransformer(mappings=mapping)

        df_transformed = x.transform(df)

        ta.equality.assert_frame_equal_msg(
            actual=df_transformed,
            expected=expected,
            msg_tag="expected output from mapping transformer",
        )

    @pytest.mark.parametrize(
        ("df", "expected"),
        ta.pandas.adjusted_dataframe_params(d.create_df_1(), expected_df_2()),
    )
    def test_non_specified_values_unchanged(self, df, expected):
        """Test that values not specified in mappings are left unchanged in transform."""
        mapping = {"a": {1: 5, 2: 6, 3: 7}, "b": {"a": "z", "b": "y", "c": "x"}}

        x = MappingTransformer(mappings=mapping)

        df_transformed = x.transform(df)

        ta.equality.assert_frame_equal_msg(
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
            "a": dict(mapping["a"]),
            "b": dict(mapping["b"]),
        }

        x = MappingTransformer(mappings=mapping)

        x.transform(df)

        ta.equality.assert_equal_dispatch(
            actual=x.mappings,
            expected=preserve_original_value_mapping,
            msg="MappingTransformer.transform has changed self.mappings unexpectedly",
        )

    @pytest.mark.parametrize(
        ("mapping", "input_col_name", "output_col_type_check"),
        [
            ({"a": {1: 1.1, 6: 6.6}}, "a", is_float_dtype),
            ({"a": {1: "one", 6: "six"}}, "a", is_object_dtype),
            (
                {"a": {1: True, 2: True, 3: True, 4: False, 5: False, 6: False}},
                "a",
                is_bool_dtype,
            ),
            (
                {"b": {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6}},
                "b",
                is_integer_dtype,
            ),
            (
                {"b": {"a": 1.1, "b": 2.2, "c": 3.3, "d": 4.4, "e": 5.5, "f": 6.6}},
                "b",
                is_float_dtype,
            ),
        ],
    )
    def test_expected_dtype_conversions(
        self,
        mapping,
        input_col_name,
        output_col_type_check,
    ):
        df = d.create_df_1()
        x = MappingTransformer(mappings=mapping)
        df = x.transform(df)

        assert output_col_type_check(df[input_col_name])

    @pytest.mark.parametrize(
        ("mapping", "input_col_name", "input_col_type"),
        [
            ({"a": {1: True, 6: False}}, "a", "int64"),
        ],
    )
    def test_unexpected_dtype_change_warning_raised(
        self,
        mapping,
        input_col_name,
        input_col_type,
    ):
        df = d.create_df_1()
        print(df["a"])

        x = MappingTransformer(mappings=mapping)

        with pytest.warns(
            UserWarning,
            match=f"MappingTransformer: This mapping changes {input_col_name} dtype from {input_col_type} to object. This is often caused by having multiple dtypes in one column, or by not mapping all values",
        ):
            x.transform(df)

    def test_unexpected_dtype_change_warning_suppressed(
        self,
        recwarn,
    ):
        df = d.create_df_1()

        mapping = {"a": {1: True, 6: False}}

        x = MappingTransformer(mappings=mapping)

        x.transform(df, suppress_dtype_warning=True)

        assert (
            len(recwarn) == 0
        ), "MappingTransformer: warning raised for dtype change with supress_dtype_warning=True"

    def test_category_dtype_is_conserved(self):
        """This is a separate test due to the behaviour of category dtypes.

        See documentation of transform method
        """
        df = d.create_df_1()
        df["b"] = df["b"].astype("category")

        mapping = mapping = {"b": {"a": "aaa", "b": "bbb"}}

        x = MappingTransformer(mappings=mapping)
        df = x.transform(df)

        assert is_categorical_dtype(df["b"])

    @pytest.mark.parametrize(
        ("mapping", "mapped_col"),
        [({"a": {99: "99", 98: "98"}}, "a"), ({"b": {"z": 99, "y": 98}}, "b")],
    )
    def test_no_applicable_mapping(self, mapping, mapped_col):
        df = d.create_df_1()

        x = MappingTransformer(mappings=mapping)

        with pytest.warns(
            UserWarning,
            match=f"MappingTransformer: No values from mapping for {mapped_col} exist in dataframe.",
        ):
            x.transform(df)

    @pytest.mark.parametrize(
        ("mapping", "mapped_col"),
        [({"a": {1: "1", 99: "99"}}, "a"), ({"b": {"a": 1, "z": 99}}, "b")],
    )
    def test_excess_mapping_values(self, mapping, mapped_col):
        df = d.create_df_1()

        x = MappingTransformer(mappings=mapping)

        with pytest.warns(
            UserWarning,
            match=f"MappingTransformer: There are values in the mapping for {mapped_col} that are not present in the dataframe",
        ):
            x.transform(df)
