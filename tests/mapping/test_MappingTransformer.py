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
from tests.mapping.test_BaseMappingTransformer import (
    BaseMappingTransformerInitTests,
    BaseMappingTransformerTransformTests,
    GenericFitTests,
    OtherBaseBehaviourTests,
)
from tubular.mapping import MappingTransformer


class TestInit(BaseMappingTransformerInitTests):
    """Tests for MappingTransformer.init()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "MappingTransformer"


class TestFit(GenericFitTests):
    """Generic tests for transformer.fit()"""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "MappingTransformer"


class TestTransform(BaseMappingTransformerTransformTests):
    """Tests for the transform method on MappingTransformer."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "MappingTransformer"

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

        mapping = {"b": {"a": "aaa", "b": "bbb"}}

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


class TestOtherBaseBehaviour(OtherBaseBehaviourTests):
    """
    Class to run tests for BaseTransformerBehaviour outside the three standard methods.

    May need to overwite specific tests in this class if the tested transformer modifies this behaviour.
    """

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "MappingTransformer"
