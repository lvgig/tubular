import numpy as np
import pandas as pd
import pytest
import test_aide as ta

import tests.test_data as d
import tubular
from tubular.mapping import CrossColumnMultiplyTransformer


class TestInit:
    """Tests for CrossColumnMultiplyTransformer.init()."""

    def test_arguments(self):
        """Test that init has expected arguments."""
        ta.functions.test_function_arguments(
            func=CrossColumnMultiplyTransformer.__init__,
            expected_arguments=["self", "adjust_column", "mappings"],
            expected_default_values=None,
        )

    def test_class_methods(self):
        """Test that CrossColumnMultiplyTransformer has transform method."""
        x = CrossColumnMultiplyTransformer(mappings={"a": {"a": 1}}, adjust_column="b")

        ta.classes.test_object_method(
            obj=x, expected_method="transform", msg="transform"
        )

    def test_inheritance(self):
        """Test that CrossColumnMultiplyTransformer inherits from BaseMappingTransformer."""
        x = CrossColumnMultiplyTransformer(mappings={"a": {"a": 1}}, adjust_column="b")

        ta.classes.assert_inheritance(x, tubular.mapping.BaseMappingTransformer)

    def test_super_init_called(self, mocker):
        """Test that init calls BaseMappingTransformer.init."""
        expected_call_args = {
            0: {
                "args": (),
                "kwargs": {"mappings": {"a": {"a": 1}}, "verbose": True, "copy": True},
            }
        }

        with ta.functions.assert_function_call(
            mocker,
            tubular.mapping.BaseMappingTransformer,
            "__init__",
            expected_call_args,
        ):
            CrossColumnMultiplyTransformer(
                mappings={"a": {"a": 1}}, adjust_column="b", verbose=True, copy=True
            )

    def test_adjust_columns_non_string_error(self):
        """Test that an exception is raised if adjust_column is not a string."""
        with pytest.raises(
            TypeError,
            match="CrossColumnMultiplyTransformer: adjust_column should be a string",
        ):
            CrossColumnMultiplyTransformer(mappings={"a": {"a": 1}}, adjust_column=1)

    def test_mapping_values_not_numeric_error(self):
        """Test that an exception is raised if mappings values are not numeric."""
        with pytest.raises(
            TypeError,
            match="CrossColumnMultiplyTransformer: mapping values must be numeric",
        ):
            CrossColumnMultiplyTransformer(
                mappings={"a": {"a": "b"}}, adjust_column="b"
            )

    def test_adjust_column_set_to_attribute(self):
        """Test that the value passed for adjust_column is saved in an attribute of the same name."""
        value = "b"

        x = CrossColumnMultiplyTransformer(
            mappings={"a": {"a": 1}}, adjust_column=value
        )

        ta.classes.test_object_attributes(
            obj=x,
            expected_attributes={"adjust_column": value},
            msg="Attributes for CrossColumnMultiplyTransformer set in init",
        )


class TestTransform:
    """Tests for the transform method on CrossColumnMultiplyTransformer."""

    def expected_df_1():
        """Expected output for test_expected_output."""
        df = pd.DataFrame(
            {"a": [1.1, 2.4, 3.9, 5.6, 7.5, 9.6], "b": ["a", "b", "c", "d", "e", "f"]}
        )

        return df

    def expected_df_2():
        """Expected output for test_non_specified_values_unchanged."""
        df = pd.DataFrame(
            {"a": [1.1, 2.4, 3, 4, 5, 6], "b": ["a", "b", "c", "d", "e", "f"]}
        )

        return df

    def expected_df_3():
        """Expected output for test_multiple_mappings_ordered_dict."""
        df = pd.DataFrame(
            {
                "a": [2.2, 4.4, 3.3, 4, 15, 21.6, 7, 8, 9, np.NaN],
                "b": ["a", "a", "a", "d", "e", "f", "g", np.NaN, np.NaN, np.NaN],
                "c": ["a", "a", "c", "c", "e", "e", "f", "g", "h", np.NaN],
            }
        )

        df["c"] = df["c"].astype("category")

        return df

    def test_arguments(self):
        """Test that transform has expected arguments."""
        ta.functions.test_function_arguments(
            func=CrossColumnMultiplyTransformer.transform,
            expected_arguments=["self", "X"],
            expected_default_values=None,
        )

    def test_check_is_fitted_call(self, mocker):
        """Test the call to check_is_fitted."""
        df = d.create_df_1()

        mapping = {"b": {"a": 1.1, "b": 1.2, "c": 1.3, "d": 1.4, "e": 1.5, "f": 1.6}}

        x = CrossColumnMultiplyTransformer(mappings=mapping, adjust_column="a")

        expected_call_args = {0: {"args": (["adjust_column"],), "kwargs": {}}}

        with ta.functions.assert_function_call(
            mocker, tubular.base.BaseTransformer, "check_is_fitted", expected_call_args
        ):
            x.transform(df)

    def test_super_transform_call(self, mocker):
        """Test the call to BaseMappingTransformer.transform."""
        df = d.create_df_1()

        mapping = {"b": {"a": 1.1, "b": 1.2, "c": 1.3, "d": 1.4, "e": 1.5, "f": 1.6}}

        x = CrossColumnMultiplyTransformer(mappings=mapping, adjust_column="a")

        expected_call_args = {0: {"args": (d.create_df_1(),), "kwargs": {}}}

        with ta.functions.assert_function_call(
            mocker,
            tubular.mapping.BaseMappingTransformer,
            "transform",
            expected_call_args,
            return_value=d.create_df_1(),
        ):
            x.transform(df)

    def test_adjust_col_not_in_x_error(self):
        """Test that an exception is raised if the adjust_column is not present in the dataframe."""
        df = d.create_df_1()

        mapping = {"b": {"a": 1.1, "b": 1.2, "c": 1.3, "d": 1.4, "e": 1.5, "f": 1.6}}

        x = CrossColumnMultiplyTransformer(mappings=mapping, adjust_column="c")

        with pytest.raises(
            ValueError, match="CrossColumnMultiplyTransformer: variable c is not in X"
        ):
            x.transform(df)

    def test_adjust_col_not_numeric_error(self):
        """Test that an exception is raised if the adjust_column is not numeric."""
        df = d.create_df_2()

        mapping = {"b": {"a": 1.1, "b": 1.2, "c": 1.3, "d": 1.4, "e": 1.5, "f": 1.6}}

        x = CrossColumnMultiplyTransformer(mappings=mapping, adjust_column="c")

        with pytest.raises(
            TypeError,
            match="CrossColumnMultiplyTransformer: variable c must have numeric dtype.",
        ):
            x.transform(df)

    @pytest.mark.parametrize(
        "df, expected",
        ta.pandas.adjusted_dataframe_params(d.create_df_1(), expected_df_1()),
    )
    def test_expected_output(self, df, expected):
        """Test that transform is giving the expected output."""
        mapping = {"b": {"a": 1.1, "b": 1.2, "c": 1.3, "d": 1.4, "e": 1.5, "f": 1.6}}

        x = CrossColumnMultiplyTransformer(mappings=mapping, adjust_column="a")

        df_transformed = x.transform(df)

        ta.equality.assert_frame_equal_msg(
            actual=df_transformed,
            expected=expected,
            msg_tag="expected output from cross column multiply transformer",
        )

    @pytest.mark.parametrize(
        "df, expected",
        ta.pandas.adjusted_dataframe_params(d.create_df_1(), expected_df_2()),
    )
    def test_non_specified_values_unchanged(self, df, expected):
        """Test that values not specified in mappings are left unchanged in transform."""
        mapping = {"b": {"a": 1.1, "b": 1.2}}

        x = CrossColumnMultiplyTransformer(mappings=mapping, adjust_column="a")

        df_transformed = x.transform(df)

        ta.equality.assert_frame_equal_msg(
            actual=df_transformed,
            expected=expected,
            msg_tag="expected output from cross column multiply transformer",
        )

    @pytest.mark.parametrize(
        "df, expected",
        ta.pandas.adjusted_dataframe_params(d.create_df_5(), expected_df_3()),
    )
    def test_multiple_mappings_expected_output(self, df, expected):
        """Test that mappings by multiple columns are both applied in transform."""
        mapping = {"b": {"a": 1.1, "f": 1.2}, "c": {"a": 2, "e": 3}}

        x = CrossColumnMultiplyTransformer(mappings=mapping, adjust_column="a")

        df_transformed = x.transform(df)

        ta.equality.assert_frame_equal_msg(
            actual=df_transformed,
            expected=expected,
            msg_tag="expected output from cross column multiply transformer",
        )

    def test_mappings_unchanged(self):
        """Test that mappings is unchanged in transform."""
        df = d.create_df_1()

        mapping = {"b": {"a": 1.1, "b": 1.2, "c": 1.3, "d": 1.4, "e": 1.5, "f": 1.6}}

        x = CrossColumnMultiplyTransformer(mappings=mapping, adjust_column="a")

        x.transform(df)

        ta.equality.assert_equal_dispatch(
            expected=mapping,
            actual=x.mappings,
            msg="CrossColumnMultiplyTransformer.transform has changed self.mappings unexpectedly",
        )
