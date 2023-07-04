import re

import numpy as np
import pandas as pd
import pytest
import test_aide as ta

import tests.test_data as d
import tubular
from tubular.numeric import LogTransformer


class TestInit:
    """Tests for LogTransformer.init()."""

    def test_arguments(self):
        """Test that init has expected arguments."""
        ta.functions.test_function_arguments(
            func=LogTransformer.__init__,
            expected_arguments=["self", "columns", "base", "add_1", "drop", "suffix"],
            expected_default_values=(None, False, True, "log"),
        )

    def test_base_type_error(self):
        """Test that an exception is raised if base is non-numeric."""
        with pytest.raises(
            ValueError,
            match=re.escape("LogTransformer: base should be numeric or None"),
        ):
            LogTransformer(
                columns=["a"],
                base="a",
            )

    def test_base_not_strictly_positive_error(self):
        """Test that an exception is raised if base is not strictly positive."""
        with pytest.raises(
            ValueError,
            match=re.escape("LogTransformer: base should be strictly positive"),
        ):
            LogTransformer(
                columns=["a"],
                base=0,
            )

    def test_class_methods(self):
        """Test that LogTransformer has transform method."""
        x = LogTransformer(columns="a")

        ta.classes.test_object_method(
            obj=x, expected_method="transform", msg="transform"
        )

    def test_inheritance(self):
        """Test that LogTransformer inherits from BaseTransformer."""
        x = LogTransformer(columns="a")

        ta.classes.assert_inheritance(x, tubular.base.BaseTransformer)

    def test_super_init_called(self, mocker):
        """Test that init calls BaseTransformer.init."""
        expected_call_args = {
            0: {
                "args": (),
                "kwargs": {"columns": ["a", "b"], "verbose": True, "copy": True},
            }
        }

        with ta.functions.assert_function_call(
            mocker, tubular.base.BaseTransformer, "__init__", expected_call_args
        ):
            LogTransformer(
                columns=["a", "b"],
                add_1=True,
                drop=True,
                suffix="_new",
                verbose=True,
                copy=True,
            )

    def test_impute_values_set_to_attribute(self):
        """Test that the value passed for impute_value is saved in an attribute of the same name."""
        x = LogTransformer(
            columns=["a", "b"],
            base=1,
            add_1=True,
            drop=False,
            suffix="new",
            verbose=True,
            copy=True,
        )

        expected_attributes = {"base": 1, "add_1": True, "drop": False, "suffix": "new"}

        ta.classes.test_object_attributes(
            obj=x,
            expected_attributes=expected_attributes,
            msg="Attributes for LogTransformer set in init",
        )


class TestTransform:
    """Tests for LogTransformer.transform()."""

    def expected_df_1():
        """Expected output of test_expected_output_1."""
        df = d.create_df_3()

        df["a_new_col"] = np.log(df["a"])
        df["b_new_col"] = np.log(df["b"])

        df.drop(columns=["a", "b"], inplace=True)

        return df

    def expected_df_2():
        """Expected output of test_expected_output_2."""
        df = d.create_df_3()

        df["a_new_col"] = np.log(df["a"] + 1)
        df["b_new_col"] = np.log(df["b"] + 1)

        df.drop(columns=["a", "b"], inplace=True)

        return df

    def expected_df_3():
        """Expected output of test_expected_output_3."""
        df = d.create_df_3()

        df["a_new_col"] = np.log(df["a"])
        df["b_new_col"] = np.log(df["b"])

        return df

    def expected_df_4():
        """Expected output of test_expected_output_4."""
        df = d.create_df_3()

        df["a_new_col"] = np.log(df["a"] + 1)
        df["b_new_col"] = np.log(df["b"] + 1)

        return df

    def expected_df_5():
        """Expected output of test_expected_output_5."""
        df = d.create_df_4()

        df["a_new_col"] = np.log(df["a"] + 1) / np.log(5)

        return df

    def expected_df_6():
        """Expected output of test_expected_output_6."""
        df = d.create_df_4()

        df["a_new_col"] = np.log(df["a"]) / np.log(7)

        df.drop("a", axis=1, inplace=True)

        return df

    def test_arguments(self):
        """Test that transform has expected arguments."""
        ta.functions.test_function_arguments(
            func=LogTransformer.transform, expected_arguments=["self", "X"]
        )

    def test_super_transform_called(self, mocker):
        """Test that BaseTransformer.transform called."""
        df = d.create_df_3()

        x = LogTransformer(columns=["a", "b"])

        expected_call_args = {0: {"args": (d.create_df_3(),), "kwargs": {}}}

        with ta.functions.assert_function_call(
            mocker,
            tubular.base.BaseTransformer,
            "transform",
            expected_call_args,
            return_value=d.create_df_3(),
        ):
            x.transform(df)

    def test_error_with_non_numeric_columns(self):
        """Test an exception is raised if transform is applied to non-numeric columns."""
        df = d.create_df_5()

        x = LogTransformer(columns=["a", "b", "c"])

        with pytest.raises(
            TypeError,
            match=r"LogTransformer: The following columns are not numeric in X; \['b', 'c'\]",
        ):
            x.transform(df)

    @pytest.mark.parametrize(
        "df, expected",
        ta.pandas.adjusted_dataframe_params(d.create_df_3(), expected_df_1()),
    )
    def test_expected_output_1(self, df, expected):
        """Test that transform is giving the expected output when not adding one and dropping original columns."""
        x1 = LogTransformer(
            columns=["a", "b"], add_1=False, drop=True, suffix="new_col"
        )

        df_transformed = x1.transform(df)

        ta.equality.assert_equal_dispatch(
            expected=expected,
            actual=df_transformed,
            msg="LogTransformer transform not adding 1 and dropping original columns",
        )

    @pytest.mark.parametrize(
        "df, expected",
        ta.pandas.adjusted_dataframe_params(d.create_df_3(), expected_df_2()),
    )
    def test_expected_output_2(self, df, expected):
        """Test that transform is giving the expected output when adding one and dropping original columns."""
        x1 = LogTransformer(columns=["a", "b"], add_1=True, drop=True, suffix="new_col")

        df_transformed = x1.transform(df)

        ta.equality.assert_equal_dispatch(
            expected=expected,
            actual=df_transformed,
            msg="LogTransformer transform adding 1 and dropping original columns",
        )

    @pytest.mark.parametrize(
        "df, expected",
        ta.pandas.adjusted_dataframe_params(d.create_df_3(), expected_df_3()),
    )
    def test_expected_output_3(self, df, expected):
        """Test that transform is giving the expected output when not adding one and not dropping original columns."""
        x1 = LogTransformer(
            columns=["a", "b"], add_1=False, drop=False, suffix="new_col"
        )

        df_transformed = x1.transform(df)

        ta.equality.assert_equal_dispatch(
            expected=expected,
            actual=df_transformed,
            msg="LogTransformer transform not adding 1 and dropping original columns",
        )

    @pytest.mark.parametrize(
        "df, expected",
        ta.pandas.adjusted_dataframe_params(d.create_df_3(), expected_df_4()),
    )
    def test_expected_output_4(self, df, expected):
        """Test that transform is giving the expected output when adding one and not dropping original columns."""
        x1 = LogTransformer(
            columns=["a", "b"], add_1=True, drop=False, suffix="new_col"
        )

        df_transformed = x1.transform(df)

        ta.equality.assert_equal_dispatch(
            expected=expected,
            actual=df_transformed,
            msg="LogTransformer transform not adding 1 and dropping original columns",
        )

    @pytest.mark.parametrize(
        "df, expected",
        ta.pandas.adjusted_dataframe_params(d.create_df_4(), expected_df_5()),
    )
    def test_expected_output_5(self, df, expected):
        """Test that transform is giving the expected output when adding one and not dropping
        original columns and using base.
        """
        x1 = LogTransformer(
            columns=["a"], base=5, add_1=True, drop=False, suffix="new_col"
        )

        df_transformed = x1.transform(df)

        ta.equality.assert_equal_dispatch(
            expected=expected,
            actual=df_transformed,
            msg="LogTransformer transform not adding 1 and dropping original columns",
        )

    @pytest.mark.parametrize(
        "df, expected",
        ta.pandas.adjusted_dataframe_params(d.create_df_4(), expected_df_6()),
    )
    def test_expected_output_6(self, df, expected):
        """Test that transform is giving the expected output when  not adding one and dropping
        original columns and using base.
        """
        x1 = LogTransformer(
            columns=["a"], base=7, add_1=False, drop=True, suffix="new_col"
        )

        df_transformed = x1.transform(df)

        ta.equality.assert_equal_dispatch(
            expected=expected,
            actual=df_transformed,
            msg="LogTransformer transform should be using base, not adding 1, and not dropping original columns",
        )

    @pytest.mark.parametrize(
        "df, columns, add_1, extra_exception_text",
        (
            [pd.DataFrame({"a": [1, 2, 0]}), ["a"], False, ""],
            [pd.DataFrame({"a": [1, 2, 0], "b": [1, 2, 3]}), ["a", "b"], False, ""],
            [pd.DataFrame({"a": [1, 2, -1]}), ["a"], True, r" \(after adding 1\)"],
            [
                pd.DataFrame({"a": [1, 2, -1], "b": [1, 2, 3]}),
                ["a", "b"],
                True,
                r" \(after adding 1\)",
            ],
            [pd.DataFrame({"b": [1, 2, -0.001]}), ["b"], False, ""],
            [
                pd.DataFrame({"b": [1, 2, -0.001], "a": [1, 2, 3]}),
                ["a", "b"],
                False,
                "",
            ],
            [pd.DataFrame({"b": [1, 2, -1.001]}), ["b"], True, r" \(after adding 1\)"],
            [
                pd.DataFrame({"b": [1, 2, -1.001], "a": [1, 2, 3]}),
                ["a", "b"],
                True,
                r" \(after adding 1\)",
            ],
        ),
    )
    def test_negative_values_raise_exception(
        self, df, columns, add_1, extra_exception_text
    ):
        """Test that an exception is raised if negative values are passed in transform."""
        x = LogTransformer(columns=columns, add_1=add_1, drop=True)

        with pytest.raises(
            ValueError,
            match=f"LogTransformer: values less than or equal to 0 in columns{extra_exception_text}, make greater than 0 before using transform",
        ):
            x.transform(df)
