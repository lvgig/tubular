import numpy as np
import pandas as pd
import pytest
import test_aide as ta

import tests.test_data as d
import tubular
from tubular.misc import SetColumnDtype


class TestSetColumnDtypeInit:
    """Tests for SetColumnDtype custom transformer."""

    def test_init_arguments(self):
        """Test that init has expected arguments."""
        ta.functions.test_function_arguments(
            func=SetColumnDtype.__init__,
            expected_arguments=[
                "self",
                "columns",
                "dtype",
            ],
            expected_default_values=None,
        )

    def test_inheritance(self):
        """Test that SetColumnDtype inherits from tubular BaseTransformer."""
        x = SetColumnDtype(columns=["a"], dtype=float)

        ta.classes.assert_inheritance(x, tubular.base.BaseTransformer)

    def test_tubular_base_transformer_super_init_called(self, mocker):
        """Test that init calls tubular BaseTransformer.init."""
        expected_call_args = {
            0: {
                "args": (["a"],),
                "kwargs": {"copy": True},
            },
        }
        with ta.functions.assert_function_call(
            mocker,
            tubular.base.BaseTransformer,
            "__init__",
            expected_call_args,
        ):
            SetColumnDtype(columns=["a"], dtype=float)

    def test_dtype_attribute_set(self):
        """Test that the value passed in the value arg is set as an attribute of the same name."""
        x = SetColumnDtype(columns=["a"], dtype=str)

        assert x.dtype == str, "unexpected value set to dtype atttribute"

    @pytest.mark.parametrize(
        "invalid_dtype",
        ["STRING", "misc_invalid", "np.int", int()],
    )
    def test_invalid_dtype_error(self, invalid_dtype):
        msg = f"SetColumnDtype: data type '{invalid_dtype}' not understood as a valid dtype"
        with pytest.raises(TypeError, match=msg):
            SetColumnDtype(columns=["a"], dtype=invalid_dtype)


class TestSetColumnDtypeTransform:
    @pytest.mark.parametrize(
        "method_name",
        [
            ("transform"),
        ],
    )
    def test_class_methods(self, method_name):
        """Test that SetColumnDtype has transform method."""
        x = SetColumnDtype(columns=["a"], dtype=float)

        ta.classes.test_object_method(
            obj=x,
            expected_method=method_name,
            msg=method_name,
        )

    def test_transform_arguments(self):
        """Test that transform has expected arguments."""
        ta.functions.test_function_arguments(
            func=SetColumnDtype.transform,
            expected_arguments=[
                "self",
                "X",
            ],
        )

    def test_super_transform_called(self, mocker):
        """Test that BaseTransformer.transform called."""
        df = d.create_df_3()

        x = SetColumnDtype(columns=["a"], dtype=float)

        expected_call_args = {0: {"args": (d.create_df_3(),), "kwargs": {}}}

        with ta.functions.assert_function_call(
            mocker,
            tubular.base.BaseTransformer,
            "transform",
            expected_call_args,
            return_value=d.create_df_3(),
        ):
            x.transform(df)

    def base_df():
        """Input dataframe from test_expected_output."""
        return pd.DataFrame(
            {
                "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, np.NaN],
                "b": [1.0, 2.0, 3.0, np.NaN, 7.0, 8.0, 9.0],
                "c": [1.0, 1.0, 2.0, 3.0, -4.0, -5.0, -6.0],
                "d": [1, 1, 2, 3, -4, -5, -6],
            },
        )

    def expected_df():
        """Define expected output from test_expected_output."""
        return pd.DataFrame(
            {
                "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, np.NaN],
                "b": [1.0, 2.0, 3.0, np.NaN, 7.0, 8.0, 9.0],
                "c": [1.0, 1.0, 2.0, 3.0, -4.0, -5.0, -6.0],
                "d": [1.0, 1.0, 2.0, 3.0, -4.0, -5.0, -6.0],
            },
        )

    @pytest.mark.parametrize(
        ("df", "expected"),
        ta.pandas.row_by_row_params(base_df(), expected_df())
        + ta.pandas.index_preserved_params(base_df(), expected_df()),
    )
    @pytest.mark.parametrize("dtype", [float, "float"])
    def test_expected_output(self, df, expected, dtype):
        """Test values are correctly set to float dtype."""
        df["a"] = df["a"].astype(str)
        df["b"] = df["b"].astype(float)
        df["c"] = df["c"].astype(int)
        df["d"] = df["d"].astype(str)

        x = SetColumnDtype(columns=["a", "b", "c", "d"], dtype=dtype)

        df_transformed = x.transform(df)

        ta.equality.assert_equal_dispatch(
            expected=expected,
            actual=df_transformed,
            msg="Check values correctly converted to float",
        )
