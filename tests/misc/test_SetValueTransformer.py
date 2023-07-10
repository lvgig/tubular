import pytest
import test_aide as ta

import tests.test_data as d
import tubular
from tubular.misc import SetValueTransformer


class TestInit:
    """Tests for the SetValueTransformer.__init__ method."""

    def test_arguments(self):
        """Test that init has expected arguments."""
        ta.functions.test_function_arguments(
            func=SetValueTransformer.__init__,
            expected_arguments=["self", "columns", "value"],
            expected_default_values=None,
        )

    def test_inheritance(self):
        """Test SetValueTransformer inherits from BaseTransformer."""
        x = SetValueTransformer(columns=["a"], value=1)

        assert isinstance(
            x,
            tubular.base.BaseTransformer,
        ), "SetValueTransformer is not instance of tubular.base.BaseTransformer"

    def test_super_init_call(self, mocker):
        """Test that BaseTransformer.init us called as expected."""
        expected_call_args = {
            0: {
                "args": (),
                "kwargs": {"columns": ["a", "b"], "verbose": False, "copy": False},
            },
        }

        with ta.functions.assert_function_call(
            mocker,
            tubular.base.BaseTransformer,
            "__init__",
            expected_call_args,
        ):
            SetValueTransformer(columns=["a", "b"], value=1, verbose=False, copy=False)

    def test_value_attribute_set(self):
        """Test that the value passed in the value arg is set as an attribute of the same name."""
        x = SetValueTransformer(columns=["a", "b"], value=1)

        assert x.value == 1, "unexpected value set to value atttribute"


class TestTransform:
    """Tests for the SetValueTransformer.transform method."""

    def expected_df_1():
        """Expected output of test_value_set_in_transform."""
        df = d.create_df_2()

        df["a"] = "a"
        df["b"] = "a"

        return df

    def test_arguments(self):
        """Test that transform has expected arguments."""
        ta.functions.test_function_arguments(
            func=SetValueTransformer.transform,
            expected_arguments=["self", "X"],
            expected_default_values=None,
        )

    def test_super_transform_called(self, mocker):
        """Test that BaseTransformer.transform called."""
        df = d.create_df_7()

        x = SetValueTransformer(columns=["a", "b"], value=1)

        expected_call_args = {0: {"args": (d.create_df_7(),), "kwargs": {}}}

        with ta.functions.assert_function_call(
            mocker,
            tubular.base.BaseTransformer,
            "transform",
            expected_call_args,
        ):
            x.transform(df)

    @pytest.mark.parametrize(
        ("df", "expected"),
        ta.pandas.adjusted_dataframe_params(d.create_df_2(), expected_df_1()),
    )
    def test_value_set_in_transform(self, df, expected):
        """Test that transform sets the value as expected."""
        x = SetValueTransformer(columns=["a", "b"], value="a")

        df_transformed = x.transform(df)

        ta.equality.assert_equal_dispatch(
            actual=df_transformed,
            expected=expected,
            msg="incorrect value after SetValueTransformer transform",
        )
