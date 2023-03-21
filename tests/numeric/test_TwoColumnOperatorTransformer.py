import pytest
import pandas as pd
import test_aide as ta
import tubular
from tubular.numeric import TwoColumnOperatorTransformer
import tests.test_data as d


@pytest.fixture(scope="module", autouse=True)
def example_transformer():

    example_transformer = TwoColumnOperatorTransformer(
        "mul",
        ["a", "b"],
        "c",
    )

    return example_transformer


class TestTwoColumnOperatorTransformerInit(object):
    """
    Tests for TwoColumnMethodTransformer.__init__()
    """

    def test_inheritance(self, example_transformer):
        """Test that TwoColumnOperatorTransformer inherits from BaseTransformer."""

        ta.classes.assert_inheritance(
            example_transformer, tubular.base.DataFrameMethodTransformer
        )

    def test_class_methods(self, example_transformer):
        """Test that TwoColumnOperatorTransformer has transform method."""

        ta.classes.test_object_method(
            obj=example_transformer, expected_method="transform", msg="transform"
        )

    def test_arguments(self):
        """Test that init has expected arguments."""

        ta.functions.test_function_arguments(
            func=TwoColumnOperatorTransformer.__init__,
            expected_arguments=[
                "self",
                "pd_method_name",
                "columns",
                "new_column_name",
                "pd_method_kwargs",
            ],
            expected_default_values=({"axis": 0},),
        )

    def test_axis_not_present_error(self):
        """Checks that an error is raised if no axis element present in pd_method_kwargs dict"""

        with pytest.raises(
            ValueError,
            match='pd_method_kwargs must contain an entry "axis" set to 0 or 1',
        ):
            TwoColumnOperatorTransformer("mul", ["a", "b"], "c", pd_method_kwargs={})

    def test_axis_not_valid_error(self):
        """Checks that an error is raised if no axis element present in pd_method_kwargs dict"""

        with pytest.raises(ValueError, match="pd_method_kwargs 'axis' must be 0 or 1"):
            TwoColumnOperatorTransformer(
                "mul", ["a", "b"], "c", pd_method_kwargs={"axis": 2}
            )

    def test_attributes(self, example_transformer):
        """Tests that the transformer has the expected attributes"""
        expected_attributes = {
            "pd_method_name": "mul",
            # 'a' is given as a list here because that's how DataFrameMethodTransformer.__init__ stores the columns attribute
            "column1_name": "a",
            "column2_name": "b",
            "new_column_name": "c",
            "pd_method_kwargs": {"axis": 0},
        }

        msg = "TwoColumneMethodTransformer object does not have expected attributes"
        ta.classes.test_object_attributes(example_transformer, expected_attributes, msg)

    def test_DataFrameMethodTransformer_init_call(self, mocker):
        """Tests that the .__init__ method is called from the parent DataFrameMethodTransformer class"""
        expected_call_args = {
            0: {
                "args": (),
                "kwargs": {
                    "new_column_name": "c",
                    "pd_method_name": "mul",
                    "columns": ["a", "b"],
                    "pd_method_kwargs": {"axis": 0},
                },
            }
        }

        with ta.functions.assert_function_call(
            mocker,
            tubular.base.DataFrameMethodTransformer,
            "__init__",
            expected_call_args,
            return_value=None,
        ):

            TwoColumnOperatorTransformer("mul", ["a", "b"], "c")


class TestTwoColumnOperatorTransformerTransform(object):
    def test_arguments(self):
        """Test that transform has expected arguments."""

        ta.functions.test_function_arguments(
            func=TwoColumnOperatorTransformer.transform,
            expected_arguments=["self", "X"],
            expected_default_values=None,
        )

    def test_BaseTransformer_transform_called(self, example_transformer, mocker):
        """Tests that the .transform method is called from the grandparent BaseTransformer class"""
        test_data = d.create_df_11()

        expected_call_args = {0: {"args": (test_data,), "kwargs": {}}}

        with ta.functions.assert_function_call(
            mocker, tubular.base.BaseTransformer, "transform", expected_call_args
        ):

            example_transformer.transform(test_data)

    @pytest.mark.parametrize(
        "pd_method_name",
        [
            ("mul"),
            ("div"),
            ("pow"),
        ],
    )
    def test_pandas_method_called(self, mocker, pd_method_name):
        """Test that the pandas method is called as expected (with kwargs passed) during transform."""
        spy = mocker.spy(pd.DataFrame, pd_method_name)

        pd_method_kwargs = {"axis": 0}

        data = d.create_df_11()
        x = TwoColumnOperatorTransformer(
            pd_method_name,
            ["a", "b"],
            "c",
        )
        x.transform(data)

        # pull out positional and keyword args to target the call
        print(spy.call_args_list)
        call_args = spy.call_args_list[0]
        call_pos_args = call_args[0]
        call_kwargs = call_args[1]

        # test keyword are as expected
        ta.equality.assert_dict_equal_msg(
            actual=call_kwargs,
            expected=pd_method_kwargs,
            msg_tag=f"""Keyword arg assert for '{pd_method_name}'""",
        )

        # test positional args are as expected
        ta.equality.assert_list_tuple_equal_msg(
            actual=call_pos_args,
            # 'a' is indexed as a list here because that's how DataFrameMethodTransformer.__init__ stores the columns attribute
            expected=(data[["a"]], data["b"]),
            msg_tag=f"""Positional arg assert for {pd_method_name}""",
        )

    @pytest.mark.parametrize(
        "pd_method_name, output",
        [
            (
                "mul",
                [4, 10, 18],
            ),
            ("div", [0.25, 0.4, 0.5]),
            ("pow", [1, 32, 729]),
        ],
    )
    def test_expected_output(self, pd_method_name, output):
        """Tests that the output given by TwoColumnOperatorTransformer is as you would expect"""
        expected = d.create_df_11()
        expected["c"] = output
        x = TwoColumnOperatorTransformer(
            pd_method_name,
            ["a", "b"],
            "c",
        )
        actual = x.transform(d.create_df_11())
        ta.equality.assert_frame_equal_msg(
            actual=actual,
            expected=expected,
            msg_tag="TwoColumnMethod transformer does not produce the expected output",
        )

    def test_non_numeric_error(self):
        x = TwoColumnOperatorTransformer(
            "mul",
            ["a", "b"],
            "c",
        )

        with pytest.raises(
            TypeError,
            match="TwoColumnOperatorTransformer: input columns in X must contain only numeric values",
        ):
            x.transform(d.create_df_8())
