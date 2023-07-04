import pytest
import test_aide as ta

import tests.test_data as d
import tubular
from tubular.comparison import EqualityChecker


@pytest.fixture(scope="module", autouse=True)
def example_transformer():
    example_transformer = EqualityChecker(columns=["a", "b"], new_col_name="d")

    return example_transformer


class TestInit:
    """Tests for the EqualityChecker.__init__ method."""

    def test_arguments(self):
        """Test that init has expected arguments."""
        ta.functions.test_function_arguments(
            func=EqualityChecker.__init__,
            expected_arguments=["self", "columns", "new_col_name", "drop_original"],
            expected_default_values=(False,),
        )

    def test_inheritance(self, example_transformer):
        """Test EqualityChecker inherits from BaseTransformer."""
        assert isinstance(
            example_transformer,
            tubular.base.BaseTransformer,
        ), "EqualityChecker is not instance of tubular.base.BaseTransformer"

    def test_super_init_call(self, mocker):
        """Test that BaseTransformer.init is called as expected."""
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
            EqualityChecker(
                columns=["a", "b"],
                new_col_name="d",
                verbose=False,
                copy=False,
            )

    def test_class_methods(self, example_transformer):
        """Test that EqualityChecker has transform method."""
        msg = "no transformation method in class"
        ta.classes.test_object_method(
            obj=example_transformer,
            expected_method="transform",
            msg=msg,
        )

    def test_value_new_col_name(self, example_transformer):
        """Test that the value passed in the new column name arg is correct."""
        assert (
            example_transformer.new_col_name == "d"
        ), "unexpected value set to new_col_name atttribute"

    def test_value_drop_original(self, example_transformer):
        """Test that the value passed in the drop_original arg is correct."""
        assert (
            not example_transformer.drop_original
        ), "unexpected value set to drop_original atttribute"

    @pytest.mark.parametrize("test_input_col_type", ["a", None])
    def test_type_error_for_columns(self, test_input_col_type):
        """Checks that an error is raised if wrong data type for argument:columns."""
        with pytest.raises(
            TypeError,
            match="columns should be list",
        ):
            EqualityChecker(columns=test_input_col_type, new_col_name="d")

    @pytest.mark.parametrize("test_input_col", [["b", "b", "b"], ["a"]])
    def test_value_error_for_columns(self, test_input_col):
        """Checks that a value error is raised where 2 cols are not supplied."""
        with pytest.raises(
            ValueError,
            match="This transformer works with two columns only",
        ):
            EqualityChecker(columns=test_input_col, new_col_name="d")

    @pytest.mark.parametrize("test_input_new_col", [123, ["a"], True])
    def test_type_error_for_new_column_name(self, test_input_new_col):
        """Checks that an error is raised if wrong data type for argument:new_col_name."""
        with pytest.raises(
            TypeError,
            match="new_col_name should be str",
        ):
            EqualityChecker(columns=["a", "b"], new_col_name=test_input_new_col)

    @pytest.mark.parametrize("test_input_drop_col", [123, ["a"], "asd"])
    def test_type_error_for_drop_column(self, test_input_drop_col):
        """Checks that an error is raised if wrong data type for argument:drop_original."""
        with pytest.raises(
            TypeError,
            match="drop_original should be bool",
        ):
            EqualityChecker(
                columns=["a", "b"],
                new_col_name="col_name",
                drop_original=test_input_drop_col,
            )


class TestTransform:
    """Tests for the EqualityChecker.transform method."""

    def test_arguments(self):
        """Test that transform has expected arguments."""
        ta.functions.test_function_arguments(
            func=EqualityChecker.transform,
            expected_arguments=["self", "X"],
            expected_default_values=None,
        )

    def test_super_transform_called(self, mocker, example_transformer):
        """Test that BaseTransformer.transform called."""
        df = d.create_df_7()

        expected_call_args = {0: {"args": (d.create_df_7(),), "kwargs": {}}}

        with ta.functions.assert_function_call(
            mocker,
            tubular.base.BaseTransformer,
            "transform",
            expected_call_args,
        ):
            example_transformer.transform(df)

    @pytest.mark.parametrize(
        "test_dataframe",
        [d.create_df_5(), d.create_df_2(), d.create_df_9()],
    )
    def test_expected_output(self, test_dataframe):
        """Tests that the output given by EqualityChecker tranformer is as you would expect
        when all cases are neither all True nor False.
        """
        expected = test_dataframe
        expected["bool_logic"] = expected["b"] == expected["c"]

        example_transformer = EqualityChecker(
            columns=["b", "c"],
            new_col_name="bool_logic",
        )
        actual = example_transformer.transform(test_dataframe)

        ta.equality.assert_frame_equal_msg(
            actual=actual,
            expected=expected,
            msg_tag="EqualityChecker transformer does not produce the expected output",
            print_actual_and_expected=True,
        )

    @pytest.mark.parametrize(
        "test_dataframe",
        [d.create_df_5(), d.create_df_2(), d.create_df_9()],
    )
    def test_expected_output_dropped(self, test_dataframe):
        """Tests that the output given by EqualityChecker tranformer is as you would expect
        when all cases are neither all True nor False.
        """
        expected = test_dataframe.copy()
        expected["bool_logic"] = expected["b"] == expected["c"]
        expected = expected.drop(["b", "c"], axis=1)

        example_transformer = EqualityChecker(
            columns=["b", "c"],
            new_col_name="bool_logic",
            drop_original=True,
        )
        actual = example_transformer.transform(test_dataframe)

        ta.equality.assert_frame_equal_msg(
            actual=actual,
            expected=expected,
            msg_tag="EqualityChecker transformer does not produce the expected output",
            print_actual_and_expected=True,
        )
