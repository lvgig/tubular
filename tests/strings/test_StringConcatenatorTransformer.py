import pytest
import test_aide as ta

import tests.test_data as d
import tubular
from tubular.strings import StringConcatenator


@pytest.fixture()
def concatenate_str():
    return StringConcatenator(columns=["a", "b"], new_column="merged_values")


class TestStringConcatenator:
    """Tests for the StringConcatenator.__init__ method."""

    def test_super_init_call(self, mocker):
        """Test that BaseTransformer.init us called as expected."""
        expected_call_args = {
            0: {
                "args": (),
                "kwargs": {"columns": ["a", "b"]},
            },
        }

        with ta.functions.assert_function_call(
            mocker,
            tubular.base.BaseTransformer,
            "__init__",
            expected_call_args,
        ):
            StringConcatenator(columns=["a", "b"], new_column="merged_column")

    def test_merged_values_attribute_set(self, concatenate_str):
        """Test that the new column name passed in the new column arg is set as an attribute of the same name."""
        assert (
            concatenate_str.new_column == "merged_values"
        ), "unexpected value set to new_column attribute"

    @pytest.mark.parametrize("new_column", [1, True, ["a", "b"], 2.0])
    def test_warning_new_column_str(self, new_column):
        """Test that an exception is raised if new_column is not a str."""
        df = d.create_df_1()

        with pytest.raises(
            TypeError,
            match="StringConcatenator: new_column should be a str",
        ):
            x = StringConcatenator(columns=["a", "b"], new_column=new_column)
            x.transform(df)

    @pytest.mark.parametrize("separator", [0.0, False, ["a", "b"], 7])
    def test_warning_seperator_str(self, separator):
        """Test that an exception is raised if separator is not a str."""
        df = d.create_df_1()

        with pytest.raises(
            TypeError,
            match="StringConcatenator: The separator should be a str",
        ):
            x = StringConcatenator(
                columns=["a", "b"],
                new_column="new_column",
                separator=separator,
            )
            x.transform(df)


class TestTransform:
    """Tests for the StringConcatenator.transform method."""

    def test_super_transform_called(self, mocker, concatenate_str):
        """Test that BaseTransformer.transform called."""
        df = d.create_df_7()

        expected_call_args = {0: {"args": (d.create_df_7(),), "kwargs": {}}}

        with ta.functions.assert_function_call(
            mocker,
            tubular.base.BaseTransformer,
            "transform",
            expected_call_args,
        ):
            concatenate_str.transform(df)

    def test_correct_df_returned_1(self, concatenate_str):
        """Test that correct df is returned after transformation."""
        df = d.create_df_1()

        df_transformed = concatenate_str.transform(df)

        expected_df = df.copy()
        expected_df["merged_values"] = ["1 a", "2 b", "3 c", "4 d", "5 e", "6 f"]

        ta.equality.assert_frame_equal_msg(
            df_transformed,
            expected_df,
            "Incorrect dataframe returned after StringConcatenator transform",
        )

    def test_correct_df_returned_2(self):
        """Test that correct df is returned after transformation."""
        df = d.create_df_1()

        x = StringConcatenator(
            columns=["a", "b"],
            new_column="merged_values",
            separator=":",
        )
        df_transformed = x.transform(df)

        expected_df = df.copy()
        expected_df["merged_values"] = ["1:a", "2:b", "3:c", "4:d", "5:e", "6:f"]

        ta.equality.assert_frame_equal_msg(
            df_transformed,
            expected_df,
            "Incorrect dataframe returned after StringConcatenator transform",
        )
