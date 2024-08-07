from tests.base_tests import (
    ColumnStrListInitTests,
    NewColumnNameInitMixintests,
    SeparatorInitMixintests,
)

# @pytest.fixture()
# def concatenate_str():
#     return StringConcatenator(columns=["a", "b"], new_column="merged_values")


class TestInit(
    SeparatorInitMixintests,
    ColumnStrListInitTests,
    NewColumnNameInitMixintests,
):
    """Generic tests for transformer.init()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "StringConcatenator"


# class TestTransform:
#     """Tests for the StringConcatenator.transform method."""

#     def test_super_transform_called(self, mocker, concatenate_str):
#         """Test that BaseTransformer.transform called."""
#         df = d.create_df_7()

#         expected_call_args = {0: {"args": (d.create_df_7(),), "kwargs": {}}}

#         with ta.functions.assert_function_call(
#             mocker,
#             tubular.base.BaseTransformer,
#             "transform",
#             expected_call_args,
#         ):
#             concatenate_str.transform(df)

#     def test_correct_df_returned_1(self, concatenate_str):
#         """Test that correct df is returned after transformation."""
#         df = d.create_df_1()

#         df_transformed = concatenate_str.transform(df)

#         expected_df = df.copy()
#         expected_df["merged_values"] = ["1 a", "2 b", "3 c", "4 d", "5 e", "6 f"]

#         ta.equality.assert_frame_equal_msg(
#             df_transformed,
#             expected_df,
#             "Incorrect dataframe returned after StringConcatenator transform",
#         )

#     def test_correct_df_returned_2(self):
#         """Test that correct df is returned after transformation."""
#         df = d.create_df_1()

#         x = StringConcatenator(
#             columns=["a", "b"],
#             new_column="merged_values",
#             separator=":",
#         )
#         df_transformed = x.transform(df)

#         expected_df = df.copy()
#         expected_df["merged_values"] = ["1:a", "2:b", "3:c", "4:d", "5:e", "6:f"]

#         ta.equality.assert_frame_equal_msg(
#             df_transformed,
#             expected_df,
#             "Incorrect dataframe returned after StringConcatenator transform",
#         )
