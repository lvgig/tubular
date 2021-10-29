import pytest
import tubular.testing.test_data as d
import tubular.testing.helpers as h

import tubular
from tubular.base import InteractionTransformer
import pandas as pd
import numpy as np


class TestInit(object):
    """Tests for DataFrameMethodTransformer.init()."""

    def test_arguments(self):
        """Test that init has expected arguments."""

        h.test_function_arguments(
            func=InteractionTransformer.__init__,
            expected_arguments=[
                "self",
                "columns",
                "new_columns_name",
                "min_degree",
                "max_degree",
            ],
            expected_default_values=(None, None, 2, 2),
        )

    def test_class_methods(self):
        """Test that DataFrameMethodTransformer has transform method."""

        x = InteractionTransformer()

        h.test_object_method(obj=x, expected_method="transform", msg="transform")

    def test_inheritance(self):
        """Test that DataFrameMethodTransformer inherits from BaseTransformer."""

        x = InteractionTransformer()

        h.assert_inheritance(x, tubular.base.BaseTransformer)

    # def test_super_init_called(self, mocker):
    #     """Test that init calls BaseTransformer.init."""
    #
    #     expected_call_args = {
    #         0: {
    #             "args": (),
    #             "kwargs": {"columns": ["b", "c"], "verbose": True, "copy": False},
    #         }
    #     }
    #
    #     with h.assert_function_call(
    #         mocker, tubular.base.BaseTransformer, "__init__", expected_call_args
    #     ):
    #
    #         DataFrameMethodTransformer(
    #             new_column_name="a",
    #             pd_method_name="sum",
    #             columns=["b", "c"],
    #             copy=False,
    #             verbose=True,
    #         )

    def test_invalid_input_type_errors(self):
        """Test that an exceptions are raised for invalid input types."""
        with pytest.raises(
            TypeError,
            match=r"unexpected type \(\<class 'float'\>\) for columns, must be str or list of strings",
        ):

            InteractionTransformer(
                columns=3.2,
                new_columns_name=["A B", "A C"],
                min_degree=2,
                max_degree=2,
            )

        with pytest.raises(
            TypeError,
            match=r"if columns is a list, all elements must be strings but got \<class 'int'\> in position 2",
        ):

            InteractionTransformer(
                columns=["A", "B", 4],
                new_columns_name=["A B", "A C"],
                min_degree=2,
                max_degree=2,
            )
        with pytest.raises(
            TypeError,
            match=r"unexpected type \(\<class 'float'\>\) for new_column_name, must be str or list of strings",
        ):

            InteractionTransformer(
                columns=["A", "B", "C"],
                new_columns_name=5.6,
                min_degree=2,
                max_degree=2,
            )

        with pytest.raises(
            TypeError,
            match=r"if new_columns_name is a list, all elements must be strings but got \<class 'int'\> in position 1",
        ):

            InteractionTransformer(
                columns=["A", "B", "C"],
                new_columns_name=["A B", 6],
                min_degree=2,
                max_degree=2,
            )
        with pytest.raises(
            TypeError,
            match=r"""unexpected type \(\<class 'str'\>\) for min_degree, must be int""",
        ):

            InteractionTransformer(
                columns=["A", "B", "C"],
                new_columns_name=["A B", "A C"],
                min_degree="2",
                max_degree=2,
            )
        with pytest.raises(
            TypeError,
            match=r"""unexpected type \(\<class 'str'\>\) for max_degree, must be int""",
        ):

            InteractionTransformer(
                columns=["A", "B", "C"],
                new_columns_name=["A B", "A C"],
                min_degree=2,
                max_degree="2",
            )

    def test_invalid_input_value_errors(self):
        """Test and exception is raised if a non pd.DataFrame method is passed for pd_method_name."""

        with pytest.raises(
            ValueError,
            match=r"""min_degree must be equal or greater than 2, got 0""",
        ):

            InteractionTransformer(
                columns=["A", "B", "C"],
                new_columns_name=["A B", "A C", "B C"],
                min_degree=0,
                max_degree=2,
            )

        with pytest.raises(
            ValueError,
            match=r"""max_degree must be equal or greater than min_degree""",
        ):

            InteractionTransformer(
                columns=["A", "B", "C"],
                new_columns_name=["A B", "A C", "B C"],
                min_degree=3,
                max_degree=2,
            )

        with pytest.raises(
            ValueError,
            match=r"""max_degree must be equal or lower than the number of expected new_columns_name""",
        ):

            InteractionTransformer(
                columns=["A", "B", "C"],
                new_columns_name=["A B", "A C", "B C"],
                min_degree=2,
                max_degree=4,
            )

    def test_attributes_set(self):
        """Test that the values passed for new_column_name, pd_method_name are saved to attributes on the object."""

        x = InteractionTransformer(
            columns=["A", "B", "C"],
            new_columns_name=["A B", "A C", "B C", "A B C"],
            min_degree=2,
            max_degree=3,
        )

        h.test_object_attributes(
            obj=x,
            expected_attributes={
                "columns": ["A", "B", "C"],
                "new_columns_name": ["A B", "A C", "B C", "A B C"],
                "min_degree": 2,
                "max_degree": 3,
            },
            msg="Attributes for InteractionTransformer set in init",
        )


class TestTransform(object):
    """Tests for DataFrameMethodTransformer.transform()."""

    def expected_df_1():
        """Expected output of test_expected_output_default_assignment."""

        df = pd.DataFrame(
            {
                "a": {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0, 4: 5.0, 5: 6.0, 6: np.NaN},
                "b": {0: 1.0, 1: 2.0, 2: 3.0, 3: np.NaN, 4: 7.0, 5: 8.0, 6: 9.0},
                "c": {0: np.NaN, 1: 1.0, 2: 2.0, 3: 3.0, 4: -4.0, 5: -5.0, 6: -6.0},
                "a b": {0: 1.0, 1: 4.0, 2: 9.0, 3: np.NaN, 4: 35.0, 5: 48.0, 6: np.NaN},
                "a c": {
                    0: np.NaN,
                    1: 2.0,
                    2: 6.0,
                    3: 12.0,
                    4: -20.0,
                    5: -30.0,
                    6: np.NaN,
                },
                "b c": {
                    0: np.NaN,
                    1: 2.0,
                    2: 6.0,
                    3: np.NaN,
                    4: -28.0,
                    5: -40.0,
                    6: -54.0,
                },
            }
        )

        return df

    def expected_df_2():
        """Expected output of test_expected_output_multiple_columns_assignment."""

        df = pd.DataFrame(
            {
                "a": {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0, 4: 5.0, 5: 6.0, 6: np.NaN},
                "b": {0: 1.0, 1: 2.0, 2: 3.0, 3: np.NaN, 4: 7.0, 5: 8.0, 6: 9.0},
                "c": {0: np.NaN, 1: 1.0, 2: 2.0, 3: 3.0, 4: -4.0, 5: -5.0, 6: -6.0},
                "a b": {0: 1.0, 1: 4.0, 2: 9.0, 3: np.NaN, 4: 35.0, 5: 48.0, 6: np.NaN},
            }
        )

        return df

    def expected_df_3():
        """Expected output of test_expected_output_multiple_columns_assignment."""

        df = pd.DataFrame(
            {
                "a": {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0, 4: 5.0, 5: 6.0, 6: np.NaN},
                "b": {0: 1.0, 1: 2.0, 2: 3.0, 3: np.NaN, 4: 7.0, 5: 8.0, 6: 9.0},
                "c": {0: np.NaN, 1: 1.0, 2: 2.0, 3: 3.0, 4: -4.0, 5: -5.0, 6: -6.0},
                "interaction_col": {
                    0: 1.0,
                    1: 4.0,
                    2: 9.0,
                    3: np.NaN,
                    4: 35.0,
                    5: 48.0,
                    6: np.NaN,
                },
            }
        )
        return df

    def expected_df_4():
        """Expected output of test_expected_output_multiple_columns_assignment."""

        df = pd.DataFrame(
            {
                "a": {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0, 4: 5.0, 5: 6.0, 6: np.NaN},
                "b": {0: 1.0, 1: 2.0, 2: 3.0, 3: np.NaN, 4: 7.0, 5: 8.0, 6: 9.0},
                "c": {0: np.NaN, 1: 1.0, 2: 2.0, 3: 3.0, 4: -4.0, 5: -5.0, 6: -6.0},
                "iter1": {
                    0: 1.0,
                    1: 4.0,
                    2: 9.0,
                    3: np.NaN,
                    4: 35.0,
                    5: 48.0,
                    6: np.NaN,
                },
                "iter2": {
                    0: np.NaN,
                    1: 2.0,
                    2: 6.0,
                    3: 12.0,
                    4: -20.0,
                    5: -30.0,
                    6: np.NaN,
                },
                "iter3": {
                    0: np.NaN,
                    1: 2.0,
                    2: 6.0,
                    3: np.NaN,
                    4: -28.0,
                    5: -40.0,
                    6: -54.0,
                },
            }
        )
        return df

    def test_arguments(self):
        """Test that transform has expected arguments."""

        h.test_function_arguments(
            func=InteractionTransformer.transform, expected_arguments=["self", "X"]
        )

    # def test_super_transform_called(self, mocker):
    #     """Test that BaseTransformer.transform called."""
    #
    #     df = d.create_df_3()
    #
    #     x = InteractionTransformer(
    #           columns=["b", "c"],new_column_name="d",
    #     )
    #
    #     expected_call_args = {0: {"args": (df.copy(),), "kwargs": {}}}
    #
    #     with h.assert_function_call(
    #         mocker, tubular.base.BaseTransformer, "transform", expected_call_args
    #     ):
    #
    #         x.transform(df)

    @pytest.mark.parametrize(
        "df, expected",
        h.row_by_row_params(d.create_df_3(), expected_df_1())
        + h.index_preserved_params(d.create_df_3(), expected_df_1()),
    )
    def test_expected_output_default_assignment(self, df, expected):
        """Test a single column assignment from transform gives expected results."""

        x = InteractionTransformer()

        df_transformed = x.transform(df)

        h.assert_equal_dispatch(
            expected=expected,
            actual=df_transformed,
            msg="InteractionTransformer default values",
        )

    @pytest.mark.parametrize(
        "df, expected",
        h.row_by_row_params(d.create_df_3(), d.create_df_3())
        + h.index_preserved_params(d.create_df_3(), d.create_df_3()),
    )
    def test_expected_output_single_column_assignment(self, df, expected):
        """Test a single column assignment from transform gives expected results."""

        x = InteractionTransformer(columns="a")

        df_transformed = x.transform(df)

        h.assert_equal_dispatch(
            expected=expected,
            actual=df_transformed,
            msg="InteractionTransformer one single column values",
        )

    @pytest.mark.parametrize(
        "df, expected",
        h.row_by_row_params(d.create_df_3(), expected_df_2())
        + h.index_preserved_params(d.create_df_3(), expected_df_2()),
    )
    def test_expected_output_multiple_columns_assignment(self, df, expected):
        """Test a multiple columns assignment from transform gives expected results."""

        x = InteractionTransformer(columns=["a", "b"])

        df_transformed = x.transform(df)

        h.assert_equal_dispatch(
            expected=expected,
            actual=df_transformed,
            msg="InteractionTransformer one single column values",
        )

    @pytest.mark.parametrize(
        "df, expected",
        h.row_by_row_params(d.create_df_3(), expected_df_3())
        + h.index_preserved_params(d.create_df_3(), expected_df_3()),
    )
    def test_expected_output_single_new_column_name_assignment(self, df, expected):
        """Test a single column assignment from transform gives expected results."""

        x = InteractionTransformer(
            columns=["a", "b"], new_columns_name="interaction_col"
        )

        df_transformed = x.transform(df)

        h.assert_equal_dispatch(
            expected=expected,
            actual=df_transformed,
            msg="InteractionTransformer one single column values",
        )

    @pytest.mark.parametrize(
        "df, expected",
        h.row_by_row_params(d.create_df_3(), expected_df_4())
        + h.index_preserved_params(d.create_df_3(), expected_df_4()),
    )
    def test_expected_output_multiple_new_column_name_assignment(self, df, expected):
        """Test a multiple columns assignment from transform gives expected results."""

        x = InteractionTransformer(
            columns=["a", "b", "c"], new_columns_name=["iter1", "iter2", "iter3"]
        )

        df_transformed = x.transform(df)

        h.assert_equal_dispatch(
            expected=expected,
            actual=df_transformed,
            msg="InteractionTransformer one single column values",
        )

    @pytest.mark.parametrize(
        "df, expected",
        h.row_by_row_params(d.create_df_3(), d.create_df_3())
        + h.index_preserved_params(d.create_df_3(), d.create_df_3()),
    )
    def test_expected_output_min_degree_greater_than_columns(self, df, expected):
        """Test a multiple columns assignment from transform gives expected results."""

        x = InteractionTransformer(columns=["a", "b"], min_degree=3, max_degree=3)

        df_transformed = x.transform(df)

        h.assert_equal_dispatch(
            expected=expected,
            actual=df_transformed,
            msg="InteractionTransformer min degree greater than nb columns",
        )

    @pytest.mark.parametrize(
        "df, expected",
        h.row_by_row_params(d.create_df_3(), expected_df_2())
        + h.index_preserved_params(d.create_df_3(), expected_df_2()),
    )
    def test_expected_output_max_degree_greater_than_columns(self, df, expected):
        """Test a multiple columns assignment from transform gives expected results."""

        x = InteractionTransformer(columns=["a", "b"], min_degree=2, max_degree=4)

        df_transformed = x.transform(df)

        h.assert_equal_dispatch(
            expected=expected,
            actual=df_transformed,
            msg="InteractionTransformer max degree greater than nb columns",
        )

    @pytest.mark.parametrize(
        "df, expected",
        h.row_by_row_params(d.create_df_3(), expected_df_4())
        + h.index_preserved_params(d.create_df_3(), expected_df_4()),
    )
    def test_expected_output_max_degree_lower_than_new_columns_name(self, df, expected):
        """Test a multiple columns assignment from transform gives expected results."""

        x = InteractionTransformer(
            columns=["a", "b", "c"],
            new_columns_name=["iter1", "iter2", "iter3", "iter4"],
            min_degree=2,
            max_degree=2,
        )

        df_transformed = x.transform(df)

        h.assert_equal_dispatch(
            expected=expected,
            actual=df_transformed,
            msg="InteractionTransformer max degree lower than nb new columns name",
        )
