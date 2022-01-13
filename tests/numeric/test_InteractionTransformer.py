import pytest
import test_aide as ta
import tests.test_data as d
import pandas as pd
import numpy as np
import re

import tubular
from tubular.numeric import InteractionTransformer


class TestInit(object):
    """Tests for InteractionTransformer.init()."""

    def test_arguments(self):
        """Test that init has expected arguments."""

        ta.functions.test_function_arguments(
            func=InteractionTransformer.__init__,
            expected_arguments=[
                "self",
                "columns",
                "min_degree",
                "max_degree",
            ],
            expected_default_values=(2, 2),
        )

    def test_class_methods(self):
        """Test that InteractionTransformer has transform method."""

        x = InteractionTransformer(columns=["a", "b"])

        ta.classes.test_object_method(
            obj=x, expected_method="transform", msg="transform"
        )

    def test_inheritance(self):
        """Test that InteractionTransformer inherits from BaseTransformer."""

        x = InteractionTransformer(columns=["a", "b"])

        ta.classes.assert_inheritance(x, tubular.base.BaseTransformer)

    def test_super_init_called(self, mocker):
        """Test that init calls BaseTransformer.init."""

        expected_call_args = {
            0: {
                "args": (),
                "kwargs": {"columns": ["b", "c"], "verbose": True, "copy": False},
            }
        }

        with ta.functions.assert_function_call(
            mocker, tubular.base.BaseTransformer, "__init__", expected_call_args
        ):
            try:
                InteractionTransformer(
                    columns=["b", "c"],
                    min_degree=2,
                    max_degree=2,
                    copy=False,
                    verbose=True,
                )
            except AttributeError:
                pass

    def test_invalid_input_type_errors(self):
        """Test that an exceptions are raised for invalid input types."""
        with pytest.raises(
            TypeError,
            match=re.escape(
                "columns must be a string or list with the columns to be pre-processed (if specified)"
            ),
        ):

            InteractionTransformer(
                columns=3.2,
                min_degree=2,
                max_degree=2,
            )

        with pytest.raises(
            TypeError,
            match=re.escape(
                "each element of columns should be a single (string) column name"
            ),
        ):

            InteractionTransformer(
                columns=["A", "B", 4],
                min_degree=2,
                max_degree=2,
            )

        with pytest.raises(
            TypeError,
            match=r"""unexpected type \(\<class 'str'\>\) for min_degree, must be int""",
        ):

            InteractionTransformer(
                columns=["A", "B", "C"],
                min_degree="2",
                max_degree=2,
            )
        with pytest.raises(
            TypeError,
            match=r"""unexpected type \(\<class 'str'\>\) for max_degree, must be int""",
        ):

            InteractionTransformer(
                columns=["A", "B", "C"],
                min_degree=2,
                max_degree="2",
            )

    def test_invalid_input_value_errors(self):
        """Test and exception is raised if degrees or columns provided are inconsistent."""
        with pytest.raises(
            ValueError,
            match=r"""umber of columns must be equal or greater than 2, got 1 column.""",
        ):

            InteractionTransformer(
                columns=["A"],
            )

        with pytest.raises(
            ValueError,
            match=r"""min_degree must be equal or greater than 2, got 0""",
        ):

            InteractionTransformer(
                columns=["A", "B", "C"],
                min_degree=0,
                max_degree=2,
            )

        with pytest.raises(
            ValueError,
            match=r"""max_degree must be equal or greater than min_degree""",
        ):

            InteractionTransformer(
                columns=["A", "B", "C"],
                min_degree=3,
                max_degree=2,
            )
        # NEW
        with pytest.raises(
            ValueError,
            match=r"""max_degree must be equal or lower than number of columns""",
        ):

            InteractionTransformer(
                columns=["A", "B", "C"],
                min_degree=3,
                max_degree=4,
            )

    def test_attributes_set(self):
        """Test that the values passed for columns, degrees are saved to attributes on the object."""

        x = InteractionTransformer(
            columns=["A", "B", "C"],
            min_degree=2,
            max_degree=3,
        )

        ta.classes.test_object_attributes(
            obj=x,
            expected_attributes={
                "columns": ["A", "B", "C"],
                "min_degree": 2,
                "max_degree": 3,
            },
            msg="Attributes for InteractionTransformer set in init",
        )


class TestTransform(object):
    """Tests for InteractionTransformer.transform()."""

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

    def test_arguments(self):
        """Test that transform has expected arguments."""

        ta.functions.test_function_arguments(
            func=InteractionTransformer.transform, expected_arguments=["self", "X"]
        )

    def test_super_transform_called(self, mocker):
        """Test that BaseTransformer.transform called."""

        df = d.create_df_3()

        x = InteractionTransformer(columns=["b", "c"])

        expected_call_args = {0: {"args": (df.copy(),), "kwargs": {}}}

        with ta.functions.assert_function_call(
            mocker, tubular.base.BaseTransformer, "transform", expected_call_args
        ):

            x.transform(df)

    @pytest.mark.parametrize(
        "df, expected",
        ta.pandas.adjusted_dataframe_params(d.create_df_3(), expected_df_1()),
    )
    def test_expected_output_default_assignment(self, df, expected):
        """Test default values and multiple columns assignment from transform gives expected results."""

        x = InteractionTransformer(columns=["a", "b", "c"])

        df_transformed = x.transform(df)

        ta.equality.assert_equal_dispatch(
            expected=expected,
            actual=df_transformed,
            msg="InteractionTransformer default values",
        )

    @pytest.mark.parametrize(
        "df, expected",
        ta.pandas.adjusted_dataframe_params(d.create_df_3(), expected_df_2()),
    )
    def test_expected_output_multiple_columns_assignment(self, df, expected):
        """Test a multiple columns assignment from transform gives expected results."""

        x = InteractionTransformer(columns=["a", "b"])

        df_transformed = x.transform(df)

        ta.equality.assert_equal_dispatch(
            expected=expected,
            actual=df_transformed,
            msg="InteractionTransformer one single column values",
        )
