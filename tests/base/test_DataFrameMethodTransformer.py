import pytest
import test_aide as ta
import tests.test_data as d
import pandas as pd
import numpy as np

import tubular
from tubular.base import DataFrameMethodTransformer


class TestInit(object):
    """Tests for DataFrameMethodTransformer.init()."""

    def test_arguments(self):
        """Test that init has expected arguments."""

        ta.functions.test_function_arguments(
            func=DataFrameMethodTransformer.__init__,
            expected_arguments=[
                "self",
                "new_column_name",
                "pd_method_name",
                "columns",
                "pd_method_kwargs",
                "drop_original",
            ],
            expected_default_values=({}, False),
        )

    def test_class_methods(self):
        """Test that DataFrameMethodTransformer has transform method."""

        x = DataFrameMethodTransformer(
            new_column_name="a", pd_method_name="sum", columns=["b", "c"]
        )

        ta.classes.test_object_method(
            obj=x, expected_method="transform", msg="transform"
        )

    def test_inheritance(self):
        """Test that DataFrameMethodTransformer inherits from BaseTransformer."""

        x = DataFrameMethodTransformer(
            new_column_name="a", pd_method_name="sum", columns=["b", "c"]
        )

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

            DataFrameMethodTransformer(
                new_column_name="a",
                pd_method_name="sum",
                columns=["b", "c"],
                copy=False,
                verbose=True,
            )

    def test_invalid_input_type_errors(self):
        """Test that an exceptions are raised for invalid input types."""

        with pytest.raises(
            TypeError,
            match=r"unexpected type \(\<class 'int'\>\) for pd_method_name, expecting str",
        ):

            DataFrameMethodTransformer(
                new_column_name="a", pd_method_name=1, columns=["b", "c"]
            )

        with pytest.raises(
            TypeError,
            match=r"unexpected type \(\<class 'float'\>\) for new_column_name, must be str or list of strings",
        ):

            DataFrameMethodTransformer(
                new_column_name=1.0, pd_method_name="sum", columns=["b", "c"]
            )

        with pytest.raises(
            TypeError,
            match=r"if new_column_name is a list, all elements must be strings but got \<class 'float'\> in position 1",
        ):

            DataFrameMethodTransformer(
                new_column_name=["a", 1.0], pd_method_name="sum", columns=["b", "c"]
            )

        with pytest.raises(
            TypeError,
            match=r"""pd_method_kwargs should be a dict but got type \<class 'int'\>""",
        ):

            DataFrameMethodTransformer(
                new_column_name=["a", "b"],
                pd_method_name="sum",
                columns=["b", "c"],
                pd_method_kwargs=1,
            )

        with pytest.raises(
            TypeError,
            match=r"""unexpected type \(\<class 'int'\>\) for pd_method_kwargs key in position 1, must be str""",
        ):

            DataFrameMethodTransformer(
                new_column_name=["a", "b"],
                pd_method_name="sum",
                columns=["b", "c"],
                pd_method_kwargs={"a": 1, 2: "b"},
            )

        with pytest.raises(
            TypeError,
            match=r"unexpected type \(\<class 'int'\>\) for drop_original, expecting bool",
        ):

            DataFrameMethodTransformer(
                new_column_name="a",
                pd_method_name="sum",
                columns=["b", "c"],
                drop_original=30,
            )

    def test_exception_raised_non_pandas_method_passed(self):
        """Test and exception is raised if a non pd.DataFrame method is passed for pd_method_name."""

        with pytest.raises(
            AttributeError,
            match="""error accessing "b" method on pd.DataFrame object - pd_method_name should be a pd.DataFrame method""",
        ):

            DataFrameMethodTransformer(
                new_column_name="a", pd_method_name="b", columns=["b", "c"]
            )

    def test_attributes_set(self):
        """Test that the values passed for new_column_name, pd_method_name are saved to attributes on the object."""

        x = DataFrameMethodTransformer(
            new_column_name="a",
            pd_method_name="sum",
            columns=["b", "c"],
            drop_original=True,
        )

        ta.classes.test_object_attributes(
            obj=x,
            expected_attributes={
                "new_column_name": "a",
                "pd_method_name": "sum",
                "drop_original": True,
            },
            msg="Attributes for DataFrameMethodTransformer set in init",
        )


class TestTransform(object):
    """Tests for DataFrameMethodTransformer.transform()."""

    def expected_df_1():
        """Expected output of test_expected_output_single_columns_assignment."""

        df = pd.DataFrame(
            {
                "a": [1, 2, 3, 4, 5, 6, np.NaN],
                "b": [1, 2, 3, np.NaN, 7, 8, 9],
                "c": [np.NaN, 1, 2, 3, -4, -5, -6],
                "d": [1.0, 3.0, 5.0, 3.0, 3.0, 3.0, 3.0],
            }
        )

        return df

    def expected_df_2():
        """Expected output of test_expected_output_multi_columns_assignment."""

        df = pd.DataFrame(
            {
                "a": [1, 2, 3, 4, 5, 6, np.NaN],
                "b": [1, 2, 3, np.NaN, 7, 8, 9],
                "c": [np.NaN, 1, 2, 3, -4, -5, -6],
                "d": [0.5, 1.0, 1.5, np.NaN, 3.5, 4.0, 4.5],
                "e": [np.NaN, 0.5, 1.0, 1.5, -2.0, -2.5, -3.0],
            }
        )

        return df

    def test_arguments(self):
        """Test that transform has expected arguments."""

        ta.functions.test_function_arguments(
            func=DataFrameMethodTransformer.transform, expected_arguments=["self", "X"]
        )

    def test_super_transform_called(self, mocker):
        """Test that BaseTransformer.transform called."""

        df = d.create_df_3()

        x = DataFrameMethodTransformer(
            new_column_name="d", pd_method_name="sum", columns=["b", "c"]
        )

        expected_call_args = {0: {"args": (df.copy(),), "kwargs": {}}}

        with ta.functions.assert_function_call(
            mocker, tubular.base.BaseTransformer, "transform", expected_call_args
        ):

            x.transform(df)

    @pytest.mark.parametrize(
        "df, expected",
        ta.pandas.adjusted_dataframe_params(d.create_df_3(), expected_df_1()),
    )
    def test_expected_output_single_columns_assignment(self, df, expected):
        """Test a single column output from transform gives expected results."""

        x = DataFrameMethodTransformer(
            new_column_name="d",
            pd_method_name="sum",
            columns=["b", "c"],
            pd_method_kwargs={"axis": 1},
        )

        df_transformed = x.transform(df)

        ta.equality.assert_equal_dispatch(
            expected=expected,
            actual=df_transformed,
            msg="DataFrameMethodTransformer sum columns b and c",
        )

    @pytest.mark.parametrize(
        "df, expected",
        ta.pandas.adjusted_dataframe_params(d.create_df_3(), expected_df_2()),
    )
    def test_expected_output_multi_columns_assignment(self, df, expected):
        """Test a multiple column output from transform gives expected results."""

        x = DataFrameMethodTransformer(
            new_column_name=["d", "e"],
            pd_method_name="div",
            columns=["b", "c"],
            pd_method_kwargs={"other": 2},
        )

        df_transformed = x.transform(df)

        ta.equality.assert_equal_dispatch(
            expected=expected,
            actual=df_transformed,
            msg="DataFrameMethodTransformer divide by 2 columns b and c",
        )

    @pytest.mark.parametrize(
        "df, new_column_name, pd_method_name, columns, pd_method_kwargs",
        [
            (d.create_df_3(), ["d", "e"], "div", ["b", "c"], {"other": 2}),
            (d.create_df_3(), "d", "sum", ["b", "c"], {"axis": 1}),
            (d.create_df_3(), ["d", "e"], "cumprod", ["b", "c"], {"axis": 1}),
            (d.create_df_3(), ["d", "e", "f"], "mod", ["a", "b", "c"], {"other": 2}),
            (d.create_df_3(), ["d", "e", "f"], "le", ["a", "b", "c"], {"other": 0}),
            (d.create_df_3(), ["d", "e"], "abs", ["a", "b"], {}),
        ],
    )
    def test_pandas_method_called(
        self, mocker, df, new_column_name, pd_method_name, columns, pd_method_kwargs
    ):
        """Test that the pandas method is called as expected (with kwargs passed) during transform."""

        spy = mocker.spy(pd.DataFrame, pd_method_name)

        x = DataFrameMethodTransformer(
            new_column_name=new_column_name,
            pd_method_name=pd_method_name,
            columns=columns,
            pd_method_kwargs=pd_method_kwargs,
        )

        x.transform(df)

        # pull out positional and keyword args to target the call
        call_args = spy.call_args_list[0]
        call_pos_args = call_args[0]
        call_kwargs = call_args[1]

        # test keyword are as expected
        ta.equality.assert_dict_equal_msg(
            actual=call_kwargs,
            expected=pd_method_kwargs,
            msg_tag=f"""Keyword arg assert for {pd_method_name}""",
        )

        # test positional args are as expected
        ta.equality.assert_list_tuple_equal_msg(
            actual=call_pos_args,
            expected=(df[columns],),
            msg_tag=f"""Positional arg assert for {pd_method_name}""",
        )

    def test_original_columns_dropped_when_specified(self):
        """Test DataFrameMethodTransformer.transform drops original columns get when specified."""

        df = d.create_df_3()

        x = DataFrameMethodTransformer(
            new_column_name="a_b_sum",
            pd_method_name="sum",
            columns=["a", "b"],
            drop_original=True,
        )

        x.fit(df)

        df_transformed = x.transform(df)

        assert ("a" not in df_transformed.columns.values) and (
            "b" not in df_transformed.columns.values
        ), "original columns not dropped"

    def test_original_columns_kept_when_specified(self):
        """Test DataFrameMethodTransformer.transform keeps original columns when specified."""

        df = d.create_df_3()

        x = DataFrameMethodTransformer(
            new_column_name="a_b_sum",
            pd_method_name="sum",
            columns=["a", "b"],
            drop_original=False,
        )

        x.fit(df)

        df_transformed = x.transform(df)

        assert ("a" in df_transformed.columns.values) and (
            "b" in df_transformed.columns.values
        ), "original columns not kept"
