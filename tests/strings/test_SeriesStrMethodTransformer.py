import pytest
import tubular.testing.test_data as d
import tubular.testing.helpers as h

import tubular
from tubular.strings import SeriesStrMethodTransformer
import pandas as pd


class TestInit(object):
    """Tests for SeriesStrMethodTransformer.init()."""

    def test_arguments(self):
        """Test that init has expected arguments."""

        h.test_function_arguments(
            func=SeriesStrMethodTransformer.__init__,
            expected_arguments=[
                "self",
                "new_column_name",
                "pd_method_name",
                "columns",
                "pd_method_kwargs",
            ],
            expected_default_values=({},),
        )

    def test_class_methods(self):
        """Test that SeriesStrMethodTransformer has transform method."""

        x = SeriesStrMethodTransformer(
            new_column_name="a", pd_method_name="find", columns=["b"]
        )

        h.test_object_method(obj=x, expected_method="transform", msg="transform")

    def test_inheritance(self):
        """Test that SeriesStrMethodTransformer inherits from BaseTransformer."""

        x = SeriesStrMethodTransformer(
            new_column_name="a", pd_method_name="find", columns=["b"]
        )

        h.assert_inheritance(x, tubular.base.BaseTransformer)

    def test_super_init_called(self, mocker):
        """Test that init calls BaseTransformer.init."""

        expected_call_args = {
            0: {
                "args": (),
                "kwargs": {"columns": ["b"], "verbose": True, "copy": False},
            }
        }

        with h.assert_function_call(
            mocker, tubular.base.BaseTransformer, "__init__", expected_call_args
        ):

            SeriesStrMethodTransformer(
                new_column_name="a",
                pd_method_name="find",
                columns=["b"],
                copy=False,
                verbose=True,
            )

    def test_invalid_input_type_errors(self):
        """Test that an exceptions are raised for invalid input types."""

        with pytest.raises(
            ValueError,
            match="columns arg should contain only 1 column name but got 2",
        ):

            SeriesStrMethodTransformer(
                new_column_name="a", pd_method_name=1, columns=["b", "c"]
            )

        with pytest.raises(
            TypeError,
            match=r"unexpected type \(\<class 'int'\>\) for pd_method_name, expecting str",
        ):

            SeriesStrMethodTransformer(
                new_column_name="a", pd_method_name=1, columns=["b"]
            )

        with pytest.raises(
            TypeError,
            match=r"unexpected type \(\<class 'float'\>\) for new_column_name, must be str",
        ):

            SeriesStrMethodTransformer(
                new_column_name=1.0, pd_method_name="find", columns=["b"]
            )

        with pytest.raises(
            TypeError,
            match=r"""pd_method_kwargs should be a dict but got type \<class 'int'\>""",
        ):

            SeriesStrMethodTransformer(
                new_column_name="a",
                pd_method_name="find",
                columns=["b"],
                pd_method_kwargs=1,
            )

        with pytest.raises(
            TypeError,
            match=r"""unexpected type \(\<class 'int'\>\) for pd_method_kwargs key in position 1, must be str""",
        ):

            SeriesStrMethodTransformer(
                new_column_name="a",
                pd_method_name="find",
                columns=["b"],
                pd_method_kwargs={"a": 1, 2: "b"},
            )

    def test_exception_raised_non_pandas_method_passed(self):
        """Test and exception is raised if a non pd.Series.str method is passed for pd_method_name."""

        with pytest.raises(
            AttributeError,
            match="""error accessing "str.b" method on pd.Series object - pd_method_name should be a pd.Series.str method""",
        ):

            SeriesStrMethodTransformer(
                new_column_name="a", pd_method_name="b", columns=["b"]
            )

    def test_attributes_set(self):
        """Test that the values passed for new_column_name, pd_method_name are saved to attributes on the object."""

        x = SeriesStrMethodTransformer(
            new_column_name="a",
            pd_method_name="find",
            columns=["b"],
            pd_method_kwargs={"d": 1},
        )

        h.test_object_attributes(
            obj=x,
            expected_attributes={
                "new_column_name": "a",
                "pd_method_name": "find",
                "pd_method_kwargs": {"d": 1},
            },
            msg="Attributes for SeriesStrMethodTransformer set in init",
        )


class TestTransform(object):
    """Tests for SeriesStrMethodTransformer.transform()."""

    def expected_df_1():
        """Expected output of test_expected_output_no_overwrite."""

        df = d.create_df_7()

        df["b_new"] = df["b"].str.find(sub="a")

        return df

    def expected_df_2():
        """Expected output of test_expected_output_overwrite."""

        df = d.create_df_7()

        df["b"] = df["b"].str.pad(width=10)

        return df

    def test_arguments(self):
        """Test that transform has expected arguments."""

        h.test_function_arguments(
            func=SeriesStrMethodTransformer.transform, expected_arguments=["self", "X"]
        )

    def test_super_transform_called(self, mocker):
        """Test that BaseTransformer.transform called."""

        df = d.create_df_7()

        x = SeriesStrMethodTransformer(
            new_column_name="cc", pd_method_name="find", columns=["c"]
        )

        expected_call_args = {0: {"args": (d.create_df_7(),), "kwargs": {}}}

        with h.assert_function_call(
            mocker, tubular.base.BaseTransformer, "transform", expected_call_args
        ):

            x.transform(df)

    @pytest.mark.parametrize(
        "df, expected",
        h.row_by_row_params(d.create_df_7(), expected_df_1())
        + h.index_preserved_params(d.create_df_7(), expected_df_1()),
    )
    def test_expected_output_no_overwrite(self, df, expected):
        """Test a single column output from transform gives expected results, when not overwriting the original column."""

        x = SeriesStrMethodTransformer(
            new_column_name="b_new",
            pd_method_name="find",
            columns=["b"],
            pd_method_kwargs={"sub": "a"},
        )

        df_transformed = x.transform(df)

        h.assert_frame_equal_msg(
            actual=df_transformed,
            expected=expected,
            msg_tag="Unexpected values in SeriesStrMethodTransformer.transform with find, not overwriting original column",
        )

    @pytest.mark.parametrize(
        "df, expected",
        h.row_by_row_params(d.create_df_7(), expected_df_2())
        + h.index_preserved_params(d.create_df_7(), expected_df_2()),
    )
    def test_expected_output_overwrite(self, df, expected):
        """Test a single column output from transform gives expected results, when overwriting the original column."""

        x = SeriesStrMethodTransformer(
            new_column_name="b",
            pd_method_name="pad",
            columns=["b"],
            pd_method_kwargs={"width": 10},
        )

        df_transformed = x.transform(df)

        h.assert_frame_equal_msg(
            actual=df_transformed,
            expected=expected,
            msg_tag="Unexpected values in SeriesStrMethodTransformer.transform with pad, overwriting original column",
        )

    @pytest.mark.parametrize(
        "df, new_column_name, pd_method_name, columns, pd_method_kwargs",
        [
            (d.create_df_7(), "b_new", "find", ["b"], {"sub": "a"}),
            (
                d.create_df_7(),
                "c_slice",
                "slice",
                ["c"],
                {"start": 0, "stop": 1, "step": 1},
            ),
            (d.create_df_7(), "b_upper", "upper", ["b"], {}),
        ],
    )
    def test_pandas_method_called(
        self, mocker, df, new_column_name, pd_method_name, columns, pd_method_kwargs
    ):
        """Test that the pandas.Series.str method is called as expected (with kwargs passed) during transform."""

        spy = mocker.spy(pd.Series.str, pd_method_name)

        x = SeriesStrMethodTransformer(
            new_column_name=new_column_name,
            pd_method_name=pd_method_name,
            columns=columns,
            pd_method_kwargs=pd_method_kwargs,
        )

        x.transform(df)

        # pull out positional and keyword args to target the call
        call_args = spy.call_args_list[0]
        call_kwargs = call_args[1]

        # test keyword are as expected
        h.assert_dict_equal_msg(
            actual=call_kwargs,
            expected=pd_method_kwargs,
            msg_tag=f"""Keyword arg assert for {pd_method_name}""",
        )

    def test_attributes_unchanged_by_transform(self):
        """Test that attributes set in init are unchanged by the transform method."""

        df = d.create_df_7()

        x = SeriesStrMethodTransformer(
            new_column_name="b",
            pd_method_name="pad",
            columns=["b"],
            pd_method_kwargs={"width": 10},
        )

        x2 = SeriesStrMethodTransformer(
            new_column_name="b",
            pd_method_name="pad",
            columns=["b"],
            pd_method_kwargs={"width": 10},
        )

        x.transform(df)

        assert (
            x.new_column_name == x2.new_column_name
        ), "new_column_name changed by SeriesDtMethodTransformer.transform"
        assert (
            x.pd_method_name == x2.pd_method_name
        ), "pd_method_name changed by SeriesDtMethodTransformer.transform"
        assert (
            x.columns == x2.columns
        ), "columns changed by SeriesDtMethodTransformer.transform"
        assert (
            x.pd_method_kwargs == x2.pd_method_kwargs
        ), "pd_method_kwargs changed by SeriesDtMethodTransformer.transform"
