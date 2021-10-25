import pytest
import test_aide as ta
import tests.test_data as d

import tubular
from tubular.dates import SeriesDtMethodTransformer


class TestInit(object):
    """Tests for SeriesDtMethodTransformer.init()."""

    def test_arguments(self):
        """Test that init has expected arguments."""

        ta.function_helpers.test_function_arguments(
            func=SeriesDtMethodTransformer.__init__,
            expected_arguments=[
                "self",
                "new_column_name",
                "pd_method_name",
                "column",
                "pd_method_kwargs",
            ],
            expected_default_values=({},),
        )

    def test_class_methods(self):
        """Test that SeriesDtMethodTransformer has transform method."""

        x = SeriesDtMethodTransformer(
            new_column_name="a", pd_method_name="year", column="b"
        )

        ta.class_helpers.test_object_method(
            obj=x, expected_method="transform", msg="transform"
        )

    def test_inheritance(self):
        """Test that SeriesDtMethodTransformer inherits from BaseTransformer."""

        x = SeriesDtMethodTransformer(
            new_column_name="a", pd_method_name="year", column="b"
        )

        ta.class_helpers.assert_inheritance(x, tubular.base.BaseTransformer)

    def test_super_init_called(self, mocker):
        """Test that init calls BaseTransformer.init."""

        expected_call_args = {
            0: {"args": (), "kwargs": {"columns": "b", "verbose": True, "copy": False}}
        }

        with ta.function_helpers.assert_function_call(
            mocker, tubular.base.BaseTransformer, "__init__", expected_call_args
        ):

            SeriesDtMethodTransformer(
                new_column_name="a",
                pd_method_name="year",
                column="b",
                copy=False,
                verbose=True,
            )

    def test_invalid_input_type_errors(self):
        """Test that an exceptions are raised for invalid input types."""

        with pytest.raises(
            TypeError,
            match=r"column should be a str but got \<class 'list'\>",
        ):

            SeriesDtMethodTransformer(
                new_column_name="a", pd_method_name=1, column=["b", "c"]
            )

        with pytest.raises(
            TypeError,
            match=r"unexpected type \(\<class 'int'\>\) for pd_method_name, expecting str",
        ):

            SeriesDtMethodTransformer(new_column_name="a", pd_method_name=1, column="b")

        with pytest.raises(
            TypeError,
            match=r"unexpected type \(\<class 'float'\>\) for new_column_name, must be str",
        ):

            SeriesDtMethodTransformer(
                new_column_name=1.0, pd_method_name="year", column="b"
            )

        with pytest.raises(
            TypeError,
            match=r"""pd_method_kwargs should be a dict but got type \<class 'int'\>""",
        ):

            SeriesDtMethodTransformer(
                new_column_name="a",
                pd_method_name="year",
                column="b",
                pd_method_kwargs=1,
            )

        with pytest.raises(
            TypeError,
            match=r"""unexpected type \(\<class 'int'\>\) for pd_method_kwargs key in position 1, must be str""",
        ):

            SeriesDtMethodTransformer(
                new_column_name="a",
                pd_method_name="year",
                column="b",
                pd_method_kwargs={"a": 1, 2: "b"},
            )

    def test_exception_raised_non_pandas_method_passed(self):
        """Test and exception is raised if a non pd.Series.dt method is passed for pd_method_name."""

        with pytest.raises(
            AttributeError,
            match="""error accessing "dt.b" method on pd.Series object - pd_method_name should be a pd.Series.dt method""",
        ):

            SeriesDtMethodTransformer(
                new_column_name="a", pd_method_name="b", column="b"
            )

    def test_attributes_set(self):
        """Test that the values passed for new_column_name, pd_method_name are saved to attributes on the object."""

        x = SeriesDtMethodTransformer(
            new_column_name="a",
            pd_method_name="year",
            column="b",
            pd_method_kwargs={"d": 1},
        )

        ta.class_helpers.test_object_attributes(
            obj=x,
            expected_attributes={
                "new_column_name": "a",
                "pd_method_name": "year",
                "pd_method_kwargs": {"d": 1},
            },
            msg="Attributes for SeriesDtMethodTransformer set in init",
        )

    @pytest.mark.parametrize(
        "pd_method_name, callable_attr", [("year", False), ("to_period", True)]
    )
    def test_callable_attribute_set(self, pd_method_name, callable_attr):
        """Test the _callable attribute is set to True if pd.Series.dt.pd_method_name is callable."""

        x = SeriesDtMethodTransformer(
            new_column_name="a",
            pd_method_name=pd_method_name,
            column="b",
            pd_method_kwargs={"d": 1},
        )

        ta.class_helpers.test_object_attributes(
            obj=x,
            expected_attributes={"_callable": callable_attr},
            msg="_callable attribute for SeriesDtMethodTransformer set in init",
        )


class TestTransform(object):
    """Tests for SeriesDtMethodTransformer.transform()."""

    def expected_df_1():
        """Expected output of test_expected_output_no_overwrite."""

        df = d.create_datediff_test_df()

        df["a_year"] = [1993, 2000, 2018, 2018, 2018, 2018, 2018, 1985]

        return df

    def expected_df_2():
        """Expected output of test_expected_output_overwrite."""

        df = d.create_datediff_test_df()

        df["a"] = [1993, 2000, 2018, 2018, 2018, 2018, 2018, 1985]

        return df

    def expected_df_3():
        """Expected output of test_expected_output_callable."""

        df = d.create_datediff_test_df()

        df["b_new"] = df["b"].dt.to_period("M")

        return df

    def test_arguments(self):
        """Test that transform has expected arguments."""

        ta.function_helpers.test_function_arguments(
            func=SeriesDtMethodTransformer.transform, expected_arguments=["self", "X"]
        )

    def test_super_transform_called(self, mocker):
        """Test that BaseTransformer.transform called."""

        df = d.create_datediff_test_df()

        x = SeriesDtMethodTransformer(
            new_column_name="a2", pd_method_name="year", column="a"
        )

        expected_call_args = {
            0: {"args": (d.create_datediff_test_df(),), "kwargs": {}}
        }

        with ta.function_helpers.assert_function_call(
            mocker, tubular.base.BaseTransformer, "transform", expected_call_args
        ):

            x.transform(df)

    @pytest.mark.parametrize(
        "df, expected",
        ta.pandas_helpers.row_by_row_params(
            d.create_datediff_test_df(), expected_df_1()
        )
        + ta.pandas_helpers.index_preserved_params(
            d.create_datediff_test_df(), expected_df_1()
        ),
    )
    def test_expected_output_no_overwrite(self, df, expected):
        """Test a single column output from transform gives expected results, when not overwriting the original column."""

        x = SeriesDtMethodTransformer(
            new_column_name="a_year",
            pd_method_name="year",
            column="a",
            pd_method_kwargs={},
        )

        df_transformed = x.transform(df)

        ta.equality_helpers.assert_frame_equal_msg(
            actual=df_transformed,
            expected=expected,
            msg_tag="Unexpected values in SeriesDtMethodTransformer.transform with find, not overwriting original column",
        )

    @pytest.mark.parametrize(
        "df, expected",
        ta.pandas_helpers.row_by_row_params(
            d.create_datediff_test_df(), expected_df_2()
        )
        + ta.pandas_helpers.index_preserved_params(
            d.create_datediff_test_df(), expected_df_2()
        ),
    )
    def test_expected_output_overwrite(self, df, expected):
        """Test a single column output from transform gives expected results, when overwriting the original column."""

        x = SeriesDtMethodTransformer(
            new_column_name="a",
            pd_method_name="year",
            column="a",
            pd_method_kwargs={},
        )

        df_transformed = x.transform(df)

        ta.equality_helpers.assert_frame_equal_msg(
            actual=df_transformed,
            expected=expected,
            msg_tag="Unexpected values in SeriesDtMethodTransformer.transform with pad, overwriting original column",
        )

    @pytest.mark.parametrize(
        "df, expected",
        ta.pandas_helpers.row_by_row_params(
            d.create_datediff_test_df(), expected_df_3()
        )
        + ta.pandas_helpers.index_preserved_params(
            d.create_datediff_test_df(), expected_df_3()
        ),
    )
    def test_expected_output_callable(self, df, expected):
        """Test transform gives expected results, when pd_method_name is a callable."""

        x = SeriesDtMethodTransformer(
            new_column_name="b_new",
            pd_method_name="to_period",
            column="b",
            pd_method_kwargs={"freq": "M"},
        )

        df_transformed = x.transform(df)

        ta.equality_helpers.assert_frame_equal_msg(
            actual=df_transformed,
            expected=expected,
            msg_tag="Unexpected values in SeriesDtMethodTransformer.transform with to_period",
        )

    def test_attributes_unchanged_by_transform(self):
        """Test that attributes set in init are unchanged by the transform method."""

        df = d.create_datediff_test_df()

        x = SeriesDtMethodTransformer(
            new_column_name="b_new",
            pd_method_name="to_period",
            column="b",
            pd_method_kwargs={"freq": "M"},
        )

        x2 = SeriesDtMethodTransformer(
            new_column_name="b_new",
            pd_method_name="to_period",
            column="b",
            pd_method_kwargs={"freq": "M"},
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
