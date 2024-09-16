import numpy as np
import pytest
import test_aide as ta

import tests.test_data as d
from tubular.dates import SeriesDtMethodTransformer


class TestInit:
    """Tests for SeriesDtMethodTransformer.init()."""

    def test_invalid_input_type_errors(self):
        """Test that an exceptions are raised for invalid input types."""
        with pytest.raises(
            TypeError,
            match=r"SeriesDtMethodTransformer: column should be a str but got \<class 'list'\>",
        ):
            SeriesDtMethodTransformer(
                new_column_name="a",
                pd_method_name=1,
                column=["b", "c"],
            )

        with pytest.raises(
            TypeError,
            match=r"SeriesDtMethodTransformer: unexpected type \(\<class 'int'\>\) for pd_method_name, expecting str",
        ):
            SeriesDtMethodTransformer(new_column_name="a", pd_method_name=1, column="b")

        with pytest.raises(
            TypeError,
            match=r"SeriesDtMethodTransformer: new_column_name should be str",
        ):
            SeriesDtMethodTransformer(
                new_column_name=1.0,
                pd_method_name="year",
                column="b",
            )

        with pytest.raises(
            TypeError,
            match=r"""SeriesDtMethodTransformer: pd_method_kwargs should be a dict but got type \<class 'int'\>""",
        ):
            SeriesDtMethodTransformer(
                new_column_name="a",
                pd_method_name="year",
                column="b",
                pd_method_kwargs=1,
            )

        with pytest.raises(
            TypeError,
            match=r"""SeriesDtMethodTransformer: unexpected type \(\<class 'int'\>\) for pd_method_kwargs key in position 1, must be str""",
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
            match="""SeriesDtMethodTransformer: error accessing "dt.b" method on pd.Series object - pd_method_name should be a pd.Series.dt method""",
        ):
            SeriesDtMethodTransformer(
                new_column_name="a",
                pd_method_name="b",
                column="b",
            )

    def test_attributes_set(self):
        """Test that the values passed for new_column_name, pd_method_name are saved to attributes on the object."""
        x = SeriesDtMethodTransformer(
            new_column_name="a",
            pd_method_name="year",
            column="b",
            pd_method_kwargs={"d": 1},
        )

        ta.classes.test_object_attributes(
            obj=x,
            expected_attributes={
                "column": "b",
                "new_column_name": "a",
                "pd_method_name": "year",
                "pd_method_kwargs": {"d": 1},
            },
            msg="Attributes for SeriesDtMethodTransformer set in init",
        )

    @pytest.mark.parametrize(
        ("pd_method_name", "callable_attr"),
        [("year", False), ("to_period", True)],
    )
    def test_callable_attribute_set(self, pd_method_name, callable_attr):
        """Test the _callable attribute is set to True if pd.Series.dt.pd_method_name is callable."""
        x = SeriesDtMethodTransformer(
            new_column_name="a",
            pd_method_name=pd_method_name,
            column="b",
            pd_method_kwargs={"d": 1},
        )

        ta.classes.test_object_attributes(
            obj=x,
            expected_attributes={"_callable": callable_attr},
            msg="_callable attribute for SeriesDtMethodTransformer set in init",
        )


class TestTransform:
    """Tests for SeriesDtMethodTransformer.transform()."""

    def expected_df_1():
        """Expected output of test_expected_output_no_overwrite."""
        df = d.create_datediff_test_df()

        df["a_year"] = np.array(
            [1993, 2000, 2018, 2018, 2018, 2018, 2018, 1985],
            dtype=np.int32,
        )

        return df

    def expected_df_2():
        """Expected output of test_expected_output_overwrite."""
        df = d.create_datediff_test_df()

        df["a"] = np.array(
            [1993, 2000, 2018, 2018, 2018, 2018, 2018, 1985],
            dtype=np.int32,
        )

        return df

    def expected_df_3():
        """Expected output of test_expected_output_callable."""
        df = d.create_datediff_test_df()

        df["b_new"] = df["b"].dt.to_period("M")

        return df

    @pytest.mark.parametrize(
        ("bad_column", "bad_type"),
        [
            ("numeric_col", "int64"),
            ("string_col", "object"),
            ("bool_col", "bool"),
            ("empty_col", "object"),
            ("date_col", "date"),
        ],
    )
    def test_input_data_check_column_errors(self, bad_column, bad_type):
        """Check that errors are raised on a variety of different non datatypes"""
        x = SeriesDtMethodTransformer(
            new_column_name="a2",
            pd_method_name="year",
            column=bad_column,
        )

        df = d.create_date_diff_incorrect_dtypes()

        msg = rf"{x.classname()}: {x.columns[0]} type should be in \['datetime64'\] but got {bad_type}"

        with pytest.raises(TypeError, match=msg):
            x.transform(df)

    @pytest.mark.parametrize(
        ("df", "expected"),
        ta.pandas.adjusted_dataframe_params(
            d.create_datediff_test_df(),
            expected_df_1(),
        ),
    )
    def test_expected_output_no_overwrite(self, df, expected):
        """Test a single column output from transform gives expected results, when not overwriting the original column."""
        x = SeriesDtMethodTransformer(
            new_column_name="a_year",
            pd_method_name="year",
            column="a",
            pd_method_kwargs=None,
        )

        df_transformed = x.transform(df)

        ta.equality.assert_frame_equal_msg(
            actual=df_transformed,
            expected=expected,
            msg_tag="Unexpected values in SeriesDtMethodTransformer.transform with find, not overwriting original column",
        )

    @pytest.mark.parametrize(
        ("df", "expected"),
        ta.pandas.adjusted_dataframe_params(
            d.create_datediff_test_df(),
            expected_df_2(),
        ),
    )
    def test_expected_output_overwrite(self, df, expected):
        """Test a single column output from transform gives expected results, when overwriting the original column."""
        x = SeriesDtMethodTransformer(
            new_column_name="a",
            pd_method_name="year",
            column="a",
            pd_method_kwargs=None,
        )

        df_transformed = x.transform(df)

        ta.equality.assert_frame_equal_msg(
            actual=df_transformed,
            expected=expected,
            msg_tag="Unexpected values in SeriesDtMethodTransformer.transform with pad, overwriting original column",
        )

    @pytest.mark.parametrize(
        ("df", "expected"),
        ta.pandas.adjusted_dataframe_params(
            d.create_datediff_test_df(),
            expected_df_3(),
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

        ta.equality.assert_frame_equal_msg(
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
