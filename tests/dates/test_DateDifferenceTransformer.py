import datetime

import numpy as np
import pandas as pd
import pytest
import test_aide as ta

import tests.test_data as d
import tubular
from tubular.dates import DateDifferenceTransformer


class TestInit:
    """Tests for DateDifferenceTransformer.init()."""

    def test_super_init_called(self, mocker):
        """Test that init calls BaseTransformer.init."""
        expected_call_args = {
            0: {
                "args": (),
                "kwargs": {
                    "columns": ["dummy_1", "dummy_2"],
                    "verbose": False,
                    "copy": None,
                },
            },
        }

        with ta.functions.assert_function_call(
            mocker,
            tubular.base.BaseTransformer,
            "__init__",
            expected_call_args,
        ):
            DateDifferenceTransformer(
                column_lower="dummy_1",
                column_upper="dummy_2",
                new_column_name="dummy_3",
                units="D",
            )

    def test_column_lower_type_error(self):
        """Test that an exception is raised if column_lower is not a str."""
        with pytest.raises(
            TypeError,
            match="DateDifferenceTransformer: column_lower must be a str",
        ):
            DateDifferenceTransformer(
                column_lower=123,
                column_upper="dummy_2",
                new_column_name="dummy_3",
                units="D",
                verbose=False,
            )

    def test_column_2_type_error(self):
        """Test that an exception is raised if column_upper is not a str."""
        with pytest.raises(
            TypeError,
            match="DateDifferenceTransformer: column_upper must be a str",
        ):
            DateDifferenceTransformer(
                column_lower="dummy_1",
                column_upper=123,
                new_column_name="dummy_3",
                units="D",
                verbose=False,
            )

    def test_new_column_name_type_error(self):
        """Test that an exception is raised if new_column_name is not a str."""
        with pytest.raises(
            TypeError,
            match="DateDifferenceTransformer: new_column_name must be a str",
        ):
            DateDifferenceTransformer(
                column_lower="dummy_1",
                column_upper="dummy_2",
                new_column_name=123,
                units="D",
                verbose=False,
            )

    def test_units_type_error(self):
        """Test that an exception is raised if new_column_name is not a str."""
        with pytest.raises(
            TypeError,
            match="DateDifferenceTransformer: units must be a str",
        ):
            DateDifferenceTransformer(
                column_lower="dummy_1",
                column_upper="dummy_2",
                new_column_name="dummy_3",
                units=123,
                verbose=False,
            )

    def test_units_values_error(self):
        """Test that an exception is raised if the value of inits is not one of accepted_values_units."""
        with pytest.raises(
            ValueError,
            match=r"DateDifferenceTransformer: units must be one of \['D', 'h', 'm', 's'\], got y",
        ):
            DateDifferenceTransformer(
                column_lower="dummy_1",
                column_upper="dummy_2",
                new_column_name="dummy_3",
                units="y",
                verbose=False,
            )

    def test_inputs_set_to_attribute(self):
        """Test that the value passed for new_column_name and units are saved in attributes of the same name."""
        x = DateDifferenceTransformer(
            column_lower="dummy_1",
            column_upper="dummy_2",
            new_column_name="value_1",
            units="D",
            verbose=False,
        )

        ta.classes.test_object_attributes(
            obj=x,
            expected_attributes={
                "columns": ["dummy_1", "dummy_2"],
                "new_column_name": "value_1",
                "units": "D",
                "verbose": False,
            },
            msg="Attributes for DateDifferenceTransformer set in init",
        )

    def test_inputs_set_to_attribute_name_not_set(self):
        """Test that the value passed for new_column_new_column_name and units are saved in attributes of the same new_column_name."""
        x = DateDifferenceTransformer(
            column_lower="dummy_1",
            column_upper="dummy_2",
            units="D",
            verbose=False,
        )

        ta.classes.test_object_attributes(
            obj=x,
            expected_attributes={
                "column_lower": "dummy_1",
                "column_upper": "dummy_2",
                "columns": ["dummy_1", "dummy_2"],
                "new_column_name": "dummy_2_dummy_1_datediff_D",
                "units": "D",
                "verbose": False,
            },
            msg="Attributes for DateDifferenceTransformer set in init",
        )


class TestTransform:
    """Tests for DateDifferenceTransformer.transform()."""

    def expected_df_3():
        """Expected output for test_expected_output_units_D."""
        return pd.DataFrame(
            {
                "a": [
                    datetime.datetime(
                        1993,
                        9,
                        27,
                        11,
                        58,
                        58,
                        tzinfo=datetime.timezone.utc,
                    ),
                    datetime.datetime(
                        2000,
                        3,
                        19,
                        12,
                        59,
                        59,
                        tzinfo=datetime.timezone.utc,
                    ),
                    datetime.datetime(
                        2018,
                        11,
                        10,
                        11,
                        59,
                        59,
                        tzinfo=datetime.timezone.utc,
                    ),
                    datetime.datetime(
                        2018,
                        10,
                        10,
                        11,
                        59,
                        59,
                        tzinfo=datetime.timezone.utc,
                    ),
                    datetime.datetime(
                        2018,
                        10,
                        10,
                        11,
                        59,
                        59,
                        tzinfo=datetime.timezone.utc,
                    ),
                    datetime.datetime(
                        2018,
                        10,
                        10,
                        10,
                        59,
                        59,
                        tzinfo=datetime.timezone.utc,
                    ),
                    datetime.datetime(
                        2018,
                        12,
                        10,
                        11,
                        59,
                        59,
                        tzinfo=datetime.timezone.utc,
                    ),
                    datetime.datetime(
                        1985,
                        7,
                        23,
                        11,
                        59,
                        59,
                        tzinfo=datetime.timezone.utc,
                    ),
                ],
                "b": [
                    datetime.datetime(
                        2020,
                        5,
                        1,
                        12,
                        59,
                        59,
                        tzinfo=datetime.timezone.utc,
                    ),
                    datetime.datetime(
                        2019,
                        12,
                        25,
                        11,
                        58,
                        58,
                        tzinfo=datetime.timezone.utc,
                    ),
                    datetime.datetime(
                        2018,
                        11,
                        10,
                        11,
                        59,
                        59,
                        tzinfo=datetime.timezone.utc,
                    ),
                    datetime.datetime(
                        2018,
                        11,
                        10,
                        11,
                        59,
                        59,
                        tzinfo=datetime.timezone.utc,
                    ),
                    datetime.datetime(
                        2018,
                        9,
                        10,
                        9,
                        59,
                        59,
                        tzinfo=datetime.timezone.utc,
                    ),
                    datetime.datetime(
                        2015,
                        11,
                        10,
                        11,
                        59,
                        59,
                        tzinfo=datetime.timezone.utc,
                    ),
                    datetime.datetime(
                        2015,
                        11,
                        10,
                        12,
                        59,
                        59,
                        tzinfo=datetime.timezone.utc,
                    ),
                    datetime.datetime(
                        2015,
                        7,
                        23,
                        11,
                        59,
                        59,
                        tzinfo=datetime.timezone.utc,
                    ),
                ],
                "D": [
                    9713.042372685186,
                    7219.957627314815,
                    0.0,
                    31.0,
                    -30.083333333333332,
                    -1064.9583333333333,
                    -1125.9583333333333,
                    10957.0,
                ],
            },
        )

    def expected_df_4():
        """Expected output for test_expected_output_units_h."""
        return pd.DataFrame(
            {
                "a": [
                    datetime.datetime(
                        1993,
                        9,
                        27,
                        11,
                        58,
                        58,
                        tzinfo=datetime.timezone.utc,
                    ),
                    datetime.datetime(
                        2000,
                        3,
                        19,
                        12,
                        59,
                        59,
                        tzinfo=datetime.timezone.utc,
                    ),
                    datetime.datetime(
                        2018,
                        11,
                        10,
                        11,
                        59,
                        59,
                        tzinfo=datetime.timezone.utc,
                    ),
                    datetime.datetime(
                        2018,
                        10,
                        10,
                        11,
                        59,
                        59,
                        tzinfo=datetime.timezone.utc,
                    ),
                    datetime.datetime(
                        2018,
                        10,
                        10,
                        11,
                        59,
                        59,
                        tzinfo=datetime.timezone.utc,
                    ),
                    datetime.datetime(
                        2018,
                        10,
                        10,
                        10,
                        59,
                        59,
                        tzinfo=datetime.timezone.utc,
                    ),
                    datetime.datetime(
                        2018,
                        12,
                        10,
                        11,
                        59,
                        59,
                        tzinfo=datetime.timezone.utc,
                    ),
                    datetime.datetime(
                        1985,
                        7,
                        23,
                        11,
                        59,
                        59,
                        tzinfo=datetime.timezone.utc,
                    ),
                ],
                "b": [
                    datetime.datetime(
                        2020,
                        5,
                        1,
                        12,
                        59,
                        59,
                        tzinfo=datetime.timezone.utc,
                    ),
                    datetime.datetime(
                        2019,
                        12,
                        25,
                        11,
                        58,
                        58,
                        tzinfo=datetime.timezone.utc,
                    ),
                    datetime.datetime(
                        2018,
                        11,
                        10,
                        11,
                        59,
                        59,
                        tzinfo=datetime.timezone.utc,
                    ),
                    datetime.datetime(
                        2018,
                        11,
                        10,
                        11,
                        59,
                        59,
                        tzinfo=datetime.timezone.utc,
                    ),
                    datetime.datetime(
                        2018,
                        9,
                        10,
                        9,
                        59,
                        59,
                        tzinfo=datetime.timezone.utc,
                    ),
                    datetime.datetime(
                        2015,
                        11,
                        10,
                        11,
                        59,
                        59,
                        tzinfo=datetime.timezone.utc,
                    ),
                    datetime.datetime(
                        2015,
                        11,
                        10,
                        12,
                        59,
                        59,
                        tzinfo=datetime.timezone.utc,
                    ),
                    datetime.datetime(
                        2015,
                        7,
                        23,
                        11,
                        59,
                        59,
                        tzinfo=datetime.timezone.utc,
                    ),
                ],
                "h": [
                    233113.01694444445,
                    173278.98305555555,
                    0.0,
                    744.0,
                    -722.0,
                    -25559.0,
                    -27023.0,
                    262968.0,
                ],
            },
        )

    def expected_df_5():
        """Expected output for test_expected_output_units_m."""
        return pd.DataFrame(
            {
                "a": [
                    datetime.datetime(
                        1993,
                        9,
                        27,
                        11,
                        58,
                        58,
                        tzinfo=datetime.timezone.utc,
                    ),
                    datetime.datetime(
                        2000,
                        3,
                        19,
                        12,
                        59,
                        59,
                        tzinfo=datetime.timezone.utc,
                    ),
                    datetime.datetime(
                        2018,
                        11,
                        10,
                        11,
                        59,
                        59,
                        tzinfo=datetime.timezone.utc,
                    ),
                    datetime.datetime(
                        2018,
                        10,
                        10,
                        11,
                        59,
                        59,
                        tzinfo=datetime.timezone.utc,
                    ),
                    datetime.datetime(
                        2018,
                        10,
                        10,
                        11,
                        59,
                        59,
                        tzinfo=datetime.timezone.utc,
                    ),
                    datetime.datetime(
                        2018,
                        10,
                        10,
                        10,
                        59,
                        59,
                        tzinfo=datetime.timezone.utc,
                    ),
                    datetime.datetime(
                        2018,
                        12,
                        10,
                        11,
                        59,
                        59,
                        tzinfo=datetime.timezone.utc,
                    ),
                    datetime.datetime(
                        1985,
                        7,
                        23,
                        11,
                        59,
                        59,
                        tzinfo=datetime.timezone.utc,
                    ),
                ],
                "b": [
                    datetime.datetime(
                        2020,
                        5,
                        1,
                        12,
                        59,
                        59,
                        tzinfo=datetime.timezone.utc,
                    ),
                    datetime.datetime(
                        2019,
                        12,
                        25,
                        11,
                        58,
                        58,
                        tzinfo=datetime.timezone.utc,
                    ),
                    datetime.datetime(
                        2018,
                        11,
                        10,
                        11,
                        59,
                        59,
                        tzinfo=datetime.timezone.utc,
                    ),
                    datetime.datetime(
                        2018,
                        11,
                        10,
                        11,
                        59,
                        59,
                        tzinfo=datetime.timezone.utc,
                    ),
                    datetime.datetime(
                        2018,
                        9,
                        10,
                        9,
                        59,
                        59,
                        tzinfo=datetime.timezone.utc,
                    ),
                    datetime.datetime(
                        2015,
                        11,
                        10,
                        11,
                        59,
                        59,
                        tzinfo=datetime.timezone.utc,
                    ),
                    datetime.datetime(
                        2015,
                        11,
                        10,
                        12,
                        59,
                        59,
                        tzinfo=datetime.timezone.utc,
                    ),
                    datetime.datetime(
                        2015,
                        7,
                        23,
                        11,
                        59,
                        59,
                        tzinfo=datetime.timezone.utc,
                    ),
                ],
                "m": [
                    13986781.016666668,
                    10396738.983333332,
                    0.0,
                    44640.0,
                    -43320.0,
                    -1533540.0,
                    -1621380.0,
                    15778080.0,
                ],
            },
        )

    def expected_df_6():
        """Expected output for test_expected_output_units_s."""
        return pd.DataFrame(
            {
                "a": [
                    datetime.datetime(
                        1993,
                        9,
                        27,
                        11,
                        58,
                        58,
                        tzinfo=datetime.timezone.utc,
                    ),
                    datetime.datetime(
                        2000,
                        3,
                        19,
                        12,
                        59,
                        59,
                        tzinfo=datetime.timezone.utc,
                    ),
                    datetime.datetime(
                        2018,
                        11,
                        10,
                        11,
                        59,
                        59,
                        tzinfo=datetime.timezone.utc,
                    ),
                    datetime.datetime(
                        2018,
                        10,
                        10,
                        11,
                        59,
                        59,
                        tzinfo=datetime.timezone.utc,
                    ),
                    datetime.datetime(
                        2018,
                        10,
                        10,
                        11,
                        59,
                        59,
                        tzinfo=datetime.timezone.utc,
                    ),
                    datetime.datetime(
                        2018,
                        10,
                        10,
                        10,
                        59,
                        59,
                        tzinfo=datetime.timezone.utc,
                    ),
                    datetime.datetime(
                        2018,
                        12,
                        10,
                        11,
                        59,
                        59,
                        tzinfo=datetime.timezone.utc,
                    ),
                    datetime.datetime(
                        1985,
                        7,
                        23,
                        11,
                        59,
                        59,
                        tzinfo=datetime.timezone.utc,
                    ),
                ],
                "b": [
                    datetime.datetime(
                        2020,
                        5,
                        1,
                        12,
                        59,
                        59,
                        tzinfo=datetime.timezone.utc,
                    ),
                    datetime.datetime(
                        2019,
                        12,
                        25,
                        11,
                        58,
                        58,
                        tzinfo=datetime.timezone.utc,
                    ),
                    datetime.datetime(
                        2018,
                        11,
                        10,
                        11,
                        59,
                        59,
                        tzinfo=datetime.timezone.utc,
                    ),
                    datetime.datetime(
                        2018,
                        11,
                        10,
                        11,
                        59,
                        59,
                        tzinfo=datetime.timezone.utc,
                    ),
                    datetime.datetime(
                        2018,
                        9,
                        10,
                        9,
                        59,
                        59,
                        tzinfo=datetime.timezone.utc,
                    ),
                    datetime.datetime(
                        2015,
                        11,
                        10,
                        11,
                        59,
                        59,
                        tzinfo=datetime.timezone.utc,
                    ),
                    datetime.datetime(
                        2015,
                        11,
                        10,
                        12,
                        59,
                        59,
                        tzinfo=datetime.timezone.utc,
                    ),
                    datetime.datetime(
                        2015,
                        7,
                        23,
                        11,
                        59,
                        59,
                        tzinfo=datetime.timezone.utc,
                    ),
                ],
                "s": [
                    839206861.0,
                    623804339.0,
                    0.0,
                    2678400.0,
                    -2599200.0,
                    -92012400.0,
                    -97282800.0,
                    946684800.0,
                ],
            },
        )

    def expected_df_7():
        """Expected output for test_expected_output_nulls."""
        return pd.DataFrame(
            {
                "a": [
                    datetime.datetime(
                        1993,
                        9,
                        27,
                        11,
                        58,
                        58,
                        tzinfo=datetime.timezone.utc,
                    ),
                    np.nan,
                ],
                "b": [
                    np.nan,
                    datetime.datetime(
                        2019,
                        12,
                        25,
                        11,
                        58,
                        58,
                        tzinfo=datetime.timezone.utc,
                    ),
                ],
                "D": [
                    np.nan,
                    np.nan,
                ],
            },
            index=[0, 1],
        )

    @pytest.mark.parametrize(
        ("columns, bad_col"),
        [
            (["date_col", "numeric_col"], 1),
            (["date_col", "string_col"], 1),
            (["date_col", "bool_col"], 1),
            (["date_col", "empty_col"], 1),
            (["numeric_col", "date_col"], 0),
            (["string_col", "date_col"], 0),
            (["bool_col", "date_col"], 0),
            (["empty_col", "date_col"], 0),
        ],
    )
    def test_input_data_check_column_errors(self, columns, bad_col):
        """Check that errors are raised on a variety of different non date datatypes"""
        x = DateDifferenceTransformer(
            column_lower=columns[0],
            column_upper=columns[1],
            new_column_name="c",
        )
        df = d.create_date_diff_incorrect_dtypes()

        msg = f"{x.classname()}: {columns[bad_col]} should be datetime64 or date type but got {df[columns[bad_col]].dtype}"

        with pytest.raises(TypeError, match=msg):
            x.transform(df)

    def test_super_transform_called(self, mocker):
        """Test that BaseTransformer.transform called."""
        df = d.create_datediff_test_df()

        x = DateDifferenceTransformer(
            column_lower="a",
            column_upper="b",
            new_column_name="D",
            units="D",
            verbose=False,
        )

        expected_call_args = {0: {"args": (d.create_datediff_test_df(),), "kwargs": {}}}

        with ta.functions.assert_function_call(
            mocker,
            tubular.base.BaseTransformer,
            "transform",
            expected_call_args,
            return_value=d.create_datediff_test_df(),
        ):
            x.transform(df)

    @pytest.mark.parametrize(
        ("columns, datetime_col, date_col"),
        [
            (["date_col_1", "datetime_col_2"], 1, 0),
            (["datetime_col_1", "date_col_2"], 0, 1),
        ],
    )
    def test_cast_to_date_warning(self, columns, datetime_col, date_col):
        "Test that transform raises a warning if one column is a date and one is datetime"

        x = DateDifferenceTransformer(
            column_lower=columns[0],
            column_upper=columns[1],
            new_column_name="c",
        )

        msg = f"""
                {x.classname()}: temporarily cast {columns[datetime_col]} from datetime64 to date before transforming in order to match {columns[date_col]}.

                Some precision may be lost from {columns[datetime_col]}. Original column not changed.
                """

        with pytest.warns(UserWarning, match=msg):
            x.transform(d.create_date_diff_different_dtypes())

    @pytest.mark.parametrize(
        ("df", "expected"),
        ta.pandas.adjusted_dataframe_params(
            d.create_datediff_test_df(),
            expected_df_3(),
        ),
    )
    def test_expected_output_units_D(self, df, expected):
        """Test that the output is expected from transform, when units is D.

        This tests positive month gaps, negative month gaps, and missing values.

        """
        x = DateDifferenceTransformer(
            column_lower="a",
            column_upper="b",
            new_column_name="D",
            units="D",
            verbose=False,
        )

        df_transformed = x.transform(df)

        ta.equality.assert_equal_dispatch(
            expected=expected,
            actual=df_transformed,
            msg="Unexpected values in DateDifferenceYearTransformer.transform",
        )

    @pytest.mark.parametrize(
        ("df", "expected"),
        ta.pandas.adjusted_dataframe_params(
            d.create_datediff_test_df(),
            expected_df_4(),
        ),
    )
    def test_expected_output_units_h(self, df, expected):
        """Test that the output is expected from transform, when units is h.

        This tests positive month gaps, negative month gaps, and missing values.

        """
        x = DateDifferenceTransformer(
            column_lower="a",
            column_upper="b",
            new_column_name="h",
            units="h",
            verbose=False,
        )

        df_transformed = x.transform(df)

        ta.equality.assert_equal_dispatch(
            expected=expected,
            actual=df_transformed,
            msg="Unexpected values in DateDifferenceYearTransformer.transform",
        )

    @pytest.mark.parametrize(
        ("df", "expected"),
        ta.pandas.adjusted_dataframe_params(
            d.create_datediff_test_df(),
            expected_df_5(),
        ),
    )
    def test_expected_output_units_m(self, df, expected):
        """Test that the output is expected from transform, when units is m.

        This tests positive month gaps, negative month gaps, and missing values.

        """
        x = DateDifferenceTransformer(
            column_lower="a",
            column_upper="b",
            new_column_name="m",
            units="m",
            verbose=False,
        )

        df_transformed = x.transform(df)

        ta.equality.assert_equal_dispatch(
            expected=expected,
            actual=df_transformed,
            msg="Unexpected values in DateDifferenceYearTransformer.transform",
        )

    @pytest.mark.parametrize(
        ("df", "expected"),
        ta.pandas.adjusted_dataframe_params(
            d.create_datediff_test_df(),
            expected_df_6(),
        ),
    )
    def test_expected_output_units_s(self, df, expected):
        """Test that the output is expected from transform, when units is s.

        This tests positive month gaps, negative month gaps, and missing values.

        """
        x = DateDifferenceTransformer(
            column_lower="a",
            column_upper="b",
            new_column_name="s",
            units="s",
            verbose=False,
        )

        df_transformed = x.transform(df)

        ta.equality.assert_equal_dispatch(
            expected=expected,
            actual=df_transformed,
            msg="Unexpected values in DateDifferenceYearTransformer.transform",
        )

    @pytest.mark.parametrize(
        ("df", "expected"),
        ta.pandas.adjusted_dataframe_params(
            d.create_datediff_test_nulls_df(),
            expected_df_7(),
        ),
    )
    def test_expected_output_nulls(self, df, expected):
        """Test that the output is expected from transform, when columns are nulls."""
        x = DateDifferenceTransformer(
            column_lower="a",
            column_upper="b",
            new_column_name="D",
            units="D",
            verbose=False,
        )

        df_transformed = x.transform(df)

        ta.equality.assert_frame_equal_msg(
            actual=df_transformed,
            expected=expected,
            msg_tag="Unexpected values in DateDifferenceTransformer.transform (nulls)",
        )

    @pytest.mark.parametrize(
        ("columns, output"),
        [
            (["date_col_1", "date_col_2"], "dates output"),
            (["date_col_1", "datetime_col_2"], "dates output"),
            (["datetime_col_1", "date_col_2"], "dates output"),
            (["datetime_col_1", "datetime_col_2"], "datetime output"),
        ],
    )
    def test_expcected_output_different_date_types(self, columns, output):
        "Test that transform works for different date datatype combinations"

        x = DateDifferenceTransformer(
            column_lower=columns[0],
            column_upper=columns[1],
            new_column_name="D",
            units="D",
            verbose=False,
        )

        output_col = d.expected_date_diff_df_3()[output]
        df = d.create_date_diff_different_dtypes_2()
        expected = df.copy()
        expected["D"] = output_col

        df_transformed = x.transform(df)

        ta.equality.assert_frame_equal_msg(
            actual=df_transformed,
            expected=expected,
            msg_tag=f"Unexpected values in DateDifferenceTransformer.transform between {columns[0]} and {columns[1]}",
        )
