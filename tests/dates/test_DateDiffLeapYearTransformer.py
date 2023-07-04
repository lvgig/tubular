import datetime

import numpy as np
import pandas as pd
import pytest
import test_aide as ta

import tests.test_data as d
import tubular
from tubular.dates import DateDiffLeapYearTransformer


class TestCalculateAge:
    """Tests for the calculate_age function in dates.py."""

    def test_row_type_error(self):
        """Test that an exception is raised if row is not a pd.Series."""
        row = "dummy_row"
        date_transformer = DateDiffLeapYearTransformer(
            column_lower="a",
            column_upper="b",
            new_column_name="c",
            drop_cols=True,
        )

        with pytest.raises(
            TypeError,
            match="DateDiffLeapYearTransformer: row should be a pd.Series",
        ):
            date_transformer.calculate_age(row=row)

    def test_null_replacement(self):
        """Test correct value is replaced using null_replacement."""
        row = pd.Series({"a": np.NaN, "b": np.NaN})
        date_transformer = DateDiffLeapYearTransformer(
            column_lower="a",
            column_upper="b",
            new_column_name="c",
            drop_cols=True,
            missing_replacement="missing_replacement",
        )

        val = date_transformer.calculate_age(row=row)

        assert val == "missing_replacement"

    def test_upper_column_type_error(self):
        """Test that an exception is raised if uppder date value is not a datetime object."""
        row = pd.Series({"a": datetime.date(2020, 5, 10), "b": "dummy_val"})
        date_transformer = DateDiffLeapYearTransformer(
            column_lower="a",
            column_upper="b",
            new_column_name="c",
            drop_cols=True,
        )

        with pytest.raises(
            TypeError,
            match="DateDiffLeapYearTransformer: upper column values should be datetime.datetime or datetime.date objects",
        ):
            date_transformer.calculate_age(row=row)

    def test_lower_column_type_error(self):
        """Test that an exception is raised if lower date value is not a datetime object."""
        row = pd.Series({"a": "dummy_val", "b": datetime.date(2020, 5, 10)})
        date_transformer = DateDiffLeapYearTransformer(
            column_lower="a",
            column_upper="b",
            new_column_name="c",
            drop_cols=True,
        )

        with pytest.raises(
            TypeError,
            match="DateDiffLeapYearTransformer: lower column values should be datetime.datetime or datetime.date objects",
        ):
            date_transformer.calculate_age(row=row)


class TestInit:
    """Tests for DateDiffLeapYearTransformer.init()."""

    def test_arguments(self):
        """Test that init has expected arguments."""
        ta.functions.test_function_arguments(
            func=DateDiffLeapYearTransformer.__init__,
            expected_arguments=[
                "self",
                "column_lower",
                "column_upper",
                "new_column_name",
                "drop_cols",
                "missing_replacement",
            ],
            expected_default_values=(None,),
        )

    def test_class_methods(self):
        """Test that DateDiffLeapYearTransformer has transform method."""
        x = DateDiffLeapYearTransformer(
            column_lower="dummy_1",
            column_upper="dummy_2",
            new_column_name="dummy_3",
            drop_cols=True,
        )

        ta.classes.test_object_method(
            obj=x,
            expected_method="transform",
            msg="transform",
        )
        ta.classes.test_object_method(
            obj=x,
            expected_method="calculate_age",
            msg="calculate_message",
        )

    def test_inheritance(self):
        """Test that DateDiffLeapYearTransformer inherits from BaseTransformer."""
        x = DateDiffLeapYearTransformer(
            column_lower="dummy_1",
            column_upper="dummy_2",
            new_column_name="dummy_3",
            drop_cols=True,
        )

        ta.classes.assert_inheritance(x, tubular.base.BaseTransformer)

    def test_super_init_called(self, mocker):
        """Test that init calls BaseTransformer.init."""
        expected_call_args = {
            0: {
                "args": (),
                "kwargs": {
                    "columns": ["dummy_1", "dummy_2"],
                    "verbose": True,
                    "copy": True,
                },
            },
        }

        with ta.functions.assert_function_call(
            mocker,
            tubular.base.BaseTransformer,
            "__init__",
            expected_call_args,
        ):
            DateDiffLeapYearTransformer(
                column_lower="dummy_1",
                column_upper="dummy_2",
                new_column_name="dummy_3",
                drop_cols=True,
                verbose=True,
                copy=True,
            )

    def test_column_lower_type_error(self):
        """Test that an exception is raised if column_lower is not a str."""
        with pytest.raises(
            TypeError,
            match="DateDiffLeapYearTransformer: column_lower should be a str",
        ):
            DateDiffLeapYearTransformer(
                column_lower=123,
                column_upper="dummy_2",
                new_column_name="dummy_3",
                drop_cols=True,
            )

    def test_column_upper_type_error(self):
        """Test that an exception is raised if column_upper is not a str."""
        with pytest.raises(
            TypeError,
            match="DateDiffLeapYearTransformer: column_upper should be a str",
        ):
            DateDiffLeapYearTransformer(
                column_lower="dummy_1",
                column_upper=123,
                new_column_name="dummy_3",
                drop_cols=True,
            )

    def test_new_column_name_type_error(self):
        """Test that an exception is raised if new_column_name is not a str."""
        with pytest.raises(
            TypeError,
            match="DateDiffLeapYearTransformer: new_column_name should be a str",
        ):
            DateDiffLeapYearTransformer(
                column_lower="dummy_1",
                column_upper="dummy_2",
                new_column_name=123,
                drop_cols=True,
            )

    def test_drop_cols_type_error(self):
        """Test that an exception is raised if drop_cols is not a bool."""
        with pytest.raises(
            TypeError,
            match="DateDiffLeapYearTransformer: drop_cols should be a bool",
        ):
            DateDiffLeapYearTransformer(
                column_lower="dummy_1",
                column_upper="dummy_2",
                new_column_name="dummy_3",
                drop_cols=123,
            )

    def test_missing_replacement_type_error(self):
        """Test that an exception is raised if missing_replacement is not the correct type."""
        with pytest.raises(
            TypeError,
            match="DateDiffLeapYearTransformer: if not None, missing_replacement should be an int, float or string",
        ):
            DateDiffLeapYearTransformer(
                column_lower="dummy_1",
                column_upper="dummy_2",
                new_column_name="dummy_3",
                drop_cols=False,
                missing_replacement=[1, 2, 3],
            )

    def test_inputs_set_to_attribute(self):
        """Test that the value passed for new_column_name and drop_cols are saved in attributes of the same name."""
        value_1 = "test_name"
        value_2 = True

        x = DateDiffLeapYearTransformer(
            column_lower="dummy_1",
            column_upper="dummy_2",
            new_column_name=value_1,
            drop_cols=value_2,
            missing_replacement="dummy_3",
        )

        ta.classes.test_object_attributes(
            obj=x,
            expected_attributes={
                "column_lower": "dummy_1",
                "column_upper": "dummy_2",
                "new_column_name": value_1,
                "drop_cols": value_2,
                "missing_replacement": "dummy_3",
            },
            msg="Attributes for DateDiffLeapYearTransformer set in init",
        )


class TestTransform:
    """Tests for DateDiffLeapYearTransformer.transform()."""

    def expected_df_1():
        """Expected output for test_expected_output_drop_cols_true."""
        df = pd.DataFrame(
            {
                "c": [
                    26,
                    19,
                    0,
                    0,
                    0,
                    -2,
                    -3,
                    30,
                ],
            },
        )

        return df

    def expected_df_2():
        """Expected output for test_expected_output_drop_cols_false."""
        df = pd.DataFrame(
            {
                "a": [
                    datetime.date(1993, 9, 27),  # day/month greater than
                    datetime.date(2000, 3, 19),  # day/month less than
                    datetime.date(2018, 11, 10),  # same day
                    datetime.date(2018, 10, 10),  # same year day/month greater than
                    datetime.date(2018, 10, 10),  # same year day/month less than
                    datetime.date(2018, 10, 10),  # negative day/month less than
                    datetime.date(2018, 12, 10),  # negative day/month greater than
                    datetime.date(
                        1985,
                        7,
                        23,
                    ),  # large gap, this is incorrect with timedelta64 solutions
                ],
                "b": [
                    datetime.date(2020, 5, 1),
                    datetime.date(2019, 12, 25),
                    datetime.date(2018, 11, 10),
                    datetime.date(2018, 11, 10),
                    datetime.date(2018, 9, 10),
                    datetime.date(2015, 11, 10),
                    datetime.date(2015, 11, 10),
                    datetime.date(2015, 7, 23),
                ],
                "c": [
                    26,
                    19,
                    0,
                    0,
                    0,
                    -2,
                    -3,
                    30,
                ],
            },
        )

        return df

    def expected_df_3():
        """Expected output for test_expected_output_nulls."""
        df = pd.DataFrame(
            {
                "a": [
                    np.NaN,
                ],
                "b": [
                    np.NaN,
                ],
                "c": [None],
            },
        )

        return df

    def test_arguments(self):
        """Test that transform has expected arguments."""
        ta.functions.test_function_arguments(
            func=DateDiffLeapYearTransformer.transform,
            expected_arguments=["self", "X"],
        )

    def test_super_transform_called(self, mocker):
        """Test that BaseTransformer.transform called."""
        df = d.create_date_test_df()

        x = DateDiffLeapYearTransformer(
            column_lower="a",
            column_upper="b",
            new_column_name="c",
            drop_cols=True,
        )

        expected_call_args = {0: {"args": (d.create_date_test_df(),), "kwargs": {}}}

        with ta.functions.assert_function_call(
            mocker,
            tubular.base.BaseTransformer,
            "transform",
            expected_call_args,
            return_value=d.create_date_test_df(),
        ):
            x.transform(df)

    @pytest.mark.parametrize(
        ("df", "expected"),
        ta.pandas.adjusted_dataframe_params(d.create_date_test_df(), expected_df_1()),
    )
    def test_expected_output_drop_cols_true(self, df, expected):
        """Test that the output is expected from transform, when drop_cols is True.

        This tests positive year gaps, negative year gaps, and missing values.

        """
        x = DateDiffLeapYearTransformer(
            column_lower="a",
            column_upper="b",
            new_column_name="c",
            drop_cols=True,
        )

        df_transformed = x.transform(df)

        ta.equality.assert_frame_equal_msg(
            actual=df_transformed,
            expected=expected,
            msg_tag="Unexpected values in DateDiffLeapYearTransformer.transform (with drop_cols)",
        )

    @pytest.mark.parametrize(
        ("df", "expected"),
        ta.pandas.adjusted_dataframe_params(d.create_date_test_df(), expected_df_2()),
    )
    def test_expected_output_drop_cols_false(self, df, expected):
        """Test that the output is expected from transform, when drop_cols is False.

        This tests positive year gaps , negative year gaps, and missing values.

        """
        x = DateDiffLeapYearTransformer(
            column_lower="a",
            column_upper="b",
            new_column_name="c",
            drop_cols=False,
        )

        df_transformed = x.transform(df)

        ta.equality.assert_frame_equal_msg(
            actual=df_transformed,
            expected=expected,
            msg_tag="Unexpected values in DateDiffLeapYearTransformer.transform (without drop_cols)",
        )

    @pytest.mark.parametrize(
        ("df", "expected"),
        ta.pandas.adjusted_dataframe_params(
            d.create_date_test_nulls_df(),
            expected_df_3(),
        ),
    )
    def test_expected_output_nulls(self, df, expected):
        """Test that the output is expected from transform, when columns are nulls."""
        x = DateDiffLeapYearTransformer(
            column_lower="a",
            column_upper="b",
            new_column_name="c",
            drop_cols=False,
        )

        df_transformed = x.transform(df)

        ta.equality.assert_frame_equal_msg(
            actual=df_transformed,
            expected=expected,
            msg_tag="Unexpected values in DateDiffLeapYearTransformer.transform (nulls)",
        )
