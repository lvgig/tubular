import datetime

import numpy as np
import pandas
import pandas as pd
import pytest
import test_aide as ta

import tests.test_data as d
import tubular
from tubular.dates import ToDatetimeTransformer


class TestInit:
    """Tests for ToDatetimeTransformer.init()."""

    def test_arguments(self):
        """Test that init has expected arguments."""
        ta.functions.test_function_arguments(
            func=ToDatetimeTransformer.__init__,
            expected_arguments=[
                "self",
                "column",
                "new_column_name",
                "to_datetime_kwargs",
            ],
            expected_default_values=({},),
        )

    def test_class_methods(self):
        """Test that ToDatetimeTransformer has fit and transform methods."""
        to_dt = ToDatetimeTransformer(column="a", new_column_name="b")

        ta.classes.test_object_method(
            obj=to_dt,
            expected_method="transform",
            msg="transform",
        )

    def test_inheritance(self):
        """Test that ToDatetimeTransformer inherits from BaseTransformer."""
        to_dt = ToDatetimeTransformer(column="a", new_column_name="b")

        ta.classes.assert_inheritance(to_dt, tubular.base.BaseTransformer)

    def test_super_init_called(self, mocker):
        """Test that init calls BaseTransformer.init."""
        expected_call_args = {
            0: {
                "args": (),
                "kwargs": {
                    "columns": ["a"],
                    "copy": True,
                    "verbose": False,
                },
            },
        }

        with ta.functions.assert_function_call(
            mocker,
            tubular.base.BaseTransformer,
            "__init__",
            expected_call_args,
        ):
            ToDatetimeTransformer(
                column="a",
                new_column_name="b",
                verbose=False,
                copy=True,
            )

    def test_column_type_error(self):
        """Test that an exception is raised if column is not a str."""
        with pytest.raises(
            TypeError,
            match="ToDatetimeTransformer: column should be a single str giving the column to transform to datetime",
        ):
            ToDatetimeTransformer(
                column=["a"],
                new_column_name="a",
            )

    def test_new_column_name_type_error(self):
        """Test that an exception is raised if new_column_name is not a str."""
        with pytest.raises(
            TypeError,
            match="ToDatetimeTransformer: new_column_name must be a str",
        ):
            ToDatetimeTransformer(column="b", new_column_name=1)

    def test_to_datetime_kwargs_type_error(self):
        """Test that an exception is raised if to_datetime_kwargs is not a dict."""
        with pytest.raises(
            TypeError,
            match=r"""ToDatetimeTransformer: to_datetime_kwargs should be a dict but got type \<class 'int'\>""",
        ):
            ToDatetimeTransformer(column="b", new_column_name="a", to_datetime_kwargs=1)

    def test_to_datetime_kwargs_key_type_error(self):
        """Test that an exception is raised if to_datetime_kwargs has keys which are not str."""
        with pytest.raises(
            TypeError,
            match=r"""ToDatetimeTransformer: unexpected type \(\<class 'int'\>\) for to_datetime_kwargs key in position 1, must be str""",
        ):
            ToDatetimeTransformer(
                new_column_name="a",
                column="b",
                to_datetime_kwargs={"a": 1, 2: "b"},
            )

    def test_inputs_set_to_attribute(self):
        """Test that the values passed in init are set to attributes."""
        to_dt = ToDatetimeTransformer(
            column="b",
            new_column_name="a",
            to_datetime_kwargs={"a": 1, "b": 2},
        )

        ta.classes.test_object_attributes(
            obj=to_dt,
            expected_attributes={
                "column": "b",
                "columns": ["b"],
                "new_column_name": "a",
                "to_datetime_kwargs": {"a": 1, "b": 2},
            },
            msg="Attributes for ToDatetimeTransformer set in init",
        )


class TestTransform:
    """Tests for ToDatetimeTransformer.transform()."""

    def expected_df_1():
        """Expected output for test_expected_output."""
        return pd.DataFrame(
            {
                "a": [1950, 1960, 2000, 2001, np.NaN, 2010],
                "b": [1, 2, 3, 4, 5, np.NaN],
                "a_Y": [
                    datetime.datetime(1950, 1, 1, tzinfo=datetime.timezone.utc),
                    datetime.datetime(1960, 1, 1, tzinfo=datetime.timezone.utc),
                    datetime.datetime(2000, 1, 1, tzinfo=datetime.timezone.utc),
                    datetime.datetime(2001, 1, 1, tzinfo=datetime.timezone.utc),
                    pd.NaT,
                    datetime.datetime(2010, 1, 1, tzinfo=datetime.timezone.utc),
                ],
                "b_m": [
                    datetime.datetime(1900, 1, 1, tzinfo=datetime.timezone.utc),
                    datetime.datetime(1900, 2, 1, tzinfo=datetime.timezone.utc),
                    datetime.datetime(1900, 3, 1, tzinfo=datetime.timezone.utc),
                    datetime.datetime(1900, 4, 1, tzinfo=datetime.timezone.utc),
                    datetime.datetime(1900, 5, 1, tzinfo=datetime.timezone.utc),
                    pd.NaT,
                ],
            },
        )

    def test_arguments(self):
        """Test that transform has expected arguments."""
        ta.functions.test_function_arguments(
            func=ToDatetimeTransformer.transform,
            expected_arguments=["self", "X"],
        )

    def test_super_transform_call(self, mocker):
        """Test the call to BaseTransformer.transform is as expected."""
        df = d.create_datediff_test_df()

        to_dt = ToDatetimeTransformer(column="a", new_column_name="Y")

        expected_call_args = {0: {"args": (d.create_datediff_test_df(),), "kwargs": {}}}

        with ta.functions.assert_function_call(
            mocker,
            tubular.base.BaseTransformer,
            "transform",
            expected_call_args,
            return_value=d.create_datediff_test_df(),
        ):
            to_dt.transform(df)

    def test_to_datetime_call(self, mocker):
        """Test the call to pandas.to_datetime is as expected."""
        df = d.create_to_datetime_test_df()

        to_dt = ToDatetimeTransformer(
            column="a",
            new_column_name="a_Y",
            to_datetime_kwargs={"format": "%Y"},
        )

        expected_call_args = {
            0: {
                "args": (d.create_to_datetime_test_df()["a"],),
                "kwargs": {"format": "%Y"},
            },
        }

        with ta.functions.assert_function_call(
            mocker,
            pandas,
            "to_datetime",
            expected_call_args,
            return_value=pd.to_datetime(d.create_to_datetime_test_df()["a"]),
        ):
            to_dt.transform(df)

    def test_output_from_to_datetime_assigned_to_column(self, mocker):
        """Test that the output from pd.to_datetime is assigned to column with name new_column_name."""
        df = d.create_to_datetime_test_df()

        to_dt = ToDatetimeTransformer(
            column="a",
            new_column_name="a_new",
            to_datetime_kwargs={"format": "%Y"},
        )

        to_datetime_output = [1, 2, 3, 4, 5, 6]

        mocker.patch("pandas.to_datetime", return_value=to_datetime_output)

        df_transformed = to_dt.transform(df)

        assert (
            df_transformed["a_new"].tolist() == to_datetime_output
        ), "unexpected values assigned to a_new column"

    @pytest.mark.parametrize(
        ("df", "expected"),
        ta.pandas.adjusted_dataframe_params(
            d.create_to_datetime_test_df(),
            expected_df_1(),
        ),
    )
    def test_expected_output(self, df, expected):
        """Test input data is transformed as expected."""
        to_dt_1 = ToDatetimeTransformer(
            column="a",
            new_column_name="a_Y",
            to_datetime_kwargs={"format": "%Y", "utc": datetime.timezone.utc},
        )

        to_dt_2 = ToDatetimeTransformer(
            column="b",
            new_column_name="b_m",
            to_datetime_kwargs={"format": "%m", "utc": datetime.timezone.utc},
        )

        df_transformed = to_dt_1.transform(df)
        df_transformed = to_dt_2.transform(df_transformed)

        print(df_transformed)
        print(expected)

        ta.equality.assert_equal_dispatch(
            expected=expected,
            actual=df_transformed,
            msg="ToDatetimeTransformer.transform output",
        )
