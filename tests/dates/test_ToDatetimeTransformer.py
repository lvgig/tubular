import datetime

import numpy as np
import pandas as pd
import pytest
import test_aide as ta

from tubular.dates import ToDatetimeTransformer


class TestInit:
    """Tests for ToDatetimeTransformer.init()."""

    def test_column_type_error(self):
        """Test that an exception is raised if column is not a str."""
        with pytest.raises(
            TypeError,
            match=r"ToDatetimeTransformer: each element of columns should be a single \(string\) column name",
        ):
            ToDatetimeTransformer(
                column=["a"],
                new_column_name="a",
            )

    @pytest.mark.xfail
    def test_new_column_name_type_error(self):
        """Test that an exception is raised if new_column_name is not a str."""
        with pytest.raises(
            TypeError,
            match="ToDatetimeTransformer: new_column_name should be str",
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
                "a": [1950, 1960, 2000, 2001, np.nan, 2010],
                "b": [1, 2, 3, 4, 5, np.nan],
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

    def create_to_datetime_test_df():
        """Create DataFrame to be used in the ToDatetimeTransformer tests."""
        return pd.DataFrame(
            {"a": [1950, 1960, 2000, 2001, np.nan, 2010], "b": [1, 2, 3, 4, 5, np.nan]},
        )

    @pytest.mark.parametrize(
        ("df", "expected"),
        ta.pandas.adjusted_dataframe_params(
            create_to_datetime_test_df(),
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
