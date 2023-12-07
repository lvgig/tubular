import re

import numpy as np
import pandas as pd
import pytest
import test_aide as ta

import tests.test_data as d
import tubular
from tubular.dates import DatetimeInfoExtractor


@pytest.fixture()
def timeofday_extractor():
    return DatetimeInfoExtractor(columns=["a"], include=["timeofday"])


@pytest.fixture()
def timeofmonth_extractor():
    return DatetimeInfoExtractor(columns=["a"], include=["timeofmonth"])


@pytest.fixture()
def timeofyear_extractor():
    return DatetimeInfoExtractor(columns=["a"], include=["timeofyear"])


@pytest.fixture()
def dayofweek_extractor():
    return DatetimeInfoExtractor(columns=["a"], include=["dayofweek"])


class TestExtractDatetimeInfoInit:
    def test_assert_inheritance(self):
        """Test that ExtractDatetimeInfo inherits from BaseTransformer."""
        x = DatetimeInfoExtractor(columns=["a"])

        ta.classes.assert_inheritance(x, tubular.base.BaseTransformer)

    def test_super_init_called(self, mocker):
        """Test that init calls BaseTransformer.init."""
        expected_call_args = {
            0: {
                "args": (),
                "kwargs": {"columns": ["a"]},
            },
        }

        with ta.functions.assert_function_call(
            mocker,
            tubular.base.BaseTransformer,
            "__init__",
            expected_call_args,
        ):
            DatetimeInfoExtractor(columns=["a"])

    def test_values_passed_in_init_set_to_attribute(self):
        """Test that the values passed in init are saved in an attribute of the same name."""
        x = DatetimeInfoExtractor(
            columns=["a"],
            include=["timeofmonth", "timeofday"],
            datetime_mappings={"timeofday": {"am": range(0, 12), "pm": range(12, 24)}},
        )

        ta.classes.test_object_attributes(
            obj=x,
            expected_attributes={
                "columns": ["a"],
                "include": ["timeofmonth", "timeofday"],
                "datetime_mappings": {
                    "timeofday": {"am": range(0, 12), "pm": range(12, 24)},
                },
            },
            msg="Attributes for ExtractDatetimeInfo set in init",
        )

    @pytest.mark.parametrize("incorrect_type_include", [2, 3.0, "invalid", "dayofweek"])
    def test_error_when_include_not_list(self, incorrect_type_include):
        """Test that an exception is raised when value include variable is not a list."""
        with pytest.raises(
            TypeError,
            match="include should be List",
        ):
            DatetimeInfoExtractor(columns=["a"], include=incorrect_type_include)

    def test_error_when_invalid_include_option(self):
        """Test that an exception is raised when include contains incorrect values."""
        with pytest.raises(
            ValueError,
            match=r'elements in include should be in \["timeofday", "timeofmonth", "timeofyear", "dayofweek"\]',
        ):
            DatetimeInfoExtractor(
                columns=["a"],
                include=["timeofday", "timeofmonth", "invalid_option"],
            )

    @pytest.mark.parametrize(
        "incorrect_type_datetime_mappings",
        [2, 3.0, ["a", "b"], "dayofweek"],
    )
    def test_error_when_datetime_mappings_not_dict(
        self,
        incorrect_type_datetime_mappings,
    ):
        """Test that an exception is raised when datetime_mappings is not a dict."""
        with pytest.raises(
            TypeError,
            match="datetime_mappings should be Dict",
        ):
            DatetimeInfoExtractor(
                columns=["a"],
                datetime_mappings=incorrect_type_datetime_mappings,
            )

    @pytest.mark.parametrize(
        "incorrect_type_datetime_mappings_values",
        [{"timeofday": 2}],
    )
    def test_error_when_datetime_mapping_value_not_dict(
        self,
        incorrect_type_datetime_mappings_values,
    ):
        """Test that an exception is raised when values in datetime_mappings are not dict."""
        with pytest.raises(
            TypeError,
            match="values in datetime_mappings should be dict",
        ):
            DatetimeInfoExtractor(
                columns=["a"],
                datetime_mappings=incorrect_type_datetime_mappings_values,
            )

    @pytest.mark.parametrize(
        ("include", "incorrect_datetime_mappings_keys"),
        [
            (["timeofyear"], {"invalid_key": {"valid_mapping": "valid_output"}}),
            (["timeofyear"], {"dayofweek": {"day": range(7)}}),
            (
                ["timeofyear"],
                {"timeofyear": {"month": range(12)}, "timeofday": {"hour": range(24)}},
            ),
        ],
    )
    def test_error_when_datetime_mapping_key_not_in_include(
        self,
        include,
        incorrect_datetime_mappings_keys,
    ):
        """Test that an exception is raised when keys in datetime_mappings are not in include."""
        with pytest.raises(
            ValueError,
            match="keys in datetime_mappings should be in include",
        ):
            DatetimeInfoExtractor(
                columns=["a"],
                include=include,
                datetime_mappings=incorrect_datetime_mappings_keys,
            )

    @pytest.mark.parametrize(
        ("incomplete_mappings", "expected_exception"),
        [
            (
                {"timeofday": {"mapped": range(23)}},
                re.escape(
                    "timeofday mapping dictionary should contain mapping for all hours between 0-23. {23} are missing",
                ),
            ),
            (
                {"timeofmonth": {"mapped": range(1, 31)}},
                re.escape(
                    "timeofmonth mapping dictionary should contain mapping for all days between 1-31. {31} are missing",
                ),
            ),
            (
                {"timeofyear": {"mapped": range(1, 12)}},
                re.escape(
                    "timeofyear mapping dictionary should contain mapping for all months between 1-12. {12} are missing",
                ),
            ),
            (
                {"dayofweek": {"mapped": range(6)}},
                re.escape(
                    "dayofweek mapping dictionary should contain mapping for all days between 0-6. {6} are missing",
                ),
            ),
        ],
    )
    def test_error_when_incomplete_mappings_passed(
        self,
        incomplete_mappings,
        expected_exception,
    ):
        """Test that error is raised when incomplete mappings are passed."""
        with pytest.raises(ValueError, match=expected_exception):
            DatetimeInfoExtractor(columns=["a"], datetime_mappings=incomplete_mappings)


class TestMapValues:
    @pytest.mark.parametrize("incorrect_type_input", ["2", [1, 2]])
    def test_incorrect_type_input(self, incorrect_type_input, timeofday_extractor):
        """Test that an error is raised if input is the wrong type."""
        with pytest.raises(
            TypeError,
            match="DatetimeInfoExtractor: value should be float or int",
        ):
            timeofday_extractor._map_values(incorrect_type_input, "timeofday")

    @pytest.mark.parametrize("incorrect_size_input", [-2, 30, 5.6, 11.2])
    def test_out_of_bounds_or_fractional_input(
        self,
        incorrect_size_input,
        timeofday_extractor,
    ):
        """Test that an error is raised when value is outside of 0-23 range."""
        with pytest.raises(
            ValueError,
            match="DatetimeInfoExtractor: value for timeofday mapping  in self._map_values should be an integer value in 0-23",
        ):
            timeofday_extractor._map_values(incorrect_size_input, "timeofday")

    @pytest.mark.parametrize(
        ("valid_hour", "hour_time_of_day"),
        [
            (0, "night"),
            (5, "night"),
            (6, "morning"),
            (11, "morning"),
            (12, "afternoon"),
            (17, "afternoon"),
            (18, "evening"),
            (23, "evening"),
        ],
    )
    def test_valid_inputs_timeofday(
        self,
        valid_hour,
        hour_time_of_day,
        timeofday_extractor,
    ):
        """Trial test to check all in one go."""
        output = timeofday_extractor._map_values(valid_hour, "timeofday")

        assert output == hour_time_of_day, "expected {}, output {}".format(
            hour_time_of_day,
            output,
        )

    @pytest.mark.parametrize(
        ("valid_day", "day_time_of_month"),
        [
            (1, "start"),
            (6, "start"),
            (10, "start"),
            (11, "middle"),
            (16, "middle"),
            (20, "middle"),
            (21, "end"),
            (21, "end"),
            (31, "end"),
        ],
    )
    def test_valid_inputs_timeofmonth(
        self,
        valid_day,
        day_time_of_month,
        timeofmonth_extractor,
    ):
        """Test that correct values are return with valid inputs."""
        output = timeofmonth_extractor._map_values(valid_day, "timeofmonth")
        assert output == day_time_of_month, "expected {}, output {}".format(
            day_time_of_month,
            output,
        )

    @pytest.mark.parametrize(
        ("valid_month", "month_time_of_year"),
        [
            (1, "winter"),
            (3, "spring"),
            (4, "spring"),
            (6, "summer"),
            (7, "summer"),
            (9, "autumn"),
            (10, "autumn"),
            (12, "winter"),
        ],
    )
    def test_valid_inputs_timeofyear(
        self,
        valid_month,
        month_time_of_year,
        timeofyear_extractor,
    ):
        """Test that correct values are return with valid inputs."""
        output = timeofyear_extractor._map_values(valid_month, "timeofyear")
        assert output == month_time_of_year, "expected {}, output {}".format(
            month_time_of_year,
            output,
        )

    @pytest.mark.parametrize(
        ("valid_day", "dayofweek"),
        [
            (0, "monday"),
            (2, "wednesday"),
            (4, "friday"),
            (6, "sunday"),
        ],
    )
    def test_valid_inputs_dayofweek(self, valid_day, dayofweek, dayofweek_extractor):
        """Test that correct values are return with valid inputs."""
        output = dayofweek_extractor._map_values(valid_day, "dayofweek")
        assert output == dayofweek, f"expected {dayofweek}, output {output}"

    def test_valid_nan_output(self, timeofday_extractor):
        """Test that correct values are return with valid inputs."""
        output = timeofday_extractor._map_values(np.nan, "timeofday")
        print(output)
        assert np.isnan(
            output,
        ), f"passing np.nan should result in np.nan, instead received {output}"


class TestTransform:
    def test_super_transform_called(self, mocker):
        """Test that init calls BaseTransformer.init."""
        df = d.create_date_test_df()
        df = df.astype("datetime64[ns]")

        expected_call_args = {
            0: {
                "args": (df,),
                "kwargs": {},
            },
        }

        with ta.functions.assert_function_call(
            mocker,
            tubular.base.BaseTransformer,
            "transform",
            expected_call_args,
            return_value=df,
        ):
            x = DatetimeInfoExtractor(columns=["a"], include=["dayofweek"])

            x.transform(df)

    @pytest.mark.parametrize(
            ("columns"),
            [
                ["numeric_col"],
                ["string_col"],
                ["bool_col"],
                ["empty_col"],
            ],
    )
    def test_input_data_check_column_errors(self, columns):
        """ Check that errors are raised on a variety of different non datatypes"""
        x = DatetimeInfoExtractor(columns=columns)

        df = d.create_date_diff_incorrect_dtypes()

        msg = f"{x.classname()}: {columns[0]} should be datetime64 or date type but got {df[columns[0]].dtype}"

        with pytest.raises(TypeError, match=msg):
            x.transform(df)

    def test_cast_to_date_warning(self):
        "Test that transform raises a warning if column is date but not datetime"

        x = DatetimeInfoExtractor(columns="date_col_1")

        column = "date_col_1"

        msg = (
                    f"""
                    {x.classname()}: temporarily cast {column} from datetime64 to date before transforming in order to apply the datetime method.

                    This will artificially increase the precision of each data point in the column. Original column not changed.
                    """
        )

        with pytest.warns(UserWarning, match = msg):
            x.transform(d.create_date_diff_different_dtypes())

    def test_correct_col_returned(self):
        """Test that the added column is correct."""
        df = d.create_date_test_df()
        df = df.astype("datetime64[ns]")

        x = DatetimeInfoExtractor(columns=["b"], include=["timeofyear"])
        transformed = x.transform(df)

        expected_output = pd.Series(
            [
                "spring",
                "winter",
                "autumn",
                "autumn",
                "autumn",
                "autumn",
                "autumn",
                "summer",
            ],
            name="b_timeofyear",
        )

        ta.equality.assert_series_equal_msg(
            transformed["b_timeofyear"],
            expected_output,
            "incorrect series returned",
            print_actual_and_expected=True,
        )

    def test_map_values_calls(self, mocker):
        """Test all intermediary methods are being called correct number of times."""
        # df is 8 rows long so each intermediate function must have 8 calls
        df = d.create_date_test_df()
        df = df.astype("datetime64[ns]")

        mocked_map_values = mocker.spy(DatetimeInfoExtractor, "_map_values")

        x = DatetimeInfoExtractor(
            columns=["b"],
            include=["timeofday", "timeofyear", "timeofmonth", "dayofweek"],
        )
        x.transform(df)

        assert mocked_map_values.call_count == 32

    def test_correct_df_returned_datetime_input(self):
        """Test that correct df is returned after transformation."""
        df = d.create_date_test_df()
        df.loc[0, "b"] = np.nan
        df = df.astype("datetime64[ns]")

        x = DatetimeInfoExtractor(columns=["b"], include=["timeofmonth", "timeofyear"])
        transformed = x.transform(df)

        expected = df.copy()
        expected["b_timeofmonth"] = [
            np.nan,
            "end",
            "start",
            "start",
            "start",
            "start",
            "start",
            "end",
        ]
        expected["b_timeofyear"] = [
            np.nan,
            "winter",
            "autumn",
            "autumn",
            "autumn",
            "autumn",
            "autumn",
            "summer",
        ]

        ta.equality.assert_frame_equal_msg(
            transformed,
            expected,
            "incorrect dataframe returned",
        )

    def test_correct_df_returned_date_input(self):
        """Test that correct df is returned after transformation."""
        df = d.create_date_test_df()
        df.loc[0, "b"] = np.nan

        x = DatetimeInfoExtractor(columns=["b"], include=["timeofmonth", "timeofyear"])
        transformed = x.transform(df)

        expected = df.copy()
        expected["b_timeofmonth"] = [
            np.nan,
            "end",
            "start",
            "start",
            "start",
            "start",
            "start",
            "end",
        ]
        expected["b_timeofyear"] = [
            np.nan,
            "winter",
            "autumn",
            "autumn",
            "autumn",
            "autumn",
            "autumn",
            "summer",
        ]

        ta.equality.assert_frame_equal_msg(
            transformed,
            expected,
            "incorrect dataframe returned",
        )
