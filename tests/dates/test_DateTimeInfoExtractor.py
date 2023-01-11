import pytest
import re
import test_aide as ta
import tests.test_data as d

import pandas as pd
import numpy as np

import tubular
from tubular.dates import DatetimeInfoExtractor


@pytest.fixture
def timeofday_extractor():
    return DatetimeInfoExtractor(columns=["a"], include=["timeofday"])


@pytest.fixture
def timeofmonth_extractor():
    return DatetimeInfoExtractor(columns=["a"], include=["timeofmonth"])


@pytest.fixture
def timeofyear_extractor():
    return DatetimeInfoExtractor(columns=["a"], include=["timeofyear"])


@pytest.fixture
def dayofweek_extractor():
    return DatetimeInfoExtractor(columns=["a"], include=["dayofweek"])


class TestExtractDatetimeInfoInit(object):
    def test_assert_inheritance(self):
        """Test that ExtractDatetimeInfo inherits from BaseTransformer."""

        x = DatetimeInfoExtractor(columns=["a"])

        ta.classes.assert_inheritance(x, tubular.base.BaseTransformer)

    def test_arguments(self):
        """Test that init has the expected arguments"""

        default_include = [
            "timeofday",
            "timeofmonth",
            "timeofyear",
            "dayofweek",
        ]
        ta.functions.test_function_arguments(
            func=DatetimeInfoExtractor.__init__,
            expected_arguments=["self", "columns", "include", "datetime_mappings"],
            expected_default_values=(
                default_include,
                {},
            ),
        )

    def test_super_init_called(self, mocker):
        """Test that init calls BaseTransformer.init."""

        expected_call_args = {
            0: {
                "args": (),
                "kwargs": {"columns": ["a"]},
            }
        }

        with ta.functions.assert_function_call(
            mocker, tubular.base.BaseTransformer, "__init__", expected_call_args
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
                    "timeofday": {"am": range(0, 12), "pm": range(12, 24)}
                },
            },
            msg="Attributes for ExtractDatetimeInfo set in init",
        )

    def test_class_methods(self):
        """Test that DatetimeInfoExtractor has fit and transform methods."""

        x = DatetimeInfoExtractor(columns=["a"])

        ta.classes.test_object_method(
            obj=x, expected_method="identify_timeofday", msg="identify_timeofday"
        )
        ta.classes.test_object_method(
            obj=x, expected_method="identify_timeofmonth", msg="identify_timeofmonth"
        )
        ta.classes.test_object_method(
            obj=x, expected_method="identify_timeofyear", msg="identify_timeofyear"
        )
        ta.classes.test_object_method(
            obj=x, expected_method="identify_dayofweek", msg="identify_dayofweek"
        )
        ta.classes.test_object_method(
            obj=x, expected_method="transform", msg="transform"
        )

    @pytest.mark.parametrize("incorrect_type_include", [2, 3.0, "invalid", "dayofweek"])
    def test_error_when_include_not_list(self, incorrect_type_include):
        """Test that an exception is raised when value include variable is not a list"""

        with pytest.raises(
            TypeError,
            match="include should be List",
        ):
            DatetimeInfoExtractor(columns=["a"], include=incorrect_type_include)

    def test_error_when_invalid_include_option(self):
        """Test that an exception is raised when include contains incorrect values"""

        with pytest.raises(
            ValueError,
            match=r'elements in include should be in \["timeofday", "timeofmonth", "timeofyear", "dayofweek"\]',
        ):
            DatetimeInfoExtractor(
                columns=["a"], include=["timeofday", "timeofmonth", "invalid_option"]
            )

    @pytest.mark.parametrize(
        "incorrect_type_datetime_mappings", [2, 3.0, ["a", "b"], "dayofweek"]
    )
    def test_error_when_datetime_mappings_not_dict(
        self, incorrect_type_datetime_mappings
    ):
        """Test that an exception is raised when datetime_mappings is not a dict"""

        with pytest.raises(
            TypeError,
            match="datetime_mappings should be Dict",
        ):
            DatetimeInfoExtractor(
                columns=["a"], datetime_mappings=incorrect_type_datetime_mappings
            )

    @pytest.mark.parametrize(
        "incorrect_type_datetime_mappings_values", [{"timeofday": 2}]
    )
    def test_error_when_datetime_mapping_value_not_dict(
        self, incorrect_type_datetime_mappings_values
    ):
        """Test that an exception is raised when values in datetime_mappings are not dict"""

        with pytest.raises(
            TypeError,
            match="values in datetime_mappings should be dict",
        ):
            DatetimeInfoExtractor(
                columns=["a"], datetime_mappings=incorrect_type_datetime_mappings_values
            )

    @pytest.mark.parametrize(
        "include, incorrect_datetime_mappings_keys",
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
        self, include, incorrect_datetime_mappings_keys
    ):
        """Test that an exception is raised when keys in datetime_mappings are not in include"""

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
        "incomplete_mappings, expected_exception",
        [
            (
                {"timeofday": {"mapped": range(23)}},
                re.escape(
                    "timeofday mapping dictionary should contain mapping for all hours between 0-23. {23} are missing"
                ),
            ),
            (
                {"timeofmonth": {"mapped": range(1, 31)}},
                re.escape(
                    "timeofmonth mapping dictionary should contain mapping for all days between 1-31. {31} are missing"
                ),
            ),
            (
                {"timeofyear": {"mapped": range(1, 12)}},
                re.escape(
                    "timeofyear mapping dictionary should contain mapping for all months between 1-12. {12} are missing"
                ),
            ),
            (
                {"dayofweek": {"mapped": range(6)}},
                re.escape(
                    "dayofweek mapping dictionary should contain mapping for all days between 0-6. {6} are missing"
                ),
            ),
        ],
    )
    def test_error_when_incomplete_mappings_passed(
        self, incomplete_mappings, expected_exception
    ):
        """Test that error is raised when incomplete mappings are passed"""

        with pytest.raises(ValueError, match=expected_exception):
            DatetimeInfoExtractor(columns=["a"], datetime_mappings=incomplete_mappings)


class TestIdentifyTimeOfDay(object):
    def test_arguments(self):
        """Test that identify_timeofday has the expected arguments"""

    ta.functions.test_function_arguments(
        func=DatetimeInfoExtractor.identify_timeofday,
        expected_arguments=["self", "hour"],
        expected_default_values=None,
    )

    @pytest.mark.parametrize("incorrect_type_input", ["2", [1, 2]])
    def test_incorrect_type_input(self, incorrect_type_input, timeofday_extractor):
        """Test that an error is raised if input is the wrong type"""

        with pytest.raises(TypeError, match="hour should be float or int"):

            timeofday_extractor.identify_timeofday(incorrect_type_input)

    @pytest.mark.parametrize("incorrect_size_input", [-2, 30, 5.6, 11.2])
    def test_out_of_bounds_or_fractional_input(
        self, incorrect_size_input, timeofday_extractor
    ):
        """Test that an error is raised when value is outside of 0-23 range"""

        with pytest.raises(ValueError, match="hour should be a whole value in 0-23"):
            timeofday_extractor.identify_timeofday(incorrect_size_input)

    @pytest.mark.parametrize(
        "valid_hour, hour_time_of_day",
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
    def test_valid_inputs(self, valid_hour, hour_time_of_day, timeofday_extractor):
        """Trial test to check all in one go"""

        output = timeofday_extractor.identify_timeofday(valid_hour)

        assert output == hour_time_of_day, "expected {}, output {}".format(
            hour_time_of_day, output
        )

    def test_valid_nan_output(self, timeofday_extractor):
        """Test that correct values are return with valid inputs"""
        output = timeofday_extractor.identify_timeofday(np.nan)
        print(output)
        assert np.isnan(
            output
        ), f"passing np.nan should result in np.nan, instead received {output}"


class TestIdentifyTimeOfMonth(object):
    def test_arguments(self):
        """Test that identify_timeofmonth has the expected arguments"""

        ta.functions.test_function_arguments(
            func=DatetimeInfoExtractor.identify_timeofmonth,
            expected_arguments=["self", "day"],
            expected_default_values=None,
        )

    @pytest.mark.parametrize("incorrect_type_input", ["2", [1, 2]])
    def test_incorrect_type_input(self, incorrect_type_input, timeofmonth_extractor):
        """Test that an error is raised if input is the wrong type"""

        with pytest.raises(TypeError, match="day should be float or int"):
            timeofmonth_extractor.identify_timeofmonth(incorrect_type_input)

    @pytest.mark.parametrize("incorrect_size_input", [-2, 40, 5.6, 11.2])
    def test_out_of_bounds_or_fractional_input(
        self, incorrect_size_input, timeofmonth_extractor
    ):
        """Test that an error is raised when value is outside of 1-31 range"""

        with pytest.raises(ValueError, match="day should be a whole number in 1-31"):
            timeofmonth_extractor.identify_timeofmonth(incorrect_size_input)

    @pytest.mark.parametrize(
        "valid_day, day_time_of_month",
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
    def test_valid_inputs(self, valid_day, day_time_of_month, timeofmonth_extractor):
        """Test that correct values are return with valid inputs"""
        output = timeofmonth_extractor.identify_timeofmonth(valid_day)
        assert output == day_time_of_month, "expected {}, output {}".format(
            day_time_of_month, output
        )

    def test_valid_nan_output(self, timeofmonth_extractor):
        """Test that correct values are return with valid inputs"""
        output = timeofmonth_extractor.identify_timeofmonth(np.nan)
        assert np.isnan(
            output
        ), f"passing np.nan should result in np.nan, instead received {output}"


class TestIdentifyTimeOfYear(object):
    def test_arguments(self):
        """Test that identify_timeofyear has the expected arguments"""

        ta.functions.test_function_arguments(
            func=DatetimeInfoExtractor.identify_timeofyear,
            expected_arguments=["self", "month"],
            expected_default_values=None,
        )

    @pytest.mark.parametrize("incorrect_type_input", ["2", [1, 2]])
    def test_incorrect_type_input(self, incorrect_type_input, timeofyear_extractor):
        """Test that an error is raised if input is the wrong type"""

        with pytest.raises(TypeError, match="month should be float or int"):
            timeofyear_extractor.identify_timeofyear(incorrect_type_input)

    @pytest.mark.parametrize("incorrect_size_input", [-2, 13, 5.6, 11.2])
    def test_out_of_bounds_or_fractional_input(
        self, incorrect_size_input, timeofyear_extractor
    ):
        """Test that an error is raised when value is outside of 1-12 range"""

        with pytest.raises(ValueError, match="month should be a whole number in 1-12"):
            timeofyear_extractor.identify_timeofyear(incorrect_size_input)

    @pytest.mark.parametrize(
        "valid_month, month_time_of_year",
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
    def test_valid_inputs(self, valid_month, month_time_of_year, timeofyear_extractor):
        """Test that correct values are return with valid inputs"""
        output = timeofyear_extractor.identify_timeofyear(valid_month)
        assert output == month_time_of_year, "expected {}, output {}".format(
            month_time_of_year, output
        )

    def test_valid_nan_output(self, timeofyear_extractor):
        """Test that correct values are return with valid inputs"""
        output = timeofyear_extractor.identify_timeofyear(np.nan)
        assert np.isnan(
            output
        ), f"passing np.nan should result in np.nan, instead recieved {output}"


class TestIdentifyDayOfWeek(object):
    def test_arguments(self):
        """Test that identify_dayofweek has the expected arguments"""

        ta.functions.test_function_arguments(
            func=DatetimeInfoExtractor.identify_dayofweek,
            expected_arguments=["self", "day"],
            expected_default_values=None,
        )

    @pytest.mark.parametrize("incorrect_type_input", ["2", [1, 2]])
    def test_incorrect_type_input(self, incorrect_type_input, dayofweek_extractor):
        """Test that an error is raised if input is the wrong type"""

        with pytest.raises(TypeError, match="day should be float or int"):
            dayofweek_extractor.identify_dayofweek(incorrect_type_input)

    @pytest.mark.parametrize("incorrect_size_input", [-2, 8, 5.6, 11.2])
    def test_out_of_boundsor_fractional_input(
        self, incorrect_size_input, dayofweek_extractor
    ):
        """Test that an error is raised when value is outside of 0-6 range"""

        with pytest.raises(ValueError, match="day should be in 0-6"):
            dayofweek_extractor.identify_dayofweek(incorrect_size_input)

    @pytest.mark.parametrize(
        "valid_day, dayofweek",
        [
            (0, "monday"),
            (2, "wednesday"),
            (4, "friday"),
            (6, "sunday"),
        ],
    )
    def test_valid_inputs(self, valid_day, dayofweek, dayofweek_extractor):
        """Test that correct values are return with valid inputs"""
        output = dayofweek_extractor.identify_dayofweek(valid_day)
        assert output == dayofweek, "expected {}, output {}".format(dayofweek, output)

    def test_valid_nan_output(self, dayofweek_extractor):
        """Test that correct values are return with valid inputs"""
        output = dayofweek_extractor.identify_dayofweek(np.nan)
        assert np.isnan(
            output
        ), f"passing np.nan should result in np.nan, instead received {output}"


class TestTransform(object):
    def test_arguments(self):
        """Test that init has the expected arguments"""

        ta.functions.test_function_arguments(
            func=DatetimeInfoExtractor.transform,
            expected_arguments=["self", "X"],
            expected_default_values=None,
        )

    def test_super_transform_called(self, mocker):
        """Test that init calls BaseTransformer.init."""

        df = d.create_date_test_df()
        df = df.astype("datetime64[ns]")

        expected_call_args = {
            0: {
                "args": (df,),
                "kwargs": {},
            }
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

    def test_non_datetime_column(self):
        """Test that error is raised if input columns do not contain datetime values"""

        df = d.create_df_1()  # Mix of int values

        x = DatetimeInfoExtractor(columns=["a"], include=["dayofweek"])

        with pytest.raises(
            TypeError, match="values in {} should be datetime".format("a")
        ):
            x.transform(df),

    def test_correct_col_returned(self):
        """Test that the added column is correct"""

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

    def test_intermediary_method_calls(self, mocker):
        """Test all intermediary methods are being called correct number of times"""

        # df is 8 rows long so each intermediate function must have 8 calls
        df = d.create_date_test_df()
        df = df.astype("datetime64[ns]")

        mocked_tod = mocker.spy(DatetimeInfoExtractor, "identify_timeofday")
        mocked_toy = mocker.spy(DatetimeInfoExtractor, "identify_timeofyear")
        mocked_tom = mocker.spy(DatetimeInfoExtractor, "identify_timeofmonth")
        mocked_doy = mocker.spy(DatetimeInfoExtractor, "identify_dayofweek")

        x = DatetimeInfoExtractor(
            columns=["b"],
            include=["timeofday", "timeofyear", "timeofmonth", "dayofweek"],
        )
        x.transform(df)

        assert mocked_tod.call_count == 8
        assert mocked_toy.call_count == 8
        assert mocked_tom.call_count == 8
        assert mocked_doy.call_count == 8

    def test_correct_df_returned(self):
        """Test that correct df is returned after transformation"""

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
            transformed, expected, "incorrect dataframe returned"
        )
