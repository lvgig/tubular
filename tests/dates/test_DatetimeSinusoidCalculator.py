import re

import numpy as np
import pandas as pd
import pytest
import test_aide as ta

import tests.test_data as d
import tubular
from tubular.dates import DatetimeSinusoidCalculator


@pytest.fixture(scope="module", autouse=True)
def example_transformer():
    return DatetimeSinusoidCalculator("a", "cos", "hour", 24)


class TestDatetimeSinusoidCalculatorInit:
    """Tests for DateDifferenceTransformer.init()."""

    def test_super_init_called(self, mocker):
        """Test that init calls BaseTransformer.init."""

        expected_call_args = {
            0: {
                "args": ("a",),
                "kwargs": {
                    "copy": True,
                },
            }
        }

        with ta.functions.assert_function_call(
            mocker, tubular.base.BaseTransformer, "__init__", expected_call_args
        ):
            DatetimeSinusoidCalculator(
                "a",
                "cos",
                "hour",
                24,
            )

    @pytest.mark.parametrize("incorrect_type_method", [2, 2.0, True, {"a": 4}])
    def test_method_type_error(self, incorrect_type_method):
        """Test that an exception is raised if method is not a str or a list."""

        with pytest.raises(
            TypeError,
            match="method must be a string or list but got {}".format(
                type(incorrect_type_method)
            ),
        ):
            DatetimeSinusoidCalculator(
                "a",
                incorrect_type_method,
                "hour",
                24,
            )

    @pytest.mark.parametrize("incorrect_type_units", [2, 2.0, True, ["help"]])
    def test_units_type_error(self, incorrect_type_units):
        """Test that an exception is raised if units is not a str or a dict."""

        with pytest.raises(
            TypeError,
            match="units must be a string or dict but got {}".format(
                type(incorrect_type_units)
            ),
        ):
            DatetimeSinusoidCalculator(
                "a",
                "cos",
                incorrect_type_units,
                24,
            )

    @pytest.mark.parametrize("incorrect_type_period", ["2", True, ["help"]])
    def test_period_type_error(self, incorrect_type_period):
        """Test that an error is raised if period is not an int or a float or a dictionary"""

        with pytest.raises(
            TypeError,
            match="period must be an int, float or dict but got {}".format(
                type(incorrect_type_period)
            ),
        ):
            DatetimeSinusoidCalculator(
                "a",
                "cos",
                "hour",
                incorrect_type_period,
            )

    @pytest.mark.parametrize(
        "incorrect_dict_types_period",
        [{"str": True}, {2: "str"}, {2: 2}, {"str": ["str"]}],
    )
    def test_period_dict_type_error(self, incorrect_dict_types_period):
        """Test that an error is raised if period dict is not a str:int or str:float kv pair"""

        with pytest.raises(
            TypeError,
            match="period dictionary key value pair must be str:int or str:float but got keys: {} and values: {}".format(
                set(type(k) for k in incorrect_dict_types_period.keys()),
                set(type(v) for v in incorrect_dict_types_period.values()),
            ),
        ):
            DatetimeSinusoidCalculator(
                "a",
                "cos",
                "hour",
                incorrect_dict_types_period,
            )

    @pytest.mark.parametrize(
        "incorrect_dict_types_units",
        [
            {"str": True},
            {2: "str"},
            {"str": 2},
            {2: 2},
            {"str": True},
            {"str": ["str"]},
        ],
    )
    def test_units_dict_type_error(self, incorrect_dict_types_units):
        """Test that an error is raised if units dict is not a str:str kv pair"""

        with pytest.raises(
            TypeError,
            match="units dictionary key value pair must be strings but got keys: {} and values: {}".format(
                set(type(k) for k in incorrect_dict_types_units.keys()),
                set(type(v) for v in incorrect_dict_types_units.values()),
            ),
        ):
            DatetimeSinusoidCalculator(
                "a",
                "cos",
                incorrect_dict_types_units,
                24,
            )

    @pytest.mark.parametrize("incorrect_dict_units", [{"str": "tweet"}])
    def test_units_dict_value_error(self, incorrect_dict_units):
        """Test that an error is raised if units dict value is not from the valid units list."""

        with pytest.raises(
            ValueError,
            match="units dictionary values must be one of 'year', 'month', 'day', 'hour', 'minute', 'second', 'microsecond' but got {}".format(
                set(incorrect_dict_units.values())
            ),
        ):
            DatetimeSinusoidCalculator(
                "a",
                "cos",
                incorrect_dict_units,
                24,
            )

    @pytest.mark.parametrize(
        "incorrect_dict_columns_period",
        [{"ham": 24}, {"str": 34.0}],
    )
    def test_period_dict_col_error(self, incorrect_dict_columns_period):
        """Test that an error is raised if period dict keys are not equal to columns"""

        with pytest.raises(
            ValueError,
            match="period dictionary keys must be the same as columns but got {}".format(
                set(incorrect_dict_columns_period.keys()),
            ),
        ):
            DatetimeSinusoidCalculator(
                ["vegan_sausages", "carrots", "peas"],
                "cos",
                "hour",
                incorrect_dict_columns_period,
            )

    @pytest.mark.parametrize(
        "incorrect_dict_columns_unit",
        [{"sausage_roll": "hour"}],
    )
    def test_unit_dict_col_error(self, incorrect_dict_columns_unit):
        """Test that an error is raised if unit dict keys is not equal to columns"""

        with pytest.raises(
            ValueError,
            match="unit dictionary keys must be the same as columns but got {}".format(
                set(incorrect_dict_columns_unit.keys()),
            ),
        ):
            DatetimeSinusoidCalculator(
                ["vegan_sausages", "carrots", "peas"],
                "cos",
                incorrect_dict_columns_unit,
                6,
            )

    def test_valid_method_value_error(self):
        """Test that a value error is raised if method is not sin, cos or a list containing both."""
        method = "tan"

        with pytest.raises(
            ValueError,
            match='Invalid method {} supplied, should be "sin", "cos" or a list containing both'.format(
                method
            ),
        ):
            DatetimeSinusoidCalculator(
                "a",
                method,
                "year",
                24,
            )

    def test_valid_units_value_error(self):
        """Test that a value error is raised if the unit supplied is not in the valid units list."""
        units = "five"
        valid_unit_list = [
            "year",
            "month",
            "day",
            "hour",
            "minute",
            "second",
            "microsecond",
        ]

        with pytest.raises(
            ValueError,
            match=re.escape(
                "Invalid units {} supplied, should be in {}".format(
                    units, valid_unit_list
                )
            ),
        ):
            DatetimeSinusoidCalculator(
                "a",
                "cos",
                units,
                24,
            )

    def test_attributes(self, example_transformer):
        """Test that the value passed for new_column_name and units are saved in attributes of the same name."""

        ta.classes.test_object_attributes(
            obj=example_transformer,
            expected_attributes={
                "columns": ["a"],
                "units": "hour",
                "period": 24,
            },
            msg="Attributes for DateDifferenceTransformer set in init",
        )


class TestDatetimeSinusoidCalculatorTransform:
    def test_arguments(self):
        """Test that transform has expected arguments."""

        ta.functions.test_function_arguments(
            func=DatetimeSinusoidCalculator.transform,
            expected_arguments=["self", "X"],
            expected_default_values=None,
        )

    def test_datetime_type_error(self):
        """Tests that an error is raised if the column passed to the transformer is not a datetime column."""
        not_datetime = pd.DataFrame({"a": [1, 2, 3]})
        column = "a"
        message = re.escape(
            f"{column} should be datetime64[ns] type but got {not_datetime[column].dtype}"
        )
        with pytest.raises(TypeError, match=message):
            x = DatetimeSinusoidCalculator(
                "a",
                "cos",
                "year",
                24,
            )

            x.transform(not_datetime)

    def test_BaseTransformer_transform_called(self, example_transformer, mocker):
        test_data = d.create_datediff_test_df()

        expected_call_args = {0: {"args": (test_data,), "kwargs": {}}}

        with ta.functions.assert_function_call(
            mocker,
            tubular.base.BaseTransformer,
            "transform",
            expected_call_args,
            return_value=test_data,
        ):
            example_transformer.transform(test_data)

    def test_cos_called_with_correct_args(self, mocker):
        """Tests that the correct numpy method is called on the correct column - also implicitly checks that the column has been transformed
        into the correct units through the value of the argument."""

        method = "cos"

        data = d.create_datediff_test_df()
        column_in_desired_unit = data["a"].dt.month
        cos_argument = column_in_desired_unit * (2.0 * np.pi / 12)

        spy = mocker.spy(np, method)

        x = DatetimeSinusoidCalculator(
            "a",
            "cos",
            "month",
            12,
        )
        x.transform(data)

        # pull out positional args to target the call

        call_args = spy.call_args_list[0][0]

        # test positional args are as expected
        ta.equality.assert_list_tuple_equal_msg(
            actual=call_args,
            expected=(cos_argument,),
            msg_tag=f"""Positional arg assert for {method}""",
        )

    @pytest.mark.parametrize(
        "transformer",
        [
            DatetimeSinusoidCalculator(
                "a",
                "cos",
                "month",
                12,
            ),
            DatetimeSinusoidCalculator(
                [
                    "a",
                    "b",
                ],
                "cos",
                "month",
                12,
            ),
        ],
    )
    def test_expected_output_single_method(self, transformer):
        expected = d.create_datediff_test_df()
        for column in transformer.columns:
            column_in_desired_unit = expected[column].dt.month
            cos_argument = column_in_desired_unit * (2.0 * np.pi / 12)
            new_col_name = "cos_12_month_" + column
            expected[new_col_name] = cos_argument.apply(np.cos)

        x = transformer
        actual = x.transform(d.create_datediff_test_df())
        ta.equality.assert_frame_equal_msg(
            actual=actual,
            expected=expected,
            msg_tag="DatetimeSinusoidCalculator transformer does not produce the expected output",
        )

    def test_expected_output_both_methods(self):
        expected = d.create_datediff_test_df()

        transformer = DatetimeSinusoidCalculator(
            [
                "a",
                "b",
            ],
            ["sin", "cos"],
            "month",
            12,
        )

        for column in transformer.columns:
            column_in_desired_unit = expected[column].dt.month
            method_ready_column = column_in_desired_unit * (2.0 * np.pi / 12)
            new_cos_col_name = "cos_12_month_" + column
            new_sin_col_name = "sin_12_month_" + column
            expected[new_sin_col_name] = method_ready_column.apply(np.sin)
            expected[new_cos_col_name] = method_ready_column.apply(np.cos)

        actual = transformer.transform(d.create_datediff_test_df())
        ta.equality.assert_frame_equal_msg(
            actual=actual,
            expected=expected,
            msg_tag="DatetimeSinusoidCalculator transformer does not produce the expected output",
        )

    def test_expected_output_dict_units(self):
        expected = d.create_datediff_test_df()

        transformer = DatetimeSinusoidCalculator(
            [
                "a",
                "b",
            ],
            ["sin"],
            {"a": "month", "b": "day"},
            12,
        )

        a_in_desired_unit = expected["a"].dt.month
        b_in_desired_unit = expected["b"].dt.day
        a_method_ready_column = a_in_desired_unit * (2.0 * np.pi / 12)
        b_method_ready_column = b_in_desired_unit * (2.0 * np.pi / 12)
        a_col_name = "sin_12_month_a"
        b_col_name = "sin_12_day_b"
        expected[a_col_name] = a_method_ready_column.apply(np.sin)
        expected[b_col_name] = b_method_ready_column.apply(np.sin)

        actual = transformer.transform(d.create_datediff_test_df())
        ta.equality.assert_frame_equal_msg(
            actual=actual,
            expected=expected,
            msg_tag="DatetimeSinusoidCalculator transformer does not produce the expected output",
        )

    def test_expected_output_dict_period(self):
        expected = d.create_datediff_test_df()

        transformer = DatetimeSinusoidCalculator(
            [
                "a",
                "b",
            ],
            ["sin"],
            "month",
            {"a": 12, "b": 24},
        )

        a_in_desired_unit = expected["a"].dt.month
        b_in_desired_unit = expected["b"].dt.month
        a_method_ready_column = a_in_desired_unit * (2.0 * np.pi / 12)
        b_method_ready_column = b_in_desired_unit * (2.0 * np.pi / 24)
        a_col_name = "sin_12_month_a"
        b_col_name = "sin_24_month_b"
        expected[a_col_name] = a_method_ready_column.apply(np.sin)
        expected[b_col_name] = b_method_ready_column.apply(np.sin)

        actual = transformer.transform(d.create_datediff_test_df())
        ta.equality.assert_frame_equal_msg(
            actual=actual,
            expected=expected,
            msg_tag="DatetimeSinusoidCalculator transformer does not produce the expected output",
        )

    def test_expected_output_dict_both(self):
        expected = d.create_datediff_test_df()

        transformer = DatetimeSinusoidCalculator(
            [
                "a",
                "b",
            ],
            ["sin"],
            {"a": "month", "b": "day"},
            {"a": 12, "b": 24},
        )

        a_in_desired_unit = expected["a"].dt.month
        b_in_desired_unit = expected["b"].dt.day
        a_method_ready_column = a_in_desired_unit * (2.0 * np.pi / 12)
        b_method_ready_column = b_in_desired_unit * (2.0 * np.pi / 24)
        a_col_name = "sin_12_month_a"
        b_col_name = "sin_24_day_b"
        expected[a_col_name] = a_method_ready_column.apply(np.sin)
        expected[b_col_name] = b_method_ready_column.apply(np.sin)

        actual = transformer.transform(d.create_datediff_test_df())
        ta.equality.assert_frame_equal_msg(
            actual=actual,
            expected=expected,
            msg_tag="DatetimeSinusoidCalculator transformer does not produce the expected output",
        )

    def test_expected_output_dict_both_with_both_methods(self):
        expected = d.create_datediff_test_df()

        transformer = DatetimeSinusoidCalculator(
            [
                "a",
                "b",
            ],
            ["sin", "cos"],
            {"a": "month", "b": "day"},
            {"a": 12, "b": 24},
        )

        a_in_desired_unit = expected["a"].dt.month
        b_in_desired_unit = expected["b"].dt.day
        a_method_ready_column = a_in_desired_unit * (2.0 * np.pi / 12)
        b_method_ready_column = b_in_desired_unit * (2.0 * np.pi / 24)
        a_sin_col_name = "sin_12_month_a"
        a_cos_col_name = "cos_12_month_a"
        b_sin_col_name = "sin_24_day_b"
        b_cos_col_name = "cos_24_day_b"
        expected[a_sin_col_name] = a_method_ready_column.apply(np.sin)
        expected[a_cos_col_name] = a_method_ready_column.apply(np.cos)
        expected[b_sin_col_name] = b_method_ready_column.apply(np.sin)
        expected[b_cos_col_name] = b_method_ready_column.apply(np.cos)

        actual = transformer.transform(d.create_datediff_test_df())
        ta.equality.assert_frame_equal_msg(
            actual=actual,
            expected=expected,
            msg_tag="DatetimeSinusoidCalculator transformer does not produce the expected output",
        )
