import pytest
import tests.test_data as d
import test_aide as ta
import re

import tubular
from tubular.dates import DatetimeSinusoidCalculator
import pandas as pd
import numpy as np


@pytest.fixture(scope="module", autouse=True)
def example_transformer():

    return DatetimeSinusoidCalculator("a", "cos", "hour", 24)


class TestDatetimeSinusoidCalculatorInit(object):
    """Tests for DateDifferenceTransformer.init()."""

    def test_arguments(self):
        """Test that init has expected arguments."""

        ta.functions.test_function_arguments(
            func=DatetimeSinusoidCalculator.__init__,
            expected_arguments=[
                "self",
                "columns",
                "method",
                "units",
                "period",
            ],
            expected_default_values=(2 * np.pi,),
        )

    def test_class_methods(self, example_transformer):
        """Test that DateDifferenceTransformer has a transform method."""

        ta.classes.test_object_method(
            obj=example_transformer, expected_method="transform", msg="transform"
        )

    def test_inheritance(self, example_transformer):
        """Test that DateDifferenceTransformer inherits from BaseTransformer."""

        ta.classes.assert_inheritance(example_transformer, tubular.base.BaseTransformer)

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
            match="units must be a string or dictionary but got {}".format(
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

    @pytest.mark.parametrize("incorrect_dict_types_units", [
        {"str":True}, {2:"str"}, {"str":2}, {2:2}, {"str":["str"]}])
    def test_units_dict_type_error(self, incorrect_dict_types_units):
        """Test that an error is raised if units dict is not a str:str kv pair"""

        with pytest.raises(
            TypeError,
            match="units dictionary key value pair must be strings but got {} {}".format(
                set(type(k) for k in incorrect_dict_types_units.keys()), set(type(v) for v in incorrect_dict_types_units.values())
            ),
        ):

            DatetimeSinusoidCalculator(
                "a",
                "cos",
                incorrect_dict_types_units,
                24,
            )
    @pytest.mark.parametrize("incorrect_dict_units", [
        {"str":"tweet"}])
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


###
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


class TestDatetimeSinusoidCalculatorTransform(object):
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
            new_col_name = "cos_" + column
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
            new_cos_col_name = "cos_" + column
            new_sin_col_name = "sin_" + column
            expected[new_sin_col_name] = method_ready_column.apply(np.sin)
            expected[new_cos_col_name] = method_ready_column.apply(np.cos)

        actual = transformer.transform(d.create_datediff_test_df())
        ta.equality.assert_frame_equal_msg(
            actual=actual,
            expected=expected,
            msg_tag="DatetimeSinusoidCalculator transformer does not produce the expected output",
        )
