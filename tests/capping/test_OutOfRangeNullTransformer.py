import numpy as np
import pytest
import test_aide as ta

import tests.test_data as d
import tubular
from tubular.capping import OutOfRangeNullTransformer


class TestInit:
    """Tests for OutOfRangeNullTransformer.init()."""

    def test_arguments(self):
        """Test that init has expected arguments."""
        ta.functions.test_function_arguments(
            func=OutOfRangeNullTransformer.__init__,
            expected_arguments=[
                "self",
                "capping_values",
                "quantiles",
                "weights_column",
            ],
            expected_default_values=(None, None, None),
        )

    @pytest.mark.parametrize(
        "method_name",
        [("fit"), ("set_replacement_values")],
    )
    def test_class_methods(self, method_name):
        """Test that OutOfRangeNullTransformer has transform fit and set_replacement_values methods."""
        x = OutOfRangeNullTransformer(capping_values={"a": [1, 3]})

        ta.classes.test_object_method(
            obj=x, expected_method=method_name, msg=method_name
        )

    def test_inheritance(self):
        """Test that OutOfRangeNullTransformer inherits from CappingTransformer."""
        x = OutOfRangeNullTransformer(capping_values={"a": [1, 3]})

        ta.classes.assert_inheritance(x, tubular.capping.CappingTransformer)

    @pytest.mark.parametrize(
        "capping_values, quantiles, weights_column, verbose, copy",
        [
            ({"a": [1, 3], "b": [None, -1]}, None, None, True, True),
            ({"a": [1, 3], "b": [None, -1]}, None, "aa", True, False),
            (None, {"a": [None, 1], "b": [0.2, None]}, "aa", False, True),
        ],
    )
    def test_super_init_called(
        self, mocker, capping_values, quantiles, weights_column, verbose, copy
    ):
        """Test that init calls CappingTransformer.init."""
        spy = mocker.spy(tubular.capping.CappingTransformer, "__init__")

        x = OutOfRangeNullTransformer(
            capping_values=capping_values,
            quantiles=quantiles,
            weights_column=weights_column,
            verbose=verbose,
            copy=copy,
        )

        assert (
            spy.call_count == 1
        ), "unexpected number of calls to CappingTransformer.__init__"

        call_args = spy.call_args_list[0]
        call_pos_args = call_args[0]
        call_kwargs = call_args[1]

        assert call_pos_args == (
            x,
        ), "unexpected positional args in CappingTransformer.__init__ call"

        expected_kwargs = {
            "capping_values": capping_values,
            "quantiles": quantiles,
            "weights_column": weights_column,
            "verbose": verbose,
            "copy": copy,
        }

        assert (
            call_kwargs == expected_kwargs
        ), "unexpected kwargs in CappingTransformer.__init__ call"

    def test_set_replacement_values_called(self, mocker):
        """Test that init calls OutOfRangeNullTransformer.set_replacement_values during init."""
        expected_call_args = {0: {"args": (), "kwargs": {}}}

        with ta.functions.assert_function_call(
            mocker,
            tubular.capping.OutOfRangeNullTransformer,
            "set_replacement_values",
            expected_call_args,
        ):
            OutOfRangeNullTransformer(
                quantiles={"c": [0, 0.99], "d": [None, 0.01]}, verbose=True, copy=True
            )


class TestFit:
    """Tests for OutOfRangeNullTransformer.fit()."""

    def test_arguments(self):
        """Test that fit has expected arguments."""
        ta.functions.test_function_arguments(
            func=OutOfRangeNullTransformer.fit,
            expected_arguments=["self", "X", "y"],
            expected_default_values=(None,),
        )

    def test_super_fit_call(self, mocker):
        """Test the call to CappingTransformer.fit."""
        spy = mocker.spy(tubular.capping.CappingTransformer, "fit")

        df = d.create_df_9()

        x = OutOfRangeNullTransformer(
            quantiles={"a": [0.1, 1], "b": [0.5, None]}, weights_column="c"
        )

        x.fit(df)

        assert (
            spy.call_count == 1
        ), "unexpected number of calls to CappingTransformer.fit"

        call_args = spy.call_args_list[0]
        call_pos_args = call_args[0]
        call_kwargs = call_args[1]

        assert call_pos_args == (
            x,
        ), "unexpected positional args in CappingTransformer.fit call"

        expected_kwargs = {"X": d.create_df_9(), "y": None}

        ta.equality.assert_equal_dispatch(
            expected=expected_kwargs,
            actual=call_kwargs,
            msg="unexpected kwargs in CappingTransformer.fit call",
        )

    def test_set_replacement_values_called(self, mocker):
        """Test that init calls OutOfRangeNullTransformer.set_replacement_values during fit."""
        df = d.create_df_9()

        x = OutOfRangeNullTransformer(
            quantiles={"a": [0.1, 1], "b": [0.5, None]}, weights_column="c"
        )

        expected_call_args = {0: {"args": (), "kwargs": {}}}

        with ta.functions.assert_function_call(
            mocker,
            tubular.capping.OutOfRangeNullTransformer,
            "set_replacement_values",
            expected_call_args,
        ):
            x.fit(df)

    def test_fit_returns_self(self):
        """Test fit returns self?."""
        df = d.create_df_9()

        x = OutOfRangeNullTransformer(
            quantiles={"a": [0.1, 1], "b": [0.5, None]}, weights_column="c"
        )

        x_fitted = x.fit(df)

        assert (
            x_fitted is x
        ), "Returned value from OutOfRangeNullTransformer.fit not as expected."


class TestSetReplacementValues:
    """Test for the OutOfRangeNullTransformer.set_replacement_values() method."""

    def test_arguments(self):
        """Test that set_replacement_values has expected arguments."""
        ta.functions.test_function_arguments(
            func=OutOfRangeNullTransformer.set_replacement_values,
            expected_arguments=["self"],
            expected_default_values=None,
        )

    @pytest.mark.parametrize(
        "value_to_set, expected_replacement_values",
        [
            (
                {"a": [0, 1], "b": [None, 1], "c": [3, None]},
                {"a": [np.NaN, np.NaN], "b": [None, np.NaN], "c": [np.NaN, None]},
            ),
            ({}, {}),
            ({"a": [None, 0.1]}, {"a": [None, np.NaN]}),
        ],
    )
    def test_expected_replacement_values_set(
        self, value_to_set, expected_replacement_values
    ):
        """Test the _replacement_values attribute is modified as expected given the prior values of the attribute."""
        x = OutOfRangeNullTransformer(capping_values={"a": [0, 1]})

        x._replacement_values = value_to_set

        x.set_replacement_values()

        # also tests that capping_values is not modified
        ta.classes.test_object_attributes(
            obj=x,
            expected_attributes={
                "_replacement_values": expected_replacement_values,
                "capping_values": {"a": [0, 1]},
            },
            msg="attributes not as expected after running set_replacement_values",
        )
