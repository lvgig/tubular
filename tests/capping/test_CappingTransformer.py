import pytest
import test_aide as ta
import tests.test_data as d
import pandas as pd
import numpy as np

import tubular
from tubular.capping import CappingTransformer


class TestInit(object):
    """Tests for CappingTransformer.init()."""

    def test_arguments(self):
        """Test that init has expected arguments."""

        ta.functions.test_function_arguments(
            func=CappingTransformer.__init__,
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
        [
            ("transform"),
            ("fit"),
            ("check_capping_values_dict"),
            ("weighted_quantile"),
            ("prepare_quantiles"),
        ],
    )
    def test_class_methods(self, method_name):
        """Test that CappingTransformer has transform method."""

        x = CappingTransformer(capping_values={"a": [1, 3]})

        ta.classes.test_object_method(
            obj=x, expected_method=method_name, msg=method_name
        )

    def test_inheritance(self):
        """Test that CappingTransformer inherits from BaseTransformer."""

        x = CappingTransformer(capping_values={"a": [1, 3]})

        ta.classes.assert_inheritance(x, tubular.base.BaseTransformer)

    def test_capping_values_quantiles_both_none_error(self):
        """Test that an exception is raised if both capping_values and quantiles are passed as None."""

        with pytest.raises(
            ValueError,
            match="CappingTransformer: both capping_values and quantiles are None, either supply capping values in the "
            "capping_values argument or supply quantiles that can be learnt in the fit method",
        ):

            CappingTransformer(capping_values=None, quantiles=None)

    def test_capping_values_quantiles_both_specified_error(self):
        """Test that an exception is raised if both capping_values and quantiles are specified."""

        with pytest.raises(
            ValueError,
            match="CappingTransformer: both capping_values and quantiles are not None, supply one or the other",
        ):

            CappingTransformer(
                capping_values={"a": [1, 4]}, quantiles={"a": [0.2, 0.4]}
            )

    @pytest.mark.parametrize("out_range_value", [(-2), (1.2)])
    def test_quantiles_outside_range_error(self, out_range_value):
        """Test that an exception is raised if quanties contain values outisde [0, 1] range."""

        with pytest.raises(
            ValueError,
            match=rf"CappingTransformer: quantile values must be in the range \[0, 1\] but got {out_range_value} for key f",
        ):

            CappingTransformer(
                quantiles={"e": [0.1, 0.9], "f": [out_range_value, None]}
            )

    def test_super_init_called_capping_values(self, mocker):
        """Test that init calls BaseTransformer.init when capping_values are passed."""

        expected_call_args = {
            0: {
                "args": (),
                "kwargs": {"columns": ["a", "b"], "verbose": True, "copy": True},
            }
        }

        with ta.functions.assert_function_call(
            mocker, tubular.base.BaseTransformer, "__init__", expected_call_args
        ):

            CappingTransformer(
                capping_values={"a": [1, 3], "b": [None, -1]}, verbose=True, copy=True
            )

    def test_super_init_called_quantiles(self, mocker):
        """Test that init calls BaseTransformer.init when quantiles are passed."""

        expected_call_args = {
            0: {
                "args": (),
                "kwargs": {"columns": ["c", "d"], "verbose": True, "copy": True},
            }
        }

        with ta.functions.assert_function_call(
            mocker, tubular.base.BaseTransformer, "__init__", expected_call_args
        ):

            CappingTransformer(
                quantiles={"c": [0, 0.99], "d": [None, 0.01]}, verbose=True, copy=True
            )

    def test_check_capping_values_dict_called_quantiles(self, mocker):
        """Test that init calls check_capping_values_dict when quantiles are passed."""

        expected_call_args = {
            0: {
                "args": ({"c": [0, 0.99], "d": [None, 0.01]}, "quantiles"),
                "kwargs": {},
            }
        }

        with ta.functions.assert_function_call(
            mocker,
            tubular.capping.CappingTransformer,
            "check_capping_values_dict",
            expected_call_args,
        ):

            CappingTransformer(quantiles={"c": [0, 0.99], "d": [None, 0.01]})

    def test_check_capping_values_dict_called_capping_values(self, mocker):
        """Test that init calls check_capping_values_dict when capping_values are passed."""

        expected_call_args = {
            0: {
                "args": ({"a": [1, 3], "b": [None, -1]}, "capping_values"),
                "kwargs": {},
            }
        }

        with ta.functions.assert_function_call(
            mocker,
            tubular.capping.CappingTransformer,
            "check_capping_values_dict",
            expected_call_args,
        ):

            CappingTransformer(capping_values={"a": [1, 3], "b": [None, -1]})

    def test_values_passed_in_init_set_to_attribute_capping_values(self):
        """Test that the capping_values passed in init are saved in an attribute of the same name."""

        capping_values_dict = {"a": [1, 3], "b": [None, -1]}

        x = CappingTransformer(capping_values=capping_values_dict)

        ta.classes.test_object_attributes(
            obj=x,
            expected_attributes={
                "capping_values": capping_values_dict,
                "weights_column": None,
                "quantiles": None,
                "_replacement_values": capping_values_dict,
            },
            msg="capping_values attribute for CappingTransformer set in init",
        )

    def test_values_passed_in_init_set_to_attribute_quantiles(self):
        """Test that the capping_values passed in init are saved in an attribute of the same name."""

        quantiles_dict = {"a": [0.2, 1], "b": [None, 0.9]}

        x = CappingTransformer(quantiles=quantiles_dict)

        ta.classes.test_object_attributes(
            obj=x,
            expected_attributes={
                "quantiles": quantiles_dict,
                "weights_column": None,
                "capping_values": {},
                "_replacement_values": {},
            },
            msg="quantiles attribute for CappingTransformer set in init",
        )


class TestCheckCappingValuesDict(object):
    """Tests for the CappingTransformer.check_capping_values_dict() method."""

    def test_arguments(self):
        """Test that check_capping_values_dict has expected arguments."""

        ta.functions.test_function_arguments(
            func=CappingTransformer.check_capping_values_dict,
            expected_arguments=["self", "capping_values_dict", "dict_name"],
            expected_default_values=None,
        )

    def test_capping_values_not_dict_error(self):
        """Test that an exception is raised if capping_values_dict is not a dict."""

        x = CappingTransformer(capping_values={"a": [1, 3], "b": [None, -1]})

        with pytest.raises(
            TypeError,
            match="CappingTransformer: aaa should be dict of columns and capping values",
        ):

            x.check_capping_values_dict(
                capping_values_dict=("a", [1, 3], "b", [None, -1]), dict_name="aaa"
            )

    def test_capping_values_non_str_key_error(self):
        """Test that an exception is raised if capping_values_dict has any non str keys."""

        x = CappingTransformer(capping_values={"a": [1, 3], "b": [None, -1]})

        with pytest.raises(
            TypeError,
            match=r"CappingTransformer: all keys in bbb should be str, but got \<class 'int'\>",
        ):

            x.check_capping_values_dict(
                capping_values_dict={"a": [1, 3], 1: [None, -1]}, dict_name="bbb"
            )

    def test_capping_values_non_list_item_error(self):
        """Test that an exception is raised if capping_values_dict has any non list items."""

        x = CappingTransformer(capping_values={"a": [1, 3], "b": [None, -1]})

        with pytest.raises(
            TypeError,
            match=r"CappingTransformer: each item in ccc should be a list, but got \<class 'tuple'\> for key b",
        ):

            x.check_capping_values_dict(
                capping_values_dict={"a": [1, 3], "b": (None, -1)}, dict_name="ccc"
            )

    def test_capping_values_non_length_2_list_item_error(self):
        """Test that an exception is raised if capping_values_dict has any non length 2 list items."""

        x = CappingTransformer(capping_values={"a": [1, 3], "b": [None, -1]})

        with pytest.raises(
            ValueError,
            match="CappingTransformer: each item in ddd should be length 2, but got 1 for key b",
        ):

            x.check_capping_values_dict(
                capping_values_dict={"a": [1, 3], "b": [None]}, dict_name="ddd"
            )

    def test_capping_values_non_numeric_error(self):
        """Test that an exception is raised if capping_values_dict contains any non-nulls and non-numeric values."""

        x = CappingTransformer(capping_values={"a": [1, 3], "b": [None, -1]})

        with pytest.raises(
            TypeError,
            match=r"CappingTransformer: each item in eee lists must contain numeric values or None, got \<class 'str'\> for key a",
        ):

            x.check_capping_values_dict(
                capping_values_dict={"b": [1, 3], "a": [None, "a"]}, dict_name="eee"
            )

    def test_lower_value_gte_upper_value_error(self):
        """Test that an exception is raised if capping_values_dict[0] >= capping_values_dict[1]."""

        x = CappingTransformer(capping_values={"a": [1, 2], "b": [None, -1]})

        with pytest.raises(
            ValueError,
            match="CappingTransformer: lower value is greater than or equal to upper value for key a",
        ):

            x.check_capping_values_dict(
                capping_values_dict={"a": [4, 3], "b": [None, -1]}, dict_name="eee"
            )

    @pytest.mark.parametrize("value", [(np.NaN), (np.inf), (-np.inf)])
    def test_capping_value_nan_inf_error(self, value):
        """Test that an exception is raised if capping_values are np.nan or np.inf values."""

        x = CappingTransformer(capping_values={"a": [1, 3], "b": [None, 1]})

        with pytest.raises(
            ValueError,
            match="CappingTransformer: item in eee lists contains numpy NaN or Inf values",
        ):

            x.check_capping_values_dict(
                capping_values_dict={"b": [1, 3], "a": [None, value]}, dict_name="eee"
            )

    def test_capping_values_both_null_error(self):
        """Test that an exception is raised if both capping_values are null."""

        x = CappingTransformer(capping_values={"a": [1, 3], "b": [None, -1]})

        with pytest.raises(
            ValueError, match="CappingTransformer: both values are None for key a"
        ):

            x.check_capping_values_dict(
                capping_values_dict={"a": [None, None], "b": [None, 1]}, dict_name="eee"
            )


class TestFit(object):
    """Tests for CappingTransformer.fit()."""

    def test_arguments(self):
        """Test that fit has expected arguments."""

        ta.functions.test_function_arguments(
            func=CappingTransformer.fit,
            expected_arguments=["self", "X", "y"],
            expected_default_values=(None,),
        )

    def test_quantiles_none_error(self):
        """Test that an exception is raised if quantiles is None when fit is run."""

        with pytest.warns(
            UserWarning,
            match="CappingTransformer: quantiles not set so no fitting done in CappingTransformer",
        ):

            df = d.create_df_3()

            x = CappingTransformer(capping_values={"a": [2, 5], "b": [-1, 8]})

            x.fit(df)

    def test_super_fit_call(self, mocker):
        """Test the call to BaseTransformer.fit."""

        df = d.create_df_9()

        x = CappingTransformer(
            quantiles={"a": [0.1, 1], "b": [0.5, None]}, weights_column="c"
        )

        expected_call_args = {0: {"args": (d.create_df_9(), None), "kwargs": {}}}

        with ta.functions.assert_function_call(
            mocker, tubular.base.BaseTransformer, "fit", expected_call_args
        ):

            x.fit(df)

    def test_prepare_quantiles_call_weight(self, mocker):
        """Test the call to prepare_quantiles if weights_column is set."""

        df = d.create_df_9()

        x = CappingTransformer(
            quantiles={"a": [0.1, 1], "b": [0.5, None]}, weights_column="c"
        )

        expected_call_args = {
            0: {
                "args": (
                    d.create_df_9()["a"],
                    [0.1, 1],
                    d.create_df_9()["c"],
                ),
                "kwargs": {},
            },
            1: {
                "args": (
                    d.create_df_9()["b"],
                    [0.5, None],
                    d.create_df_9()["c"],
                ),
                "kwargs": {},
            },
        }

        with ta.functions.assert_function_call(
            mocker,
            tubular.capping.CappingTransformer,
            "prepare_quantiles",
            expected_call_args,
        ):

            x.fit(df)

    def test_prepare_quantiles_call_no_weight(self, mocker):
        """Test the call to prepare_quantiles if weights_column is not set."""

        df = d.create_df_9()

        x = CappingTransformer(quantiles={"a": [0.1, 1], "b": [0.5, None]})

        expected_call_args = {
            0: {
                "args": (d.create_df_9()["a"], [0.1, 1], None),
                "kwargs": {},
            },
            1: {
                "args": (d.create_df_9()["b"], [0.5, None], None),
                "kwargs": {},
            },
        }

        with ta.functions.assert_function_call(
            mocker,
            tubular.capping.CappingTransformer,
            "prepare_quantiles",
            expected_call_args,
        ):

            x.fit(df)

    @pytest.mark.parametrize("weights_column", [("c"), (None)])
    def test_prepare_quantiles_output_set_attributes(self, mocker, weights_column):
        """Test the output of prepare_quantiles is set to capping_values and_replacement_values attributes."""

        df = d.create_df_9()

        x = CappingTransformer(
            quantiles={"a": [0.1, 1], "b": [0.5, None]}, weights_column=weights_column
        )

        mocked_return_values = [["aaaa", "bbbb"], [1234, None]]

        mocker.patch(
            "tubular.capping.CappingTransformer.prepare_quantiles",
            side_effect=mocked_return_values,
        )

        x.fit(df)

        ta.classes.test_object_attributes(
            obj=x,
            expected_attributes={
                "capping_values": {
                    "a": mocked_return_values[0],
                    "b": mocked_return_values[1],
                },
                "_replacement_values": {
                    "a": mocked_return_values[0],
                    "b": mocked_return_values[1],
                },
            },
            msg="weighted_quantile output set to capping_values, _replacement_values attributes",
        )

    @pytest.mark.parametrize("weights_column", [(None), ("c")])
    @pytest.mark.parametrize("quantiles", [([0.2, 0.8]), ([None, 0.5]), ([0.6, None])])
    def test_quantile_combinations_handled(self, quantiles, weights_column):
        """Test that a given combination of None and non-None quantile values can be calculated successfully."""

        df = d.create_df_9()

        x = CappingTransformer(
            quantiles={"a": quantiles}, weights_column=weights_column
        )

        try:

            x.fit(df)

        except Exception as err:

            pytest.fail(
                f"unexpected exception when calling fit with quantiles {quantiles} - {err}"
            )


class TestPrepareQuantiles(object):
    """Tests for the CappingTransformer.prepare_quantiles method."""

    def test_arguments(self):
        """Test that transform has expected arguments."""

        ta.functions.test_function_arguments(
            func=CappingTransformer.prepare_quantiles,
            expected_arguments=["self", "values", "quantiles", "sample_weight"],
            expected_default_values=(None,),
        )

    @pytest.mark.parametrize(
        "values, quantiles, sample_weight, expected_quantiles",
        [
            (
                d.create_df_9()["a"],
                [0.1, 0.6],
                d.create_df_9()["c"],
                [0.1, 0.6],
            ),
            (
                d.create_df_9()["b"],
                [0.1, None],
                d.create_df_9()["c"],
                [0.1],
            ),
            (
                d.create_df_9()["a"],
                [None, 0.6],
                d.create_df_9()["c"],
                [0.6],
            ),
            (d.create_df_9()["b"], [0.1, 0.6], None, [0.1, 0.6]),
            (d.create_df_9()["a"], [0.1, None], None, [0.1]),
            (d.create_df_9()["b"], [None, 0.6], None, [0.6]),
        ],
    )
    def test_weighted_quantile_call(
        self, mocker, values, quantiles, sample_weight, expected_quantiles
    ):
        """Test the call to weighted_quantile, inlcuding the filtering out of None values."""

        x = CappingTransformer(quantiles={"a": [0.1, 1], "b": [0.5, None]})

        mocked = mocker.patch("tubular.capping.CappingTransformer.weighted_quantile")

        x.prepare_quantiles(values, quantiles, sample_weight)

        assert (
            mocked.call_count == 1
        ), f"unexpected number of calls to weighted_quantile, expecting 1 but got {mocked.call_count}"

        call_args = mocked.call_args_list[0]
        call_pos_args = call_args[0]
        call_kwargs = call_args[1]

        expected_pos_args = (values, expected_quantiles, sample_weight)

        assert (
            call_pos_args == expected_pos_args
        ), f"unexpected positional args in call to weighted_quantile, expecting {expected_pos_args} but got {call_pos_args}"

        assert (
            call_kwargs == {}
        ), f"unexpected kwargs in call to weighted_quantile, expecting None but got {call_kwargs}"

    @pytest.mark.parametrize(
        "values, quantiles, sample_weight, expected_results",
        [
            (
                d.create_df_9()["a"],
                [0.1, 0.6],
                d.create_df_9()["c"],
                ["aaaa"],
            ),
            (
                d.create_df_9()["b"],
                [0.1, None],
                d.create_df_9()["c"],
                ["aaaa", None],
            ),
            (
                d.create_df_9()["a"],
                [None, 0.6],
                d.create_df_9()["c"],
                [None, "aaaa"],
            ),
            (d.create_df_9()["b"], [0.1, 0.6], None, ["aaaa"]),
            (d.create_df_9()["a"], [0.1, None], None, ["aaaa", None]),
            (d.create_df_9()["b"], [None, 0.6], None, [None, "aaaa"]),
        ],
    )
    def test_output_from_weighted_quantile_returned(
        self, mocker, values, quantiles, sample_weight, expected_results
    ):
        """Test the output from weighted_quantile is returned from the function, inlcuding None values added back in."""

        x = CappingTransformer(quantiles={"a": [0.1, 1], "b": [0.5, None]})

        mocker.patch(
            "tubular.capping.CappingTransformer.weighted_quantile",
            return_value=["aaaa"],
        )

        results = x.prepare_quantiles(values, quantiles, sample_weight)

        assert (
            results == expected_results
        ), f"unexpected value returned from prepare_quantiles, expecting {results} but got {expected_results}"


class TestTransform(object):
    """Tests for CappingTransformer.transform()."""

    def expected_df_1():
        """Expected output from test_expected_output_min_and_max."""

        df = pd.DataFrame(
            {
                "a": [2, 2, 3, 4, 5, 5, np.NaN],
                "b": [1, 2, 3, np.NaN, 7, 7, 7],
                "c": [np.NaN, 1, 2, 3, 0, 0, 0],
            }
        )

        return df

    def expected_df_2():
        """Expected output from test_expected_output_max."""

        df = pd.DataFrame(
            {
                "a": [2, 2, 3, 4, 5, 6, 7, np.NaN],
                "b": ["a", "b", "c", "d", "e", "f", "g", np.NaN],
                "c": ["a", "b", "c", "d", "e", "f", "g", np.NaN],
            }
        )

        df["c"] = df["c"].astype("category")

        return df

    def test_arguments(self):
        """Test that transform has expected arguments."""

        ta.functions.test_function_arguments(
            func=CappingTransformer.transform, expected_arguments=["self", "X"]
        )

    def test_check_is_fitted_call_count(self, mocker):
        """Test there are 2 calls to BaseTransformer check_is_fitted in transform."""

        df = d.create_df_3()

        x = CappingTransformer(capping_values={"a": [2, 5], "b": [-1, 8]})

        with ta.functions.assert_function_call_count(
            mocker, tubular.base.BaseTransformer, "check_is_fitted", 2
        ):

            x.transform(df)

    def test_check_is_fitted_call_1(self, mocker):
        """Test the first call to BaseTransformer check_is_fitted in transform."""

        df = d.create_df_3()

        x = CappingTransformer(capping_values={"a": [2, 5], "b": [-1, 8]})

        expected_call_args = {
            0: {"args": (["capping_values"],), "kwargs": {}},
            1: {"args": (["_replacement_values"],), "kwargs": {}},
        }

        with ta.functions.assert_function_call(
            mocker, tubular.base.BaseTransformer, "check_is_fitted", expected_call_args
        ):

            x.transform(df)

    def test_super_transform_called(self, mocker):
        """Test that BaseTransformer.transform called."""

        df = d.create_df_3()

        x = CappingTransformer(capping_values={"a": [2, 5], "b": [-1, 8]})

        expected_call_args = {0: {"args": (d.create_df_3(),), "kwargs": {}}}

        with ta.functions.assert_function_call(
            mocker,
            tubular.base.BaseTransformer,
            "transform",
            expected_call_args,
            return_value=d.create_df_3(),
        ):

            x.transform(df)

    def test_learnt_values_not_modified(self):
        """Test that the replacements from fit are not changed in transform."""

        capping_values_dict = {"a": [2, 5], "b": [-1, 8]}

        df = d.create_df_3()

        x = CappingTransformer(capping_values_dict)

        x.transform(df)

        ta.classes.test_object_attributes(
            obj=x,
            expected_attributes={"capping_values": capping_values_dict},
            msg="Attributes for CappingTransformer set in init",
        )

    @pytest.mark.parametrize(
        "df, expected",
        ta.pandas.adjusted_dataframe_params(d.create_df_3(), expected_df_1()),
    )
    def test_expected_output_min_and_max_combinations(self, df, expected):
        """Test that capping is applied correctly in transform."""

        x = CappingTransformer(
            capping_values={"a": [2, 5], "b": [None, 7], "c": [0, None]}
        )

        df_transformed = x.transform(df)

        ta.equality.assert_frame_equal_msg(
            actual=df_transformed,
            expected=expected,
            msg_tag="Unexpected values in CappingTransformer.transform",
        )

    @pytest.mark.parametrize(
        "df, expected",
        ta.pandas.adjusted_dataframe_params(d.create_df_4(), expected_df_2()),
    )
    def test_non_cap_column_left_untouched(self, df, expected):
        """Test that capping is applied only to specific columns, others remain the same."""

        x = CappingTransformer(capping_values={"a": [2, 10]})

        df_transformed = x.transform(df)

        ta.equality.assert_frame_equal_msg(
            actual=df_transformed,
            expected=expected,
            msg_tag="Unexpected values in CappingTransformer.transform, with columns meant to not be transformed",
        )

    def test_non_numeric_column_error(self):
        """Test that transform will raise an error if a column to transform is not numeric."""

        df = d.create_df_5()

        x = CappingTransformer(capping_values={"a": [2, 5], "b": [-1, 8], "c": [-1, 8]})

        with pytest.raises(
            TypeError,
            match=r"CappingTransformer: The following columns are not numeric in X; \['b', 'c'\]",
        ):

            x.transform(df)

    def test_quantile_not_fit_error(self):
        """Test that transform will raise an error if quantiles are specified in init but fit is not run before calling transform."""

        df = d.create_df_9()

        x = CappingTransformer(quantiles={"a": [0.2, 1], "b": [0, 1]})

        with pytest.raises(
            ValueError,
            match="CappingTransformer: capping_values attribute is an empty dict - perhaps the fit method has not been run yet",
        ):

            x.transform(df)

    def test_replacement_values_dict_not_set_error(self):
        """Test that transform will raise an error if _replacement_values is an empty dict."""

        df = d.create_df_9()

        x = CappingTransformer(quantiles={"a": [0.2, 1], "b": [0, 1]})

        # manually set attribute to get past the capping_values attribute is an empty dict exception
        x.capping_values = {"a": [1, 4]}

        with pytest.raises(
            ValueError,
            match="CappingTransformer: _replacement_values attribute is an empty dict - perhaps the fit method has not been run yet",
        ):

            x.transform(df)

    def test_attributes_unchanged_from_transform(self):
        """Test that attributes are unchanged after transform is run."""

        df = d.create_df_9()

        x = CappingTransformer(quantiles={"a": [0.2, 1], "b": [0, 1]})

        x.fit(df)

        x2 = CappingTransformer(quantiles={"a": [0.2, 1], "b": [0, 1]})

        x2.fit(df)

        x2.transform(df)

        assert (
            x.capping_values == x2.capping_values
        ), "capping_values attribute modified in transform"
        assert (
            x._replacement_values == x2._replacement_values
        ), "_replacement_values attribute modified in transform"
        assert (
            x.weights_column == x2.weights_column
        ), "weights_column attribute modified in transform"
        assert x.quantiles == x2.quantiles, "quantiles attribute modified in transform"


class TestWeightedQuantile(object):
    """Tests for the CappingTransformer.weighted_quantile method."""

    def test_arguments(self):
        """Test that transform has expected arguments."""

        ta.functions.test_function_arguments(
            func=CappingTransformer.weighted_quantile,
            expected_arguments=["self", "values", "quantiles", "sample_weight"],
            expected_default_values=(None,),
        )

    @pytest.mark.parametrize(
        "values, sample_weight, quantiles, expected_quantiles",
        [
            (
                [1, 2, 3],
                [1, 1, 1],
                [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 3.0],
            ),
            (
                [1, 2, 3],
                [0, 1, 0],
                [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
            ),
            (
                [1, 2, 3],
                [1, 1, 0],
                [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0],
            ),
            (
                [1, 2, 3, 4, 5],
                [1, 1, 1, 1, 1],
                [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                [1.0, 1.0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
            ),
            ([1, 2, 3, 4, 5], [1, 0, 1, 0, 1], [0, 0.5, 1.0], [1.0, 2.0, 5.0]),
        ],
    )
    def test_expected_output(
        self, values, sample_weight, quantiles, expected_quantiles
    ):
        """Test that weighted_quantile gives the expected outputs."""

        x = CappingTransformer(capping_values={"a": [2, 10]})

        values = pd.Series(values)

        actual = x.weighted_quantile(values, quantiles, sample_weight)

        # round to 1dp to avoid mismatches due to numerical precision
        actual_rounded_1_dp = list(np.round(actual, 1))

        assert (
            actual_rounded_1_dp == expected_quantiles
        ), "unexpected weighted quantiles calculated"

    def test_zero_total_weight_error(self):
        """Test that an exception is raised if the total sample weights are 0."""

        x = CappingTransformer(capping_values={"a": [2, 10]})

        with pytest.raises(
            ValueError,
            match="CappingTransformer: total sample weights are not greater than 0",
        ):

            x.weighted_quantile([2, 3, 4, 5], [0, 1], [0, 0])

    def test_null_values_in_weights_error(self):
        """Test that an exception is raised if there are null values in sample_weight."""

        x = CappingTransformer(capping_values={"a": [2, 10]})

        with pytest.raises(
            ValueError, match="CappingTransformer: null values in sample weights"
        ):

            x.weighted_quantile([2, 3, 4, 5], [0, 1], [3, np.NaN])

    def test_inf_values_in_weights_error(self):
        """Test that an exception is raised if there are inf values in sample_weight."""

        x = CappingTransformer(capping_values={"a": [2, 10]})

        with pytest.raises(
            ValueError, match="CappingTransformer: infinite values in sample weights"
        ):

            x.weighted_quantile([2, 3, 4, 5], [0, 1], [2, np.inf])

        with pytest.raises(
            ValueError, match="CappingTransformer: infinite values in sample weights"
        ):

            x.weighted_quantile([2, 3, 4, 5], [0, 1], [1, -np.inf])

    def test_negative_values_in_weights_error(self):
        """Test that an exception is raised if there are negative values in sample_weight."""

        x = CappingTransformer(capping_values={"a": [2, 10]})

        with pytest.raises(
            ValueError, match="CappingTransformer: negative weights in sample weights"
        ):

            x.weighted_quantile([2, 3, 4, 5], [0, 1], [2, -0.01])
