import pytest
import test_aide as ta
import tests.test_data as d
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler

import tubular
from tubular.numeric import ScalingTransformer


class TestInit(object):
    """Tests for ScalingTransformer.init()."""

    def test_arguments(self):
        """Test that init has expected arguments."""

        ta.functions.test_function_arguments(
            func=ScalingTransformer.__init__,
            expected_arguments=["self", "columns", "scaler", "scaler_kwargs"],
            expected_default_values=({},),
        )

    def test_inheritance(self):
        """Test that ScalingTransformer inherits from BaseTransformer."""

        x = ScalingTransformer(columns=["a"], scaler="standard")

        ta.classes.assert_inheritance(x, tubular.base.BaseTransformer)

    def test_to_scaler_kwargs_type_error(self):
        """Test that an exception is raised if scaler_kwargs is not a dict."""

        with pytest.raises(
            TypeError,
            match=r"""scaler_kwargs should be a dict but got type \<class 'int'\>""",
        ):

            ScalingTransformer(columns="b", scaler="standard", scaler_kwargs=1)

    def test_scaler_kwargs_key_type_error(self):
        """Test that an exception is raised if scaler_kwargs has keys which are not str."""

        with pytest.raises(
            TypeError,
            match=r"""unexpected type \(\<class 'int'\>\) for scaler_kwargs key in position 1, must be str""",
        ):

            ScalingTransformer(
                columns="b",
                scaler="standard",
                scaler_kwargs={"a": 1, 2: "b"},
            )

    def test_to_scaler_non_allowed_value_error(self):
        """Test that an exception is raised if scaler is not one of the allowed values."""

        with pytest.raises(
            ValueError,
            match=r"""scaler should be one of; \['min_max', 'max_abs', 'standard'\]""",
        ):

            ScalingTransformer(columns="b", scaler="zzz", scaler_kwargs={"a": 1})

    @pytest.mark.parametrize(
        "scaler, scaler_type",
        [
            ("min_max", MinMaxScaler),
            ("max_abs", MaxAbsScaler),
            ("standard", StandardScaler),
        ],
    )
    def test_scaler_attribute_type(self, scaler, scaler_type):
        """Test that the scaler attribute is set to the correct type given what is passed when initialising the transformer."""

        x = ScalingTransformer(columns="b", scaler=scaler)

        assert (
            type(x.scaler) is scaler_type
        ), f"unexpected scaler set in init for {scaler}"

    @pytest.mark.parametrize(
        "scaler, scaler_type_str, scaler_kwargs_value",
        [
            ("min_max", "MinMaxScaler", {"copy": False, "feature_range": (0.5, 1.5)}),
            ("max_abs", "MaxAbsScaler", {"copy": False}),
            (
                "standard",
                "StandardScaler",
                {"copy": False, "with_mean": True, "with_std": True},
            ),
        ],
    )
    def test_scaler_initialised_with_scaler_kwargs(
        self, mocker, scaler, scaler_type_str, scaler_kwargs_value
    ):
        """Test that the scaler is initialised with the scaler_kwargs arguments."""

        mocked = mocker.patch(
            f"sklearn.preprocessing.{scaler_type_str}.__init__", return_value=None
        )

        ScalingTransformer(
            columns="b", scaler=scaler, scaler_kwargs=scaler_kwargs_value
        )

        assert mocked.call_count == 1, "unexpected number of calls to init"

        call_args = mocked.call_args_list[0]
        call_pos_args = call_args[0]
        call_kwargs = call_args[1]

        assert (
            call_pos_args == ()
        ), f"unexpected positional args in {scaler_type_str} init call"

        assert (
            call_kwargs == scaler_kwargs_value
        ), f"unexpected kwargs in {scaler_type_str} init call"

    def test_super_init_called(self, mocker):
        """Test that super.__init__ called."""

        expected_call_args = {
            0: {
                "args": (),
                "kwargs": {"columns": ["a", "b"], "copy": True, "verbose": False},
            }
        }

        with ta.functions.assert_function_call(
            mocker, tubular.base.BaseTransformer, "__init__", expected_call_args
        ):

            ScalingTransformer(
                columns=["a", "b"], scaler="standard", copy=True, verbose=False
            )


class TestCheckNumericColumns(object):
    """Tests for the check_numeric_columns method."""

    def test_arguments(self):
        """Test that check_numeric_columns has expected arguments."""

        ta.functions.test_function_arguments(
            func=ScalingTransformer.check_numeric_columns,
            expected_arguments=["self", "X"],
            expected_default_values=None,
        )

    def test_exception_raised(self):
        """Test an exception is raised if non numeric columns are passed in X."""

        df = d.create_df_2()

        x = ScalingTransformer(columns=["a", "b", "c"], scaler="standard")

        with pytest.raises(
            TypeError,
            match=r"""The following columns are not numeric in X; \['b', 'c'\]""",
        ):

            x.check_numeric_columns(df)

    def test_X_returned(self):
        """Test that the input X is returned from the method."""

        df = d.create_df_2()

        x = ScalingTransformer(columns=["a"], scaler="standard")

        df_returned = x.check_numeric_columns(df)

        ta.equality.assert_equal_dispatch(
            expected=df,
            actual=df_returned,
            msg="unexepcted object returned from check_numeric_columns",
        )


class TestFit(object):
    """Tests for ScalingTransformer.fit()."""

    def test_arguments(self):
        """Test that fit has expected arguments."""

        ta.functions.test_function_arguments(
            func=ScalingTransformer.fit,
            expected_arguments=["self", "X", "y"],
            expected_default_values=(None,),
        )

    def test_super_fit_call(self, mocker):
        """Test the call to BaseTransformer.fit."""

        df = d.create_df_2()

        x = ScalingTransformer(columns=["a"], scaler="standard")

        expected_call_args = {0: {"args": (d.create_df_2(), None), "kwargs": {}}}

        with ta.functions.assert_function_call(
            mocker, tubular.base.BaseTransformer, "fit", expected_call_args
        ):

            x.fit(df)

    def test_check_numeric_columns_call(self, mocker):
        """Test the call to ScalingTransformer.check_numeric_columns."""

        df = d.create_df_2()

        x = ScalingTransformer(columns=["a"], scaler="standard")

        expected_call_args = {0: {"args": (d.create_df_2(),), "kwargs": {}}}

        with ta.functions.assert_function_call(
            mocker,
            tubular.numeric.ScalingTransformer,
            "check_numeric_columns",
            expected_call_args,
            return_value=d.create_df_2(),
        ):

            x.fit(df)

    @pytest.mark.parametrize(
        "scaler, scaler_type_str",
        [
            ("min_max", "MinMaxScaler"),
            ("max_abs", "MaxAbsScaler"),
            ("standard", "StandardScaler"),
        ],
    )
    def test_scaler_fit_call(self, mocker, scaler, scaler_type_str):
        """Test that the call to the scaler.fit method."""

        df = d.create_df_3()

        x = ScalingTransformer(
            columns=["b", "c"], scaler=scaler, scaler_kwargs={"copy": True}
        )

        mocked = mocker.patch(
            f"sklearn.preprocessing.{scaler_type_str}.fit", return_value=None
        )

        x.fit(df)

        assert mocked.call_count == 1, "unexpected number of calls to scaler fit"

        call_args = mocked.call_args_list[0]
        call_pos_args = call_args[0]
        call_kwargs = call_args[1]

        expected_positional_args = (df[["b", "c"]],)

        ta.equality.assert_equal_dispatch(
            expected=expected_positional_args,
            actual=call_pos_args,
            msg=f"unexpected positional args in {scaler_type_str} fit call",
        )

        assert call_kwargs == {}, f"unexpected kwargs in {scaler_type_str} fit call"

    def test_return_self(self):
        """Test that fit returns self."""

        df = d.create_df_2()

        x = ScalingTransformer(columns=["a"], scaler="standard")

        x_fitted = x.fit(df)

        assert (
            x_fitted is x
        ), "return value from ScalingTransformer.fit not as expected (self)."


class TestTransform(object):
    """Tests for ScalingTransformer.transform()."""

    def test_arguments(self):
        """Test that transform has expected arguments."""

        ta.functions.test_function_arguments(
            func=ScalingTransformer.transform,
            expected_arguments=["self", "X"],
            expected_default_values=None,
        )

    def test_super_transform_called(self, mocker):
        """Test that BaseTransformer.transform called."""

        df = d.create_df_2()

        x = ScalingTransformer(columns=["a"], scaler="standard")

        x.fit(df)

        expected_call_args = {0: {"args": (d.create_df_2(),), "kwargs": {}}}

        with ta.functions.assert_function_call(
            mocker,
            tubular.base.BaseTransformer,
            "transform",
            expected_call_args,
            return_value=d.create_df_2(),
        ):

            x.transform(df)

    def test_check_numeric_columns_call(self, mocker):
        """Test the call to ScalingTransformer.check_numeric_columns."""

        df = d.create_df_2()

        x = ScalingTransformer(columns=["a"], scaler="standard", copy=True)

        x.fit(df)

        expected_call_args = {0: {"args": (d.create_df_2(),), "kwargs": {}}}

        with ta.functions.assert_function_call(
            mocker,
            tubular.base.BaseTransformer,
            "transform",
            expected_call_args,
            return_value=d.create_df_2(),
        ):

            x.transform(df)

    @pytest.mark.parametrize(
        "scaler, scaler_type_str",
        [
            ("min_max", "MinMaxScaler"),
            ("max_abs", "MaxAbsScaler"),
            ("standard", "StandardScaler"),
        ],
    )
    def test_scaler_transform_call(self, mocker, scaler, scaler_type_str):
        """Test that the call to the scaler.transform method."""

        df = d.create_df_3()

        x = ScalingTransformer(
            columns=["b", "c"], scaler=scaler, scaler_kwargs={"copy": True}
        )

        x.fit(df)

        mocked = mocker.patch(
            f"sklearn.preprocessing.{scaler_type_str}.transform",
            return_value=df[["b", "c"]],
        )

        x.transform(df)

        assert mocked.call_count == 1, "unexpected number of calls to scaler fit"

        call_args = mocked.call_args_list[0]
        call_pos_args = call_args[0]
        call_kwargs = call_args[1]

        expected_positional_args = (df[["b", "c"]],)

        ta.equality.assert_equal_dispatch(
            expected=expected_positional_args,
            actual=call_pos_args,
            msg=f"unexpected positional args in {scaler_type_str} transform call",
        )

        assert (
            call_kwargs == {}
        ), f"unexpected kwargs in {scaler_type_str} transform call"

    @pytest.mark.parametrize(
        "scaler, scaler_type_str",
        [
            ("min_max", "MinMaxScaler"),
            ("max_abs", "MaxAbsScaler"),
            ("standard", "StandardScaler"),
        ],
    )
    def test_output_from_scaler_transform_set_to_columns(
        self, mocker, scaler, scaler_type_str
    ):
        """Test that the call to the scaler.transform method."""

        df = d.create_df_3()

        x = ScalingTransformer(
            columns=["b", "c"], scaler=scaler, scaler_kwargs={"copy": True}
        )

        x.fit(df)

        scaler_transform_output = pd.DataFrame(
            {"b": [1, 2, 3, 4, 5, 6, 7], "c": [7, 6, 5, 4, 3, 2, 1]}
        )

        mocker.patch(
            f"sklearn.preprocessing.{scaler_type_str}.transform",
            return_value=scaler_transform_output,
        )

        df_transformed = x.transform(df)

        ta.equality.assert_equal_dispatch(
            expected=scaler_transform_output,
            actual=df_transformed[["b", "c"]],
            msg=f"output from {scaler_type_str} transform not assigned to columns",
        )

    @pytest.mark.parametrize("columns", [("b"), ("c"), (["b", "c"])])
    @pytest.mark.parametrize(
        "scaler, scaler_type_str",
        [
            ("min_max", "MinMaxScaler"),
            ("max_abs", "MaxAbsScaler"),
            ("standard", "StandardScaler"),
        ],
    )
    def test_return_type(self, scaler, scaler_type_str, columns):
        """Test that transform returns a pd.DataFrame."""

        df = d.create_df_3()

        x = ScalingTransformer(
            columns=columns, scaler=scaler, scaler_kwargs={"copy": True}
        )

        x.fit(df)

        df_transformed = x.transform(df)

        assert (
            type(df_transformed) is pd.DataFrame
        ), "unexpected output type from transform"
