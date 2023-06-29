import pytest
import test_aide as ta

import tests.test_data as d
import tubular
from tubular.imputers import ArbitraryImputer


class TestInit(object):
    """Tests for ArbitraryImputer.init()."""

    def test_arguments(self):
        """Test that init has expected arguments."""

        ta.functions.test_function_arguments(
            func=ArbitraryImputer.__init__,
            expected_arguments=["self", "impute_value", "columns"],
            expected_default_values=None,
        )

    def test_class_methods(self):
        """Test that ArbitraryImputer has transform method."""

        x = ArbitraryImputer(impute_value=1, columns="a")

        ta.classes.test_object_method(
            obj=x, expected_method="transform", msg="transform"
        )

    def test_inheritance(self):
        """Test that ArbitraryImputer inherits from BaseTransformer."""

        x = ArbitraryImputer(impute_value=1, columns="a")

        ta.classes.assert_inheritance(x, tubular.imputers.BaseImputer)

    def test_super_init_called(self, mocker):
        """Test that init calls BaseTransformer.init."""

        expected_call_args = {
            0: {"args": (), "kwargs": {"columns": "a", "verbose": True, "copy": True}}
        }

        with ta.functions.assert_function_call(
            mocker, tubular.base.BaseTransformer, "__init__", expected_call_args
        ):
            ArbitraryImputer(impute_value=1, columns="a", verbose=True, copy=True)

    def test_columns_none_error(self):
        """Test that an exception is raised if columns is passed as None."""

        with pytest.raises(
            ValueError,
            match="ArbitraryImputer: columns must be specified in init for ArbitraryImputer",
        ):
            ArbitraryImputer(impute_value=1, columns=None)

    def test_impute_value_type_error(self):
        """Test that an exception is raised if impute_value is not an int, float or str."""

        with pytest.raises(
            ValueError,
            match="ArbitraryImputer: impute_value should be a single value .*",
        ):
            ArbitraryImputer(impute_value={}, columns="a")

    def test_impute_values_set_to_attribute(self):
        """Test that the value passed for impute_value is saved in an attribute of the same name."""

        value = 1

        x = ArbitraryImputer(impute_value=value, columns="a")

        ta.classes.test_object_attributes(
            obj=x,
            expected_attributes={"impute_value": value, "impute_values_": {}},
            msg="Attributes for ArbitraryImputer set in init",
        )


class TestTransform(object):
    """Tests for ArbitraryImputer.transform()."""

    def test_arguments(self):
        """Test that transform has expected arguments."""

        ta.functions.test_function_arguments(
            func=ArbitraryImputer.transform, expected_arguments=["self", "X"]
        )

    def test_check_is_fitted_called(self, mocker):
        """Test that BaseTransformer check_is_fitted called."""

        df = d.create_df_1()

        x = ArbitraryImputer(impute_value=1, columns="a")

        expected_call_args = {0: {"args": (["impute_value"],), "kwargs": {}}}

        with ta.functions.assert_function_call(
            mocker, tubular.base.BaseTransformer, "check_is_fitted", expected_call_args
        ):
            x.transform(df)

    def test_super_transform_called(self, mocker):
        """Test that BaseImputer.transform called."""

        df = d.create_df_2()

        x = ArbitraryImputer(impute_value=1, columns="a")

        expected_call_args = {0: {"args": (d.create_df_2(),), "kwargs": {}}}

        with ta.functions.assert_function_call(
            mocker, tubular.imputers.BaseImputer, "transform", expected_call_args
        ):
            x.transform(df)

    def test_impute_values_set(self, mocker):
        """Test that impute_values_ are set with imput_value in transform."""

        df = d.create_df_2()

        x = ArbitraryImputer(impute_value=1, columns=["a", "b", "c"])

        # mock BaseImputer.transform so it does not run
        mocker.patch.object(
            tubular.imputers.BaseImputer, "transform", return_value=1234
        )

        x.transform(df)

        assert x.impute_values_ == {
            "a": 1,
            "b": 1,
            "c": 1,
        }, "impute_values_ not set with imput_value in transform"

    def test_impute_value_unchanged(self):
        """Test that self.impute_value is unchanged after transform."""

        df = d.create_df_1()

        value = 1

        x = ArbitraryImputer(impute_value=value, columns="a")

        x.transform(df)

        ta.classes.test_object_attributes(
            obj=x,
            expected_attributes={"impute_value": value},
            msg="impute_value changed in transform",
        )

    def test_super_columns_check_called(self, mocker):
        """Test that BaseTransformer.columns_check called."""

        df = d.create_df_2()

        x = ArbitraryImputer(impute_value=-1, columns="a")

        expected_call_args = {0: {"args": (d.create_df_2(),), "kwargs": {}}}

        with ta.functions.assert_function_call(
            mocker, tubular.base.BaseTransformer, "columns_check", expected_call_args
        ):
            x.transform(df)

    # Unit testing to check if downcast datatypes of columns is preserved after imputation is done
    def test_impute_value_preserve_dtype(self):
        """Testing downcast dtypes of columns are preserved after imputation using the create_downcast_df dataframe.

        Explicitly setting the dtype of "a" to int8 and "b" to float16 and check if the dtype of the columns are preserved after imputation.
        """
        df = (
            d.create_downcast_df()
        )  # By default the dtype of "a" and "b" are int64 and float64 respectively

        # Imputing the dataframe
        x = ArbitraryImputer(impute_value=1, columns=["a", "b"])

        # Setting the dtype of "a" to int8 and "b" to float16
        df["a"] = df["a"].astype("int8")
        df["b"] = df["b"].astype("float16")

        # Checking if the dtype of "a" and "b" are int8 and float16 respectively
        assert df["a"].dtype == "int8"
        assert df["b"].dtype == "float16"

        # Impute the dataframe
        df = x.transform(df)

        # Checking if the dtype of "a" and "b" are int8 and float16 respectively after imputation
        assert df["a"].dtype == "int8"
        assert df["b"].dtype == "float16"
