import pytest
import test_aide as ta
import tests.test_data as d
from unittest import mock
import pandas
import re

import tubular
from tubular.base import BaseTransformer


class TestInit(object):
    """Tests for BaseTransformer.__init__()."""

    def test_arguments(self):
        """Test that init has expected arguments."""

        ta.functions.test_function_arguments(
            func=BaseTransformer.__init__,
            expected_arguments=["self", "columns", "copy", "verbose"],
            expected_default_values=(None, True, False),
        )

    def test_default_attributes_set_in_init(self):
        """Test correct default values for attributes set in init."""

        x = BaseTransformer()

        expected_attributes = {
            "version_": tubular._version.__version__,
            "verbose": False,
            "columns": None,
            "copy": True,
        }

        ta.classes.test_object_attributes(
            obj=x,
            expected_attributes=expected_attributes,
            msg="Default attributes set in init",
        )

    def test_attributes_set_from_passed_values(self):
        """Test attributes set from values passed in init have the correct values."""

        expected_attributes = {
            "columns": ["a", "b", "c"],
            "copy": False,
            "verbose": True,
        }

        x = BaseTransformer(**expected_attributes)

        ta.classes.test_object_attributes(
            obj=x,
            expected_attributes=expected_attributes,
            msg="Attributes set in init from passed values",
        )

    def test_columns_str_to_list(self):
        """Test columns is converted to list if passed as string."""

        x = BaseTransformer(columns="a")

        expected_attributes = {"columns": ["a"]}

        ta.classes.test_object_attributes(
            obj=x,
            expected_attributes=expected_attributes,
            msg="String put in list for columns",
        )

    def test_class_methods(self):
        """Test that BaseTransformer has fit and transform methods."""

        x = BaseTransformer()

        ta.classes.test_object_method(obj=x, expected_method="fit", msg="fit")

        ta.classes.test_object_method(
            obj=x, expected_method="transform", msg="transform"
        )

        ta.classes.test_object_method(
            obj=x, expected_method="columns_set_or_check", msg="columns_set_or_check"
        )

        ta.classes.test_object_method(
            obj=x, expected_method="columns_check", msg="columns_check"
        )

    def test_verbose_non_bool_error(self):
        """Test an error is raised if verbose is not specified as a bool."""

        with pytest.raises(TypeError, match="verbose must be a bool"):

            BaseTransformer(verbose=1)

    def test_copy_non_bool_error(self):
        """Test an error is raised if copy is not specified as a bool."""

        with pytest.raises(TypeError, match="copy must be a bool"):

            BaseTransformer(copy=1)

    def test_columns_empty_list_error(self):
        """Test an error is raised if columns is specified as an empty list."""

        with pytest.raises(ValueError):

            BaseTransformer(columns=[])

    def test_columns_list_element_error(self):
        """Test an error is raised if columns list contains non-string elements."""

        with pytest.raises(
            TypeError,
            match=re.escape(
                "each element of columns should be a single (string) column name"
            ),
        ):

            BaseTransformer(columns=[[], "a"])

    def test_columns_non_string_error(self):
        """Test an error is raised if columns is not passed as a string."""

        with pytest.raises(
            TypeError,
            match=re.escape(
                "columns must be a string or list with the columns to be pre-processed (if specified)"
            ),
        ):

            BaseTransformer(columns=1)


class TestFit(object):
    """Tests for BaseTransformer.fit()."""

    def test_arguments(self):
        """Test that fit has expected arguments."""

        ta.functions.test_function_arguments(
            func=BaseTransformer.fit,
            expected_arguments=["self", "X", "y"],
            expected_default_values=(None,),
        )

    def test_fit_returns_self(self):
        """Test fit returns self?"""

        df = d.create_df_1()

        x = BaseTransformer(columns="a")

        x_fitted = x.fit(df)

        assert x_fitted is x, "Returned value from BaseTransformer.fit not as expected."

    def test_X_non_df_error(self):
        """Test an error is raised if X is not passed as a pd.DataFrame."""

        x = BaseTransformer(columns="a")

        with pytest.raises(TypeError, match="X should be a pd.DataFrame"):

            x.fit("a")

    def test_non_pd_type_error(self):
        """Test an error is raised if y is not passed as a pd.Series."""

        df = d.create_df_1()

        x = BaseTransformer(columns="a")

        with pytest.raises(
            TypeError, match="unexpected type for y, should be a pd.Series"
        ):

            x.fit(X=df, y=[1, 2, 3, 4, 5, 6])

    def test_columns_set_or_check_called(self, mocker):
        """Test that self.columns_set_or_check is called during fit."""

        df = d.create_df_1()

        x = BaseTransformer(columns="a")

        expected_call_args = {0: {"args": (df,), "kwargs": {}}}

        with ta.functions.assert_function_call(
            mocker,
            tubular.base.BaseTransformer,
            "columns_set_or_check",
            expected_call_args,
        ):

            x.fit(X=df)

    def test_X_no_rows_error(self):
        """Test an error is raised if X has no rows."""

        x = BaseTransformer(columns="a")

        df = pandas.DataFrame(columns=["a"])

        with pytest.raises(ValueError, match=re.escape("X has no rows; (0, 1)")):

            x.fit(X=df)

    def test_y_no_rows_error(self):
        """Test an error is raised if X has no rows."""

        x = BaseTransformer(columns="a")

        df = pandas.DataFrame({"a": 1}, index=[0])

        with pytest.raises(ValueError, match=re.escape("y is empty; (0,)")):

            x.fit(X=df, y=pandas.Series(name="b", dtype=object))

    def test_unexpected_kwarg_error(self):

        with pytest.raises(
            TypeError,
            match=re.escape(
                "__init__() got an unexpected keyword argument 'unexpected_kwarg'"
            ),
        ):

            BaseTransformer(columns="a", unexpected_kwarg="spanish inquisition")


class TestTransform(object):
    """Tests for BaseTransformer.transform()."""

    def test_arguments(self):
        """Test that transform has expected arguments."""

        ta.functions.test_function_arguments(
            func=BaseTransformer.transform, expected_arguments=["self", "X"]
        )

    def test_columns_check_called(self, mocker):
        """Test that self.columns_check is called during transform."""

        df = d.create_df_1()

        x = BaseTransformer(columns="a")

        expected_call_args = {0: {"args": (df,), "kwargs": {}}}

        with ta.functions.assert_function_call(
            mocker, tubular.base.BaseTransformer, "columns_check", expected_call_args
        ):

            x.transform(X=df)

    def test_non_pd_type_error(self):
        """Test an error is raised if y is not passed as a pd.DataFrame."""

        x = BaseTransformer(columns="a")

        with pytest.raises(TypeError, match="X should be a pd.DataFrame"):

            x.transform(X=[1, 2, 3, 4, 5, 6])

    def test_df_copy_called(self, mocker):
        """Test pd.DataFrame.copy is called if copy is True."""

        df = d.create_df_1()

        x = BaseTransformer(columns="a", copy=True)

        expected_call_args = {0: {"args": (), "kwargs": {}}}

        with ta.functions.assert_function_call(
            mocker, pandas.DataFrame, "copy", expected_call_args, return_value=df
        ):

            x.transform(X=df)

    def test_no_rows_error(self):
        """Test an error is raised if X has no rows."""

        x = BaseTransformer(columns="a")

        df = pandas.DataFrame(columns=["a"])

        with pytest.raises(ValueError, match=re.escape("X has no rows; (0, 1)")):

            x.transform(df)

    @pytest.mark.parametrize(
        "df, expected",
        ta.pandas.adjusted_dataframe_params(d.create_df_1(), d.create_df_1()),
    )
    def test_X_returned(self, df, expected):
        """Test that X is returned from transform."""

        x = BaseTransformer(columns="a", copy=True)

        df_transformed = x.transform(X=df)

        ta.equality.assert_equal_dispatch(
            expected=expected,
            actual=df_transformed,
            msg="Check X returned from transform",
        )


class TestColumnsCheck(object):
    """Tests for columns_check method."""

    def test_arguments(self):
        """Test that columns_check has expected arguments."""

        ta.functions.test_function_arguments(
            func=BaseTransformer.columns_check, expected_arguments=["self", "X"]
        )

    def test_non_pd_df_error(self):
        """Test an error is raised if X is not passed as a pd.DataFrame."""

        x = BaseTransformer(columns="a")

        with pytest.raises(TypeError, match="X should be a pd.DataFrame"):

            x.columns_check(X=[1, 2, 3, 4, 5, 6])

    def test_columns_none_error(self):
        """Test an error is raised if self.columns is None."""

        df = d.create_df_1()

        x = BaseTransformer(columns=None)

        assert x.columns is None, f"self.columns should be None but got {x.columns}"

        with pytest.raises(ValueError):

            x.columns_check(X=df)

    def test_columns_str_error(self):
        """Test an error is raised if self.columns is not a list."""

        df = d.create_df_1()

        x = BaseTransformer(columns=None)

        x.columns = "a"

        with pytest.raises(TypeError, match="self.columns should be a list"):

            x.columns_check(X=df)

    def test_columns_not_in_X_error(self):
        """Test an error is raised if self.columns contains a value not in X."""

        df = d.create_df_1()

        x = BaseTransformer(columns=["a", "z"])

        with pytest.raises(ValueError):

            x.columns_check(X=df)


class TestColumnsSetOrCheck(object):
    """Tests for columns_set_or_check method."""

    def test_arguments(self):
        """Test that columns_set_or_check has expected arguments."""

        ta.functions.test_function_arguments(
            func=BaseTransformer.columns_set_or_check, expected_arguments=["self", "X"]
        )

    def test_non_pd_df_error(self):
        """Test an error is raised if X is not passed as a pd.DataFrame."""

        x = BaseTransformer(columns="a")

        with pytest.raises(TypeError, match="X should be a pd.DataFrame"):

            x.columns_set_or_check(X=[1, 2, 3, 4, 5, 6])

    def test_columns_set_to_all_columns_when_none(self):
        """Test that X.columns are set to self.columns if self.columns is None when function called."""

        df = d.create_df_1()

        x = BaseTransformer(columns=None)

        x.columns_set_or_check(X=df)

        ta.equality.assert_equal_dispatch(
            expected=list(df.columns.values),
            actual=x.columns,
            msg="x.columns set when None",
        )


class TestCheckIsFitted(object):
    """Tests for the check_is_fitted method."""

    def test_check_is_fitted_call(self):
        """Test the call to tubular.base.check_is_fitted (sklearn.utils.validation.check_is_fitted)."""

        x = BaseTransformer(columns=None)

        with mock.patch("tubular.base.check_is_fitted") as mocked_method:

            attributes = "columns"

            x.check_is_fitted(attributes)

            assert (
                mocked_method.call_count == 1
            ), f"Incorrect number of calls to tubular.base.check_is_fitted -\n  Expected: 1\n  Actual: {mocked_method.call_count}"

            call_1_args = mocked_method.call_args_list[0]
            call_1_pos_args = call_1_args[0]
            call_1_kwargs = call_1_args[1]

            ta.equality.assert_dict_equal_msg(
                actual=call_1_kwargs,
                expected={},
                msg_tag="Keyword arg assert for tubular.base.check_is_fitted",
            )

            assert (
                len(call_1_pos_args) == 2
            ), f"Incorrect number of positional arguments in check_is_fitted call -\n  Expected: 2\n  Actual: {len(call_1_pos_args)}"

            assert (
                call_1_pos_args[0] is x
            ), f"Incorrect first positional arg in check_is_fitted call -\n  Expected: {x}\n  Actual: {call_1_pos_args[0]}"

            assert (
                call_1_pos_args[1] == attributes
            ), f"Incorrect second positional arg in check_is_fitted call -\n  Expected: {attributes}\n  Actual: {call_1_pos_args[1]}"


class TestCombineXy:
    """Tests for the BaseTransformer._combine_X_y method."""

    def test_arguments(self):
        """Test that columns_check has expected arguments."""

        ta.functions.test_function_arguments(
            func=BaseTransformer._combine_X_y, expected_arguments=["self", "X", "y"]
        )

    def test_X_not_DataFrame_error(self):
        """Test an exception is raised if X is not a pd.DataFrame."""

        x = BaseTransformer(columns=["a"])

        with pytest.raises(TypeError, match="X should be a pd.DataFrame"):

            x._combine_X_y(X=1, y=pandas.Series([1, 2]))

    def test_y_not_Series_error(self):
        """Test an exception is raised if y is not a pd.Series."""

        x = BaseTransformer(columns=["a"])

        with pytest.raises(TypeError, match="y should be a pd.Series"):

            x._combine_X_y(X=pandas.DataFrame({"a": [1, 2]}), y=1)

    def test_X_and_y_different_number_of_rows_error(self):
        """Test an exception is raised if X and y have different numbers of rows."""

        x = BaseTransformer(columns=["a"])

        with pytest.raises(
            ValueError,
            match=re.escape("X and y have different numbers of rows (2 vs 1)"),
        ):

            x._combine_X_y(X=pandas.DataFrame({"a": [1, 2]}), y=pandas.Series([2]))

    def test_X_and_y_different_indexes_warning(self):
        """Test a warning is raised if X and y have different indexes, but the output is still X and y."""

        x = BaseTransformer(columns=["a"])

        with pytest.warns(UserWarning, match="X and y do not have equal indexes"):

            result = x._combine_X_y(
                X=pandas.DataFrame({"a": [1, 2]}, index=[1, 2]), y=pandas.Series([2, 4])
            )

        expected_output = pandas.DataFrame(
            {"a": [1, 2], "_temporary_response": [2, 4]}, index=[1, 2]
        )

        pandas.testing.assert_frame_equal(result, expected_output)

    def test_output_same_indexes(self):
        """Test output is correct if X and y have the same index."""

        x = BaseTransformer(columns=["a"])

        result = x._combine_X_y(
            X=pandas.DataFrame({"a": [1, 2]}, index=[1, 2]),
            y=pandas.Series([2, 4], index=[1, 2]),
        )

        expected_output = pandas.DataFrame(
            {"a": [1, 2], "_temporary_response": [2, 4]}, index=[1, 2]
        )

        pandas.testing.assert_frame_equal(result, expected_output)
