import pytest
import test_aide as ta

import tests.test_data as d
import tubular
from tubular.base import BaseTransformer
from tubular.nominal import BaseNominalTransformer


class TestInit:
    """Test for BaseNominalTransformer object."""

    def test_class_methods(self):
        """Test that BaseNominalTransformer has columns_set_or_check method."""
        x = BaseNominalTransformer()

        ta.classes.test_object_method(
            obj=x,
            expected_method="columns_set_or_check",
            msg="columns_set_or_check",
        )

        ta.classes.test_object_method(
            obj=x,
            expected_method="check_mappable_rows",
            msg="check_mappable_rows",
        )

    def test_inheritance(self):
        """Test that BaseNominalTransformer inherits from BaseTransformer."""
        x = BaseNominalTransformer()

        ta.classes.assert_inheritance(x, BaseTransformer)


class TestNominalColumnSetOrCheck:
    """Tests for BaseNominalTransformer.columns_set_or_check method."""

    def test_columns_none_get_cat_columns(self):
        """If self.columns is None then object and categorical columns are set as self.columns."""
        df = d.create_df_4()

        x = BaseNominalTransformer()

        x.columns = None

        x.columns_set_or_check(df)

        ta.equality.assert_equal_dispatch(
            expected=["b", "c"],
            actual=x.columns,
            msg="nominal columns getting",
        )

    def test_columns_none_no_cat_columns_error(self):
        """If self.columns is None and there are no object and categorical columns then an exception is raised."""
        df = d.create_1_int_column_df()

        x = BaseNominalTransformer()

        x.columns = None

        with pytest.raises(ValueError):
            x.columns_set_or_check(df)

    def test_columns_check_called(self, mocker):
        """Test call to tubular.base.BaseTransformer.columns_check."""

        class JointInheritanceClass(BaseNominalTransformer, BaseTransformer):
            """Class to use in TestNominalColumnSetOrCheck.test_columns_check_called, which inherits from
            BaseNominalTransformer and BaseTransformer in order to test that the columns_check method
            from BaseTransformer is called.
            """

        df = d.create_df_1()

        x = JointInheritanceClass()

        x.columns = ["a"]

        expected_call_args = {0: {"args": (d.create_df_1(),), "kwargs": {}}}

        with ta.functions.assert_function_call(
            mocker,
            tubular.base.BaseTransformer,
            "columns_check",
            expected_call_args,
        ):
            x.columns_set_or_check(df)


class TestCheckMappableRows:
    """Tests for the BaseNominalTransformer.check_mappable_rows method."""

    def test_exception_raised(self):
        """Test an exception is raised if non-mappable rows are present in X."""
        df = d.create_df_1()

        x = BaseNominalTransformer()
        x.columns = ["a", "b"]
        x.mappings = {
            "a": {1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7},
            "b": {"a": 1, "c": 2, "d": 3, "e": 4, "f": 5},
        }

        with pytest.raises(
            ValueError,
            match="BaseNominalTransformer: nulls would be introduced into column b from levels not present in mapping",
        ):
            x.check_mappable_rows(df)
