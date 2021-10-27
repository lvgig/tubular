import pandas as pd
import test_aide as ta
import tests.test_data as d

import tubular
from tubular.mapping import BaseMappingTransformMixin


class TestInit(object):
    """Tests for BaseMappingTransformMixin.init()."""

    def test_class_methods(self):
        """Test that BaseMappingTransformMixin has transform method."""

        x = BaseMappingTransformMixin()

        ta.classes.test_object_method(
            obj=x, expected_method="transform", msg="transform method"
        )

    def test_inheritance(self):
        """Test that BaseMappingTransformMixin inherits from BaseTransformer."""

        x = BaseMappingTransformMixin()

        ta.classes.assert_inheritance(x, tubular.base.BaseTransformer)


class TestTransform(object):
    """Tests for BaseMappingTransformMixin.transform()."""

    def test_check_is_fitted_call(self, mocker):
        """Test the call to check_is_fitted."""

        df = d.create_df_1()

        mapping = {
            "a": {1: "a", 2: "b", 3: "c", 4: "d", 5: "e", 6: "f"},
            "b": {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6},
        }

        x = BaseMappingTransformMixin()
        x.columns = ["a", "b"]
        x.mappings = mapping

        expected_call_args = {0: {"args": (["mappings"],), "kwargs": {}}}

        with ta.functions.assert_function_call(
            mocker, tubular.base.BaseTransformer, "check_is_fitted", expected_call_args
        ):

            x.transform(df)

    def test_super_transform_call(self, mocker):
        """Test the call to BaseTransformer.transform."""

        df = d.create_df_1()

        mapping = {
            "a": {1: "a", 2: "b", 3: "c", 4: "d", 5: "e", 6: "f"},
            "b": {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6},
        }

        x = BaseMappingTransformMixin()
        x.columns = ["a", "b"]
        x.mappings = mapping

        expected_call_args = {0: {"args": (d.create_df_1(),), "kwargs": {}}}

        with ta.functions.assert_function_call(
            mocker, tubular.base.BaseTransformer, "transform", expected_call_args
        ):

            x.transform(df)

    def test_pd_series_map_call(self, mocker):
        """Test the call to pd.Series.map."""

        spy = mocker.spy(pd.Series, "map")

        df = d.create_df_1()

        mapping = {
            "a": {1: "a", 2: "b", 3: "c", 4: "d", 5: "e", 6: "f"},
            "b": {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6},
        }

        x = BaseMappingTransformMixin()
        x.columns = ["a", "b"]
        x.mappings = mapping

        x.transform(df)

        assert spy.call_count == 2, "unexpected number of calls to pd.Series.map"

        call_args = spy.call_args_list[0]

        call_pos_arg = call_args[0]
        call_kwargs = call_args[1]

        assert call_kwargs == {}

        expected_pos_args = (df["a"], mapping["a"])

        ta.equality.assert_equal_dispatch(
            expected_pos_args,
            call_pos_arg,
            "positional args in first pd.Series.map call not correct",
        )

        call_args = spy.call_args_list[1]

        call_pos_arg = call_args[0]
        call_kwargs = call_args[1]

        assert call_kwargs == {}

        expected_pos_args = (df["b"], mapping["b"])

        ta.equality.assert_equal_dispatch(
            expected_pos_args,
            call_pos_arg,
            "positional args in second pd.Series.map call not correct",
        )
