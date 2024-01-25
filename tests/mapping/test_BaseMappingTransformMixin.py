import pandas as pd
import test_aide as ta

import tests.test_data as d
from tubular.mapping import BaseMappingTransformMixin


class TestInit:
    """Tests for BaseMappingTransformMixin.init().
    Currently nothing to test."""


class TestTransform:
    """Tests for BaseMappingTransformMixin.transform()."""

    # TODO replace this with a behaviour test
    def test_pd_series_replace_call(self, mocker):
        """Test the call to pd.Series.replace."""
        spy = mocker.spy(pd.Series, "replace")

        df = d.create_df_1()

        mapping = {
            "a": {1: "a", 2: "b", 3: "c", 4: "d", 5: "e", 6: "f"},
            "b": {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6},
        }

        x = BaseMappingTransformMixin(columns=["a", "b"])

        x.mappings = mapping

        x.transform(df)

        assert spy.call_count == 2, "unexpected number of calls to pd.Series.replace"

        call_args = spy.call_args_list[0]

        call_pos_arg = call_args[0]
        call_kwargs = call_args[1]

        # not totally sure where this kwarg is injected from but regex=False is the
        # desired default behaviour for Series.replace in this case so leaving for now.
        # Likely to do with the way DataFrame.replace is implemented.
        assert call_kwargs == {"regex": False}

        # pd.Series.replace separates dict into to replace and value lists
        expected_pos_args = (
            df["a"],
            [1, 2, 3, 4, 5, 6],
            ["a", "b", "c", "d", "e", "f"],
        )

        ta.equality.assert_equal_dispatch(
            expected_pos_args,
            call_pos_arg,
            "positional args in first pd.Series.replace call not correct",
        )

        call_args = spy.call_args_list[1]

        call_pos_arg = call_args[0]
        call_kwargs = call_args[1]

        assert call_kwargs == {"regex": False}

        # pd.Series.replace separates dict into to replace and value lists
        expected_pos_args = (
            df["b"],
            ["a", "b", "c", "d", "e", "f"],
            [1, 2, 3, 4, 5, 6],
        )

        ta.equality.assert_equal_dispatch(
            expected_pos_args,
            call_pos_arg,
            "positional args in second pd.Series.replace call not correct",
        )
