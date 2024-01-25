import re

import numpy as np
import pandas as pd
import pytest
import test_aide as ta

import tests.test_data as d
from tubular.base import DataFrameMethodTransformer


class TestInit:
    """Tests for DataFrameMethodTransformer.init()."""

    @pytest.mark.parametrize("not_dictionary", ["a", [1, 2], True, 1.5])
    def test_exception_raised_pd_method_kwargs_not_dict(self, not_dictionary):
        """Test an exception is raised if pd_method_kwargs not a dict"""

        with pytest.raises(
            TypeError,
            match=f"DataFrameMethodTransformer: pd_method_kwargs should be a dict but got type {type(not_dictionary)}",
        ):
            DataFrameMethodTransformer(
                new_column_names="a",
                pd_method_name="b",
                columns=["b", "c"],
                pd_method_kwargs=not_dictionary,
            )

    @pytest.mark.parametrize("not_string", [1, True, 1.5])
    def test_exception_raised_pd_method_kwargs_key_not_string(self, not_string):
        """Test an exception is raised if a pd_method_kwarg key is not a string"""

        pd_method_kwargs = {
            "other": 2,
            not_string: 1,
        }

        with pytest.raises(
            TypeError,
            match=re.escape(
                f"DataFrameMethodTransformer: unexpected type ({type(not_string)}) for pd_method_kwargs key in position {1}, must be str",
            ),
        ):
            DataFrameMethodTransformer(
                new_column_names="a",
                pd_method_name="b",
                columns=["b", "c"],
                pd_method_kwargs=pd_method_kwargs,
            )

    @pytest.mark.parametrize("not_string", [{"a": 1}, [1, 2], 1, True, 1.5])
    def test_exception_raised_pd_method_name_not_string(self, not_string):
        """Test an exception is raised if pd_method_name is not a string"""

        with pytest.raises(
            TypeError,
            match=re.escape(
                f"DataFrameMethodTransformer: unexpected type ({type(not_string)}) for pd_method_name, expecting str",
            ),
        ):
            DataFrameMethodTransformer(
                new_column_names="a",
                pd_method_name=not_string,
                columns=["b", "c"],
            )

    def test_exception_raised_non_pandas_method_passed(self):
        """Test an exception is raised if a non pd.DataFrame method is passed for pd_method_name."""
        with pytest.raises(
            AttributeError,
            match="""DataFrameMethodTransformer: error accessing "b" method on pd.DataFrame object - pd_method_name should be a pd.DataFrame method""",
        ):
            DataFrameMethodTransformer(
                new_column_names="a",
                pd_method_name="b",
                columns=["b", "c"],
            )

    @pytest.mark.parametrize("not_bool", [{"a": 1}, [1, 2], 1, "True", 1.5])
    def test_exception_raised_drop_original_not_bool(self, not_bool):
        """Test an exception is raised if pd_method_name is not a string"""

        with pytest.raises(
            TypeError,
            match=re.escape(
                f"DataFrameMethodTransformer: unexpected type ({type(not_bool)}) for drop_original, expecting bool",
            ),
        ):
            DataFrameMethodTransformer(
                new_column_names="a",
                pd_method_name="sum",
                columns=["b", "c"],
                drop_original=not_bool,
            )


class TestTransform:
    """Tests for DataFrameMethodTransformer.transform()."""

    def expected_df_1():
        """Expected output of test_expected_output_single_columns_assignment."""
        return pd.DataFrame(
            {
                "a": [1, 2, 3, 4, 5, 6, np.NaN],
                "b": [1, 2, 3, np.NaN, 7, 8, 9],
                "c": [np.NaN, 1, 2, 3, -4, -5, -6],
                "d": [1.0, 3.0, 5.0, 3.0, 3.0, 3.0, 3.0],
            },
        )

    def expected_df_2():
        """Expected output of test_expected_output_multi_columns_assignment."""
        return pd.DataFrame(
            {
                "a": [1, 2, 3, 4, 5, 6, np.NaN],
                "b": [1, 2, 3, np.NaN, 7, 8, 9],
                "c": [np.NaN, 1, 2, 3, -4, -5, -6],
                "d": [0.5, 1.0, 1.5, np.NaN, 3.5, 4.0, 4.5],
                "e": [np.NaN, 0.5, 1.0, 1.5, -2.0, -2.5, -3.0],
            },
        )

    @pytest.mark.parametrize(
        ("df", "expected"),
        ta.pandas.adjusted_dataframe_params(d.create_df_3(), expected_df_1()),
    )
    def test_expected_output_single_columns_assignment(self, df, expected):
        """Test a single column output from transform gives expected results."""
        x = DataFrameMethodTransformer(
            new_column_names="d",
            pd_method_name="sum",
            columns=["b", "c"],
            pd_method_kwargs={"axis": 1},
        )

        df_transformed = x.transform(df)

        ta.equality.assert_equal_dispatch(
            expected=expected,
            actual=df_transformed,
            msg="DataFrameMethodTransformer sum columns b and c",
        )

    @pytest.mark.parametrize(
        ("df", "expected"),
        ta.pandas.adjusted_dataframe_params(d.create_df_3(), expected_df_2()),
    )
    def test_expected_output_multi_columns_assignment(self, df, expected):
        """Test a multiple column output from transform gives expected results."""
        x = DataFrameMethodTransformer(
            new_column_names=["d", "e"],
            pd_method_name="div",
            columns=["b", "c"],
            pd_method_kwargs={"other": 2},
        )

        df_transformed = x.transform(df)

        ta.equality.assert_equal_dispatch(
            expected=expected,
            actual=df_transformed,
            msg="DataFrameMethodTransformer divide by 2 columns b and c",
        )

    def test_original_columns_dropped_when_specified(self):
        """Test DataFrameMethodTransformer.transform drops original columns get when specified."""
        df = d.create_df_3()

        x = DataFrameMethodTransformer(
            new_column_names="a_b_sum",
            pd_method_name="sum",
            columns=["a", "b"],
            drop_original=True,
        )

        x.fit(df)

        df_transformed = x.transform(df)

        assert ("a" not in df_transformed.columns.to_numpy()) and (
            "b" not in df_transformed.columns.to_numpy()
        ), "original columns not dropped"

    def test_original_columns_kept_when_specified(self):
        """Test DataFrameMethodTransformer.transform keeps original columns when specified."""
        df = d.create_df_3()

        x = DataFrameMethodTransformer(
            new_column_names="a_b_sum",
            pd_method_name="sum",
            columns=["a", "b"],
            drop_original=False,
        )

        x.fit(df)

        df_transformed = x.transform(df)

        assert ("a" in df_transformed.columns.to_numpy()) and (
            "b" in df_transformed.columns.to_numpy()
        ), "original columns not kept"
