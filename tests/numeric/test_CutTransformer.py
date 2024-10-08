import pandas as pd
import pytest
import test_aide as ta

import tests.test_data as d
from tests.numeric.test_BaseNumericTransformer import (
    BaseNumericTransformerInitTests,
    BaseNumericTransformerTransformTests,
)
from tubular.numeric import CutTransformer


class TestInit(BaseNumericTransformerInitTests):
    """Tests for CutTransformer.init()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "CutTransformer"


class TestTransform(BaseNumericTransformerTransformTests):
    """Tests for CutTransformer.transform()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "CutTransformer"

    def expected_df_1():
        """Expected output for test_expected_output."""
        df = d.create_df_9()

        df["d"] = pd.Categorical(
            values=["c", "b", "a", "d", "e", "f"],
            categories=["a", "b", "c", "d", "e", "f"],
            ordered=True,
        )

        return df

    def test_output_from_cut_assigned_to_column(self, mocker):
        """Test that the output from pd.cut is assigned to column with name new_column_name."""
        df = d.create_df_9()

        x = CutTransformer(column="c", new_column_name="c_new", cut_kwargs={"bins": 2})

        cut_output = [1, 2, 3, 4, 5, 6]

        mocker.patch("pandas.cut", return_value=cut_output)

        df_transformed = x.transform(df)

        assert (
            df_transformed["c_new"].tolist() == cut_output
        ), "unexpected values assigned to c_new column"

    @pytest.mark.parametrize(
        ("df", "expected"),
        ta.pandas.adjusted_dataframe_params(d.create_df_9(), expected_df_1()),
    )
    def test_expected_output(self, df, expected):
        """Test input data is transformed as expected."""
        cut_1 = CutTransformer(
            column="c",
            new_column_name="d",
            cut_kwargs={
                "bins": [0, 1, 2, 3, 4, 5, 6],
                "labels": ["a", "b", "c", "d", "e", "f"],
            },
        )

        df_transformed = cut_1.transform(df)

        ta.equality.assert_equal_dispatch(
            expected=expected,
            actual=df_transformed,
            msg="CutTransformer.transform output",
        )
