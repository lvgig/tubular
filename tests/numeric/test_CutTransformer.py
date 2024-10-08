import re

import pandas as pd
import pytest
import test_aide as ta

import tests.test_data as d
from tubular.numeric import CutTransformer


class TestInit:
    """Tests for CutTransformer.init()."""

    def test_column_type_error(self):
        """Test that an exception is raised if column is not a str."""
        with pytest.raises(
            TypeError,
            match=re.escape(
                "CutTransformer: column arg (name of column) should be a single str giving the column to discretise",
            ),
        ):
            CutTransformer(
                column=["a"],
                new_column_name="a",
            )

    def test_new_column_name_type_error(self):
        """Test that an exception is raised if new_column_name is not a str."""
        with pytest.raises(
            TypeError,
            match="CutTransformer: new_column_name must be a str",
        ):
            CutTransformer(column="b", new_column_name=1)

    def test_cut_kwargs_type_error(self):
        """Test that an exception is raised if cut_kwargs is not a dict."""
        with pytest.raises(
            TypeError,
            match=r"""cut_kwargs should be a dict but got type \<class 'int'\>""",
        ):
            CutTransformer(column="b", new_column_name="a", cut_kwargs=1)

    def test_cut_kwargs_key_type_error(self):
        """Test that an exception is raised if cut_kwargs has keys which are not str."""
        with pytest.raises(
            TypeError,
            match=r"""CutTransformer: unexpected type \(\<class 'int'\>\) for cut_kwargs key in position 1, must be str""",
        ):
            CutTransformer(
                new_column_name="a",
                column="b",
                cut_kwargs={"a": 1, 2: "b"},
            )


class TestTransform:
    """Tests for CutTransformer.transform()."""

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
