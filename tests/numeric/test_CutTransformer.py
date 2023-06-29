import pytest
import test_aide as ta
import tests.test_data as d
import re
import pandas
import pandas as pd

import tubular
from tubular.numeric import CutTransformer


class TestInit(object):
    """Tests for CutTransformer.init()."""

    def test_arguments(self):
        """Test that init has expected arguments."""

        ta.functions.test_function_arguments(
            func=CutTransformer.__init__,
            expected_arguments=["self", "column", "new_column_name", "cut_kwargs"],
            expected_default_values=({},),
        )

    def test_class_methods(self):
        """Test that CutTransformer has transform method."""

        x = CutTransformer(column="a", new_column_name="b")

        ta.classes.test_object_method(
            obj=x, expected_method="transform", msg="transform"
        )

    def test_inheritance(self):
        """Test that CutTransformer inherits from BaseTransformer."""

        x = CutTransformer(column="a", new_column_name="b")

        ta.classes.assert_inheritance(x, tubular.base.BaseTransformer)

    def test_super_init_called(self, mocker):
        """Test that init calls BaseTransformer.init."""

        expected_call_args = {
            0: {
                "args": (),
                "kwargs": {"columns": ["a"], "copy": True, "verbose": False},
            }
        }

        with ta.functions.assert_function_call(
            mocker, tubular.base.BaseTransformer, "__init__", expected_call_args
        ):
            CutTransformer(column="a", new_column_name="b", verbose=False, copy=True)

    def test_column_type_error(self):
        """Test that an exception is raised if column is not a str."""

        with pytest.raises(
            TypeError,
            match=re.escape(
                "CutTransformer: column arg (name of column) should be a single str giving the column to discretise"
            ),
        ):
            CutTransformer(
                column=["a"],
                new_column_name="a",
            )

    def test_new_column_name_type_error(self):
        """Test that an exception is raised if new_column_name is not a str."""

        with pytest.raises(
            TypeError, match="CutTransformer: new_column_name must be a str"
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

    def test_inputs_set_to_attribute(self):
        """Test that the values passed in init are set to attributes."""

        x = CutTransformer(
            column="b",
            new_column_name="a",
            cut_kwargs={"a": 1, "b": 2},
        )

        ta.classes.test_object_attributes(
            obj=x,
            expected_attributes={
                "column": "b",
                "columns": ["b"],
                "new_column_name": "a",
                "cut_kwargs": {"a": 1, "b": 2},
            },
            msg="Attributes for CutTransformer set in init",
        )


class TestTransform(object):
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

    def test_arguments(self):
        """Test that transform has expected arguments."""

        ta.functions.test_function_arguments(
            func=CutTransformer.transform, expected_arguments=["self", "X"]
        )

    def test_super_transform_call(self, mocker):
        """Test the call to BaseTransformer.transform is as expected."""

        df = d.create_df_9()

        x = CutTransformer(column="a", new_column_name="Y", cut_kwargs={"bins": 3})

        expected_call_args = {0: {"args": (d.create_df_9(),), "kwargs": {}}}

        with ta.functions.assert_function_call(
            mocker,
            tubular.base.BaseTransformer,
            "transform",
            expected_call_args,
            return_value=d.create_df_9(),
        ):
            x.transform(df)

    def test_pd_cut_call(self, mocker):
        """Test the call to pd.cut is as expected."""

        df = d.create_df_9()

        x = CutTransformer(
            column="a",
            new_column_name="a_cut",
            cut_kwargs={"bins": 3, "right": False, "precision": 2},
        )

        expected_call_args = {
            0: {
                "args": (d.create_df_9()["a"],),
                "kwargs": {"bins": 3, "right": False, "precision": 2},
            }
        }

        with ta.functions.assert_function_call(
            mocker, pandas, "cut", expected_call_args, return_value=[1, 2, 3, 4, 5, 6]
        ):
            x.transform(df)

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
        "df, expected",
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

    def test_non_numeric_column_error(self):
        """Test that an exception is raised if the column to discretise is not numeric."""

        df = d.create_df_8()

        x = CutTransformer(column="b", new_column_name="d")

        with pytest.raises(
            TypeError,
            match="CutTransformer: b should be a numeric dtype but got object",
        ):
            x.transform(df)
