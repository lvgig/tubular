import pytest
import tubular.testing.test_data as d
import tubular.testing.helpers as h

import pandas as pd
import sklearn

import tubular
from tubular.nominal import OneHotEncodingTransformer


class TestInit(object):
    """Tests for OneHotEncodingTransformer.init()."""

    def test_arguments(self):
        """Test that init has expected arguments."""

        h.test_function_arguments(
            func=OneHotEncodingTransformer.__init__,
            expected_arguments=[
                "self",
                "columns",
                "separator",
                "drop_original",
                "copy",
                "verbose",
            ],
            expected_default_values=(None, "_", False, True, False),
        )

    def test_class_methods(self):
        """Test that OneHotEncodingTransformer has fit and transform methods."""

        x = OneHotEncodingTransformer()

        h.test_object_method(obj=x, expected_method="fit", msg="fit")
        h.test_object_method(obj=x, expected_method="transform", msg="transform")

    def test_inheritance(self):
        """Test that OneHotEncodingTransformer inherits from BaseNominalTransformer and sklean's OneHotEncoder."""

        x = OneHotEncodingTransformer()

        h.assert_inheritance(x, tubular.nominal.BaseNominalTransformer)
        h.assert_inheritance(x, sklearn.preprocessing.OneHotEncoder)

    def test_super_init_called(self, mocker):
        """Test that init calls BaseNominalTransformer.init.

        Note, not using h.assert_function_call for this as it does not handle self being passed to BaseNominalTransformer.init.
        """

        expected_keyword_args = {"columns": None, "verbose": True, "copy": True}

        mocker.patch("tubular.nominal.BaseNominalTransformer.__init__")

        x = OneHotEncodingTransformer(columns=None, verbose=True, copy=True)

        assert (
            tubular.nominal.BaseNominalTransformer.__init__.call_count == 1
        ), f"Not enough calls to BaseNominalTransformer.__init__ -\n  Expected: 1\n  Actual: {tubular.nominal.BaseNominalTransformer.__init__.call_count}"

        call_args = tubular.nominal.BaseNominalTransformer.__init__.call_args_list[0]
        call_pos_args = call_args[0]
        call_kwargs = call_args[1]

        h.assert_equal_dispatch(
            expected=expected_keyword_args,
            actual=call_kwargs,
            msg="kwargs for BaseNominalTransformer.__init__ in OneHotEncodingTransformer.init",
        )

        assert (
            len(call_pos_args) == 1
        ), f"Unepxected number of positional args in BaseNominalTransformer.__init__ call -\n  Expected: 1\n  Actual: {len(call_pos_args)}"

        assert (
            call_pos_args[0] is x
        ), f"Unexpected positional arg (self) in BaseNominalTransformer.__init__ call -\n  Expected: self\n  Actual: {call_pos_args[0]}"

    def test_one_hot_encoder_init_called(self, mocker):
        """Test that init calls OneHotEncoder.init.

        Again not using h.assert_function_call for this as it does not handle self being passed to OneHotEncoder.init
        """

        expected_keyword_args = {"sparse": False, "handle_unknown": "ignore"}

        mocker.patch("sklearn.preprocessing.OneHotEncoder.__init__")

        x = OneHotEncodingTransformer(
            columns=None, verbose=True, copy=True, separator="x", drop_original=True
        )

        assert (
            sklearn.preprocessing.OneHotEncoder.__init__.call_count == 1
        ), f"Not enough calls to OneHotEncoder.__init__ -\n  Expected: 1\n  Actual: {sklearn.preprocessing.OneHotEncoder.__init__.call_count}"

        call_args = sklearn.preprocessing.OneHotEncoder.__init__.call_args_list[0]
        call_pos_args = call_args[0]
        call_kwargs = call_args[1]

        h.assert_equal_dispatch(
            expected=expected_keyword_args,
            actual=call_kwargs,
            msg="kwargs for OneHotEncoder.__init__ in OneHotEncodingTransformer.init",
        )

        assert (
            len(call_pos_args) == 1
        ), f"Unepxected number of positional args in OneHotEncoder.__init__ call -\n  Expected: 1\n  Actual: {len(call_pos_args)}"

        assert (
            call_pos_args[0] is x
        ), f"Unexpected positional arg (self) in OneHotEncoder.__init__ call -\n  Expected: self\n  Actual: {call_pos_args[0]}"

    def test_values_passed_in_init_set_to_attribute(self):
        """Test that the values passed in init are saved in an attribute of the same name."""

        x = OneHotEncodingTransformer(
            columns=None, verbose=True, copy=True, separator="x", drop_original=True
        )

        h.test_object_attributes(
            obj=x,
            expected_attributes={"separator": "x", "drop_original": True},
            msg="Attributes for OneHotEncodingTransformer set in init",
        )


class TestFit(object):
    """Tests for OneHotEncodingTransformer.fit()"""

    def test_arguments(self):
        """Test that init fit expected arguments."""

        h.test_function_arguments(
            func=OneHotEncodingTransformer.fit,
            expected_arguments=["self", "X", "y"],
            expected_default_values=(None,),
        )

    def test_columns_set_or_check_called(self, mocker):
        """Test that fit calls BaseNominalTransformer.columns_set_or_check."""

        df = d.create_df_1()

        x = OneHotEncodingTransformer(columns="b")

        expected_call_args = {0: {"args": (d.create_df_1(),), "kwargs": {}}}

        with h.assert_function_call(
            mocker,
            tubular.nominal.BaseNominalTransformer,
            "columns_set_or_check",
            expected_call_args,
        ):

            x.fit(df)

    def test_base_nominal_transformer_fit_called(self, mocker):
        """Test that fit calls BaseNominalTransformer.fit."""

        expected_keyword_args = {"X": d.create_df_1(), "y": None}

        df = d.create_df_1()

        x = OneHotEncodingTransformer(columns="b")

        mocker.patch("tubular.nominal.BaseNominalTransformer.fit")

        x.fit(df)

        assert (
            tubular.nominal.BaseNominalTransformer.fit.call_count == 1
        ), f"Not enough calls to BaseNominalTransformer.fit -\n  Expected: 1\n  Actual: {tubular.nominal.BaseNominalTransformer.fit.call_count}"

        call_args = tubular.nominal.BaseNominalTransformer.fit.call_args_list[0]
        call_pos_args = call_args[0]
        call_kwargs = call_args[1]

        h.assert_equal_dispatch(
            expected=expected_keyword_args,
            actual=call_kwargs,
            msg="kwargs for BaseNominalTransformer.fit in OneHotEncodingTransformer.init",
        )

        assert (
            len(call_pos_args) == 1
        ), f"Unepxected number of positional args in BaseNominalTransformer.fit call -\n  Expected: 1\n  Actual: {len(call_pos_args)}"

        assert (
            call_pos_args[0] is x
        ), f"Unexpected positional arg (self) in BaseNominalTransformer.fit call -\n  Expected: self\n  Actual: {call_pos_args[0]}"

    def test_one_hot_encoder_fit_called(self, mocker):
        """Test that fit calls OneHotEncoder.fit."""

        expected_keyword_args = {"X": d.create_df_1()[["b"]], "y": None}

        df = d.create_df_1()

        x = OneHotEncodingTransformer(columns="b")

        mocker.patch("sklearn.preprocessing.OneHotEncoder.fit")

        x.fit(df)

        assert (
            sklearn.preprocessing.OneHotEncoder.fit.call_count == 1
        ), f"Not enough calls to OneHotEncoder.fit -\n  Expected: 1\n  Actual: {sklearn.preprocessing.OneHotEncoder.fit.call_count}"

        call_args = sklearn.preprocessing.OneHotEncoder.fit.call_args_list[0]
        call_pos_args = call_args[0]
        call_kwargs = call_args[1]

        h.assert_equal_dispatch(
            expected=expected_keyword_args,
            actual=call_kwargs,
            msg="kwargs for OneHotEncoder.fit in OneHotEncodingTransformer.init",
        )

        assert (
            len(call_pos_args) == 1
        ), f"Unepxected number of positional args in OneHotEncoder.fit call -\n  Expected: 1\n  Actual: {len(call_pos_args)}"

        assert (
            call_pos_args[0] is x
        ), f"Unexpected positional arg (self) in OneHotEncoder.fit call -\n  Expected: self\n  Actual: {call_pos_args[0]}"

    def test_nulls_in_X_error(self):
        """Test that an exception is raised if X has nulls in column to be fit on."""

        df = d.create_df_2()

        x = OneHotEncodingTransformer(columns=["b", "c"])

        with pytest.raises(
            ValueError, match="column b has nulls - replace before proceeding"
        ):

            x.fit(df)

    def test_fields_with_over_100_levels_error(self):
        """Test that OneHotEncodingTransformer.fit on fields with more than 100 levels raises error."""

        df = d.prepare_boston_df()
        df = df.loc[df["CRIM"].notnull(), ["CHAS_cat", "CRIM"]]

        x = OneHotEncodingTransformer(columns=["CHAS_cat", "CRIM"])

        with pytest.raises(
            ValueError,
            match="column CRIM has over 100 unique values - consider another type of encoding",
        ):

            x.fit(df)

    def test_fit_returns_self(self):
        """Test fit returns self?"""

        df = d.create_df_1()

        x = OneHotEncodingTransformer(columns="b")

        x_fitted = x.fit(df)

        assert (
            x_fitted is x
        ), "Returned value from OneHotEncodingTransformer.fit not as expected."

    def test_fit_not_changing_data(self):
        """Test fit does not change X."""

        df = d.create_df_1()

        x = OneHotEncodingTransformer(columns="b")

        x.fit(df)

        h.assert_equal_dispatch(
            expected=d.create_df_1(),
            actual=df,
            msg="Check X not changing during fit",
        )


class TestTransform(object):
    """Tests for OneHotEncodingTransformer.transform()."""

    def expected_df_1():
        """Expected output for test_expected_output."""

        df = pd.DataFrame(
            {
                "a": [4, 2, 2, 1, 3],
                "b": ["x", "z", "y", "x", "x"],
                "c": ["c", "a", "a", "c", "b"],
            }
        )

        df["c"] = df["c"].astype("category")

        df["b_x"] = [1.0, 0.0, 0.0, 1.0, 1.0]
        df["b_y"] = [0.0, 0.0, 1.0, 0.0, 0.0]
        df["b_z"] = [0.0, 1.0, 0.0, 0.0, 0.0]

        return df

    def expected_df_2():
        """Expected output for test_unseen_categories_encoded_as_all_zeroes."""

        df = pd.DataFrame(
            {
                "a": [1, 5, 2, 3, 3],
                "b": ["w", "w", "z", "y", "x"],
                "c": ["a", "a", "c", "b", "a"],
            },
            index=[10, 15, 200, 251, 59],
        )

        df["c"] = df["c"].astype("category")

        df["a_1"] = [1.0, 0.0, 0.0, 0.0, 0.0]
        df["a_2"] = [0.0, 0.0, 1.0, 0.0, 0.0]
        df["a_3"] = [0.0, 0.0, 0.0, 1.0, 1.0]
        df["a_4"] = [0.0, 0.0, 0.0, 0.0, 0.0]
        df["b_x"] = [0.0, 0.0, 0.0, 0.0, 1.0]
        df["b_y"] = [0.0, 0.0, 0.0, 1.0, 0.0]
        df["b_z"] = [0.0, 0.0, 1.0, 0.0, 0.0]
        df["c_a"] = [1.0, 1.0, 0.0, 0.0, 1.0]
        df["c_b"] = [0.0, 0.0, 0.0, 1.0, 0.0]
        df["c_c"] = [0.0, 0.0, 1.0, 0.0, 0.0]

        return df

    def test_arguments(self):
        """Test that transform has expected arguments."""

        h.test_function_arguments(
            func=OneHotEncodingTransformer.transform, expected_arguments=["self", "X"]
        )

    def test_columns_check_call(self, mocker):
        """Test the first call to BaseTransformer columns_check."""

        df = d.create_df_1()

        x = OneHotEncodingTransformer(columns="b")

        x.fit(df)

        expected_call_args = {0: {"args": (d.create_df_1(),), "kwargs": {}}}

        with h.assert_function_call(
            mocker, tubular.base.BaseTransformer, "columns_check", expected_call_args
        ):

            x.transform(df)

    def test_check_is_fitted_first_call(self, mocker):
        """Test the calls to BaseTransformer check_is_fitted."""

        df = d.create_df_1()

        x = OneHotEncodingTransformer(columns="b")

        x.fit(df)

        expected_call_args = {
            0: {"args": (["separator"],), "kwargs": {}},
            1: {"args": (["drop_original"],), "kwargs": {}},
        }

        with h.assert_function_call(
            mocker, tubular.base.BaseTransformer, "check_is_fitted", expected_call_args
        ):

            x.transform(df)

    def test_non_numeric_column_error_1(self):
        """Test that transform will raise an error if a column to transform has nulls."""

        df_train = d.create_df_1()
        df_test = d.create_df_2()

        x = OneHotEncodingTransformer(columns=["b"])

        x.fit(df_train)

        with pytest.raises(
            ValueError, match="column b has nulls - replace before proceeding"
        ):

            x.transform(df_test)

    def test_base_nominal_transformer_transform_called(self, mocker):
        """Test that BaseNominalTransformer.transform called."""

        df = d.create_df_1()

        x = OneHotEncodingTransformer(columns="b")

        x.fit(df)

        mocker.patch(
            "tubular.nominal.BaseNominalTransformer.transform",
            return_value=d.create_df_1(),
        )

        x.transform(df)

        assert (
            tubular.nominal.BaseNominalTransformer.transform.call_count == 1
        ), f"Not enough calls to BaseNominalTransformer.transform -\n  Expected: 1\n  Actual: {tubular.nominal.BaseNominalTransformer.transform.call_count}"

        call_args = tubular.nominal.BaseNominalTransformer.transform.call_args_list[0]
        call_pos_args = call_args[0]
        call_kwargs = call_args[1]

        h.assert_equal_dispatch(
            expected={},
            actual=call_kwargs,
            msg="kwargs for BaseNominalTransformer.transform in OneHotEncodingTransformer.init",
        )

        expected_pos_args = (x, d.create_df_1())

        assert (
            len(call_pos_args) == 2
        ), f"Unepxected number of positional args in BaseNominalTransformer.transform call -\n  Expected: 2\n  Actual: {len(call_pos_args)}"

        h.assert_frame_equal_msg(
            expected=expected_pos_args[1],
            actual=call_pos_args[1],
            msg_tag="X positional arg in BaseNominalTransformer.transform call",
        )

        assert (
            expected_pos_args[0] == call_pos_args[0]
        ), "self positional arg in BaseNominalTransformer.transform call"

    def test_one_hot_encoder_transform_called(self, mocker):
        """Test that OneHotEncoder.transform called."""

        df = d.create_df_1()

        x = OneHotEncodingTransformer(columns="b")

        x.fit(df)

        mocker.patch("sklearn.preprocessing.OneHotEncoder.transform")

        x.transform(df)

        assert (
            sklearn.preprocessing.OneHotEncoder.transform.call_count == 1
        ), f"Not enough calls to OneHotEncoder.transform -\n  Expected: 1\n  Actual: {sklearn.preprocessing.OneHotEncoder.transform.call_count}"

        call_args = sklearn.preprocessing.OneHotEncoder.transform.call_args_list[0]
        call_pos_args = call_args[0]
        call_kwargs = call_args[1]

        h.assert_equal_dispatch(
            expected={},
            actual=call_kwargs,
            msg="kwargs for OneHotEncodingTransformer.transform in BaseTransformer.init",
        )

        assert (
            len(call_pos_args) == 2
        ), f"Unepxected number of positional args in OneHotEncodingTransformer.transform call -\n  Expected: 2\n  Actual: {len(call_pos_args)}"

        assert (
            call_pos_args[0] is x
        ), f"Unexpected positional arg (self, index 1) in OneHotEncodingTransformer.transform call -\n  Expected: self\n  Actual: {call_pos_args[0]}"

        h.assert_frame_equal_msg(
            expected=d.create_df_1()[["b"]],
            actual=call_pos_args[1],
            msg_tag="X positional arg in OneHotEncodingTransformer.transform call",
        )

    @pytest.mark.parametrize(
        "df_test, expected",
        h.row_by_row_params(d.create_df_7(), expected_df_1())
        + h.index_preserved_params(d.create_df_7(), expected_df_1()),
    )
    def test_expected_output(self, df_test, expected):
        """Test that OneHotEncodingTransformer.transform encodes the feature correctly.

        Also tests that OneHotEncodingTransformer.transform does not modify unrelated columns.
        """

        # transformer is fit on the whole dataset separately from the input df to work with the decorators
        df_train = d.create_df_7()
        x = OneHotEncodingTransformer(columns="b")
        x.fit(df_train)

        df_transformed = x.transform(df_test)

        h.assert_frame_equal_msg(
            expected=expected,
            actual=df_transformed,
            msg_tag="Unspecified columns changed in transform",
        )

    def test_categories_not_modified(self):
        """Test that the categories from fit are not changed in transform."""

        df_train = d.create_df_1()
        df_test = d.create_df_7()

        x = OneHotEncodingTransformer(columns=["a", "b"], verbose=False)
        x2 = OneHotEncodingTransformer(columns=["a", "b"], verbose=False)

        x.fit(df_train)
        x2.fit(df_train)

        x.transform(df_test)

        h.assert_equal_dispatch(
            expected=list(x2.categories_[0]),
            actual=list(x.categories_[0]),
            msg="categories_ (index 0) modified during transform",
        )

        h.assert_equal_dispatch(
            expected=list(x2.categories_[1]),
            actual=list(x.categories_[1]),
            msg="categories_ (index 1) modified during transform",
        )

    def test_renaming_feature_works_as_expected(self):
        """Test OneHotEncodingTransformer.transform() is renaming features correctly."""

        df = d.create_df_7()
        df = df[["b", "c"]]

        x = OneHotEncodingTransformer(
            columns=["b", "c"], separator="|", drop_original=True
        )

        x.fit(df)

        df_transformed = x.transform(df)

        h.assert_equal_dispatch(
            expected=["b|x", "b|y", "b|z", "c|a", "c|b", "c|c"],
            actual=list(df_transformed.columns.values),
            msg="renaming columns feature in OneHotEncodingTransformer.transform",
        )

    def test_warning_generated_by_unseen_categories(self):
        """Test OneHotEncodingTransformer.transform triggers a warning for unseen categories."""

        df_train = d.create_df_7()
        df_test = d.create_df_8()

        x = OneHotEncodingTransformer(verbose=True)

        x.fit(df_train)

        with pytest.warns(Warning):

            x.transform(df_test)

    @pytest.mark.parametrize(
        "df_test, expected",
        h.row_by_row_params(d.create_df_8(), expected_df_2())
        + h.index_preserved_params(d.create_df_8(), expected_df_2()),
    )
    def test_unseen_categories_encoded_as_all_zeroes(self, df_test, expected):
        """Test OneHotEncodingTransformer.transform encodes unseen categories correctly (all 0s)."""

        # transformer is fit on the whole dataset separately from the input df to work with the decorators
        df_train = d.create_df_7()
        x = OneHotEncodingTransformer(columns=["a", "b", "c"], verbose=False)
        x.fit(df_train)

        df_transformed = x.transform(df_test)

        h.assert_equal_dispatch(
            expected=expected,
            actual=df_transformed,
            msg="unseen category rows not encoded as 0s",
        )

    def test_original_columns_dropped_when_specified(self):
        """Test OneHotEncodingTransformer.transform drops original columns get when specified."""

        df = d.create_df_7()

        x = OneHotEncodingTransformer(columns=["a", "b", "c"], drop_original=True)

        x.fit(df)

        df_transformed = x.transform(df)

        h.assert_equal_dispatch(
            expected=["a", "b", "c"],
            actual=[
                x for x in df.columns.values if x not in df_transformed.columns.values
            ],
            msg="original columns not dropped",
        )

    def test_original_columns_kept_when_specified(self):
        """Test OneHotEncodingTransformer.transform keeps original columns when specified."""

        df = d.create_df_7()

        x = OneHotEncodingTransformer(drop_original=False)

        x.fit(df)

        df_transformed = x.transform(df)

        h.assert_equal_dispatch(
            expected=list(set()),
            actual=list(set(["a", "b", "c"]) - set(df_transformed.columns)),
            msg="original columns not kept",
        )
