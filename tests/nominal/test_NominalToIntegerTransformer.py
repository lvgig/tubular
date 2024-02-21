import pandas as pd
import pytest
import test_aide as ta

import tests.test_data as d
import tubular
from tubular.nominal import NominalToIntegerTransformer


class TestInit:
    """Tests for NominalToIntegerTransformer.init()."""

    def test_start_encoding_not_int_error(self):
        """Test that an exception is raised if start_encoding is not an int."""
        with pytest.raises(ValueError):
            NominalToIntegerTransformer(columns="a", start_encoding="a")


class TestFit:
    """Tests for NominalToIntegerTransformer.fit()."""

    def test_super_fit_called(self, mocker):
        """Test that fit calls BaseTransformer.fit."""
        df = d.create_df_1()

        x = NominalToIntegerTransformer(columns=["a", "b"])

        spy = mocker.spy(tubular.base.BaseTransformer, "fit")

        x.fit(df)

        assert spy.call_count == 1, "unexpected number of calls to BaseTransformer.fit"

        call_args = spy.call_args_list[0]
        call_pos_args = call_args[0]
        call_kwargs = call_args[1]

        expected_kwargs = {}

        assert (
            call_kwargs == expected_kwargs
        ), "unexpected kwargs in BaseTransformer.fit call"

        expected_pos_args = (x, d.create_df_1(), None)

        assert len(expected_pos_args) == len(
            call_pos_args,
        ), "unexpected # positional args in BaseTransformer.fit call"

        assert (
            expected_pos_args[0] == call_pos_args[0]
        ), "unexpected 1st positional arg in BaseTransformer.fit call"

        ta.equality.assert_equal_dispatch(
            expected_pos_args[1:3],
            call_pos_args[1:3],
            "unexpected 2nd, 3rd positional arg in BaseTransformer.fit call",
        )

    def test_learnt_values(self):
        """Test that the impute values learnt during fit are expected."""
        df = d.create_df_1()

        x = NominalToIntegerTransformer(columns=["a", "b"], start_encoding=1)

        x.fit(df)

        ta.classes.test_object_attributes(
            obj=x,
            expected_attributes={
                "mappings": {
                    "a": {k: i for i, k in enumerate(df["a"].unique(), 1)},
                    "b": {k: i for i, k in enumerate(df["b"].unique(), 1)},
                },
            },
            msg="mappings attribute",
        )

    def test_fit_returns_self(self):
        """Test fit returns self?."""
        df = d.create_df_1()

        x = NominalToIntegerTransformer(columns=["a", "b"])

        x_fitted = x.fit(df)

        assert (
            x_fitted is x
        ), "Returned value from NominalToIntegerTransformer.fit not as expected."

    def test_fit_not_changing_data(self):
        """Test fit does not change X."""
        df = d.create_df_1()

        x = NominalToIntegerTransformer(columns=["a", "b"])

        x.fit(df)

        ta.equality.assert_equal_dispatch(
            expected=d.create_df_1(),
            actual=df,
            msg="Check X not changing during fit",
        )


class TestTransform:
    """Tests for NominalToIntegerTransformer.transform()."""

    def expected_df_1():
        """Expected output for test_expected_output."""
        df = pd.DataFrame(
            {"a": [1, 2, 3, 4, 5, 6], "b": ["a", "b", "c", "d", "e", "f"]},
        )

        df["a"] = df["a"].replace({k: i for i, k in enumerate(df["a"].unique())})

        df["b"] = df["b"].replace({k: i for i, k in enumerate(df["b"].unique())})

        return df

    def test_check_is_fitted_called(self, mocker):
        """Test that BaseTransformer check_is_fitted called."""
        df = d.create_df_1()

        x = NominalToIntegerTransformer(columns=["a", "b"])

        x.fit(df)

        expected_call_args = {0: {"args": (["mappings"],), "kwargs": {}}}

        with ta.functions.assert_function_call(
            mocker,
            tubular.base.BaseTransformer,
            "check_is_fitted",
            expected_call_args,
        ):
            x.transform(df)

    def test_not_dataframe_error_raised(self):
        df = d.create_df_1()

        x = NominalToIntegerTransformer(columns=["a", "b"])
        x.fit(df, df["a"])

        with pytest.raises(
            TypeError,
            match=f"{x.classname()}: X should be a pd.DataFrame",
        ):
            x.transform(X=[1, 2, 3, 4, 5, 6])

    def test_super_transform_called(self, mocker):
        """Test that BaseTransformer.transform called."""
        df = d.create_df_1()

        x = NominalToIntegerTransformer(columns="a")

        x.fit(df)

        expected_call_args = {0: {"args": (d.create_df_1(),), "kwargs": {}}}

        with ta.functions.assert_function_call(
            mocker,
            tubular.base.BaseTransformer,
            "transform",
            expected_call_args,
            return_value=d.create_df_1(),
        ):
            x.transform(df)

    def test_learnt_values_not_modified(self):
        """Test that the mappings from fit are not changed in transform."""
        df = d.create_df_1()

        x = NominalToIntegerTransformer(columns=["a", "b"])

        x.fit(df)

        x2 = NominalToIntegerTransformer(columns=["a", "b"])

        x2.fit_transform(df)

        ta.equality.assert_equal_dispatch(
            expected=x.mappings,
            actual=x2.mappings,
            msg="Impute values not changed in transform",
        )

    def test_non_mappable_rows_raises_error(self):
        """Test that rows that cannot be mapped result in an exception."""
        df = d.create_df_1()

        x = NominalToIntegerTransformer(columns=["a", "b"])

        x.fit(df)

        df["a"] = df["a"] + 1

        with pytest.raises(
            ValueError,
            match="NominalToIntegerTransformer: nulls would be introduced into column a from levels not present in mapping",
        ):
            x.transform(df)

    @pytest.mark.parametrize(
        ("df", "expected"),
        ta.pandas.adjusted_dataframe_params(d.create_df_1(), expected_df_1()),
    )
    def test_expected_output(self, df, expected):
        """Test that the output is expected from transform."""
        x = NominalToIntegerTransformer(columns=["a", "b"])

        # set the mapping dict directly rather than fitting x on df so test works with helpers
        x.mappings = {
            "a": {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5},
            "b": {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5},
        }

        df_transformed = x.transform(df)

        ta.equality.assert_frame_equal_msg(
            actual=df_transformed,
            expected=expected,
            msg_tag="Unexpected values in NominalToIntegerTransformer.transform",
        )


class TestInverseTransform:
    """Tests for NominalToIntegerTransformer.inverse_transform()."""

    def test_check_is_fitted_called(self, mocker):
        """Test that BaseTransformer check_is_fitted called."""
        df = d.create_df_1()

        x = NominalToIntegerTransformer(columns=["a", "b"])

        x.fit(df)

        df_transformed = x.transform(df)

        expected_call_args = {0: {"args": (["mappings"],), "kwargs": {}}}

        with ta.functions.assert_function_call(
            mocker,
            tubular.base.BaseTransformer,
            "check_is_fitted",
            expected_call_args,
        ):
            x.inverse_transform(df_transformed)

    @pytest.mark.parametrize(
        ("df", "expected"),
        ta.pandas.adjusted_dataframe_params(d.create_df_1(), d.create_df_1()),
    )
    def test_expected_output(self, df, expected):
        """Test that transform then inverse_transform gets back to the original df."""
        x = NominalToIntegerTransformer(columns=["a", "b"])

        # set the mapping dict directly rather than fitting x on df so test works with helpers
        x.mappings = {
            "a": {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5},
            "b": {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5},
        }

        df_transformed = x.transform(df)

        df_transformed_back = x.inverse_transform(df_transformed)

        ta.equality.assert_frame_equal_msg(
            actual=df_transformed_back,
            expected=expected,
            msg_tag="transform reverse does not get back to original",
        )

    def test_non_mappable_rows_raises_error(self):
        """Test that rows that cannot be mapped result in an exception."""
        x = NominalToIntegerTransformer(columns=["a", "b"])

        df = d.create_df_1()

        x.fit(df)

        df_transformed = x.transform(df)

        df_transformed["b"] = df_transformed["b"] + 1

        with pytest.raises(
            ValueError,
            match="NominalToIntegerTransformer: nulls introduced from levels not present in mapping for column: b",
        ):
            x.inverse_transform(df_transformed)

    def test_learnt_values_not_modified(self):
        """Test that the mappings from fit are not changed in inverse_transform."""
        df = d.create_df_1()

        x = NominalToIntegerTransformer(columns=["a", "b"])

        x.fit(df)

        x2 = NominalToIntegerTransformer(columns=["a", "b"])

        x2.fit(df)

        df_transformed = x2.transform(df)

        x2.inverse_transform(df_transformed)

        ta.equality.assert_equal_dispatch(
            expected=x.mappings,
            actual=x2.mappings,
            msg="Impute values not changed in inverse_transform",
        )
