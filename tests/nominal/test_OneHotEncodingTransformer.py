import narwhals as nw
import pytest
from test_BaseNominalTransformer import GenericNominalTransformTests

import tests.test_data as d
from tests.base_tests import (
    ColumnStrListInitTests,
    DropOriginalInitMixinTests,
    DropOriginalTransformMixinTests,
    GenericFitTests,
    GenericTransformTests,
    SeparatorInitMixintests,
)
from tests.utils import assert_frame_equal_dispatch, dataframe_init_dispatch
from tubular.nominal import OneHotEncodingTransformer


class TestInit(
    SeparatorInitMixintests,
    DropOriginalInitMixinTests,
    ColumnStrListInitTests,
):
    """Generic tests for transformer.init()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "OneHotEncodingTransformer"


class TestFit(GenericFitTests):
    """Generic tests for transformer.fit()"""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "OneHotEncodingTransformer"

    @pytest.mark.parametrize(
        "library",
        ["pandas", "polars"],
    )
    def test_nulls_in_X_error(self, library):
        """Test that an exception is raised if X has nulls in column to be fit on."""
        df = d.create_df_2(library=library)

        transformer = OneHotEncodingTransformer(columns=["b", "c"])

        with pytest.raises(
            ValueError,
            match="OneHotEncodingTransformer: column b has nulls - replace before proceeding",
        ):
            transformer.fit(df)

    @pytest.mark.parametrize(
        "library",
        ["pandas", "polars"],
    )
    def test_fields_with_over_100_levels_error(self, library):
        """Test that OneHotEncodingTransformer.fit on fields with more than 100 levels raises error."""
        df_dict = {"a": [1] * 101, "b": list(range(101))}

        df = dataframe_init_dispatch(library=library, dataframe_dict=df_dict)

        transformer = OneHotEncodingTransformer(columns=["a", "b"])

        with pytest.raises(
            ValueError,
            match="OneHotEncodingTransformer: column b has over 100 unique values - consider another type of encoding",
        ):
            transformer.fit(df)


class TestTransform(
    DropOriginalTransformMixinTests,
    GenericNominalTransformTests,
    GenericTransformTests,
):
    """Tests for transformer.transform."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "OneHotEncodingTransformer"

    def create_OneHotEncoderTransformer_test_df_1(self, library="pandas"):
        """Create DataFrame to test OneHotEncoderTransformer

        binary columns are representative of transformed output of column b

        Parameters
        ----------
        library : str
            Whether to return polars of pandas df

        """

        df_dict = {
            "a": [4, 2, 2, 1, 3],
            "b": ["x", "z", "y", "x", "x"],
            "c": ["c", "a", "a", "c", "b"],
            "b_x": [1.0, 0.0, 0.0, 1.0, 1.0],
            "b_y": [0.0, 0.0, 1.0, 0.0, 0.0],
            "b_z": [0.0, 1.0, 0.0, 0.0, 0.0],
        }

        df = dataframe_init_dispatch(library=library, dataframe_dict=df_dict)

        df = nw.from_native(df)
        df = df.with_columns(nw.col("c").cast(nw.dtypes.Categorical))

        return df.to_native()

    def create_OneHotEncoderTransformer_test_df_2(self, library="pandas"):
        """Create DataFrame to test OneHotEncoderTransformer

        binary columns are representative of transformed output of all columns

        Parameters
        ----------
        library : str
            Whether to return polars of pandas df

        """

        df_dict = {
            "a": [1, 5, 2, 3, 3],
            "b": ["w", "w", "z", "y", "x"],
            "c": ["a", "a", "c", "b", "a"],
            "a_1": [1.0, 0.0, 0.0, 0.0, 0.0],
            "a_2": [0.0, 0.0, 1.0, 0.0, 0.0],
            "a_3": [0.0, 0.0, 0.0, 1.0, 1.0],
            "a_4": [0.0, 0.0, 0.0, 0.0, 0.0],
            "b_x": [0.0, 0.0, 0.0, 0.0, 1.0],
            "b_y": [0.0, 0.0, 0.0, 1.0, 0.0],
            "b_z": [0.0, 0.0, 1.0, 0.0, 0.0],
            "c_a": [1.0, 1.0, 0.0, 0.0, 1.0],
            "c_b": [0.0, 0.0, 0.0, 1.0, 0.0],
            "c_c": [0.0, 0.0, 1.0, 0.0, 0.0],
        }

        df = dataframe_init_dispatch(dataframe_dict=df_dict, library=library)

        df = nw.from_native(df)
        df = df.with_columns(nw.col("c").cast(nw.dtypes.Categorical))

        return df.to_native()

    def test_non_mappable_rows_exception_raised(self):
        """Test inherited from GenericBaseNominalTransformerTests needs to be overwritten,
        inherited test tests the mapping attribute, which OHE transfomer doesn't have.
        """

    @pytest.mark.parametrize(
        "library",
        ["pandas", "polars"],
    )
    def test_non_numeric_column_error_1(self, library):
        """Test that transform will raise an error if a column to transform has nulls."""
        df_train = d.create_df_1(library=library)
        df_test = d.create_df_2(library=library)

        transformer = OneHotEncodingTransformer(columns=["b"])

        transformer.fit(df_train)

        with pytest.raises(
            ValueError,
            match="OneHotEncodingTransformer: column b has nulls - replace before proceeding",
        ):
            transformer.transform(df_test)

    @pytest.mark.parametrize(
        "library",
        ["pandas", "polars"],
    )
    def test_expected_output(self, library):
        """Test that OneHotEncodingTransformer.transform encodes the feature correctly.

        Also tests that OneHotEncodingTransformer.transform does not modify unrelated columns.
        """
        # transformer is fit on the whole dataset separately from the input df to work with the decorators
        columns = ["b"]
        df_train = d.create_df_7(library=library)
        df_train = nw.from_native(df_train)

        df_test = df_train.clone()
        expected = self.create_OneHotEncoderTransformer_test_df_1(library=library)

        transformer = OneHotEncodingTransformer(columns=columns)
        transformer.fit(df_train)

        df_transformed = transformer.transform(df_test.to_native())

        expected = nw.from_native(expected)
        for col in [
            column + f"_{value}"
            for column in columns
            for value in df_train.select(nw.col(column).unique())
            .get_column(column)
            .to_list()
        ]:
            expected = expected.with_columns(nw.col(col).cast(nw.Boolean))

        assert_frame_equal_dispatch(expected.to_native(), df_transformed)

        # also test single row transform
        for i in range(len(df_test)):
            df_transformed_row = transformer.transform(df_test[[i]].to_native())
            df_expected_row = expected[[i]].to_native()

            assert_frame_equal_dispatch(
                df_transformed_row,
                df_expected_row,
            )

    @pytest.mark.parametrize(
        "library",
        ["pandas", "polars"],
    )
    def test_categories_not_modified(self, library):
        """Test that the categories from fit are not changed in transform."""
        df_train = d.create_df_1(library=library)
        df_test = d.create_df_7(library=library)

        transformer = OneHotEncodingTransformer(columns=["a", "b"], verbose=False)
        transformer2 = OneHotEncodingTransformer(columns=["a", "b"], verbose=False)

        transformer.fit(df_train)
        transformer2.fit(df_train)

        transformer.transform(df_test)

        assert (
            transformer2.categories_ == transformer.categories_
        ), f"categories_ modified during transform, pre transform had {transformer2.categories_} but post transform has {transformer.categories_}"

    @pytest.mark.parametrize(
        "library",
        ["pandas", "polars"],
    )
    def test_renaming_feature_works_as_expected(self, library):
        """Test OneHotEncodingTransformer.transform() is renaming features correctly."""
        df = d.create_df_7(library=library)
        df = df[["b", "c"]]

        transformer = OneHotEncodingTransformer(
            columns=["b", "c"],
            separator="|",
            drop_original=True,
        )

        transformer.fit(df)

        df_transformed = transformer.transform(df)

        expected_columns = ["b|x", "b|y", "b|z", "c|a", "c|b", "c|c"]

        df_transformed = nw.from_native(df_transformed)
        actual_columns = df_transformed.columns

        assert (
            set(expected_columns) == set(actual_columns)
        ), f"renaming columns feature in OneHotEncodingTransformer.transform, expected {expected_columns} but got {actual_columns}"

    @pytest.mark.parametrize(
        "library",
        ["pandas", "polars"],
    )
    def test_warning_generated_by_unseen_categories(self, library):
        """Test OneHotEncodingTransformer.transform triggers a warning for unseen categories."""
        df_train = d.create_df_7(library=library)
        df_test = d.create_df_8(library=library)

        transformer = OneHotEncodingTransformer(columns=["a", "b", "c"], verbose=True)

        transformer.fit(df_train)

        with pytest.warns(UserWarning, match="unseen categories"):
            transformer.transform(df_test)

    @pytest.mark.parametrize(
        "library",
        ["pandas", "polars"],
    )
    def test_unseen_categories_encoded_as_all_zeroes(self, library):
        """Test OneHotEncodingTransformer.transform encodes unseen categories correctly (all 0s)."""
        # transformer is fit on the whole dataset separately from the input df to work with the decorators
        df_train = d.create_df_7(library=library)

        columns = ["a", "b", "c"]
        x = OneHotEncodingTransformer(columns=columns, verbose=False)
        x.fit(df_train)

        df_test = d.create_df_8(library=library)
        expected = self.create_OneHotEncoderTransformer_test_df_2(library=library)

        df_transformed = x.transform(df_test)

        df_train = nw.from_native(df_train)
        expected = nw.from_native(expected)

        for col in [
            column + f"_{value}"
            for column in columns
            for value in df_train.select(nw.col(column).unique())
            .get_column(column)
            .to_list()
        ]:
            expected = expected.with_columns(nw.col(col).cast(nw.Boolean))

        column_order = expected.columns
        assert_frame_equal_dispatch(expected.to_native(), df_transformed[column_order])

        # also test single row transform
        df_test = nw.from_native(df_test)
        for i in range(len(df_test)):
            df_transformed_row = x.transform(df_test[[i]].to_native())
            df_expected_row = expected[[i]].to_native()

            assert_frame_equal_dispatch(
                df_transformed_row[column_order],
                df_expected_row,
            )
