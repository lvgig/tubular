from copy import deepcopy

import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.exceptions import NotFittedError

import tests.test_data as d
from tests import utils as u
from tests.base_tests import (
    ColumnStrListInitTests,
    GenericFitTests,
    OtherBaseBehaviourTests,
)


@pytest.fixture
def library(request):
    return request.param


@pytest.fixture()
def test_fit_df(request):
    library = getattr(request, "param", "pandas")
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            "b": ["a", "b", "c", "d", "e", "f", np.nan],
            "c": ["a", "b", "c", "d", "e", "f", np.nan],
        },
    )
    if library == "polars":
        return pl.from_pandas(df)
    return df


@pytest.fixture()
def expected_df_3(request):
    library = getattr(request, "param", "pandas")
    df3 = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, np.nan],
            "b": ["a", "b", "c", "d", "e", "f", "g"],
            "c": ["a", "b", "c", "d", "e", "f", "f"],
        },
    )
    df3["c"] = df3["c"].astype("category")
    if library == "polars":
        return pl.from_pandas(df3)
    return df3


@pytest.fixture()
def expected_df_2(request):
    library = getattr(request, "param", "pandas")
    df2 = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, np.nan],
            "b": ["a", "b", "c", "d", "e", "f", "g"],
            "c": ["a", "b", "c", "d", "e", "f", np.nan],
        },
    )
    df2["c"] = df2["c"].astype("category")
    if library == "polars":
        return pl.from_pandas(df2)
    return df2


@pytest.fixture()
def expected_df_1(request):
    library = getattr(request, "param", "pandas")
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            "b": ["a", "b", "c", "d", "e", "f", np.nan],
            "c": ["a", "b", "c", "d", "e", "f", np.nan],
        },
    )
    df["c"] = df["c"].astype("category")
    if library == "polars":
        return pl.from_pandas(df)
    return df


class GenericImputerTransformTests:
    @pytest.mark.parametrize("test_fit_df", ["pandas", "polars"], indirect=True)
    def test_not_fitted_error_raised(self, test_fit_df, initialized_transformers):
        if initialized_transformers[self.transformer_name].FITS:
            with pytest.raises(NotFittedError):
                initialized_transformers[self.transformer_name].transform(test_fit_df)

    @pytest.mark.parametrize("library", ["pandas", "polars"], indirect=True)
    def test_impute_value_unchanged(self, library, initialized_transformers):
        """Test that self.impute_value is unchanged after transform."""
        create_df_1 = d.create_df_1(library=library)
        transformer = initialized_transformers[self.transformer_name]
        transformer.impute_values_ = {"b": 1}
        impute_values = deepcopy(transformer.impute_values_)

        transformer.transform(create_df_1)

        assert (
            transformer.impute_values_ == impute_values
        ), "impute_values_ changed in transform"

    @pytest.mark.parametrize(
        ("library", "expected_df_1"),
        [("pandas", "pandas"), ("polars", "polars")],
        indirect=["expected_df_1"],
    )
    def test_expected_output_1(self, library, expected_df_1, initialized_transformers):
        """Test that transform is giving the expected output when applied to float column."""
        # Create the DataFrame using the library parameter
        create_df_2 = d.create_df_2(library=library)

        # Initialize the transformer
        transformer = initialized_transformers[self.transformer_name]
        transformer.impute_values_ = {"a": 7}
        transformer.columns = ["a"]

        # Transform the DataFrame
        df_transformed = transformer.transform(create_df_2)

        # Check outcomes for single rows
        for i in range(len(df_transformed)):
            if isinstance(df_transformed, pd.DataFrame):
                u.assert_frame_equal_dispatch(
                    df_transformed.iloc[[i]],
                    expected_df_1.iloc[[i]],
                )
            else:  # Assuming Polars DataFrame
                u.assert_frame_equal_dispatch(df_transformed[i], expected_df_1[i])

    @pytest.mark.parametrize(
        ("library", "expected_df_2"),
        [("pandas", "pandas"), ("polars", "polars")],
        indirect=["expected_df_2"],
    )
    def test_expected_output_2(self, library, expected_df_2, initialized_transformers):
        """Test that transform is giving the expected output when applied to object column."""
        # Create the DataFrame using the library parameter
        create_df_2 = d.create_df_2(library=library)

        # Initialize the transformer
        transformer = initialized_transformers[self.transformer_name]
        transformer.impute_values_ = {"b": "g"}
        transformer.columns = ["b"]

        # Transform the DataFrame
        df_transformed = transformer.transform(create_df_2)

        # Check outcomes for single rows
        for i in range(len(df_transformed)):
            if isinstance(df_transformed, pd.DataFrame):
                u.assert_frame_equal_dispatch(
                    df_transformed.iloc[[i]],
                    expected_df_2.iloc[[i]],
                )
            else:  # Assuming Polars DataFrame
                u.assert_frame_equal_dispatch(df_transformed[i], expected_df_2[i])

    @pytest.mark.parametrize(
        ("library", "expected_df_3"),
        [("pandas", "pandas"), ("polars", "polars")],
        indirect=["expected_df_3"],
    )
    def test_expected_output_3(self, library, expected_df_3, initialized_transformers):
        """Test that transform is giving the expected output when applied to object and categorical columns."""
        # Create the DataFrame using the library parameter
        create_df_2 = d.create_df_2(library=library)

        # Initialize the transformer
        transformer = initialized_transformers[self.transformer_name]
        transformer.impute_values_ = {"b": "g", "c": "f"}
        transformer.columns = ["b", "c"]

        # Transform the DataFrame
        df_transformed = transformer.transform(create_df_2)

        # ArbitraryImputer will add a new categorical level to cat columns,
        # make sure expected takes this into account
        if self.transformer_name == "ArbitraryImputer" and isinstance(
            expected_df_3,
            pd.DataFrame,
        ):
            expected_df_3["c"] = expected_df_3["c"].cat.add_categories(
                transformer.impute_values_["c"],
            )

        # Check if the DataFrame matches the expected DataFrame
        u.assert_frame_equal_dispatch(df_transformed, expected_df_3)

        # Check outcomes for single rows
        for i in range(len(df_transformed)):
            if isinstance(df_transformed, pd.DataFrame):
                u.assert_frame_equal_dispatch(
                    df_transformed.iloc[[i]],
                    expected_df_3.iloc[[i]],
                )
            else:  # Assuming Polars DataFrame
                u.assert_frame_equal_dispatch(df_transformed[i], expected_df_3[i])


@pytest.fixture
def expected_df_weights():
    """Expected output for test_nulls_imputed_correctly_weights."""
    df = d.create_df_9()

    for col in ["b"]:
        df.loc[df[col].isna(), col] = 4

    return df


class GenericImputerTransformTestsWeight:
    @pytest.mark.parametrize(
        ("library", "expected_df_weights"),
        [("pandas", "pandas"), ("polars", "polars")],
        indirect=["expected_df_weights"],
    )
    def test_nulls_imputed_correctly_weights(
        self,
        library,
        expected_df_weights,
        minimal_attribute_dict,
        uninitialized_transformers,
    ):
        """Test missing values are filled with the correct values - and unrelated columns are not changed
        (when weight is used).
        """
        # Create the DataFrame using the library parameter
        df = d.create_df_9(library=library)

        args = minimal_attribute_dict[self.transformer_name].copy()
        args["weights_column"] = "c"

        transformer = uninitialized_transformers[self.transformer_name](**args)

        # Set the impute values dict directly rather than fitting x on df so test works with helpers
        transformer.impute_values_ = {"b": 4}

        df_transformed = transformer.transform(df)

        # Check if the DataFrame matches the expected DataFrame
        u.assert_frame_equal_dispatch(df_transformed, expected_df_weights)

        # Check outcomes for single rows
        for i in range(len(df_transformed)):
            if isinstance(df_transformed, pd.DataFrame):
                u.assert_frame_equal_dispatch(
                    df_transformed.iloc[[i]],
                    expected_df_weights.iloc[[i]],
                )
            else:  # Assuming Polars DataFrame
                u.assert_frame_equal_dispatch(df_transformed[i], expected_df_weights[i])

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_learnt_values_not_modified_weights(
        self,
        library,
        minimal_attribute_dict,
        uninitialized_transformers,
    ):
        """Test that the impute_values_ from fit are not changed in transform - when using weights."""
        df = d.create_df_9(library=library)

        args = minimal_attribute_dict[self.transformer_name].copy()
        args["columns"] = ["a", "b"]
        args["weights_column"] = "c"

        transformer1 = uninitialized_transformers[self.transformer_name](**args)

        transformer1.fit(df)

        transformer2 = uninitialized_transformers[self.transformer_name](**args)

        transformer2.fit_transform(df)

        # Check if the impute_values_ are the same
        assert (
            transformer1.impute_values_ == transformer2.impute_values_
        ), f"Impute values changed in transform for {self.transformer_name}"


class TestInit(ColumnStrListInitTests):
    """Generic tests for transformer.init()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseImputer"


class TestFit(GenericFitTests):
    """Generic tests for transformer.fit()"""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseImputer"


class TestTransform(GenericImputerTransformTests):
    """Tests for BaseImputer.transform."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseImputer"


class TestOtherBaseBehaviour(OtherBaseBehaviourTests):
    """
    Class to run tests for BaseTransformerBehaviour outside the three standard methods.

    May need to overwite specific tests in this class if the tested transformer modifies this behaviour.
    """

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseImputer"
