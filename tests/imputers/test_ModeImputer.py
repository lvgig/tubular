import narwhals as nw
import numpy as np
import pytest

import tests.test_data as d
from tests.base_tests import (
    ColumnStrListInitTests,
    GenericFitTests,
    GenericTransformTests,
    OtherBaseBehaviourTests,
    WeightColumnFitMixinTests,
    WeightColumnInitMixinTests,
)
from tests.imputers.test_BaseImputer import (
    GenericImputerTransformTests,
    GenericImputerTransformTestsWeight,
)
from tests.utils import dataframe_init_dispatch
from tubular.imputers import ModeImputer


def input_df_nan(library="pandas"):
    return dataframe_init_dispatch(
        dataframe_dict={"a": [None, None, None]},
        library=library,
    )


class TestInit(ColumnStrListInitTests, WeightColumnInitMixinTests):
    """Generic tests for transformer.init()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "ModeImputer"


class TestFit(WeightColumnFitMixinTests, GenericFitTests):
    """Generic tests for transformer.fit()"""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "ModeImputer"

    @pytest.mark.parametrize(
        "library",
        ["pandas", "polars"],
    )
    def test_learnt_values(self, library):
        """Test that the impute values learnt during fit are expected."""
        df = d.create_df_6(library=library)

        x = ModeImputer(columns=["a", "b", "c"])

        x.fit(df)

        expected_impute_values = {
            "a": 2,
            "b": "a",
            "c": "f",
        }

        assert (
            x.impute_values_ == expected_impute_values
        ), f"impute_values_ attribute not expected, expected {expected_impute_values} but got {x.impute_values_}"

    @pytest.mark.parametrize(
        "library",
        ["pandas", "polars"],
    )
    def test_learnt_values_tied(self, library):
        """Test that the impute values learnt during fit are expected - when mode is tied."""
        df = d.create_df_3(library=library)

        x = ModeImputer(columns=["a"])

        with pytest.warns(
            UserWarning,
            match="ModeImputer: The Mode of column a is tied, will sort in descending order and return first candidate",
        ):
            x.fit(df)

        expected_impute_values = {
            "a": 6,
        }

        assert (
            x.impute_values_ == expected_impute_values
        ), f"impute_values_ attribute not expected, expected {expected_impute_values} but got {x.impute_values_}"

    @pytest.mark.parametrize(
        "library",
        ["pandas", "polars"],
    )
    def test_learnt_values_tied_weighted(self, library):
        """
        Test that the impute values learnt during fit are expected -
        when mode is tied and transformer is weighted.
        """
        df = d.create_df_3(library=library)

        df = nw.from_native(df)
        native_namespace = nw.get_native_namespace(df)

        weights_column = "weights_column"
        x = ModeImputer(columns=["a"], weights_column=weights_column)

        # setup weights column
        df = df.with_columns(
            nw.new_series(
                name=weights_column,
                values=[1] * len(df),
                native_namespace=native_namespace,
            ),
        )

        df = nw.to_native(df)

        with pytest.warns(
            UserWarning,
            match="ModeImputer: The Mode of column a is tied, will sort in descending order and return first candidate",
        ):
            x.fit(df)

        expected_impute_values = {
            "a": 6,
        }

        assert (
            x.impute_values_ == expected_impute_values
        ), f"impute_values_ attribute not expected, expected {expected_impute_values} but got {x.impute_values_}"

    @pytest.mark.parametrize(
        "library",
        ["pandas", "polars"],
    )
    def test_nan_learnt_values(self, library):
        """Test behaviour when learnt value is None."""
        x = ModeImputer(columns=["a"])

        df = input_df_nan(library)

        with pytest.warns(
            UserWarning,
            match="ModeImputer: The Mode of column a is None",
        ):
            x.fit(df)

        expected_impute_values = {"a": None}

        assert (
            x.impute_values_ == expected_impute_values
        ), f"impute_values_ attribute not as expected, expected {expected_impute_values} but got {x.impute_values_}"

    @pytest.mark.parametrize(
        "library",
        ["pandas", "polars"],
    )
    def test_nan_learnt_values_weighted(self, library):
        """Test behaviour when learnt value is None - when weights are used."""
        weights_column = "weights_column"
        x = ModeImputer(columns=["a"], weights_column=weights_column)

        df = d.create_weighted_imputers_test_df(library=library)

        df = nw.from_native(df)
        native_namespace = nw.get_native_namespace(df)

        # replace 'a' with all null values to trigger warning
        df = df.with_columns(
            nw.new_series(
                name="a",
                values=[None] * len(df),
                native_namespace=native_namespace,
            ),
        )

        df = df.to_native()

        with pytest.warns(
            UserWarning,
            match="ModeImputer: The Mode of column a is None",
        ):
            x.fit(df)

        expected_impute_values = {"a": None}

        assert (
            x.impute_values_ == expected_impute_values
        ), f"impute_values_ attribute not as expected, expected {expected_impute_values} but got {x.impute_values_}"

    @pytest.mark.parametrize(
        "library",
        ["pandas", "polars"],
    )
    def test_learnt_values_weighted_df(self, library):
        """Test that the impute values learnt during fit are expected when df is weighted."""
        df = d.create_weighted_imputers_test_df(library=library)

        x = ModeImputer(columns=["a", "b", "c", "d"], weights_column="weights_column")

        x.fit(df)

        expected_impute_values = {
            "a": np.float64(5.0),
            "b": "e",
            "c": "f",
            "d": np.float64(1.0),
        }

        assert (
            x.impute_values_ == expected_impute_values
        ), f"impute_values_ attribute not as expected, expected {expected_impute_values} but got {x.impute_values_}"


class TestTransform(
    GenericTransformTests,
    GenericImputerTransformTestsWeight,
    GenericImputerTransformTests,
):
    """Tests for transformer.transform."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "ModeImputer"


class TestOtherBaseBehaviour(OtherBaseBehaviourTests):
    """
    Class to run tests for BaseTransformerBehaviour outside the three standard methods.

    May need to overwite specific tests in this class if the tested transformer modifies this behaviour.
    """

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "ModeImputer"
