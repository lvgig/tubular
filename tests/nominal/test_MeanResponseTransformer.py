from itertools import product

import numpy as np
import pandas as pd
import pytest
import test_aide as ta
from pandas.testing import assert_series_equal

from tests.base_tests import (
    ColumnStrListInitTests,
    GenericFitTests,
    GenericTransformTests,
    OtherBaseBehaviourTests,
    WeightColumnFitMixinTests,
    WeightColumnInitMixinTests,
)
from tubular.nominal import MeanResponseTransformer


# Dataframe used exclusively in this testing script
def create_MeanResponseTransformer_test_df():
    """Create DataFrame to use MeanResponseTransformer tests that correct values are.

    DataFrame column a is the response, the other columns are categorical columns
    of types; object, category, int, float, bool.

    """
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "b": ["a", "b", "c", "d", "e", "f"],
            "c": ["a", "b", "c", "d", "e", "f"],
            "d": [1, 2, 3, 4, 5, 6],
            "e": [1, 2, 3, 4, 5, 6.0],
            "f": [False, False, False, True, True, True],
            "multi_level_response": [
                "blue",
                "blue",
                "yellow",
                "yellow",
                "green",
                "green",
            ],
        },
    )

    df["c"] = df["c"].astype("category")

    return df


# Dataframe used exclusively in this testing script
def create_MeanResponseTransformer_test_df_unseen_levels():
    """Create DataFrame to use in MeanResponseTransformer tests that check correct values are
    generated when using transform method on data with unseen levels.
    DataFrame column a is the response, the other columns are categorical columns
    of types; object, category, int, float, bool.

    """
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 3.0],
            "b": ["a", "b", "c", "d", "e", "f", "g", "h"],
            "c": ["a", "b", "c", "d", "e", "f", "g", "h"],
            "d": [1, 2, 3, 4, 5, 6, 7, 8],
            "e": [1, 2, 3, 4, 5, 6.0, 7, 8],
            "f": [False, False, False, True, True, True, True, False],
            "multi_level_response": [
                "blue",
                "blue",
                "yellow",
                "yellow",
                "green",
                "green",
                "yellow",
                "blue",
            ],
        },
    )

    df["c"] = df["c"].astype("category")

    return df


@pytest.fixture()
def learnt_mapping_dict():
    full_dict = {}

    b_dict = {
        "b": {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0, "e": 5.0, "f": 6.0},
        "b_blue": {"a": 1.0, "b": 1.0, "c": 0.0, "d": 0.0, "e": 0.0, "f": 0.0},
        "b_yellow": {"a": 0.0, "b": 0.0, "c": 1.0, "d": 1.0, "e": 0.0, "f": 0.0},
        "b_green": {"a": 0.0, "b": 0.0, "c": 0.0, "d": 0.0, "e": 1.0, "f": 1.0},
    }

    # c matches b, but is categorical (see create_MeanResponseTransformer_test_df)
    c_dict = {
        "c" + suffix: b_dict["b" + suffix]
        for suffix in ["", "_blue", "_yellow", "_green"]
    }

    full_dict.update(b_dict)
    full_dict.update(c_dict)

    return full_dict


@pytest.fixture()
def learnt_unseen_levels_encoding_dict_mean():
    return_dict = {
        "b": (1.0 + 2.0 + 3.0 + 4.0 + 5.0 + 6.0) / 6,
        "b_blue": (1.0 + 1.0 + 0.0 + 0.0 + 0.0 + 0.0) / 6,
        "b_yellow": (0.0 + 0.0 + 1.0 + 1.0 + 0.0 + 0.0) / 6,
        "b_green": (0.0 + 0.0 + 0.0 + 0.0 + 1.0 + 1.0) / 6,
    }

    for key in return_dict:
        return_dict[key] = np.float32(return_dict[key])
    return return_dict


@pytest.fixture()
def learnt_unseen_levels_encoding_dict_median():
    return_dict = {
        "b": (3.0 + 4.0) / 2,
        "b_blue": (0.0 + 0.0) / 2,
        "b_yellow": (0.0 + 0.0) / 2,
        "b_green": (0.0 + 0.0) / 2,
    }

    for key in return_dict:
        return_dict[key] = np.float32(return_dict[key])
    return return_dict


@pytest.fixture()
def learnt_unseen_levels_encoding_dict_highest():
    return_dict = {
        "b": 6.0,
        "b_blue": 1.0,
        "b_yellow": 1.0,
        "b_green": 1.0,
    }
    for key in return_dict:
        return_dict[key] = np.float32(return_dict[key])
    return return_dict


@pytest.fixture()
def learnt_unseen_levels_encoding_dict_lowest():
    return_dict = {
        "b": 1.0,
        "b_blue": 0.0,
        "b_yellow": 0.0,
        "b_green": 0.0,
    }

    for key in return_dict:
        return_dict[key] = np.float32(return_dict[key])
    return return_dict


@pytest.fixture()
def learnt_unseen_levels_encoding_dict_arbitrary():
    return_dict = {
        "b": 22.0,
        "b_blue": 22.0,
        "b_yellow": 22.0,
        "b_green": 22.0,
    }
    for key in return_dict:
        return_dict[key] = np.float32(return_dict[key])
    return return_dict


class TestInit(ColumnStrListInitTests, WeightColumnInitMixinTests):
    """Tests for MeanResponseTransformer.init()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "MeanResponseTransformer"

    def test_prior_not_int_error(self):
        """Test that an exception is raised if prior is not an int."""
        with pytest.raises(TypeError, match="prior should be a int"):
            MeanResponseTransformer(prior="1")

    def test_prior_not_positive_int_error(self):
        """Test that an exception is raised if prior is not a positive int."""
        with pytest.raises(ValueError, match="prior should be positive int"):
            MeanResponseTransformer(prior=-1)

    @pytest.mark.parametrize("level", [{"dict": 1}, 2, 2.5])
    def test_level_wrong_type_error(self, level):
        with pytest.raises(
            TypeError,
            match=f"Level should be a NoneType, list or str but got {type(level)}",
        ):
            MeanResponseTransformer(level=level)

    def test_unseen_level_handling_incorrect_value_error(self):
        """Test that an exception is raised if unseen_level_handling is an incorrect value."""
        with pytest.raises(
            ValueError,
            match="unseen_level_handling should be the option: Mean, Median, Lowest, Highest or an arbitrary int/float value",
        ):
            MeanResponseTransformer(unseen_level_handling="AAA")

    def test_return_type_handling_incorrect_value_error(self):
        """Test that an exception is raised if return_type is an incorrect value."""
        with pytest.raises(
            ValueError,
            match="return_type should be one of: 'float64', 'float32'",
        ):
            MeanResponseTransformer(return_type="int")


class TestPriorRegularisation:
    "tests for _prior_regularisation method."

    def test_output1(self):
        "Test output of method."
        x = MeanResponseTransformer(columns="a", prior=3)

        x.fit(X=pd.DataFrame({"a": [1, 2]}), y=pd.Series([2, 3]))

        expected1 = (1 * 1 + 3 * 2.5) / (1 + 3)

        expected2 = (2 * 2 + 3 * 2.5) / (2 + 3)

        expected = pd.Series([expected1, expected2])

        output = x._prior_regularisation(
            cat_freq=pd.Series([1, 2]),
            target_means=pd.Series([1, 2]),
        )

        assert_series_equal(expected, output)

    @pytest.mark.parametrize(
        "dtype",
        ["object", "category"],
    )
    def test_output2(self, dtype):
        "Test output of method - for category and object dtypes"
        x = MeanResponseTransformer(columns="a", prior=0)

        df = pd.DataFrame({"a": ["a", "b"]})
        df["a"] = df["a"].astype(dtype)

        x.fit(X=df, y=pd.Series([2, 3]))

        expected1 = (1 * 1) / (1)

        expected2 = (2 * 2) / (2)

        expected = pd.Series([expected1, expected2])

        output = x._prior_regularisation(
            cat_freq=pd.Series([1, 2]),
            target_means=pd.Series([1, 2]),
        )

        assert_series_equal(expected, output)


class TestFit(GenericFitTests, WeightColumnFitMixinTests):
    @classmethod
    def setup_class(cls):
        cls.transformer_name = "MeanResponseTransformer"

    @pytest.mark.parametrize(
        ("level", "target_column", "unseen_level_handling"),
        [
            (None, "a", "Mean"),
            ("all", "multi_level_response", 32),
            (["yellow", "blue"], "multi_level_response", "Highest"),
        ],
    )
    def test_response_column_nulls_error(
        self,
        level,
        target_column,
        unseen_level_handling,
    ):
        """Test that an exception is raised if nulls are present in response_column."""
        df = create_MeanResponseTransformer_test_df()
        df.loc[1, target_column] = np.nan

        x = MeanResponseTransformer(
            columns=["b"],
            level=level,
            unseen_level_handling=unseen_level_handling,
        )

        with pytest.raises(
            ValueError,
            match="MeanResponseTransformer: y has 1 null values",
        ):
            x.fit(df, df[target_column])

    @pytest.mark.parametrize(
        ("level", "target_column", "unseen_level_handling"),
        [
            (None, "a", "Mean"),
            (None, "a", "Lowest"),
        ],
    )
    def test_correct_mappings_stored_numeric_response(
        self,
        learnt_mapping_dict,
        level,
        target_column,
        unseen_level_handling,
    ):
        "Test that the mapping dictionary created in fit has the correct keys and values."
        df = create_MeanResponseTransformer_test_df()
        columns = ["b", "c"]
        x = MeanResponseTransformer(
            columns=columns,
            level=level,
            unseen_level_handling=unseen_level_handling,
        )
        x.fit(df, df[target_column])

        assert x.columns == columns, "Columns attribute changed in fit"

        for column in x.columns:
            actual = x.mappings[column]
            expected = learnt_mapping_dict[column]
            assert actual == expected

    @pytest.mark.parametrize(
        ("level", "target_column", "unseen_level_handling"),
        [
            (["blue"], "multi_level_response", "Median"),
            ("all", "multi_level_response", 32),
            (["yellow", "blue"], "multi_level_response", "Highest"),
        ],
    )
    def test_correct_mappings_stored_categorical_response(
        self,
        learnt_mapping_dict,
        level,
        target_column,
        unseen_level_handling,
    ):
        "Test that the mapping dictionary created in fit has the correct keys and values."
        df = create_MeanResponseTransformer_test_df()
        columns = ["b", "c"]
        x = MeanResponseTransformer(
            columns=columns,
            level=level,
            unseen_level_handling=unseen_level_handling,
        )
        x.fit(df, df[target_column])

        if level == "all":
            expected_created_cols = {
                prefix + "_" + suffix
                for prefix, suffix in product(
                    columns,
                    df[target_column].unique().tolist(),
                )
            }

        else:
            expected_created_cols = {
                prefix + "_" + suffix for prefix, suffix in product(columns, level)
            }
        assert (
            set(x.mapped_columns) == expected_created_cols
        ), "Stored mapped columns are not as expected"

        for column in x.mapped_columns:
            actual = x.mappings[column]
            expected = learnt_mapping_dict[column]
            assert actual == expected

    @pytest.mark.parametrize(
        ("level", "target_column", "unseen_level_handling"),
        [
            (None, "a", "Mean"),
            (None, "a", "Median"),
            (None, "a", "Lowest"),
            (None, "a", "Highest"),
            (None, "a", 22.0),
            ("all", "multi_level_response", "Mean"),
            (["yellow", "blue"], "multi_level_response", "Mean"),
        ],
    )
    def test_correct_unseen_levels_encoding_dict_stored(
        self,
        learnt_unseen_levels_encoding_dict_mean,
        learnt_unseen_levels_encoding_dict_median,
        learnt_unseen_levels_encoding_dict_lowest,
        learnt_unseen_levels_encoding_dict_highest,
        learnt_unseen_levels_encoding_dict_arbitrary,
        level,
        target_column,
        unseen_level_handling,
    ):
        "Test that the unseen_levels_encoding_dict dictionary created in fit has the correct keys and values."
        df = create_MeanResponseTransformer_test_df()
        x = MeanResponseTransformer(
            columns=["b"],
            level=level,
            unseen_level_handling=unseen_level_handling,
        )
        x.fit(df, df[target_column])

        if level:
            if level == "all":
                assert set(x.unseen_levels_encoding_dict.keys()) == {
                    "b_blue",
                    "b_yellow",
                    "b_green",
                }, "Stored unseen_levels_encoding_dict keys are not as expected"

            else:
                assert set(x.unseen_levels_encoding_dict.keys()) == {
                    "b_blue",
                    "b_yellow",
                }, "Stored unseen_levels_encoding_dict keys are not as expected"

            for column in x.unseen_levels_encoding_dict:
                actual = x.unseen_levels_encoding_dict[column]
                expected = learnt_unseen_levels_encoding_dict_mean[column]
                assert actual == expected

        else:
            assert x.unseen_levels_encoding_dict.keys() == {
                "b",
            }, "Stored unseen_levels_encoding_dict key is not as expected"

            if unseen_level_handling == "Mean":
                for column in x.unseen_levels_encoding_dict:
                    actual = x.unseen_levels_encoding_dict[column]
                    expected = learnt_unseen_levels_encoding_dict_mean[column]
                    assert actual == expected

            if unseen_level_handling == "Median":
                for column in x.unseen_levels_encoding_dict:
                    actual = x.unseen_levels_encoding_dict[column]
                    expected = learnt_unseen_levels_encoding_dict_median[column]
                    assert actual == expected

            if unseen_level_handling == "Lowest":
                for column in x.unseen_levels_encoding_dict:
                    actual = x.unseen_levels_encoding_dict[column]
                    expected = learnt_unseen_levels_encoding_dict_lowest[column]
                    assert actual == expected

            if unseen_level_handling == "Highest":
                for column in x.unseen_levels_encoding_dict:
                    actual = x.unseen_levels_encoding_dict[column]
                    expected = learnt_unseen_levels_encoding_dict_highest[column]
                    assert actual == expected

            if unseen_level_handling == "Abitrary":
                for column in x.unseen_levels_encoding_dict:
                    actual = x.unseen_levels_encoding_dict[column]
                    expected = learnt_unseen_levels_encoding_dict_arbitrary[column]
                    assert actual == expected

    def test_missing_categories_ignored(self):
        "test that where a categorical column has missing levels, these do not make it into the encoding dict"

        df = create_MeanResponseTransformer_test_df()
        unobserved_value = "bla"
        df["c"] = df["c"].cat.add_categories(unobserved_value)
        target_column = "e"
        x = MeanResponseTransformer(
            columns=["c"],
        )
        x.fit(df, df[target_column])

        assert (
            unobserved_value not in x.mappings
        ), "MeanResponseTransformer should ignore unobserved levels"


class TestFitBinaryResponse(GenericFitTests, WeightColumnFitMixinTests):
    """Tests for MeanResponseTransformer.fit()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "MeanResponseTransformer"

    def test_learnt_values(self):
        """Test that the mean response values learnt during fit are expected."""
        df = create_MeanResponseTransformer_test_df()

        x = MeanResponseTransformer(columns=["b", "d", "f"])

        x.mappings = {}

        x._fit_binary_response(df, df["a"], x.columns)

        expected_mappings = {
            "b": {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0, "e": 5.0, "f": 6.0},
            "d": {1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0, 5: 5.0, 6: 6.0},
            "f": {False: 2.0, True: 5.0},
        }

        for key in expected_mappings:
            for value in expected_mappings[key]:
                expected_mappings[key][value] = x.cast_method(
                    expected_mappings[key][value],
                )

        ta.classes.test_object_attributes(
            obj=x,
            expected_attributes={
                "mappings": expected_mappings,
                "global_mean": np.float64(3.5),
            },
            msg="mappings attribute",
        )

    def test_learnt_values_prior_no_weight(self):
        """Test that the mean response values learnt during fit are expected."""
        df = create_MeanResponseTransformer_test_df()

        x = MeanResponseTransformer(columns=["b", "d", "f"], prior=5)

        x.mappings = {}

        x._fit_binary_response(df, df["a"], x.columns)

        expected_mappings = {
            "b": {
                "a": 37 / 12,
                "b": 13 / 4,
                "c": 41 / 12,
                "d": 43 / 12,
                "e": 15 / 4,
                "f": 47 / 12,
            },
            "d": {
                1: 37 / 12,
                2: 13 / 4,
                3: 41 / 12,
                4: 43 / 12,
                5: 15 / 4,
                6: 47 / 12,
            },
            "f": {False: 47 / 16, True: 65 / 16},
        }
        for key in expected_mappings:
            for value in expected_mappings[key]:
                expected_mappings[key][value] = x.cast_method(
                    expected_mappings[key][value],
                )

        ta.classes.test_object_attributes(
            obj=x,
            expected_attributes={
                "mappings": expected_mappings,
                "global_mean": np.float64(3.5),
            },
            msg="mappings attribute",
        )

    def test_learnt_values_no_prior_weight(self):
        """Test that the mean response values learnt during fit are expected if a weights column is specified."""
        df = create_MeanResponseTransformer_test_df()

        x = MeanResponseTransformer(weights_column="e", columns=["b", "d", "f"])

        x.mappings = {}

        x._fit_binary_response(df, df["a"], x.columns)

        expected_mappings = {
            "b": {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0, "e": 5.0, "f": 6.0},
            "d": {1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0, 5: 5.0, 6: 6.0},
            "f": {False: 14 / 6, True: 77 / 15},
        }

        for key in expected_mappings:
            for value in expected_mappings[key]:
                expected_mappings[key][value] = x.cast_method(
                    expected_mappings[key][value],
                )

        ta.classes.test_object_attributes(
            obj=x,
            expected_attributes={
                "mappings": expected_mappings,
            },
            msg="mappings attribute",
        )

    def test_learnt_values_prior_weight(self):
        """Test that the mean response values learnt during fit are expected - when using weight and prior."""
        df = create_MeanResponseTransformer_test_df()

        df["weight"] = [1, 1, 1, 2, 2, 2]

        x = MeanResponseTransformer(
            columns=["d", "f"],
            prior=5,
            weights_column="weight",
        )

        x.mappings = {}

        x._fit_binary_response(df, df["a"], x.columns)

        expected_mappings = {
            "d": {1: 7 / 2, 2: 11 / 3, 3: 23 / 6, 4: 4.0, 5: 30 / 7, 6: 32 / 7},
            "f": {False: 13 / 4, True: 50 / 11},
        }
        for key in expected_mappings:
            for value in expected_mappings[key]:
                expected_mappings[key][value] = x.cast_method(
                    expected_mappings[key][value],
                )

        ta.classes.test_object_attributes(
            obj=x,
            expected_attributes={
                "mappings": expected_mappings,
                "global_mean": np.float64(4.0),
            },
            msg="mappings attribute",
        )

    @pytest.mark.parametrize("prior", (1, 3, 5, 7, 9, 11, 100))
    def test_prior_logic(self, prior):
        "Test that for prior>0 encodings are closer to global mean than for prior=0."
        df = create_MeanResponseTransformer_test_df()

        df["weight"] = [1, 1, 1, 2, 2, 2]

        x_prior = MeanResponseTransformer(
            columns=["d", "f"],
            prior=prior,
            weights_column="weight",
        )

        x_no_prior = MeanResponseTransformer(
            columns=["d", "f"],
            prior=0,
            weights_column="weight",
        )

        x_prior.mappings = {}
        x_no_prior.mappings = {}

        x_prior._fit_binary_response(df, df["a"], x_prior.columns)

        x_no_prior._fit_binary_response(df, df["a"], x_no_prior.columns)

        prior_mappings = x_prior.mappings

        no_prior_mappings = x_no_prior.mappings

        global_mean = x_prior.global_mean

        assert (
            global_mean == x_no_prior.global_mean
        ), "global means for transformers with/without priors should match"

        for col in prior_mappings:
            for value in prior_mappings[col]:
                prior_encoding = prior_mappings[col][value]
                no_prior_encoding = no_prior_mappings[col][value]

                prior_mean_dist = np.abs(prior_encoding - global_mean)
                no_prior_mean_dist = np.abs(no_prior_encoding - global_mean)

                assert (
                    prior_mean_dist <= no_prior_mean_dist
                ), "encodings using priors should be closer to the global mean than without"

    @pytest.mark.parametrize(
        ("low_weight", "high_weight"),
        ((1, 2), (2, 3), (3, 4), (10, 20)),
    )
    def test_prior_logic_for_weights(self, low_weight, high_weight):
        "Test that for fixed prior a group with lower weight is moved closer to the global mean than one with higher weight."
        df = create_MeanResponseTransformer_test_df()

        # column f looks like [False, False, False, True, True, True]
        df["weight"] = [
            low_weight,
            low_weight,
            low_weight,
            high_weight,
            high_weight,
            high_weight,
        ]

        x_prior = MeanResponseTransformer(
            columns=["f"],
            prior=5,
            weights_column="weight",
        )

        x_no_prior = MeanResponseTransformer(
            columns=["f"],
            prior=0,
            weights_column="weight",
        )

        x_prior.mappings = {}
        x_no_prior.mappings = {}

        x_prior._fit_binary_response(df, df["a"], x_prior.columns)

        x_no_prior._fit_binary_response(df, df["a"], x_no_prior.columns)

        prior_mappings = x_prior.mappings

        no_prior_mappings = x_no_prior.mappings

        global_mean = x_prior.global_mean

        assert (
            global_mean == x_no_prior.global_mean
        ), "global means for transformers with/without priors should match"

        low_weight_prior_encoding = prior_mappings["f"][False]
        high_weight_prior_encoding = prior_mappings["f"][True]

        low_weight_no_prior_encoding = no_prior_mappings["f"][False]
        high_weight_no_prior_encoding = no_prior_mappings["f"][True]

        low_weight_prior_mean_dist = np.abs(low_weight_prior_encoding - global_mean)
        high_weight_prior_mean_dist = np.abs(high_weight_prior_encoding - global_mean)

        low_weight_no_prior_mean_dist = np.abs(
            low_weight_no_prior_encoding - global_mean,
        )
        high_weight_no_prior_mean_dist = np.abs(
            high_weight_no_prior_encoding - global_mean,
        )

        # check low weight group has been moved further towards mean than high weight group by prior, i.e
        # that the distance remaining is a smaller proportion of the no prior distance
        low_ratio = low_weight_prior_mean_dist / low_weight_no_prior_mean_dist
        high_ratio = high_weight_prior_mean_dist / high_weight_no_prior_mean_dist
        assert (
            low_ratio <= high_ratio
        ), "encodings for categories with lower weights should be moved closer to the global mean than those with higher weights, for fixed prior"

    def test_weights_column_missing_error(self):
        """Test that an exception is raised if weights_column is specified but not present in data for fit."""
        df = create_MeanResponseTransformer_test_df()

        x = MeanResponseTransformer(weights_column="z", columns=["b", "d", "f"])

        with pytest.raises(
            ValueError,
            match=r"weight col \(z\) is not present in columns of data",
        ):
            x._fit_binary_response(df, df["a"], x.columns)


class TestTransform(GenericTransformTests):
    """Tests for MeanResponseTransformer.transform()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "MeanResponseTransformer"

    def expected_df_1():
        """Expected output for single level response."""
        df = pd.DataFrame(
            {
                "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "b": [1, 2, 3, 4, 5, 6],
                "c": ["a", "b", "c", "d", "e", "f"],
                "d": [1, 2, 3, 4, 5, 6],
                "e": [1, 2, 3, 4, 5, 6.0],
                "f": [2, 2, 2, 5, 5, 5],
                "multi_level_response": [
                    "blue",
                    "blue",
                    "yellow",
                    "yellow",
                    "green",
                    "green",
                ],
            },
        )

        df["c"] = df["c"].astype("category")

        return df

    def expected_df_2():
        """Expected output for response with level = blue."""
        df = pd.DataFrame(
            {
                "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "c": ["a", "b", "c", "d", "e", "f"],
                "d": [1, 2, 3, 4, 5, 6],
                "e": [1, 2, 3, 4, 5, 6.0],
                "multi_level_response": [
                    "blue",
                    "blue",
                    "yellow",
                    "yellow",
                    "green",
                    "green",
                ],
                "b_blue": [1, 1, 0, 0, 0, 0],
                "f_blue": [2 / 3, 2 / 3, 2 / 3, 0, 0, 0],
            },
        )

        df["c"] = df["c"].astype("category")

        return df

    def expected_df_3():
        """Expected output for response with level = 'all'."""
        df = pd.DataFrame(
            {
                "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "c": ["a", "b", "c", "d", "e", "f"],
                "d": [1, 2, 3, 4, 5, 6],
                "e": [1, 2, 3, 4, 5, 6.0],
                "multi_level_response": [
                    "blue",
                    "blue",
                    "yellow",
                    "yellow",
                    "green",
                    "green",
                ],
                "b_blue": [1, 1, 0, 0, 0, 0],
                "f_blue": [2 / 3, 2 / 3, 2 / 3, 0, 0, 0],
                "b_green": [0, 0, 0, 0, 1, 1],
                "f_green": [0, 0, 0, 2 / 3, 2 / 3, 2 / 3],
                "b_yellow": [0, 0, 1, 1, 0, 0],
                "f_yellow": [1 / 3, 1 / 3, 1 / 3, 1 / 3, 1 / 3, 1 / 3],
            },
        )

        df["c"] = df["c"].astype("category")

        return df

    def expected_df_4():
        """Expected output for transform on dataframe with single level response and unseen levels,
        where unseen_level_handling = 'Mean'.
        """
        df = pd.DataFrame(
            {
                "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 3.0],
                "b": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 3.5, 3.5],
                "c": ["a", "b", "c", "d", "e", "f", "g", "h"],
                "d": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 3.5, 3.5],
                "e": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                "f": [2.0, 2.0, 2.0, 5.0, 5.0, 5.0, 5.0, 2.0],
                "multi_level_response": [
                    "blue",
                    "blue",
                    "yellow",
                    "yellow",
                    "green",
                    "green",
                    "yellow",
                    "blue",
                ],
            },
        )

        df["c"] = df["c"].astype("category")

        return df

    def expected_df_5():
        """Expected output for transform on dataframe with single level response and unseen levels,
        where unseen_level_handling = 'Median'.
        """
        df = pd.DataFrame(
            {
                "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 3.0],
                "b": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 3.5, 3.5],
                "c": ["a", "b", "c", "d", "e", "f", "g", "h"],
                "d": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 3.5, 3.5],
                "e": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                "f": [2.0, 2.0, 2.0, 5.0, 5.0, 5.0, 5.0, 2.0],
                "multi_level_response": [
                    "blue",
                    "blue",
                    "yellow",
                    "yellow",
                    "green",
                    "green",
                    "yellow",
                    "blue",
                ],
            },
        )

        df["c"] = df["c"].astype("category")

        return df

    def expected_df_6():
        """Expected output for transform on dataframe with single level response and unseen levels,
        where unseen_level_handling = 'Lowest'.
        """
        df = pd.DataFrame(
            {
                "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 3.0],
                "b": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 1.0],
                "c": ["a", "b", "c", "d", "e", "f", "g", "h"],
                "d": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 1.0],
                "e": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                "f": [2.0, 2.0, 2.0, 5.0, 5.0, 5.0, 5.0, 2.0],
                "multi_level_response": [
                    "blue",
                    "blue",
                    "yellow",
                    "yellow",
                    "green",
                    "green",
                    "yellow",
                    "blue",
                ],
            },
        )

        df["c"] = df["c"].astype("category")

        return df

    def expected_df_7():
        """Expected output for transform on dataframe with single level response and unseen levels,
        where unseen_level_handling = 'Highest'.
        """
        df = pd.DataFrame(
            {
                "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 3.0],
                "b": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 6.0, 6.0],
                "c": ["a", "b", "c", "d", "e", "f", "g", "h"],
                "d": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 6.0, 6.0],
                "e": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                "f": [2.0, 2.0, 2.0, 5.0, 5.0, 5.0, 5.0, 2.0],
                "multi_level_response": [
                    "blue",
                    "blue",
                    "yellow",
                    "yellow",
                    "green",
                    "green",
                    "yellow",
                    "blue",
                ],
            },
        )

        df["c"] = df["c"].astype("category")

        return df

    def expected_df_8():
        """Expected output for transform on dataframe with single level response and unseen levels,
        where unseen_level_handling set to arbitrary int/float value.
        """
        df = pd.DataFrame(
            {
                "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 3.0],
                "b": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 21.6, 21.6],
                "c": ["a", "b", "c", "d", "e", "f", "g", "h"],
                "d": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 21.6, 21.6],
                "e": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                "f": [2.0, 2.0, 2.0, 5.0, 5.0, 5.0, 5.0, 2.0],
                "multi_level_response": [
                    "blue",
                    "blue",
                    "yellow",
                    "yellow",
                    "green",
                    "green",
                    "yellow",
                    "blue",
                ],
            },
        )

        df["c"] = df["c"].astype("category")

        return df

    def expected_df_9():
        """Expected output for transform on dataframe with multi-level response with level = blue and unseen levels in data."""
        df = pd.DataFrame(
            {
                "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 3.0],
                "c": ["a", "b", "c", "d", "e", "f", "g", "h"],
                "d": [1, 2, 3, 4, 5, 6, 7, 8],
                "e": [1, 2, 3, 4, 5, 6.0, 7, 8],
                "multi_level_response": [
                    "blue",
                    "blue",
                    "yellow",
                    "yellow",
                    "green",
                    "green",
                    "yellow",
                    "blue",
                ],
                "b_blue": [1, 1, 0, 0, 0, 0, 2 / 6, 2 / 6],
                "f_blue": [2 / 3, 2 / 3, 2 / 3, 0, 0, 0, 0, 2 / 3],
            },
        )

        df["c"] = df["c"].astype("category")

        return df

    def expected_df_10():
        """Expected output for transform on dataframe with multi-level response with level = "all" and unseen levels in data."""
        df = pd.DataFrame(
            {
                "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 3.0],
                "c": ["a", "b", "c", "d", "e", "f", "g", "h"],
                "d": [1, 2, 3, 4, 5, 6, 7, 8],
                "e": [1, 2, 3, 4, 5, 6.0, 7, 8],
                "multi_level_response": [
                    "blue",
                    "blue",
                    "yellow",
                    "yellow",
                    "green",
                    "green",
                    "yellow",
                    "blue",
                ],
                "b_blue": [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
                "f_blue": [2 / 3, 2 / 3, 2 / 3, 0, 0, 0, 0, 2 / 3],
                "b_yellow": [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
                "f_yellow": [1 / 3, 1 / 3, 1 / 3, 1 / 3, 1 / 3, 1 / 3, 1 / 3, 1 / 3],
                "b_green": [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
                "f_green": [0.0, 0.0, 0.0, 2 / 3, 2 / 3, 2 / 3, 2 / 3, 0],
            },
        )

        df["c"] = df["c"].astype("category")

        return df

    @pytest.mark.parametrize(
        ("df", "expected"),
        ta.pandas.adjusted_dataframe_params(
            create_MeanResponseTransformer_test_df(),
            expected_df_1(),
        ),
    )
    def test_expected_output_binary_response(self, df, expected):
        """Test that the output is expected from transform with a binary response."""
        columns = ["b", "d", "f"]
        x = MeanResponseTransformer(columns=columns)

        # set the impute values dict directly rather than fitting x on df so test works with helpers
        x.mappings = {
            "b": {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6},
            "d": {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6},
            "f": {False: 2, True: 5},
        }

        df_transformed = x.transform(df)

        for col in columns:
            expected[col] = expected[col].astype(x.return_type)

        ta.equality.assert_frame_equal_msg(
            actual=df_transformed,
            expected=expected,
            msg_tag="Unexpected values in MeanResponseTransformer.transform",
        )

    @pytest.mark.parametrize(
        ("df", "expected"),
        ta.pandas.adjusted_dataframe_params(
            create_MeanResponseTransformer_test_df(),
            expected_df_2(),
        ),
    )
    def test_expected_output_one_multi_level(self, df, expected):
        """Test that the output is expected from transform with a multi-level response and one level selected."""
        columns = ["b", "f"]
        level = ["blue"]
        expected_created_cols = [
            prefix + "_" + suffix for prefix, suffix in product(columns, level)
        ]
        x = MeanResponseTransformer(columns=columns, level=level)

        for col in expected_created_cols:
            expected[col] = expected[col].astype(x.return_type)

        # set the impute values dict directly rather than fitting x on df so test works with helpers
        x.mappings = {
            "b_blue": {"a": 1, "b": 1, "c": 0, "d": 0, "e": 0, "f": 0},
            "f_blue": {False: 2 / 3, True: 0},
        }
        x.response_levels = level
        x.mapped_columns = list(x.mappings.keys())
        df_transformed = x.transform(df)
        new_expected_created_cols = [
            prefix + "_" + suffix
            for prefix, suffix in product(columns, x.response_levels)
        ]

        for col in new_expected_created_cols:
            expected[col] = expected[col].astype(x.return_type)

        ta.equality.assert_frame_equal_msg(
            actual=df_transformed,
            expected=expected,
            msg_tag="Unexpected values in MeanResponseTransformer.transform",
            check_like=False,
        )

    @pytest.mark.parametrize(
        ("df", "expected"),
        ta.pandas.adjusted_dataframe_params(
            create_MeanResponseTransformer_test_df(),
            expected_df_3(),
        ),
    )
    def test_expected_output_all_levels(self, df, expected):
        """Test that the output is expected from transform for a multi-level response and all levels selected."""
        columns = ["b", "f"]
        x = MeanResponseTransformer(columns=columns, level="all")

        # set the impute values dict directly rather than fitting x on df so test works with helpers
        x.mappings = {
            "b_blue": {"a": 1, "b": 1, "c": 0, "d": 0, "e": 0, "f": 0},
            "b_yellow": {"a": 0, "b": 0, "c": 1, "d": 1, "e": 0, "f": 0},
            "b_green": {"a": 0, "b": 0, "c": 0, "d": 0, "e": 1, "f": 1},
            "f_blue": {False: 2 / 3, True: 0},
            "f_yellow": {False: 1 / 3, True: 1 / 3},
            "f_green": {False: 0, True: 2 / 3},
        }

        x.response_levels = ["blue", "green", "yellow"]
        x.mapped_columns = list(x.mappings.keys())
        df_transformed = x.transform(df)
        expected_created_cols = [
            prefix + "_" + suffix
            for prefix, suffix in product(columns, x.response_levels)
        ]

        for col in expected_created_cols:
            expected[col] = expected[col].astype(x.return_type)

        ta.equality.assert_frame_equal_msg(
            actual=df_transformed,
            expected=expected,
            msg_tag="Unexpected values in MeanResponseTransformer.transform",
            check_like=False,
        )

    def test_expected_output_sigle_level_response_unseen_levels_default(self):
        """Test that the output is expected from transform with a single level response with unseen levels in data with
        unseen_level_handling set to 'None', i.e., default value.
        """
        initial_df = create_MeanResponseTransformer_test_df()
        x = MeanResponseTransformer(columns=["b", "d", "f"])
        x.fit(initial_df, initial_df["a"])
        df = create_MeanResponseTransformer_test_df_unseen_levels()
        with pytest.raises(
            ValueError,
            match="MeanResponseTransformer: nulls would be introduced into column b from levels not present in mapping",
        ):
            x.transform(df)

    @pytest.mark.parametrize(
        ("df", "expected"),
        ta.pandas.adjusted_dataframe_params(
            create_MeanResponseTransformer_test_df_unseen_levels(),
            expected_df_4(),
        ),
    )
    def test_expected_output_sigle_level_response_unseen_levels_mean(
        self,
        df,
        expected,
    ):
        """Test that the output is expected from transform with a single level response with unseen levels in data with
        unseen_level_handling set to 'Mean'.
        """
        columns = ["b", "d", "f"]
        target = "a"
        x = MeanResponseTransformer(
            columns=columns,
            unseen_level_handling="Mean",
        )

        initial_df = create_MeanResponseTransformer_test_df()
        x.fit(initial_df, initial_df[target])
        df_transformed = x.transform(df)

        for col in columns:
            expected[col] = expected[col].astype(x.return_type)

        ta.equality.assert_frame_equal_msg(
            actual=df_transformed,
            expected=expected,
            msg_tag="Unexpected values in MeanResponseTransformer.transform",
        )

    @pytest.mark.parametrize(
        ("df", "expected"),
        ta.pandas.adjusted_dataframe_params(
            create_MeanResponseTransformer_test_df_unseen_levels(),
            expected_df_5(),
        ),
    )
    def test_expected_output_sigle_level_response_unseen_levels_median(
        self,
        df,
        expected,
    ):
        """Test that the output is expected from transform with a single level response with unseen levels in data
        with unseen_level_handling set to 'Median'.
        """
        columns = ["b", "d", "f"]
        target = "a"
        x = MeanResponseTransformer(
            columns=columns,
            unseen_level_handling="Median",
        )

        initial_df = create_MeanResponseTransformer_test_df()
        x.fit(initial_df, initial_df[target])
        df_transformed = x.transform(df)

        for col in columns:
            expected[col] = expected[col].astype(x.return_type)

        ta.equality.assert_frame_equal_msg(
            actual=df_transformed,
            expected=expected,
            msg_tag="Unexpected values in MeanResponseTransformer.transform",
        )

    @pytest.mark.parametrize(
        ("df", "expected"),
        ta.pandas.adjusted_dataframe_params(
            create_MeanResponseTransformer_test_df_unseen_levels(),
            expected_df_6(),
        ),
    )
    def test_expected_output_sigle_level_response_unseen_levels_lowest(
        self,
        df,
        expected,
    ):
        """Test that the output is expected from transform with a single level response with unseen levels in data
        with unseen_level_handling set to 'Lowest'.
        """
        columns = ["b", "d", "f"]
        target = "a"
        x = MeanResponseTransformer(
            columns=columns,
            unseen_level_handling="Lowest",
        )

        initial_df = create_MeanResponseTransformer_test_df()
        x.fit(initial_df, initial_df[target])
        df_transformed = x.transform(df)

        for col in columns:
            expected[col] = expected[col].astype(x.return_type)

        ta.equality.assert_frame_equal_msg(
            actual=df_transformed,
            expected=expected,
            msg_tag="Unexpected values in MeanResponseTransformer.transform",
        )

    @pytest.mark.parametrize(
        ("df", "expected"),
        ta.pandas.adjusted_dataframe_params(
            create_MeanResponseTransformer_test_df_unseen_levels(),
            expected_df_7(),
        ),
    )
    def test_expected_output_sigle_level_response_unseen_levels_highest(
        self,
        df,
        expected,
    ):
        """Test that the output is expected from transform with a single level response with unseen levels in data
        with unseen_level_handling set to 'Highest'.
        """
        columns = ["b", "d", "f"]
        target = "a"
        x = MeanResponseTransformer(
            columns=["b", "d", "f"],
            unseen_level_handling="Highest",
        )

        initial_df = create_MeanResponseTransformer_test_df()
        x.fit(initial_df, initial_df[target])
        df_transformed = x.transform(df)

        for col in columns:
            expected[col] = expected[col].astype(x.return_type)

        ta.equality.assert_frame_equal_msg(
            actual=df_transformed,
            expected=expected,
            msg_tag="Unexpected values in MeanResponseTransformer.transform",
        )

    @pytest.mark.parametrize(
        ("df", "expected"),
        ta.pandas.adjusted_dataframe_params(
            create_MeanResponseTransformer_test_df_unseen_levels(),
            expected_df_8(),
        ),
    )
    def test_expected_output_sigle_level_response_unseen_levels_arbitrary(
        self,
        df,
        expected,
    ):
        """Test that the output is expected from transform with a single level response with unseen levels in data
        with unseen_level_handling set to an arbitrary int/float value'.
        """
        columns = ["b", "d", "f"]
        target = "a"
        x = MeanResponseTransformer(columns=columns, unseen_level_handling=21.6)

        initial_df = create_MeanResponseTransformer_test_df()
        x.fit(initial_df, initial_df[target])
        df_transformed = x.transform(df)

        for col in columns:
            expected[col] = expected[col].astype(x.return_type)

        ta.equality.assert_frame_equal_msg(
            actual=df_transformed,
            expected=expected,
            msg_tag="Unexpected values in MeanResponseTransformer.transform",
        )

    @pytest.mark.parametrize(
        ("df", "expected"),
        ta.pandas.adjusted_dataframe_params(
            create_MeanResponseTransformer_test_df_unseen_levels(),
            expected_df_9(),
        ),
    )
    def test_expected_output_one_multi_level_unseen_levels(self, df, expected):
        """Test that the output is expected from transform with a multi-level response and unseen levels and one level selected."""
        columns = ["b", "f"]
        level = ["blue"]
        expected_created_cols = [
            prefix + "_" + suffix for prefix, suffix in product(columns, level)
        ]
        x = MeanResponseTransformer(
            columns=columns,
            level=level,
            unseen_level_handling="Mean",
        )

        for col in expected_created_cols:
            expected[col] = expected[col].astype(x.return_type)

        initial_df = create_MeanResponseTransformer_test_df()
        x.fit(initial_df, initial_df["multi_level_response"])
        df_transformed = x.transform(df)

        ta.equality.assert_frame_equal_msg(
            actual=df_transformed,
            expected=expected,
            msg_tag="Unexpected values in MeanResponseTransformer.transform",
        )

    @pytest.mark.parametrize(
        ("df", "expected"),
        ta.pandas.adjusted_dataframe_params(
            create_MeanResponseTransformer_test_df_unseen_levels(),
            expected_df_10(),
        ),
    )
    def test_expected_output_all_multi_level_unseen_levels(self, df, expected):
        """Test that the output is expected from transform with a multi-level response and unseen levels and all level selected."""
        columns = ["b", "f"]
        target = "multi_level_response"
        initial_df = create_MeanResponseTransformer_test_df()
        expected_created_cols = [
            prefix + "_" + suffix
            for prefix, suffix in product(columns, initial_df[target].unique().tolist())
        ]
        x = MeanResponseTransformer(
            columns=columns,
            level="all",
            unseen_level_handling="Highest",
        )

        x.fit(initial_df, initial_df[target])
        df_transformed = x.transform(df)

        for col in expected_created_cols:
            expected[col] = expected[col].astype(x.return_type)

        ta.equality.assert_frame_equal_msg(
            actual=df_transformed,
            expected=expected,
            msg_tag="Unexpected values in MeanResponseTransformer.transform",
        )

    def test_nulls_introduced_in_transform_error(self):
        """Test that transform will raise an error if nulls are introduced."""
        df = create_MeanResponseTransformer_test_df()

        x = MeanResponseTransformer(columns=["b", "d", "f"])

        x.fit(df, df["a"])

        df["b"] = "z"

        with pytest.raises(
            ValueError,
            match="MeanResponseTransformer: nulls would be introduced into column b from levels not present in mapping",
        ):
            x.transform(df)

    @pytest.mark.parametrize(
        "prior, level, target, unseen_level_handling",
        [
            (5, "all", "c", "Mean"),
            (100, ["a", "b"], "c", "Lowest"),
            (1, None, "a", "Highest"),
            (0, None, "a", "Median"),
        ],
    )
    def test_return_type_can_be_changed(
        self,
        prior,
        level,
        target,
        unseen_level_handling,
    ):
        "Test that output return types are controlled by return_type param, this defaults to float32 so test float64 here"

        df = create_MeanResponseTransformer_test_df()

        columns = ["b", "d", "f"]
        x = MeanResponseTransformer(
            columns=columns,
            return_type="float64",
            prior=prior,
            unseen_level_handling=unseen_level_handling,
            level=level,
        )

        x.fit(df, df[target])

        output_df = x.transform(df)

        if target == "c":
            actual_levels = df[target].unique().tolist() if level == "all" else level
            expected_created_cols = [
                prefix + "_" + suffix
                for prefix, suffix in product(columns, actual_levels)
            ]

        else:
            expected_created_cols = columns

        for col in expected_created_cols:
            expected_type = x.return_type
            actual_type = output_df[col].dtype.name
            assert (
                actual_type == expected_type
            ), f"{x.classname} should output columns with type determine by the return_type param, expected {expected_type} but got {actual_type}"

    def test_learnt_values_not_modified(self):
        """Test that the mappings from fit are not changed in transform."""
        df = create_MeanResponseTransformer_test_df()

        x = MeanResponseTransformer(columns="b")

        x.fit(df, df["a"])

        x2 = MeanResponseTransformer(columns="b")

        x2.fit(df, df["a"])

        x2.transform(df)

        ta.equality.assert_equal_dispatch(
            expected=x.mappings,
            actual=x2.mappings,
            msg="Mean response values not changed in transform",
        )


class TestOtherBaseBehaviour(OtherBaseBehaviourTests):
    """
    Class to run tests for BaseTransformerBehaviour outside the three standard methods.

    May need to overwite specific tests in this class if the tested transformer modifies this behaviour.
    """

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "MeanResponseTransformer"
