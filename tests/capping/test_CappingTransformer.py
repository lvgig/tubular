import numpy as np
import pandas as pd
import pytest
import test_aide as ta

import tests.test_data as d
from tests.base_tests import OtherBaseBehaviourTests
from tests.capping.test_BaseCappingTransformer import (
    GenericCappingFitTests,
    GenericCappingInitTests,
    GenericCappingTransformTests,
)


class TestInit(GenericCappingInitTests):
    """Tests for CappingTransformer.init()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "CappingTransformer"


class TestFit(GenericCappingFitTests):
    """Tests for CappingTransformer.fit()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "CappingTransformer"

    @pytest.mark.parametrize(
        ("values", "sample_weight", "quantiles"),
        [
            ([1, 2, 3], [1, 2, 1], [0.1, 0.5]),
            ([1, 2, 3], None, [0.1, 0.5]),
            ([2, 4, 6, 8], [3, 2, 1, 1], [None, 0.5]),
            ([2, 4, 6, 8], None, [None, 0.5]),
            ([-1, -5, -10, 20, 30], [1, 2, 2, 2, 2], [0.1, None]),
            ([-1, -5, -10, 20, 40], None, [0.1, None]),
        ],
    )
    def test_replacement_values_updated(
        self,
        values,
        sample_weight,
        quantiles,
        minimal_attribute_dict,
        uninitialized_transformers,
    ):
        """Test that weighted_quantile gives the expected outputs."""

        args = minimal_attribute_dict[self.transformer_name].copy()
        args["quantiles"] = {"a": quantiles}
        args["capping_values"] = None
        args["weights_column"] = "w"

        transformer = uninitialized_transformers[self.transformer_name](**args)

        if not sample_weight:
            sample_weight = [1] * len(values)

        df = pd.DataFrame(
            {
                "a": values,
                "w": sample_weight,
            },
        )

        transformer.fit(df)

        assert (
            transformer._replacement_values == transformer.quantile_capping_values
        ), f"unexpected value for replacement_values attribute, expected {transformer.quantile_capping_values} but got {transformer.replacement_values_}"


class TestTransform(GenericCappingTransformTests):
    """Tests for CappingTransformer.transform()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "CappingTransformer"

    def expected_df_1():
        """Expected output from test_expected_output_min_and_max."""
        return pd.DataFrame(
            {
                "a": [2, 2, 3, 4, 5, 5, np.nan],
                "b": [1, 2, 3, np.nan, 7, 7, 7],
                "c": [np.nan, 1, 2, 3, 0, 0, 0],
            },
        )

    @pytest.mark.parametrize(
        ("df", "expected"),
        ta.pandas.adjusted_dataframe_params(d.create_df_3(), expected_df_1()),
    )
    def test_expected_output_min_and_max_combinations(
        self,
        df,
        expected,
        minimal_attribute_dict,
        uninitialized_transformers,
    ):
        """Test that capping is applied correctly in transform."""

        args = minimal_attribute_dict[self.transformer_name].copy()
        args["capping_values"] = {"a": [2, 5], "b": [None, 7], "c": [0, None]}

        transformer = uninitialized_transformers[self.transformer_name](**args)

        df_transformed = transformer.transform(df)

        ta.equality.assert_frame_equal_msg(
            actual=df_transformed,
            expected=expected,
            msg_tag=f"Unexpected values in {self.transformer_name}.transform",
        )


class TestOtherBaseBehaviour(OtherBaseBehaviourTests):
    """
    Class to run tests for BaseTransformerBehaviour outside the three standard methods.

    May need to overwite specific tests in this class if the tested transformer modifies this behaviour.
    """

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "CappingTransformer"

    def test_get_params_call_with_capping_values_none(
        self,
        uninitialized_transformers,
        minimal_attribute_dict,
    ):
        """Test get_params method when capping_values is None."""
        args = minimal_attribute_dict[self.transformer_name]
        args["capping_values"] = None
        args["quantiles"] = {"a": [0.1, 0.9]}
        transformer = uninitialized_transformers[self.transformer_name](**args)

        # Ensure no AttributeError is raised when calling get_params method
        try:
            transformer.get_params()
        except AttributeError as e:
            pytest.fail(f"AttributeError was raised: {e}")
