from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from tests.base_tests import (
    ColumnStrListInitTests,
    DropOriginalInitMixinTests,
    GenericFitTests,
    GenericTransformTests,
    NewColumnNameInitMixintests,
    OtherBaseBehaviourTests,
)


class DatetimeMixinTransformTests:
    """Generic tests for Datetime Transformers"""

    @pytest.mark.parametrize(
        ("bad_value", "bad_type"),
        [
            (1, "int64"),
            ("a", "object"),
            (np.nan, "float64"),
            (pd.to_datetime("01/02/2020").date(), "date"),
        ],
    )
    def test_non_datetypes_error(
        self,
        uninitialized_transformers,
        minimal_attribute_dict,
        minimal_dataframe_lookup,
        bad_value,
        bad_type,
    ):
        "Test that transform raises an error if columns contains non date types"

        args = minimal_attribute_dict[self.transformer_name].copy()

        # pull out columns arg to use below, and remove from dict
        columns = args["columns"]

        for i in range(len(columns)):
            df = deepcopy(minimal_dataframe_lookup[self.transformer_name])
            col = columns[i]
            df[col] = bad_value

            x = uninitialized_transformers[self.transformer_name](
                **args,
            )

            msg = rf"{col} type should be in \['datetime64'\] but got {bad_type}"

            with pytest.raises(
                TypeError,
                match=msg,
            ):
                x.transform(df)


class TestInit(
    NewColumnNameInitMixintests,
    DropOriginalInitMixinTests,
    ColumnStrListInitTests,
):
    """Generic tests for transformer.init()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseDatetimeTransformer"


class TestFit(GenericFitTests):
    """Generic tests for transformer.fit()"""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseDatetimeTransformer"


class TestTransform(GenericTransformTests, DatetimeMixinTransformTests):
    """Tests for BaseDatetimeTransformer.transform."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseDatetimeTransformer"


class TestOtherBaseBehaviour(OtherBaseBehaviourTests):
    """
    Class to run tests for BaseTransformerBehaviour outside the three standard methods.

    May need to overwite specific tests in this class if the tested transformer modifies this behaviour.
    """

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseDatetimeTransformer"
