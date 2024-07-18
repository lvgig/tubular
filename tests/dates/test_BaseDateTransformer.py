import pandas as pd
import pytest

import tests.test_data as d
from tests.base_tests import (
    ColumnStrListInitTests,
    GenericFitTests,
    GenericTransformTests,
    OtherBaseBehaviourTests,
)


class GenericDatesTransformTests:
    """Generic tests for Dates Transformers"""

    def test_mismatched_datetypes_error(
        self,
        columns,
        datetime_col,
        date_col,
        initialized_transformers,
    ):
        "Test that transform raises an error if one column is a date and one is datetime"

        x = initialized_transformers(self.transformer_name)

        df = d.create_date_diff_different_dtypes()
        # types don't seem to come out of the above function as expected, hard enforce
        for col in ["date_col_1", "date_col_2"]:
            df[col] = pd.to_datetime(df[col]).dt.date

        for col in ["datetime_col_1", "datetime_col_2"]:
            df[col] = pd.to_datetime(df[col])

        present_types = (
            {"datetime64", "date"} if datetime_col == 0 else {"date", "datetime64"}
        )
        msg = rf"Columns fed to datetime transformers should be \['datetime64', 'date'\] and have consistent types, but found {present_types}. Please use ToDatetimeTransformer to standardise"
        with pytest.raises(
            TypeError,
            match=msg,
        ):
            x.transform(df)


class TestInit(ColumnStrListInitTests):
    """Generic tests for transformer.init()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseDateTransformer"


class TestFit(GenericFitTests):
    """Generic tests for transformer.fit()"""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseDateTransformer"


class TestTransform(GenericTransformTests, GenericDatesTransformTests):
    """Tests for BaseDateTransformer.transform."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseDateTransformer"


class TestOtherBaseBehaviour(OtherBaseBehaviourTests):
    """
    Class to run tests for BaseTransformerBehaviour outside the three standard methods.

    May need to overwite specific tests in this class if the tested transformer modifies this behaviour.
    """

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseDateTransformer"
