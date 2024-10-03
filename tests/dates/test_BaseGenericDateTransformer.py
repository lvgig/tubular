import datetime
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


def create_date_diff_different_dtypes():
    """Dataframe with different datetime formats"""
    return pd.DataFrame(
        {
            "date_col_1": [
                datetime.date(1993, 9, 27),
                datetime.date(2000, 3, 19),
                datetime.date(2018, 11, 10),
                datetime.date(2018, 10, 10),
                datetime.date(2018, 10, 10),
                datetime.date(2018, 10, 10),
                datetime.date(2018, 12, 10),
                datetime.date(
                    1985,
                    7,
                    23,
                ),
            ],
            "date_col_2": [
                datetime.date(2020, 5, 1),
                datetime.date(2019, 12, 25),
                datetime.date(2018, 11, 10),
                datetime.date(2018, 11, 10),
                datetime.date(2018, 9, 10),
                datetime.date(2015, 11, 10),
                datetime.date(2015, 11, 10),
                datetime.date(2015, 7, 23),
            ],
            "datetime_col_1": [
                datetime.datetime(1993, 9, 27, tzinfo=datetime.timezone.utc),
                datetime.datetime(2000, 3, 19, tzinfo=datetime.timezone.utc),
                datetime.datetime(2018, 11, 10, tzinfo=datetime.timezone.utc),
                datetime.datetime(2018, 10, 10, tzinfo=datetime.timezone.utc),
                datetime.datetime(2018, 10, 10, tzinfo=datetime.timezone.utc),
                datetime.datetime(2018, 10, 10, tzinfo=datetime.timezone.utc),
                datetime.datetime(2018, 12, 10, tzinfo=datetime.timezone.utc),
                datetime.datetime(
                    1985,
                    7,
                    23,
                    tzinfo=datetime.timezone.utc,
                ),
            ],
            "datetime_col_2": [
                datetime.datetime(2020, 5, 1, tzinfo=datetime.timezone.utc),
                datetime.datetime(2019, 12, 25, tzinfo=datetime.timezone.utc),
                datetime.datetime(2018, 11, 10, tzinfo=datetime.timezone.utc),
                datetime.datetime(2018, 11, 10, tzinfo=datetime.timezone.utc),
                datetime.datetime(2018, 9, 10, tzinfo=datetime.timezone.utc),
                datetime.datetime(2015, 11, 10, tzinfo=datetime.timezone.utc),
                datetime.datetime(2015, 11, 10, tzinfo=datetime.timezone.utc),
                datetime.datetime(2015, 7, 23, tzinfo=datetime.timezone.utc),
            ],
        },
    )


class GenericDatesMixinTransformTests:
    """Generic tests for Dates Transformers"""

    @pytest.mark.parametrize(
        ("bad_value", "bad_type"),
        [
            (1, "int64"),
            ("a", "object"),
            (np.nan, "float64"),
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
        columns = args["columns"]

        for i in range(len(columns)):
            df = deepcopy(minimal_dataframe_lookup[self.transformer_name])
            print(df)
            col = columns[i]
            df[col] = bad_value

            x = uninitialized_transformers[self.transformer_name](
                **args,
            )

            msg = (
                rf"{col} type should be in \['datetime64', 'date'\] but got {bad_type}"
            )

            with pytest.raises(
                TypeError,
                match=msg,
            ):
                x.transform(df)

    @pytest.mark.parametrize(
        ("columns, datetime_col"),
        [
            (["date_col_1", "datetime_col_2"], 1),
            (["datetime_col_1", "date_col_2"], 0),
        ],
    )
    def test_mismatched_datetypes_error(
        self,
        columns,
        datetime_col,
        uninitialized_transformers,
        minimal_attribute_dict,
    ):
        "Test that transform raises an error if one column is a date and one is datetime"
        args = minimal_attribute_dict[self.transformer_name].copy()
        args["columns"] = columns

        x = uninitialized_transformers[self.transformer_name](
            **args,
        )

        df = create_date_diff_different_dtypes()
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


class TestInit(
    NewColumnNameInitMixintests,
    DropOriginalInitMixinTests,
    ColumnStrListInitTests,
):
    """Generic tests for transformer.init()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseGenericDateTransformer"


class TestFit(GenericFitTests):
    """Generic tests for transformer.fit()"""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseGenericDateTransformer"


class TestTransform(GenericTransformTests, GenericDatesMixinTransformTests):
    """Tests for BaseGenericDateTransformer.transform."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseGenericDateTransformer"


class TestOtherBaseBehaviour(OtherBaseBehaviourTests):
    """
    Class to run tests for BaseTransformerBehaviour outside the three standard methods.

    May need to overwite specific tests in this class if the tested transformer modifies this behaviour.
    """

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseGenericDateTransformer"
