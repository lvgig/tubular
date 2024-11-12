import re

import pandas as pd
import pytest

import tests.test_data as d
from tests.base_tests import GenericFitTests, GenericInitTests, GenericTransformTests


class BaseNumericTransformerInitTests(GenericInitTests):
    """
    Tests for BaseNumericTransformer.init().
    Note this deliberately avoids starting with "Tests" so that the tests are not run on import.
    """


class BaseNumericTransformerFitTests(GenericFitTests):
    """
    Tests for BaseNumericTransformer.fit().
    Note this deliberately avoids starting with "Tests" so that the tests are not run on import.
    """

    @pytest.mark.parametrize(
        ("df_generator", "bad_cols"),
        [
            (d.create_df_2, ["b"]),  # str
            (d.create_is_between_dates_df_1, ["a"]),  # datetime
            (d.create_bool_and_float_df, ["b"]),  # bool
            (d.create_df_with_none_and_nan_cols, ["b"]),  # None
        ],
    )
    def test_non_numeric_exception_raised(
        self,
        initialized_transformers,
        df_generator,
        bad_cols,
    ):
        """Test an exception is raised if self.columns are non-numeric in X."""
        df = df_generator()
        # add in 'target column' for fit
        df["c"] = [1] * len(df)

        x = initialized_transformers[self.transformer_name]
        x.columns = bad_cols

        with pytest.raises(
            TypeError,
            match=re.escape(
                f"{self.transformer_name}: The following columns are not numeric in X; {bad_cols}",
            ),
        ):
            x.fit(df, df["c"])

    @pytest.mark.parametrize(
        ("df_generator", "cols"),
        [
            (d.create_df_2, ["a"]),  # int
            (d.create_bool_and_float_df, ["a"]),  # float
            (d.create_df_with_none_and_nan_cols, ["a"]),  # nan
        ],
    )
    def test_numeric_passes(self, initialized_transformers, df_generator, cols):
        """Test check passes if self.columns numeric in X."""
        df = df_generator()
        # add in 'target column' for fit
        df["c"] = [1] * len(df)

        x = initialized_transformers[self.transformer_name]
        x.columns = cols

        x.fit(df, df["c"])


class BaseNumericTransformerTransformTests(
    GenericTransformTests,
):
    """
    Tests for the transform method on BaseNumericTransformer.
    Note this deliberately avoids starting with "Tests" so that the tests are not run on import.
    """

    @pytest.mark.parametrize(
        ("df_generator", "bad_cols"),
        [
            (d.create_df_2, ["b"]),  # str
            (d.create_is_between_dates_df_1, ["a"]),  # datetime
            (d.create_bool_and_float_df, ["b"]),  # bool
            (d.create_df_with_none_and_nan_cols, ["b"]),  # None
        ],
    )
    def test_non_numeric_exception_raised(
        self,
        initialized_transformers,
        df_generator,
        bad_cols,
    ):
        """Test an exception is raised if self.columns are non-numeric in X."""
        df = df_generator()
        # add in 'target column' for and additional numeric column fit
        df["c"] = [1] * len(df)

        x = initialized_transformers[self.transformer_name]
        x.columns = bad_cols

        # if the transformer fits, run a working fit before transform
        if x.FITS:
            # create numeric df to fit on
            numeric_df = pd.DataFrame({col: df["c"] for col in [*x.columns, "c"]})
            x.fit(numeric_df, numeric_df["c"])

        with pytest.raises(
            TypeError,
            match=re.escape(
                rf"{self.transformer_name}: The following columns are not numeric in X; {x.columns}",
            ),
        ):
            x.transform(df)

    @pytest.mark.parametrize(
        ("df_generator"),
        [
            d.create_df_2,  # int
            d.create_bool_and_float_df,  # float
            d.create_df_with_none_and_nan_cols,  # nan
        ],
    )
    def test_numeric_passes(self, initialized_transformers, df_generator):
        """Test check passes if self.columns numeric in X."""
        df = df_generator()
        # add in 'target column' for and additional numeric column fit
        df["c"] = [1] * len(df)
        df["b"] = [1] * len(df)

        x = initialized_transformers[self.transformer_name]
        x.columns = ["a", "b"]

        if x.FITS:
            # create numeric df to fit on
            numeric_df = pd.DataFrame({col: df["c"] for col in [*x.columns, "c"]})
            x.fit(numeric_df, numeric_df["c"])

        x.transform(df)


class TestInit(BaseNumericTransformerInitTests):
    """Tests for BaseNumericTransformer.init()"""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseNumericTransformer"


class TestFit(BaseNumericTransformerFitTests):
    """Tests for BaseNumericTransformer.fit()"""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseNumericTransformer"


class TestTransform(BaseNumericTransformerTransformTests):
    """Tests for BaseNumericTransformer.transform()"""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseNumericTransformer"
