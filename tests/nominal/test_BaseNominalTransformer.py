import pytest

import tests.test_data as d
from tubular.nominal import BaseNominalTransformer


class TestInit:
    """Test for BaseNominalTransformer object.
    Currently nothing to test."""


class TestCheckMappableRows:
    """Tests for the BaseNominalTransformer.check_mappable_rows method."""

    def test_exception_raised(self):
        """Test an exception is raised if non-mappable rows are present in X."""
        df = d.create_df_1()

        x = BaseNominalTransformer()
        x.columns = ["a", "b"]
        x.mappings = {
            "a": {1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7},
            "b": {"a": 1, "c": 2, "d": 3, "e": 4, "f": 5},
        }

        with pytest.raises(
            ValueError,
            match="BaseNominalTransformer: nulls would be introduced into column b from levels not present in mapping",
        ):
            x.check_mappable_rows(df)
