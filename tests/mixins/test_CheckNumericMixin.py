import pytest

from tests.test_data import create_df_2
from tubular.mixins import CheckNumericMixin


class TestCheckNumericMixin:
    "tests for CheckNumericMixin class"

    def test_check_numeric_columns(self):
        "test check_numeric_columns method raises appropriate error"

        df = create_df_2()

        obj = CheckNumericMixin()

        # this object is generally wrapped in a transformer with a .columns attr, set this here
        obj.columns = ["b", "c"]

        with pytest.raises(
            TypeError,
            match=r"CheckNumericMixin: The following columns are not numeric in X; \['b', 'c'\]",
        ):
            obj.check_numeric_columns(df)