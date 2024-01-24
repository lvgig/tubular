import pytest
import test_aide as ta

import tests.test_data as d
from tubular.base import BaseTransformer


class TestTransform:
    @pytest.mark.parametrize(
        ("df", "expected"),
        ta.pandas.adjusted_dataframe_params(d.create_df_1(), d.create_df_1()),
    )
    def test_X_returned(self, df, expected):
        """Test that X is returned from transform."""
        x = BaseTransformer(columns="a", copy=True)

        df_transformed = x.transform(X=df)

        ta.equality.assert_equal_dispatch(
            expected=expected,
            actual=df_transformed,
            msg="Check X returned from transform",
        )
