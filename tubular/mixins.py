import numpy as np
import pandas as pd


class WeightColumnMixin:
    def check_weights_column(X: pd.DataFrame, weights_column: str) -> None:
        """Helper method for validating weights column in dataframe.

        Args:
        ----
            X (pd.DataFrame): df containing weight column
            weights_column (str): name of weight column

        """
        # check if given weight is in columns
        if weights_column not in X.columns:
            msg = f"weight col ({weights_column}) is not present in columns of data"
            raise ValueError(msg)

        # check weight is numeric

        if not pd.api.types.is_numeric_dtype(X[weights_column]):
            msg = "weight column must be numeric."
            raise ValueError(msg)

        # check weight is positive

        if (X[weights_column] < 0).sum() != 0:
            msg = "weight column must be positive"
            raise ValueError(msg)

        # check weight non-null
        if X[weights_column].isna().sum() != 0:
            msg = "weight column must be non-null"
            raise ValueError(msg)

        # check weight not inf
        if np.isinf(X[weights_column]).any():
            msg = "weight column must not contain infinite values."
            raise ValueError(msg)
