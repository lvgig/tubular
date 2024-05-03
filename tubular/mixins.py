import numpy as np
import pandas as pd


class WeightColumnMixin:
    def check_weights_column(self, X: pd.DataFrame, weights_column: str) -> None:
        """Helper method for validating weights column in dataframe.

        Args:
        ----
            X (pd.DataFrame): df containing weight column
            weights_column (str): name of weight column

        """
        # check if given weight is in columns
        if weights_column not in X.columns:
            msg = f"{self.classname()}: weight col ({weights_column}) is not present in columns of data"
            raise ValueError(msg)

        # check weight is numeric

        if not pd.api.types.is_numeric_dtype(X[weights_column]):
            msg = f"{self.classname()}: weight column must be numeric."
            raise ValueError(msg)

        # check weight is positive

        if (X[weights_column] < 0).sum() != 0:
            msg = f"{self.classname()}: weight column must be positive"
            raise ValueError(msg)

        # check weight non-null
        if X[weights_column].isna().sum() != 0:
            msg = f"{self.classname()}: weight column must be non-null"
            raise ValueError(msg)

        # check weight not inf
        if np.isinf(X[weights_column]).any():
            msg = f"{self.classname()}: weight column must not contain infinite values."
            raise ValueError(msg)

        if X[weights_column].sum() <= 0:
            msg = f"{self.classname()}: total sample weights are not greater than 0"
            raise ValueError(msg)

    def check_and_set_weight(self, weights_column: str) -> None:
        """Helper method that validates and assigns the specified column name to be used as the weights_column attribute.
        This function ensures that the `weights_column` parameter is either a string representing
        the column name or None. If `weights_column` is not of type str and is not None, it raises
        a TypeError.

        Parameters:
            weights_column (str or None): The name of the column to be used as weights. If None, no weights are used.

        Raises:
            TypeError: If `weights_column` is neither a string nor None.

        Returns:
            None
        """

        if weights_column is not None and not isinstance(weights_column, str):
            msg = "weights_column should be str or None"
            raise TypeError(msg)
        self.weights_column = weights_column
