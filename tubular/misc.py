from tubular.base import BaseTransformer
import pandas as pd


class SetValueTransformer(BaseTransformer):
    """Transformer to set value of column(s) to a given value.

    This should be used if columns need to be set to a constant value.

    Parameters
    ----------
    columns: list or str
        Columns to set values.

    value : various
        Value to set.

    **kwargs
        Arbitrary keyword arguments passed onto BaseTransformer.init method.

    """

    def __init__(self, columns, value, **kwargs):

        self.value = value

        super().__init__(columns=columns, **kwargs)

    def transform(self, X):
        """Set columns to value.

        Parameters
        ----------
        X : pd.DataFrame
            Data to apply mappings to.

        Returns
        -------
        X : pd.DataFrame
            Transformed input X with columns set to value.

        """

        X = super().transform(X)

        X[self.columns] = self.value

        return X


class SetColumnDtype(BaseTransformer):
    """
    Transformer to set transform columns in a dataframe to a dtype

    Parameters
    ----------
    columns : str or list
        Columns to set dtype. Must be set or transform will not run.

    dtype : type or string
        dtype object to set columns to or a string interpretable as one by pd.api.types.pandas_dtype
        e.g. float or 'float'
    """

    def __init__(self, columns, dtype):

        super().__init__(columns, copy=True)

        self.__validate_dtype(dtype)

        self.dtype = dtype

    def transform(self, X):

        X = super().transform(X)

        X[self.columns] = X[self.columns].astype(self.dtype)

        return X

    def __validate_dtype(self, dtype: str):
        """Check string is a valid dtype"""

        try:
            pd.api.types.pandas_dtype(dtype)
        except TypeError:
            raise TypeError(
                f"{self.classname()}: data type '{dtype}' not understood as a valid dtype"
            )
