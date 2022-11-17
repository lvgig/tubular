from tubular.base import BaseTransformer


class SetValueTransformer(BaseTransformer):
    """Transformer to set value of column(s) to a given value.

    This should be used if columns need to be set to a constant value.

    Parameters
    ----------
    columns: list or str
        Columns to set values.

    value : various
        Value to set.

    copy : bool
        True if X should be copied before transforms are applied, False otherwise

    verbose : bool
        True to print statements to show which methods are being run or not.

    """

    def __init__(self, columns, value, copy=True, verbose=False):

        self.value = value

        super().__init__(columns=columns, copy=copy, verbose=verbose)

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
