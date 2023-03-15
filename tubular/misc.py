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
