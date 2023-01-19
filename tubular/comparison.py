from tubular.base import BaseTransformer
import pandas as pd


class EqualityChecker(BaseTransformer):

    """Transformer to check if two columns are equal.

    Parameters
    ----------
    columns: list
        List containing names of the two columns to check.

    new_col_name: string
        string containing the name of the new column.

    drop_original: boolean = False
        boolean representing dropping the input columns from X after checks.

    **kwargs:
        Arbitrary keyword arguments passed onto BaseTransformer.init method.

    """

    def __init__(
        self, columns: list, new_col_name: str, drop_original: bool = False, **kwargs
    ) -> None:

        super().__init__(columns=columns, **kwargs)

        if not (isinstance(columns, list)):
            raise TypeError(f"{self.classname()}: columns should be list")

        if len(columns) != 2:
            raise ValueError(
                f"{self.classname()}: This transformer works with two columns only"
            )

        if not (isinstance(new_col_name, str)):
            raise TypeError(f"{self.classname()}: new_col_name should be str")

        if not (isinstance(drop_original, bool)):
            raise TypeError(f"{self.classname()}: drop_original should be bool")

        self.new_col_name = new_col_name
        self.drop_original = drop_original

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:

        """Create a column which is populated by the boolean
        matching between two columns iterated over rows.

        Parameters
        ----------
        X : pd.DataFrame
            Data to apply mappings to.

        Returns
        -------
        X : pd.DataFrame
            Transformed input X with additional boolean column.

        """
        X = super().transform(X)

        X[self.new_col_name] = X[self.columns[0]] == X[self.columns[1]]

        if self.drop_original:

            X.drop(self.columns, axis=1, inplace=True)

        return X
