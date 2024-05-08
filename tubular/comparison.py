from __future__ import annotations

import pandas as pd  # noqa: TCH002

from tubular.base import BaseTransformer
from tubular.mixins import BaseDropOriginalMixin


class EqualityChecker(BaseDropOriginalMixin, BaseTransformer):
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
        self,
        columns: list,
        new_col_name: str,
        drop_original: bool = False,
        **kwargs: dict[str, bool],
    ) -> None:
        super().__init__(columns=columns, **kwargs)

        if not (isinstance(columns, list)):
            msg = f"{self.classname()}: columns should be list"
            raise TypeError(msg)

        if len(columns) != 2:
            msg = f"{self.classname()}: This transformer works with two columns only"
            raise ValueError(msg)

        if not (isinstance(new_col_name, str)):
            msg = f"{self.classname()}: new_col_name should be str"
            raise TypeError(msg)

        self.new_col_name = new_col_name

        BaseDropOriginalMixin.set_drop_original_column(self, drop_original)

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

        # Drop original columns
        BaseDropOriginalMixin.drop_original_column(
            self,
            X,
            self.drop_original,
            self.columns,
        )

        return X
