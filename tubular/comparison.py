from __future__ import annotations

import pandas as pd  # noqa: TCH002

from tubular.base import BaseTwoColumnTransformer
from tubular.mixins import DropOriginalMixin, NewColumnNameMixin


class EqualityChecker(DropOriginalMixin, NewColumnNameMixin, BaseTwoColumnTransformer):
    """Transformer to check if two columns are equal.

    Parameters
    ----------
    columns: list
        List containing names of the two columns to check.

    new_column_name: string
        string containing the name of the new column.

    drop_original: boolean = False
        boolean representing dropping the input columns from X after checks.

    **kwargs:
        Arbitrary keyword arguments passed onto BaseTransformer.init method.

    """

    def __init__(
        self,
        columns: list,
        new_column_name: str,
        drop_original: bool = False,
        **kwargs: dict[str, bool],
    ) -> None:
        super().__init__(columns=columns, **kwargs)

        DropOriginalMixin.set_drop_original_column(self, drop_original)
        NewColumnNameMixin.check_and_set_new_column_name(self, new_column_name)

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

        X[self.new_column_name] = X[self.columns[0]] == X[self.columns[1]]

        # Drop original columns if self.drop_original is True
        DropOriginalMixin.drop_original_column(
            self,
            X,
            self.drop_original,
            self.columns,
        )

        return X
