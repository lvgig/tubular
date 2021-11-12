"""
This module contains transformers that apply numeric functions.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler

from tubular.base import BaseTransformer


class LogTransformer(BaseTransformer):
    """Transformer to apply log transformation.

    Transformer has the option to add 1 to the columns to log and drop the
    original columns.

    Parameters
    ----------
    columns : None or str or list
        Columns to log transform.

    base : None or float/int
        Base for log transform. If None uses natural log.

    add_1 : bool
        Should a constant of 1 be added to the columns to be transformed prior to
        applying the log transform?

    drop : bool
        Should the original columns to be transformed be dropped after applying the
        log transform?

    suffix : str, default = '_log'
        The suffix to add onto the end of column names for new columns.

    **kwargs
        Arbitrary keyword arguments passed onto BaseTransformer.init method.

    Attributes
    ----------
    add_1 : bool
        The name of the column or columns to be assigned to the output of running the
        pandas method in transform.

    drop : bool
        The name of the pandas.DataFrame method to call.

    suffix : str
        The suffix to add onto the end of column names for new columns.

    """

    def __init__(
        self, columns, base=None, add_1=False, drop=True, suffix="log", **kwargs
    ):

        super().__init__(columns=columns, **kwargs)

        if base is not None:
            if not isinstance(base, (int, float)):
                raise ValueError("base should be numeric or None")
            if not base > 0:
                raise ValueError("base should be strictly positive")

        self.base = base
        self.add_1 = add_1
        self.drop = drop
        self.suffix = suffix

    def transform(self, X):
        """Applies the log transform to the specified columns.

        If the drop attribute is True then the original columns are dropped. If
        the add_1 attribute is True then the original columns + 1 are logged.

        Parameters
        ----------
        X : pd.DataFrame
            The dataframe to be transformed.

        Returns
        -------
        X : pd.DataFrame
            The dataframe with the specified columns logged, optionally dropping the original
            columns if self.drop is True.

        """

        X = super().transform(X)

        numeric_column_types = X[self.columns].apply(
            pd.api.types.is_numeric_dtype, axis=0
        )

        if not numeric_column_types.all():

            non_numeric_columns = list(
                numeric_column_types.loc[~numeric_column_types].index
            )

            raise TypeError(
                f"The following columns are not numeric in X; {non_numeric_columns}"
            )

        new_column_names = [f"{column}_{self.suffix}" for column in self.columns]

        if self.add_1:

            if (X[self.columns] <= -1).sum().sum() > 0:

                raise ValueError(
                    "values less than or equal to 0 in columns (after adding 1), make greater than 0 before using transform"
                )

            if self.base is None:

                X[new_column_names] = np.log(X[self.columns] + 1)

            else:

                X[new_column_names] = np.log(X[self.columns] + 1) / np.log(self.base)

        else:

            if (X[self.columns] <= 0).sum().sum() > 0:

                raise ValueError(
                    "values less than or equal to 0 in columns, make greater than 0 before using transform"
                )

            if self.base is None:

                X[new_column_names] = np.log(X[self.columns])

            else:

                X[new_column_names] = np.log(X[self.columns]) / np.log(self.base)

        if self.drop:

            X.drop(self.columns, axis=1, inplace=True)

        return X


class CutTransformer(BaseTransformer):
    """Class to bin a column into discrete intervals.

    Class simply uses the [pd.cut](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.cut.html)
    method on the specified column.

    Parameters
    ----------
    column : str
        Name of the column to discretise.

    new_column_name : str
        Name given to the new discrete column.

    cut_kwargs : dict, default = {}
        A dictionary of keyword arguments to be passed to the pd.cut method when it is called in transform.

    **kwargs
        Arbitrary keyword arguments passed onto BaseTransformer.init().

    """

    def __init__(self, column, new_column_name, cut_kwargs={}, **kwargs):

        if not type(column) is str:

            raise TypeError(
                "column arg (name of column) should be a single str giving the column to discretise"
            )

        if not type(new_column_name) is str:

            raise TypeError("new_column_name must be a str")

        if not type(cut_kwargs) is dict:

            raise TypeError(
                f"cut_kwargs should be a dict but got type {type(cut_kwargs)}"
            )

        else:

            for i, k in enumerate(cut_kwargs.keys()):

                if not type(k) is str:

                    raise TypeError(
                        f"unexpected type ({type(k)}) for cut_kwargs key in position {i}, must be str"
                    )

        self.cut_kwargs = cut_kwargs
        self.new_column_name = new_column_name

        # This attribute is not for use in any method, use 'columns' instead.
        # Here only as a fix to allow string representation of transformer.
        self.column = column

        super().__init__(columns=[column], **kwargs)

    def transform(self, X):
        """Discretise specified column using pd.cut.

        Parameters
        ----------
        X : pd.DataFrame
            Data with column to transform.

        """

        X = super().transform(X)

        if not pd.api.types.is_numeric_dtype(X[self.columns[0]]):

            raise TypeError(
                f"{self.columns[0]} should be a numeric dtype but got {X[self.columns[0]].dtype}"
            )

        X[self.new_column_name] = pd.cut(X[self.columns[0]], **self.cut_kwargs)

        return X


class ScalingTransformer(BaseTransformer):
    """Transformer to perform scaling of numeric columns.

    Transformer can apply min max scaling, max absolute scaling or standardisation (subtract mean and divide by std).
    The transformer uses the appropriate sklearn.preprocessing scaler.

    Parameters
    ----------
    columns : str, list or None
        Name of the columns to apply scaling to.

    scaler_type : str
        Type of scaler to use, must be one of 'min_max', 'max_abs' or 'standard'. The corresponding
        sklearn.preprocessing scaler used in each case is MinMaxScaler, MaxAbsScaler or StandardScaler.

    scaler_kwargs : dict, default = {}
        A dictionary of keyword arguments to be passed to the scaler object when it is initialised.

    **kwargs
        Arbitrary keyword arguments passed onto BaseTransformer.init().

    """

    def __init__(self, columns, scaler_type, scaler_kwargs={}, **kwargs):

        if not type(scaler_kwargs) is dict:

            raise TypeError(
                f"scaler_kwargs should be a dict but got type {type(scaler_kwargs)}"
            )

        else:

            for i, k in enumerate(scaler_kwargs.keys()):

                if not type(k) is str:

                    raise TypeError(
                        f"unexpected type ({type(k)}) for scaler_kwargs key in position {i}, must be str"
                    )

        allowed_scaler_values = ["min_max", "max_abs", "standard"]

        if scaler_type not in allowed_scaler_values:

            raise ValueError(f"scaler_type should be one of; {allowed_scaler_values}")

        if scaler_type == "min_max":

            self.scaler = MinMaxScaler(**scaler_kwargs)

        elif scaler_type == "max_abs":

            self.scaler = MaxAbsScaler(**scaler_kwargs)

        elif scaler_type == "standard":

            self.scaler = StandardScaler(**scaler_kwargs)

        # This attribute is not for use in any method
        # Here only as a fix to allow string representation of transformer.
        self.scaler_kwargs = scaler_kwargs
        self.scaler_type = scaler_type

        super().__init__(columns=columns, **kwargs)

    def check_numeric_columns(self, X):
        """Method to check all columns (specicifed in self.columns) in X are all numeric.

        Parameters
        ----------
        X : pd.DataFrame
            Data containing columns to check.

        """

        numeric_column_types = X[self.columns].apply(
            pd.api.types.is_numeric_dtype, axis=0
        )

        if not numeric_column_types.all():

            non_numeric_columns = list(
                numeric_column_types.loc[~numeric_column_types].index
            )

            raise TypeError(
                f"The following columns are not numeric in X; {non_numeric_columns}"
            )

        return X

    def fit(self, X, y=None):
        """Fit scaler to input data.

        Parameters
        ----------
        X : pd.DataFrame
            Dataframe with columns to learn scaling values from.

        y : None
            Required for pipeline.

        """

        super().fit(X, y)

        X = self.check_numeric_columns(X)

        self.scaler.fit(X[self.columns])

        return self

    def transform(self, X):
        """Transform input data X with fitted scaler.

        Parameters
        ----------
        X : pd.DataFrame
            Dataframe containing columns to be scaled.

        Returns
        -------
        X : pd.DataFrame
            Input X with columns scaled.

        """

        X = super().transform(X)

        X = self.check_numeric_columns(X)

        X[self.columns] = self.scaler.transform(X[self.columns])

        return X
