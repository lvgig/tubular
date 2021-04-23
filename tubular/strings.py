"""
This module contains transformers that apply string functions.
"""

import pandas as pd

from tubular.base import BaseTransformer


class SeriesStrMethodTransformer(BaseTransformer):
    """Tranformer that applies a pandas.Series.str method.

    Transformer assigns the output of the method to a new column. It is possible to
    supply other key word arguments to the transform method, which will be passed to the
    pandas.Series.str method being called.

    Be aware it is possible to supply incompatible arguments to init that will only be
    identified when transform is run. This is because there are many combinations of method, input
    and output sizes. Additionally some methods may only work as expected when called in
    transform with specific key word arguments.

    Parameters
    ----------
    new_column_name : str
        The name of the column to be assigned to the output of running the pd.Series.str in transform.

    pd_method_name : str
        The name of the pandas.Series.str method to call.

    columns : str
        Column to apply the transformer to. If a str is passed this is put into a list. Value passed
        in columns is saved in the columns attribute on the object. Note this has no default value so
        the user has to specify the columns when initialising the transformer. This is avoid likely
        when the user forget to set columns, in this case all columns would be picked up when super
        transform runs.

    pd_method_kwargs : dict, default = {}
        A dictionary of keyword arguments to be passed to the pd.Series.str method when it is called.

    **kwargs
        Arbitrary keyword arguments passed onto BaseTransformer.__init__().

    Attributes
    ----------
    new_column_name : str
        The name of the column or columns to be assigned to the output of running the
        pd.Series.str in transform.

    pd_method_name : str
        The name of the pd.Series.str method to call.

    """

    def __init__(
        self, new_column_name, pd_method_name, columns, pd_method_kwargs={}, **kwargs
    ):

        if type(columns) is list:

            if len(columns) > 1:

                raise ValueError(
                    f"columns arg should contain only 1 column name but got {len(columns)}"
                )

        super().__init__(columns=columns, **kwargs)

        if type(new_column_name) is not str:

            raise TypeError(
                f"unexpected type ({type(new_column_name)}) for new_column_name, must be str"
            )

        if type(pd_method_name) is not str:

            raise TypeError(
                f"unexpected type ({type(pd_method_name)}) for pd_method_name, expecting str"
            )

        if type(pd_method_kwargs) is not dict:

            raise TypeError(
                f"pd_method_kwargs should be a dict but got type {type(pd_method_kwargs)}"
            )

        else:

            for i, k in enumerate(pd_method_kwargs.keys()):

                if not type(k) is str:

                    raise TypeError(
                        f"unexpected type ({type(k)}) for pd_method_kwargs key in position {i}, must be str"
                    )

        self.new_column_name = new_column_name
        self.pd_method_name = pd_method_name
        self.pd_method_kwargs = pd_method_kwargs

        try:

            ser = pd.Series(["a"])
            getattr(ser.str, pd_method_name)

        except Exception as err:

            raise AttributeError(
                f"""error accessing "str.{pd_method_name}" method on pd.Series object - pd_method_name should be a pd.Series.str method"""
            ) from err

    def transform(self, X):
        """Transform specific column on input pandas.DataFrame (X) using the given pandas.Series.str method and
        assign the output back to column in X.

        Any keyword arguments set in the pd_method_kwargs attribute are passed onto the pd.Series.str method
        when calling it.

        Parameters
        ----------
        X : pd.DataFrame
            Data to transform.

        Returns
        -------
        X : pd.DataFrame
            Input X with additional column (self.new_column_name) added. These contain the output of
            running the pd.Series.str method.

        """

        X = super().transform(X)

        X[self.new_column_name] = getattr(X[self.columns[0]].str, self.pd_method_name)(
            **self.pd_method_kwargs
        )

        return X
