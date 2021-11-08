"""
This module contains transformers that other transformers in the package inherit
from. These transformers contain key checks to be applied in all cases.
"""

import pandas as pd
import warnings
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from tubular._version import __version__


class BaseTransformer(TransformerMixin, BaseEstimator):
    """Base tranformer class which all other transformers in the package inherit from.

    Provides fit and transform methods (required by sklearn transformers), simple input checking
    and functionality to copy X prior to transform.

    Parameters
    ----------
    columns : None or list or str, default = None
        Columns to apply the transformer to. If a str is passed this is put into a list. Value passed
        in columns is saved in the columns attribute on the object.

    copy : bool, default = True
        Should X be copied before tansforms are applied?

    verbose : bool, default = False
        Should statements be printed when methods are run?

    **kwds
        Arbitrary keyword arguments.

    Attributes
    ----------
    columns : list or None
        Either a list of str values giving which columns in a input pandas.DataFrame the transformer
        will be applied to - or None.

    copy : bool
        Should X be copied before tansforms are applied?

    verbose : bool
        Print statements to show which methods are being run or not.

    version_ : str
        Version number (__version__ attribute from _version.py).

    """

    def __init__(self, columns=None, copy=True, verbose=False, **kwargs):

        self.version_ = __version__

        if not isinstance(verbose, bool):

            raise TypeError("verbose must be a bool")

        else:

            self.verbose = verbose

        if self.verbose:

            print("BaseTransformer.__init__() called")

        if columns is None:

            self.columns = None

        else:

            # make sure columns is a single str or list of strs
            if isinstance(columns, str):

                self.columns = [columns]

            elif isinstance(columns, list):

                if not len(columns) > 0:

                    raise ValueError("columns has no values")

                for c in columns:

                    if not isinstance(c, str):

                        raise TypeError(
                            "each element of columns should be a single (string) column name"
                        )

                self.columns = columns

            else:

                raise TypeError(
                    "columns must be a string or list with the columns to be pre-processed (if specified)"
                )

        if not isinstance(copy, bool):

            raise TypeError("copy must be a bool")

        else:

            self.copy = copy

    def fit(self, X, y=None):
        """Base transformer fit method, checks X and y types. Currently only pandas DataFrames are allowed for X
        and DataFrames or Series for y.

        Fit calls the columns_set_or_check method which will set the columns attribute to all columns in X, if it
        is None.

        Parameters
        ----------
        X : pd.DataFrame
            Data to fit the transformer on.

        y : None or pd.DataFrame or pd.Series, default = None
            Optional argument only required for the transformer to work with sklearn pipelines.

        """

        if self.verbose:

            print("BaseTransformer.fit() called")

        self.columns_set_or_check(X)

        if not X.shape[0] > 0:

            raise ValueError(f"X has no rows; {X.shape}")

        if y is not None:

            if not isinstance(y, pd.Series):

                raise TypeError("unexpected type for y, should be a pd.Series")

            if not y.shape[0] > 0:

                raise ValueError(f"y is empty; {y.shape}")

        return self

    def _combine_X_y(self, X, y):
        """Combine X and y by adding a new column with the values of y to a copy of X.

        The new column response column will be called `_temporary_response`.

        This method can be used by transformers that need to use the response, y, together
        with the explanatory variables, X, in their `fit` methods.

        Parameters
        ----------
        X : pd.DataFrame
            Data containing explanatory variables.

        y : pd.Series
            Response variable.

        """

        if not isinstance(X, pd.DataFrame):

            raise TypeError("X should be a pd.DataFrame")

        if not isinstance(y, pd.Series):

            raise TypeError("y should be a pd.Series")

        if X.shape[0] != y.shape[0]:

            raise ValueError(
                f"X and y have different numbers of rows ({X.shape[0]} vs {y.shape[0]})"
            )

        if not (X.index == y.index).all():

            warnings.warn("X and y do not have equal indexes")

        X_y = X.copy()

        X_y["_temporary_response"] = y.values

        return X_y

    def transform(self, X):
        """Base transformer transform method; checks X type (pandas DataFrame only) and copies data if requested.

        Transform calls the columns_check method which will check columns in columns attribute are in X.

        Parameters
        ----------
        X : pd.DataFrame
            Data to transform with the transformer.

        Returns
        -------
        X : pd.DataFrame
            Input X, copied if specified by user.

        """

        self.columns_check(X)

        if self.verbose:

            print("BaseTransformer.transform() called")

        if self.copy:

            X = X.copy()

        if not X.shape[0] > 0:

            raise ValueError(f"X has no rows; {X.shape}")

        return X

    def check_is_fitted(self, attribute):
        """Check if particular attributes are on the object. This is useful to do before running transform to avoid
        trying to transform data without first running the fit method.

        Wrapper for utils.validation.check_is_fitted function.

        Parameters
        ----------
        attributes : List
            List of str values giving names of attribute to check exist on self.

        """

        check_is_fitted(self, attribute)

    def columns_check(self, X):
        """Method to check that the columns attribute is set and all values are present in X.

        Parameters
        ----------
        X : pd.DataFrame
            Data to check columns are in.

        """

        if not isinstance(X, pd.DataFrame):

            raise TypeError("X should be a pd.DataFrame")

        if self.columns is None:

            raise ValueError("columns not set")

        if not isinstance(self.columns, list):

            raise TypeError("self.columns should be a list")

        for c in self.columns:

            if c not in X.columns.values:

                raise ValueError("variable " + c + " is not in X")

    def columns_set_or_check(self, X):
        """Function to check or set columns attribute.

        If the columns attribute is None then set it to all columns in X. Otherwise run the columns_check method.

        Parameters
        ----------
        X : pd.DataFrame
            Data to check columns are in.

        """

        if not isinstance(X, pd.DataFrame):

            raise TypeError("X should be a pd.DataFrame")

        if self.columns is None:

            self.columns = list(X.columns.values)

        else:

            self.columns_check(X)


class ReturnKeyDict(dict):
    """Dict class that implements __missing__ method to return the key if it is not present in the dict
    when looked up.

    This is intended to be used in combination with the pd.Series.map function so that it does not
    introduce nulls if a key is not found (which is the behaviour if used with a standard dict).
    """

    def __missing__(self, key):
        """Function to return passed key.

        Parameters
        ----------
        key : various
            key to lookup from dict

        Returns
        -------
        key : input key

        """

        return key


class DataFrameMethodTransformer(BaseTransformer):
    """Tranformer that applies a pandas.DataFrame method.

    Transformer assigns the output of the method to a new column or columns. It is possible to
    supply other key word arguments to the transform method, which will be passed to the
    pandas.DataFrame method being called.

    Be aware it is possible to supply incompatible arguments to init that will only be
    identified when transform is run. This is because there are many combinations of method, input
    and output sizes. Additionally some methods may only work as expected when called in
    transform with specific key word arguments.

    Parameters
    ----------
    new_column_name : str or list of str
        The name of the column or columns to be assigned to the output of running the
        pandas method in transform.

    pd_method_name : str
        The name of the pandas.DataFrame method to call.

    columns : None or list or str
        Columns to apply the transformer to. If a str is passed this is put into a list. Value passed
        in columns is saved in the columns attribute on the object. Note this has no default value so
        the user has to specify the columns when initialising the transformer. This is avoid likely
        when the user forget to set columns, in this case all columns would be picked up when super
        transform runs.

    pd_method_kwargs : dict, default = {}
        A dictionary of keyword arguments to be passed to the pd.DataFrame method when it is called.

    drop_original : bool, default = False
        Should original columns be dropped?

    **kwargs
        Arbitrary keyword arguments passed onto BaseTransformer.__init__().

    Attributes
    ----------
    new_column_name : str or list of str
        The name of the column or columns to be assigned to the output of running the
        pandas method in transform.

    pd_method_name : str
        The name of the pandas.DataFrame method to call.

    """

    def __init__(
        self,
        new_column_name,
        pd_method_name,
        columns,
        pd_method_kwargs={},
        drop_original=False,
        **kwargs,
    ):

        super().__init__(columns=columns, **kwargs)

        if type(new_column_name) is list:

            for i, item in enumerate(new_column_name):

                if not type(item) is str:

                    raise TypeError(
                        f"if new_column_name is a list, all elements must be strings but got {type(item)} in position {i}"
                    )

        elif not type(new_column_name) is str:

            raise TypeError(
                f"unexpected type ({type(new_column_name)}) for new_column_name, must be str or list of strings"
            )

        if not type(pd_method_name) is str:

            raise TypeError(
                f"unexpected type ({type(pd_method_name)}) for pd_method_name, expecting str"
            )

        if not type(pd_method_kwargs) is dict:

            raise TypeError(
                f"pd_method_kwargs should be a dict but got type {type(pd_method_kwargs)}"
            )

        else:

            for i, k in enumerate(pd_method_kwargs.keys()):

                if not type(k) is str:

                    raise TypeError(
                        f"unexpected type ({type(k)}) for pd_method_kwargs key in position {i}, must be str"
                    )

        if not type(drop_original) is bool:

            raise TypeError(
                f"unexpected type ({type(drop_original)}) for drop_original, expecting bool"
            )

        self.new_column_name = new_column_name
        self.pd_method_name = pd_method_name
        self.pd_method_kwargs = pd_method_kwargs
        self.drop_original = drop_original

        try:

            df = pd.DataFrame()
            getattr(df, pd_method_name)

        except Exception as err:

            raise AttributeError(
                f"""error accessing "{pd_method_name}" method on pd.DataFrame object - pd_method_name should be a pd.DataFrame method"""
            ) from err

    def transform(self, X):
        """Transform input pandas DataFrame (X) using the given pandas.DataFrame method and assign the output
        back to column or columns in X.

        Any keyword arguments set in the pd_method_kwargs attribute are passed onto the pandas DataFrame method when calling it.

        Parameters
        ----------
        X : pd.DataFrame
            Data to transform.

        Returns
        -------
        X : pd.DataFrame
            Input X with additional column or columns (self.new_column_name) added. These contain the output of
            running the pandas DataFrame method.

        """

        X = super().transform(X)

        X[self.new_column_name] = getattr(X[self.columns], self.pd_method_name)(
            **self.pd_method_kwargs
        )

        if self.drop_original:

            X.drop(self.columns, axis=1, inplace=True)

        return X
