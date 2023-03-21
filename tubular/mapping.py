"""
This module contains transformers that apply different types of mappings to columns.
"""

import pandas as pd
import numpy as np
from collections import OrderedDict

from tubular.base import BaseTransformer, ReturnKeyDict


class BaseMappingTransformer(BaseTransformer):
    """Base Transformer Extension for mapping transformers.

    Parameters
    ----------
    mappings : dict
        Dictionary containing column mappings. Each value in mappings should be a dictionary
        of key (column to apply mapping to) value (mapping dict for given columns) pairs. For
        example the following dict {'a': {1: 2, 3: 4}, 'b': {'a': 1, 'b': 2}} would specify
        a mapping for column a of 1->2, 3->4 and a mapping for column b of 'a'->1, b->2.

    **kwargs
        Arbitrary keyword arguments passed onto BaseTransformer.init method.

    Attributes
    ----------
    mappings : dict
        Dictionary of mappings for each column individually. The dict passed to mappings in
        init is set to the mappings attribute.

    """

    def __init__(self, mappings, **kwargs):

        if isinstance(mappings, dict):

            if not len(mappings) > 0:

                raise ValueError("mappings has no values")

            for j in mappings.values():

                if not isinstance(j, dict):

                    raise ValueError(
                        "values in mappings dictionary should be dictionaries"
                    )

            self.mappings = mappings

        else:

            raise ValueError("mappings must be a dictionary")

        columns = list(mappings.keys())

        super().__init__(columns=columns, **kwargs)

    def transform(self, X):
        """Base mapping transformer transform method.  Checks that the mappings
        dict has been fitted and calls the BaseTransformer transform method.

        Parameters
        ----------
        X : pd.DataFrame
            Data to apply mappings to.

        Returns
        -------
        X : pd.DataFrame
            Input X, copied if specified by user.

        """

        self.check_is_fitted(["mappings"])

        X = super().transform(X)

        return X


class BaseMappingTransformMixin(BaseTransformer):
    """Mixin class to apply standard pd.Series.map transform method.

    Transformer uses the mappings attribute which should be a dict of dicts/mappings
    for each required column.

    """

    def transform(self, X):
        """Applies the mapping defined in the mappings dict to each column in the columns
        attribute.

        Parameters
        ----------
        X : pd.DataFrame
            Data with nominal columns to transform.

        Returns
        -------
        X : pd.DataFrame
            Transformed input X with levels mapped accoriding to mappings dict.

        """

        self.check_is_fitted(["mappings"])

        X = super().transform(X)

        for c in self.columns:

            X[c] = X[c].map(self.mappings[c])

        return X


class MappingTransformer(BaseMappingTransformer, BaseMappingTransformMixin):
    """Transformer to map values in columns to other values e.g. to merge two levels into one.

    Note, the MappingTransformer does not require 'self-mappings' to be defined i.e. if you want
    to map a value to itself, you can omit this value from the mappings rather than having to
    map it to itself. This is because it uses the ReturnKeyDict  type to store the mappings
    for each columns, this dict will return the key i.e. the original value in that row if it
    is not available in the mapping dict.

    This transformer inherits from BaseMappingTransformMixin as well as the BaseMappingTransformer
    in order to access the startard pd.Series.map transform function.

    Parameters
    ----------
    mappings : dict
        Dictionary containing column mappings. Each value in mappings should be a dictionary
        of key (column to apply mapping to) value (mapping dict for given columns) pairs. For
        example the following dict {'a': {1: 2, 3: 4}, 'b': {'a': 1, 'b': 2}} would specify
        a mapping for column a of 1->2, 3->4 and a mapping for column b of 'a'->1, b->2.

    **kwargs
        Arbitrary keyword arguments passed onto BaseMappingTransformer.init method.

    Attributes
    ----------
    mappings : dict
        Dictionary of mappings for each column individually. The dict passed to mappings in
        init is set to the mappings attribute.

    """

    def __init__(self, mappings, **kwargs):

        for k, v in mappings.items():

            if isinstance(v, dict):

                mappings[k] = ReturnKeyDict(v)

            else:

                raise TypeError(
                    f"each item in mappings should be a dict but got type {type(v)} for key {k}"
                )

        BaseMappingTransformer.__init__(self, mappings=mappings, **kwargs)

    def transform(self, X):
        """Transfrom the input data X according to the mappings in the mappings attribute dict.

        This method calls the BaseMappingTransformMixin.transform. Note, this transform method is
        different to some of the transform methods in the nominal module, even though they also
        use the BaseMappingTransformMixin.transform method. Here, if a value does not exist in
        the mapping it is unchanged.

        Parameters
        ----------
        X : pd.DataFrame
            Data with nominal columns to transform.

        Returns
        -------
        X : pd.DataFrame
            Transformed input X with levels mapped accoriding to mappings dict.

        """

        X = BaseMappingTransformMixin.transform(self, X)

        return X


class CrossColumnMappingTransformer(BaseMappingTransformer):
    """Transformer to adjust values in one column based on the values of another column.

    Parameters
    ----------
    adjust_column : str
        The column to be adjusted.

    mappings : dict or OrderedDict
        Dictionary containing adjustments. Each value in adjustments should be a dictionary
        of key (column to apply adjustment based on) value (adjustment dict for given columns) pairs. For
        example the following dict {'a': {1: 'a', 3: 'b'}, 'b': {'a': 1, 'b': 2}}
        would replace the values in the adjustment column based off the values in column a using the mapping
        1->'a', 3->'b' and also replace based off the values in column b using a mapping 'a'->1, 'b'->2.
        If more than one column is defined for this mapping, then this object must be an OrderedDict
        to ensure reproducability.

    **kwargs
        Arbitrary keyword arguments passed onto BaseTransformer.init method.

    Attributes
    ----------

    adjust_column : str
        Column containing the values to be adjusted.

    mappings : dict
        Dictionary of mappings for each column individually to be applied to the adjust_column.
        The dict passed to mappings in init is set to the mappings attribute.


    """

    def __init__(self, adjust_column, mappings, **kwargs):

        super().__init__(mappings=mappings, **kwargs)

        if not isinstance(adjust_column, str):

            raise TypeError("adjust_column should be a string")

        if len(mappings) > 1:

            if not isinstance(mappings, OrderedDict):

                raise TypeError(
                    "mappings should be an ordered dict for 'replace' mappings using multiple columns"
                )

        self.adjust_column = adjust_column

    def transform(self, X):
        """Transforms values in given column using the values provided in the adjustments dictionary.

        Parameters
        ----------
        X : pd.DataFrame
            Data to apply adjustments to.

        Returns
        -------
        X : pd.DataFrame
            Transformed data X with adjustments applied to specified columns.

        """

        self.check_is_fitted(["adjust_column"])

        X = super().transform(X)

        if self.adjust_column not in X.columns.values:

            raise ValueError("variable " + self.adjust_column + " is not in X")

        for i in self.columns:

            for j in self.mappings[i].keys():

                X[self.adjust_column] = np.where(
                    (X[i] == j), self.mappings[i][j], X[self.adjust_column]
                )

        return X


class CrossColumnMultiplyTransformer(BaseMappingTransformer):
    """Transformer to apply a multiplicative adjustment to values in one column based on the values of another column.

    Parameters
    ----------
    adjust_column : str
        The column to be adjusted.  The data type of this column must be int or float.

    mappings : dict
        Dictionary containing adjustments. Each value in adjustments should be a dictionary
        of key (column to apply adjustment based on) value (adjustment dict for given columns) pairs. For
        example the following dict {'a': {1: 2, 3: 5}, 'b': {'a': 0.5, 'b': 1.1}}
        would multiply the values in the adjustment column based off the values in column a using the mapping
        1->2*value, 3->5*value and also multiply based off the values in column b using a mapping
        'a'->0.5*value, 'b'->1.1*value.
        The values within the dicts defining the multipliers must have type int or float.

    **kwargs
        Arbitrary keyword arguments passed onto BaseTransformer.init method.

    Attributes
    ----------

    adjust_column : str
        Column containing the values to be adjusted.

    mappings : dict
        Dictionary of multiplicative adjustments for each column individually to be applied to the adjust_column.
        The dict passed to mappings in init is set to the mappings attribute.


    """

    def __init__(self, adjust_column, mappings, **kwargs):

        super().__init__(mappings=mappings, **kwargs)

        if not isinstance(adjust_column, str):

            raise TypeError("adjust_column should be a string")

        for j in mappings.values():

            for k in j.values():

                if type(k) not in [int, float]:

                    raise TypeError("mapping values must be numeric")

        self.adjust_column = adjust_column

    def transform(self, X):
        """Transforms values in given column using the values provided in the adjustments dictionary.

        Parameters
        ----------
        X : pd.DataFrame
            Data to apply adjustments to.

        Returns
        -------
        X : pd.DataFrame
            Transformed data X with adjustments applied to specified columns.

        """

        self.check_is_fitted(["adjust_column"])

        X = super().transform(X)

        if self.adjust_column not in X.columns.values:

            raise ValueError("variable " + self.adjust_column + " is not in X")

        if not pd.api.types.is_numeric_dtype(X[self.adjust_column]):

            raise TypeError(
                "variable " + self.adjust_column + " must have numeric dtype."
            )

        for i in self.columns:

            for j in self.mappings[i].keys():

                X[self.adjust_column] = np.where(
                    (X[i] == j),
                    X[self.adjust_column] * self.mappings[i][j],
                    X[self.adjust_column],
                )

        return X


class CrossColumnAddTransformer(BaseMappingTransformer):
    """Transformer to apply an additive adjustment to values in one column based on the values of another column.

    Parameters
    ----------
    adjust_column : str
        The column to be adjusted.  The data type of this column must be int or float.

    mappings : dict
        Dictionary containing adjustments. Each value in adjustments should be a dictionary
        of key (column to apply adjustment based on) value (adjustment dict for given columns) pairs. For
        example the following dict {'a': {1: 2, 3: 5}, 'b': {'a': 1, 'b': -5}}
        would provide an additive adjustment to the values in the adjustment column based off the values
        in column a using the mapping 1->2+value, 3->5+value and also an additive adjustment based off the
        values in column b using a mapping 'a'->1+value, 'b'->(-5)+value.
        The values within the dicts defining the values to be added must have type int or float.

    **kwargs
        Arbitrary keyword arguments passed onto BaseTransformer.init method.

    Attributes
    ----------

    adjust_column : str
        Column containing the values to be adjusted.

    mappings : dict
        Dictionary of additive adjustments for each column individually to be applied to the adjust_column.
        The dict passed to mappings in init is set to the mappings attribute.


    """

    def __init__(self, adjust_column, mappings, **kwargs):

        super().__init__(mappings=mappings, **kwargs)

        if not isinstance(adjust_column, str):

            raise TypeError("adjust_column should be a string")

        for j in mappings.values():

            for k in j.values():

                if type(k) not in [int, float]:

                    raise TypeError("mapping values must be numeric")

        self.adjust_column = adjust_column

    def transform(self, X):
        """Transforms values in given column using the values provided in the adjustments dictionary.

        Parameters
        ----------
        X : pd.DataFrame
            Data to apply adjustments to.

        Returns
        -------
        X : pd.DataFrame
            Transformed data X with adjustments applied to specified columns.

        """

        self.check_is_fitted(["adjust_column"])

        X = super().transform(X)

        if self.adjust_column not in X.columns.values:

            raise ValueError("variable " + self.adjust_column + " is not in X")

        if not pd.api.types.is_numeric_dtype(X[self.adjust_column]):

            raise TypeError(
                "variable " + self.adjust_column + " must have numeric dtype."
            )

        for i in self.columns:

            for j in self.mappings[i].keys():

                X[self.adjust_column] = np.where(
                    (X[i] == j),
                    X[self.adjust_column] + self.mappings[i][j],
                    X[self.adjust_column],
                )

        return X
