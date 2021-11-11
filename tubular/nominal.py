"""
This module contains transformers that apply encodings to nominal columns.
"""

import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import OneHotEncoder

from tubular.base import BaseTransformer
from tubular.mapping import BaseMappingTransformMixin


class BaseNominalTransformer(BaseTransformer):
    """Base nominal transformer designed inherrited from for nominal transformers.

    Contains columns_set_or_check method which overrides the columns_set_or_check method in BaseTransformer if given
    primacy in inheritance. The difference being that NominalColumnSetOrCheckMixin's columns_set_or_check only selects
    object and categorical columns from X, if the columns attribute is not set by the user.
    """

    def columns_set_or_check(self, X):
        """Function to check or set columns attribute.

        If the columns attribute is None then set it to all object and category columns in X. Otherwise run the
        columns_check method.

        Parameters
        ----------
        X : pd.DataFrame
            Data to check columns are in.

        """

        if self.columns is None:

            columns = [
                c for c in X.columns if X[c].dtype.name in ["object", "category"]
            ]

            if not len(columns) > 0:

                raise ValueError("no object or category columns in X")

            self.columns = columns

        else:

            self.columns_check(X)

    def check_mappable_rows(self, X):
        """Method to check that all the rows to apply the transformer to are able to be
        mapped according to the values in the mappings dict.

        Raises
        ------
        ValueError
            If any of the rows in a column (c) to be mapped, could not be mapped according to
            the mapping dict in mappings[c].

        """

        self.check_is_fitted(["mappings"])

        for c in self.columns:

            mappable_rows = X[c].isin([k for k in self.mappings[c].keys()]).sum()

            if mappable_rows < X.shape[0]:

                raise ValueError(
                    f"nulls would be introduced into column {c} from levels not present in mapping"
                )


class NominalToIntegerTransformer(BaseNominalTransformer, BaseMappingTransformMixin):
    """Transformer to convert columns containing nominal values into integer values.

    The nominal levels that are mapped to integers are not ordered in any way.

    Parameters
    ----------
    columns : None or str or list, default = None
        Columns to transform, if the default of None is supplied all object and category
        columns in X are used.

    start_encoding : int, default = 0
        Value to start the encoding from e.g. if start_encoding = 0 then the encoding would be
        {'A': 0, 'B': 1, 'C': 3} etc.. or if start_encoding = 5 then the same encoding would be
        {'A': 5, 'B': 6, 'C': 7}. Can be positive or negative.

    **kwargs
        Arbitrary keyword arguments passed onto BaseTransformer.init method.

    Attributes
    ----------
    start_encoding : int
        Value to start the encoding / mapping of nominal to integer from.

    mappings : dict
        Created in fit. A dict of key (column names) value (mappings between levels and integers for given
        column) pairs.

    inverse_mapping_ : dict
        Created in inverse_transform. Inverse mapping of mappings. Maps integer value back to categorical
        levels.

    """

    def __init__(self, columns=None, start_encoding=0, **kwargs):

        BaseNominalTransformer.__init__(self, columns=columns, **kwargs)

        if not isinstance(start_encoding, int):

            raise ValueError("start_encoding should be an integer")

        self.start_encoding = start_encoding

    def fit(self, X, y=None):
        """Creates mapping between nominal levels and integer values for categorical variables.

        Parameters
        ----------
        X : pd.DataFrame
            Data to fit the transformer on, this sets the nominal levels that can be mapped.

        y : None or pd.DataFrame or pd.Series, default = None
            Optional argument only required for the transformer to work with sklearn pipelines.

        """

        BaseNominalTransformer.fit(self, X, y)

        self.mappings = {}

        for c in self.columns:

            col_values = X[c].unique()

            self.mappings[c] = {
                k: i for i, k in enumerate(col_values, self.start_encoding)
            }

        return self

    def transform(self, X):
        """Transform method to apply integer encoding stored in the mappings attribute to
        each column in the columns attribute.

        This method calls the check_mappable_rows method from BaseNominalTransformer to check that
        all rows can be mapped then transform from BaseMappingTransformMixin to apply the
        standard pd.Series.map method.

        Parameters
        ----------
        X : pd.DataFrame
            Data with nominal columns to transform.

        Returns
        -------
        X : pd.DataFrame
            Transformed input X with levels mapped accoriding to mappings dict.

        """

        self.check_mappable_rows(X)

        X = BaseMappingTransformMixin.transform(self, X)

        return X

    def inverse_transform(self, X):
        """Converts integer values back to categorical / nominal values. Does the inverse of the transform method.

        Parameters
        ----------
        X : pd.DataFrame
            Data to with integer columns to convert back to catgeorical.

        Returns
        -------
        X : pd.DataFrame
            Transformed input X with integers mapped back to categorical levels.

        """

        X = BaseNominalTransformer.transform(self, X)

        self.check_is_fitted(["mappings"])

        self.inverse_mapping_ = {}

        for c in self.columns:

            # calculate the reverse mapping
            self.inverse_mapping_[c] = {v: k for k, v in self.mappings[c].items()}

            mappable_rows = (
                X[c].isin([k for k, v in self.inverse_mapping_[c].items()]).sum()
            )

            X[c] = X[c].replace(self.inverse_mapping_[c])

            if (X.shape[0] - mappable_rows) > 0:

                raise ValueError(
                    "nulls introduced from levels not present in mapping for column: "
                    + c
                )

        return X


class GroupRareLevelsTransformer(BaseNominalTransformer):
    """Transformer to group together rare levels of nominal variables into a new level,
    labelled 'rare' (by default).

    Rare levels are defined by a cut off percentage, which can either be based on the
    number of rows or sum of weights. Any levels below this cut off value will be
    grouped into the rare level.

    Parameters
    ----------
    columns : None or str or list, default = None
        Columns to transform, if the default of None is supplied all object and category
        columns in X are used.

    cut_off_percent : float, default = 0.01
        Cut off for the percent of rows or percent of weight for a level, levels below
        this value will be grouped.

    weight : None or str, default = None
        Name of weights column that should be used so cut_off_percent applies to sum of weights
        rather than number of rows.

    rare_level_name : any,default = 'rare'.
        Must be of the same type as columns.
        Label for the new 'rare' level.

    record_rare_levels : bool, default = False
        If True, an attribute called rare_levels_record_ will be added to the object. This will be a dict
        of key (column name) value (level from column considered rare according to cut_off_percent) pairs.
        Care should be taken if working with nominal variables with many levels as this could potentially
        result in many being stored in this attribute.

    **kwargs
        Arbitrary keyword arguments passed onto BaseTransformer.init method.

    Attributes
    ----------
    cut_off_percent : float
        Cut off percentage (either in terms of number of rows or sum of weight) for a given
        nominal level to be considered rare.

    mapping_ : dict
        Created in fit. A dict of non-rare levels (i.e. levels with more than cut_off_percent weight or rows)
        that is used to identify rare levels in transform.

    rare_level_name : any
        Must be of the same type as columns.
        Label for the new nominal level that will be added to group together rare levels (as
        defined by cut_off_percent).

    record_rare_levels : bool
        Should the 'rare' levels that will be grouped together be recorded? If not they will be lost
        after the fit and the only information remaining will be the 'non'rare' levels.

    rare_levels_record_ : dict
        Only created (in fit) if record_rare_levels is True. This is dict containing a list of
        levels that were grouped into 'rare' for each column the transformer was applied to.

    weight : str
        Name of weights columns to use if cut_off_percent should be in terms of sum of weight
        not number of rows.

    """

    def __init__(
        self,
        columns=None,
        cut_off_percent=0.01,
        weight=None,
        rare_level_name="rare",
        record_rare_levels=True,
        **kwargs,
    ):

        super().__init__(columns=columns, **kwargs)

        if not isinstance(cut_off_percent, float):

            raise ValueError("cut_off_percent must be a float")

        if not ((cut_off_percent > 0) & (cut_off_percent < 1)):

            raise ValueError("cut_off_percent must be > 0 and < 1")

        self.cut_off_percent = cut_off_percent

        if weight is not None:

            if not isinstance(weight, str):

                raise ValueError("weight should be a single column (str)")

        self.weight = weight

        self.rare_level_name = rare_level_name

        if not isinstance(record_rare_levels, bool):

            raise ValueError("record_rare_levels must be a bool")

        self.record_rare_levels = record_rare_levels

    def fit(self, X, y=None):
        """Records non-rare levels for categorical variables.

        When transform is called, only levels records in mapping_ during fit will remain
        unchanged - all other levels will be grouped. If record_rare_levels is True then the
        rare levels will also be recorded.

        The label for the rare levels must be of the same type as the columns.

        Parameters
        ----------
        X : pd.DataFrame
            Data to identify non-rare levels from.

        y : None or pd.DataFrame or pd.Series, default = None
            Optional argument only required for the transformer to work with sklearn pipelines.

        """

        super().fit(X, y)

        for c in self.columns:

            if X[c].dtype.name != "category":

                if pd.Series(self.rare_level_name).dtype != X[c].dtypes:

                    raise ValueError(
                        "rare_level_name must be of the same type of the columns"
                    )

        if self.weight is not None:

            if self.weight not in X.columns.values:

                raise ValueError("weight " + self.weight + " not in X")

        self.mapping_ = {}

        if self.record_rare_levels:

            self.rare_levels_record_ = {}

        if self.weight is None:

            for c in self.columns:

                col_percents = X[c].value_counts(dropna=False) / X.shape[0]

                self.mapping_[c] = list(
                    col_percents.loc[col_percents >= self.cut_off_percent].index.values
                )

                self.mapping_[c] = sorted(self.mapping_[c], key=str)

                if self.record_rare_levels:

                    self.rare_levels_record_[c] = list(
                        col_percents.loc[
                            col_percents < self.cut_off_percent
                        ].index.values
                    )

                    self.rare_levels_record_[c] = sorted(
                        self.rare_levels_record_[c], key=str
                    )

        else:

            for c in self.columns:

                cols_w_percents = X.groupby(c)[self.weight].sum()

                # nulls are excluded from pandas groupby; https://github.com/pandas-dev/pandas/issues/3729
                # so add them back in
                if cols_w_percents.sum() < X[self.weight].sum():

                    cols_w_percents[np.NaN] = X.loc[X[c].isnull(), self.weight].sum()

                cols_w_percents = cols_w_percents / X[self.weight].sum()

                self.mapping_[c] = list(
                    cols_w_percents.loc[
                        cols_w_percents >= self.cut_off_percent
                    ].index.values
                )

                self.mapping_[c] = sorted(self.mapping_[c], key=str)

                if self.record_rare_levels:

                    self.rare_levels_record_[c] = list(
                        cols_w_percents.loc[
                            cols_w_percents < self.cut_off_percent
                        ].index.values
                    )

                    self.rare_levels_record_[c] = sorted(
                        self.rare_levels_record_[c], key=str
                    )

        return self

    def transform(self, X):
        """Grouped rare levels together into a new 'rare' level.

        Parameters
        ----------
        X : pd.DataFrame
            Data to with catgeorical variables to apply rare level grouping to.

        Returns
        -------
        X : pd.DataFrame
            Transformed input X with rare levels grouped for into a new rare level.

        """

        X = super().transform(X)

        self.check_is_fitted(["mapping_"])

        for c in self.columns:

            # for categorical dtypes have to set new category for the impute values first
            # and convert back to the categorical type, other it will convert to object
            if "category" in X[c].dtype.name:

                if self.rare_level_name not in X[c].cat.categories:

                    X[c] = X[c].cat.add_categories(self.rare_level_name)

                dtype_before = X[c].dtype

                X[c] = pd.Series(
                    data=np.where(
                        X[c].isin(self.mapping_[c]), X[c], self.rare_level_name
                    ),
                    index=X.index,
                ).astype(dtype_before)

            else:
                # using np.where converts np.NaN to str value if only one row of data frame is passed
                # instead, using pd.where(), if condition true, keep original value, else replace with self.rare_level_name
                X[c] = X[c].where(X[c].isin(self.mapping_[c]), self.rare_level_name)

        return X


class MeanResponseTransformer(BaseNominalTransformer, BaseMappingTransformMixin):
    """Transformer to apply mean response encoding. This converts categorical variables to
    numeric by mapping levels to the mean response for that level.

    If a categorical variable contains null values these will not be transformed.

    Parameters
    ----------
    columns : None or str or list, default = None
        Columns to transform, if the default of None is supplied all object and category
        columns in X are used.

    weights_column : str or None
        Weights column to use when calculating the mean response.

    **kwargs
        Arbitrary keyword arguments passed onto BaseTransformer.init method.

    Attributes
    ----------
    weights_column : str or None
        Weights column to use when calculating the mean response.

    mappings : dict
        Created in fit. Dict of key (column names) value (mapping of categorical levels to numeric,
        mean response values) pairs.

    """

    def __init__(self, columns=None, weights_column=None, **kwargs):

        if weights_column is not None:

            if type(weights_column) is not str:

                raise TypeError("weights_column should be a str")

        self.weights_column = weights_column

        BaseNominalTransformer.__init__(self, columns=columns, **kwargs)

    def fit(self, X, y):
        """Identify mapping of categorical levels to mean response values.

        If the user specified the weights_column arg in when initialising the transformer
        the weighted mean response will be calculated using that column.

        Parameters
        ----------
        X : pd.DataFrame
            Data to with catgeorical variable columns to transform and also containing response_column
            column.

        y : pd.Series
            Response variable or target.

        """

        BaseNominalTransformer.fit(self, X, y)

        self.mappings = {}

        if self.weights_column is not None:

            if self.weights_column not in X.columns.values:

                raise ValueError(f"weights column {self.weights_column} not in X")

        response_null_count = y.isnull().sum()

        if response_null_count > 0:

            raise ValueError(f"y has {response_null_count} null values")

        X_y = self._combine_X_y(X, y)
        response_column = "_temporary_response"

        for c in self.columns:

            if self.weights_column is None:

                self.mappings[c] = X_y.groupby([c])[response_column].mean().to_dict()

            else:

                groupby_sum = X_y.groupby([c])[
                    [response_column, self.weights_column]
                ].sum()

                self.mappings[c] = (
                    groupby_sum[response_column] / groupby_sum[self.weights_column]
                ).to_dict()

        return self

    def transform(self, X):
        """Transform method to apply mean response encoding stored in the mappings attribute to
        each column in the columns attribute.

        This method calls the check_mappable_rows method from BaseNominalTransformer to check that
        all rows can be mapped then transform from BaseMappingTransformMixin to apply the
        standard pd.Series.map method.

        Parameters
        ----------
        X : pd.DataFrame
            Data with nominal columns to transform.

        Returns
        -------
        X : pd.DataFrame
            Transformed input X with levels mapped accoriding to mappings dict.

        """

        self.check_mappable_rows(X)

        X = BaseMappingTransformMixin.transform(self, X)

        return X


class OrdinalEncoderTransformer(BaseNominalTransformer, BaseMappingTransformMixin):
    """Transformer to encode categorical variables into ascending rank-ordered integer values variables by mapping
    it's levels to the target-mean response for that level.
    Values will be sorted in ascending order only i.e. categorical level with lowest target mean response to
    be encoded as 1, the next highest value as 2 and so on.

    If a categorical variable contains null values these will not be transformed.

    Parameters
    ----------
    columns : None or str or list, default = None
        Columns to transform, if the default of None is supplied all object and category
        columns in X are used.

    weights_column : str or None
        Weights column to use when calculating the mean response.

    **kwargs
        Arbitrary keyword arguments passed onto BaseTransformer.init method.

    Attributes
    ----------
    weights_column : str or None
        Weights column to use when calculating the mean response.

    mappings : dict
        Created in fit. Dict of key (column names) value (mapping of categorical levels to numeric,
        ordinal encoded response values) pairs.

    """

    def __init__(self, columns=None, weights_column=None, **kwargs):

        if weights_column is not None:

            if type(weights_column) is not str:

                raise TypeError("weights_column should be a str")

        self.weights_column = weights_column

        BaseNominalTransformer.__init__(self, columns=columns, **kwargs)

    def fit(self, X, y):
        """Identify mapping of categorical levels to rank-ordered integer values by target-mean in ascending order.

        If the user specified the weights_column arg in when initialising the transformer
        the weighted mean response will be calculated using that column.

        Parameters
        ----------
        X : pd.DataFrame
            Data to with catgeorical variable columns to transform and response_column column
            specified when object was initialised.

        y : pd.Series
            Response column or target.

        """

        BaseNominalTransformer.fit(self, X, y)

        self.mappings = {}

        if self.weights_column is not None:

            if self.weights_column not in X.columns.values:

                raise ValueError(f"weights column {self.weights_column} not in X")

        response_null_count = y.isnull().sum()

        if response_null_count > 0:

            raise ValueError(f"y has {response_null_count} null values")

        X_y = self._combine_X_y(X, y)
        response_column = "_temporary_response"

        for c in self.columns:

            if self.weights_column is None:

                # get the indexes of the sorted target mean-encoded dict
                _idx_target_mean = list(
                    X_y.groupby([c])[response_column]
                    .mean()
                    .sort_values(ascending=True, kind="mergesort")
                    .index
                )

                # create a dictionary whose keys are the levels of the categorical variable
                # sorted ascending by their target-mean value
                # and whose values are ascending ordinal integers
                ordinal_encoded_dict = {
                    k: _idx_target_mean.index(k) + 1 for k in _idx_target_mean
                }

                self.mappings[c] = ordinal_encoded_dict

            else:

                groupby_sum = X_y.groupby([c])[
                    [response_column, self.weights_column]
                ].sum()

                # get the indexes of the sorted target mean-encoded dict
                _idx_target_mean = list(
                    (groupby_sum[response_column] / groupby_sum[self.weights_column])
                    .sort_values(ascending=True, kind="mergesort")
                    .index
                )

                # create a dictionary whose keys are the levels of the categorical variable
                # sorted ascending by their target-mean value
                # and whose values are ascending ordinal integers
                ordinal_encoded_dict = {
                    k: _idx_target_mean.index(k) + 1 for k in _idx_target_mean
                }

                self.mappings[c] = ordinal_encoded_dict

        return self

    def transform(self, X):
        """Transform method to apply ordinal encoding stored in the mappings attribute to
        each column in the columns attribute. This maps categorical levels to rank-ordered integer values by target-mean in ascending order.

        This method calls the check_mappable_rows method from BaseNominalTransformer to check that
        all rows can be mapped then transform from BaseMappingTransformMixin to apply the
        standard pd.Series.map method.

        Parameters
        ----------
        X : pd.DataFrame
            Data to with catgeorical variable columns to transform.

        Returns
        -------
        X : pd.DataFrame
            Transformed data with levels mapped to ordinal encoded values for categorical variables.

        """

        self.check_mappable_rows(X)

        X = BaseMappingTransformMixin.transform(self, X)

        return X


class OneHotEncodingTransformer(BaseNominalTransformer, OneHotEncoder):
    """Transformer to convert cetegorical variables into dummy columns.

    Extends the sklearn OneHotEncoder class to provide easy renaming of dummy columns.

    Parameters
    ----------
    columns : str or list of strings, default = None
        Names of columns to transform

    separator : str
        Used to create dummy column names, the name will take
        the format [categorical feature][separator][category level]

    drop_original : bool, default = False
        Should original columns be dropped after creating dummy fields?

    copy : bool, default = True
        Should X be copied prior to transform?

    verbose : bool, default = True
        Should warnings/checkmarks get displayed?

    **kwargs
        Arbitrary keyword arguments passed onto sklearn OneHotEncoder.init method.

    Attributes
    ----------
    separator : str
        Separator used in naming for dummy columns.

    drop_original : bool
        Should original columns be dropped after creating dummy fields?

    """

    def __init__(
        self,
        columns=None,
        separator="_",
        drop_original=False,
        copy=True,
        verbose=False,
        **kwargs,
    ):

        BaseNominalTransformer.__init__(
            self, columns=columns, copy=copy, verbose=verbose
        )

        # Set attributes for scikit-learn'S OneHotEncoder
        OneHotEncoder.__init__(self, sparse=False, handle_unknown="ignore", **kwargs)

        # Set other class attrributes
        self.separator = separator
        self.drop_original = drop_original

    def fit(self, X, y=None):
        """Gets list of levels for each column to be transformed. This defines which dummy columns
        will be created in transform.

        Parameters
        ----------
        X : pd.DataFrame
            Data to identify levels from.

        y : None
            Ignored. This parameter exists only for compatibility with sklearn.pipeline.Pipeline.

        """

        BaseNominalTransformer.fit(self, X=X, y=y)

        # Check for nulls
        for c in self.columns:

            if X[c].isnull().sum() > 0:

                raise ValueError("column %s has nulls - replace before proceeding" % c)

        # Check each field has less than 100 categories/levels
        for c in self.columns:

            levels = X[c].unique().tolist()

            if len(levels) > 100:

                raise ValueError(
                    "column %s has over 100 unique values - consider another type of encoding"
                    % c
                )

        OneHotEncoder.fit(self, X=X[self.columns], y=y)

        return self

    def transform(self, X):
        """Create new dummy columns from categorical fields.

        Parameters
        ----------
        X : pd.DataFrame
            Data to apply one hot encoding to.

        Returns
        -------
        X_transformed : pd.DataFrame
            Transformed input X with dummy columns derived from categorical columns added. If drop_original
            = True then the original categorical columns that the dummies are created from will not be in
            the output X.

        """

        self.check_is_fitted(["separator"])
        self.check_is_fitted(["drop_original"])

        self.columns_check(X)

        # Check for nulls
        for c in self.columns:

            if X[c].isnull().sum() > 0:

                raise ValueError("column %s has nulls - replace before proceeding" % c)

        X = BaseNominalTransformer.transform(self, X)

        # Apply OHE transform
        X_transformed = OneHotEncoder.transform(self, X[self.columns])

        input_columns = self.get_feature_names(input_features=self.columns)

        X_transformed = pd.DataFrame(
            X_transformed, columns=input_columns, index=X.index
        )

        # Rename dummy fields if separator is specified
        if self.separator != "_":

            old_names = [
                c + "_" + str(lvl)
                for i, c in enumerate(self.columns)
                for lvl in self.categories_[i]
            ]
            new_names = [
                c + self.separator + str(lvl)
                for i, c in enumerate(self.columns)
                for lvl in self.categories_[i]
            ]

            X_transformed.rename(
                columns={i: j for i, j in zip(old_names, new_names)}, inplace=True
            )

        # Print warning for unseen levels
        if self.verbose:

            for i, c in enumerate(self.columns):

                unseen_levels = set(X[c].unique().tolist()) - set(self.categories_[i])

                if len(unseen_levels) > 0:

                    warnings.warn(
                        "column %s has unseen categories: %s" % (c, unseen_levels)
                    )

        # Drop original columns
        if self.drop_original:

            X.drop(self.columns, axis=1, inplace=True)

        # Concatenate original and new dummy fields
        X_transformed = pd.concat((X, X_transformed), axis=1)

        return X_transformed
