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

                raise ValueError(
                    f"{self.classname()}: no object or category columns in X"
                )

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
                    f"{self.classname()}: nulls would be introduced into column {c} from levels not present in mapping"
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

            raise ValueError(f"{self.classname()}: start_encoding should be an integer")

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
                    f"{self.classname()}: nulls introduced from levels not present in mapping for column: "
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

            raise ValueError(f"{self.classname()}: cut_off_percent must be a float")

        if not ((cut_off_percent > 0) & (cut_off_percent < 1)):

            raise ValueError(f"{self.classname()}: cut_off_percent must be > 0 and < 1")

        self.cut_off_percent = cut_off_percent

        if weight is not None:

            if not isinstance(weight, str):

                raise ValueError(
                    f"{self.classname()}: weight should be a single column (str)"
                )

        self.weight = weight

        self.rare_level_name = rare_level_name

        if not isinstance(record_rare_levels, bool):

            raise ValueError(f"{self.classname()}: record_rare_levels must be a bool")

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
                        f"{self.classname()}: rare_level_name must be of the same type of the columns"
                    )

        if self.weight is not None:

            if self.weight not in X.columns.values:

                raise ValueError(f"{self.classname()}: weight {self.weight} not in X")

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

    For a continuous or binary response the categorical columns specified will have values
    replaced with the mean response for each category.

    For an n > 1 level categorical response, up to n binary responses can be created, which in
    turn can then be used to encode each categorical column specified. This will generate up
    to n * len(columns) new columns, of with names of the form {column}_{response_level}. The
    original columns will be removed from the dataframe. This functionality is controlled using
    the 'level' parameter. Note that the above only works for a n > 1 level categorical response.
    Do not use 'level' parameter for a n > 1 level numerical response. In this case, use the standard
    mean response transformer without the 'level' parameter.

    If a categorical variable contains null values these will not be transformed.

    The same weights and prior are applied to each response level in the multi-level case.

    Parameters
    ----------
    columns : None or str or list, default = None
        Columns to transform, if the default of None is supplied all object and category
        columns in X are used.

    weights_column : str or None
        Weights column to use when calculating the mean response.

    prior : int, default = 0
        Regularisation parameter, can be thought of roughly as the size a category should be in order for
        its statistics to be considered reliable (hence default value of 0 means no regularisation).

    level : str, list or None, default = None
        Parameter to control encoding against a multi-level categorical response. For a continuous or
        binary response, leave this as None. In the multi-level case, set to 'all' to encode against every
        response level or provide a list of response levels to encode against.

    unseen_level_handling : str("Mean", "Median", "Lowest" or "Highest) or int/float, default = None
        Parameter to control the logic for handling unseen levels of the categorical features to encode in
        data when using transform method. Default value of None will output error when attempting to use transform
        on data with unseen levels in categorical columns to encode. Set this parameter to one of the options above
        in order to encode unseen levels in each categorical column with the mean, median etc. of
        each column. One can also pass an arbitrary int/float value to use for encoding unseen levels.

    **kwargs
        Arbitrary keyword arguments passed onto BaseTransformer.init method.

    Attributes
    ----------
    columns : str or list
        Categorical columns to encode in the input data.

    weights_column : str or None
        Weights column to use when calculating the mean response.

    prior : int, default = 0
        Regularisation parameter, can be thought of roughly as the size a category should be in order for
        its statistics to be considered reliable (hence default value of 0 means no regularisation).

    level : str, list or None, default = None
        Parameter to control encoding against a multi-level categorical response. If None the response will be
        treated as binary or continous, if 'all' all response levels will be encoded against and if it is a list of
        levels then only the levels specified will be encoded against.

    response_levels : list
        Only created in the mutli-level case. Generated from level, list of all the response levels to encode against.

    mappings : dict
        Created in fit. Dict of key (column names) value (mapping of categorical levels to numeric,
        mean response values) pairs.

    mapped_columns : list
        Only created in the multi-level case. A list of the new columns produced by encoded the columns in self.columns
        against multiple response levels, of the form {column}_{level}.

    transformer_dict : dict
        Only created in the mutli-level case. A dictionary of the form level : transformer containing the mean response
        transformers for each level to be encoded against.

    unseen_levels_encoding_dict: dict
        Dict containing the values (based on chosen unseen_level_handling) derived from the encoded columns to use when handling unseen levels in data passed to transform method.


    """

    def __init__(
        self,
        columns=None,
        weights_column=None,
        prior=0,
        level=None,
        unseen_level_handling=None,
        **kwargs,
    ):

        if weights_column is not None:

            if type(weights_column) is not str:

                raise TypeError(f"{self.classname()}: weights_column should be a str")

        if type(prior) is not int:

            raise TypeError(f"{self.classname()}: prior should be a int")

        if not prior >= 0:
            raise ValueError(f"{self.classname()}: prior should be positive int")

        if level:

            if not isinstance(level, str) and not isinstance(level, list):

                raise TypeError(
                    f"{self.classname()}: Level should be a NoneType, list or str but got {type(level)}"
                )
        if unseen_level_handling:
            if unseen_level_handling not in ["Mean", "Median", "Lowest", "Highest"]:
                if not isinstance(unseen_level_handling, (int, float)):
                    raise ValueError(
                        f"{self.classname()}: unseen_level_handling should be the option: Mean, Median, Lowest, Highest or an arbitrary int/float value"
                    )

        self.weights_column = weights_column
        self.prior = prior
        self.level = level
        self.unseen_level_handling = unseen_level_handling
        # TODO: set default prior to None and refactor to only use prior regularisation when it is set?

        BaseNominalTransformer.__init__(self, columns=columns, **kwargs)

    def _prior_regularisation(self, target_means, cat_freq):
        """Regularise encoding values by pushing encodings of infrequent categories towards the global mean.  If prior is zero this will return target_means unaltered.

        Parameters
        ----------
        target_means : pd.Series
            Series containing group means for levels of column in data

        cat_freq : str
            Series containing group sizes for levels of column in data

        Returns
        -------
        regularised : pd.Series
            Series of regularised encoding values
        """

        self.check_is_fitted(["global_mean"])

        regularised = (
            target_means.multiply(cat_freq, axis="index")
            + self.global_mean * self.prior
        ).divide(cat_freq + self.prior, axis="index")

        return regularised

    def _fit_binary_response(self, X, y, columns):
        """
        Function to learn the MRE mappings for a given binary or continuous response.

        Parameters
        ----------
        X : pd.DataFrame
            Data to with catgeorical variable columns to transform.

        y : pd.Series
            Binary or contionuous response variable to encode against.

        columns : list(str)
            Post transform names of columns to be encoded. In the binary or continous case
            this is just self.columns. In the multi-level case this should be of the form
            {column_in_original_data}_{response_level}, where response_level is the level
            being encoded against in this call of _fit_binary_response.
        """
        if self.weights_column is not None:

            if self.weights_column not in X.columns.values:

                raise ValueError(
                    f"{self.classname()}: weights column {self.weights_column} not in X"
                )

        response_null_count = y.isnull().sum()

        if response_null_count > 0:

            raise ValueError(
                f"{self.classname()}: y has {response_null_count} null values"
            )

        X_y = self._combine_X_y(X, y)
        response_column = "_temporary_response"

        if self.weights_column is None:

            self.global_mean = X_y[response_column].mean()

        else:

            X_y["weighted_response"] = X_y[response_column].multiply(
                X_y[self.weights_column]
            )

            self.global_mean = (
                X_y["weighted_response"].sum() / X_y[self.weights_column].sum()
            )

        for c in columns:

            if self.weights_column is None:

                group_means = X_y.groupby(c)[response_column].mean()

                group_counts = X_y.groupby(c)[response_column].size()

                self.mappings[c] = self._prior_regularisation(
                    group_means, group_counts
                ).to_dict()

            else:

                groupby_sum = X_y.groupby([c])[
                    ["weighted_response", self.weights_column]
                ].sum()

                group_weight = groupby_sum[self.weights_column]

                group_means = groupby_sum["weighted_response"] / group_weight

                self.mappings[c] = self._prior_regularisation(
                    group_means, group_weight
                ).to_dict()

    def fit(self, X, y):
        """Identify mapping of categorical levels to mean response values.

        If the user specified the weights_column arg in when initialising the transformer
        the weighted mean response will be calculated using that column.

        In the multi-level case this method learns which response levels are present and
        are to be encoded against.

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
        self.unseen_levels_encoding_dict = {}
        X_temp = X.copy()

        if self.level:

            if self.level == "all":

                self.response_levels = y.unique()

            else:

                if isinstance(self.level, str):
                    self.level = [self.level]

                if any([level not in list(y.unique()) for level in self.level]):
                    raise ValueError(
                        "Levels contains a level to encode against that is not present in the response."
                    )

                self.response_levels = self.level

            self.transformer_dict = {}
            mapped_columns = []

            for level in self.response_levels:
                mapping_columns_for_this_level = [
                    column + "_" + level for column in self.columns
                ]

                for column in self.columns:
                    X_temp[column + "_" + level] = X[column].copy()

                # keep nans to preserve null check functionality of binary response MRE transformer
                y_temp = y.apply(lambda x: x == level if not pd.isnull(x) else np.nan)

                self.transformer_dict[level] = self._fit_binary_response(
                    X_temp, y_temp, mapping_columns_for_this_level
                )

                mapped_columns += mapping_columns_for_this_level

            self.mapped_columns = list(set(mapped_columns) - set(self.columns))
            self.encoded_feature_columns = self.mapped_columns

        else:

            self._fit_binary_response(X, y, self.columns)
            self.encoded_feature_columns = self.columns

        if self.unseen_level_handling == "Mean":
            for c in self.encoded_feature_columns:
                self.unseen_levels_encoding_dict[c] = (
                    X_temp[c].map(self.mappings[c]).mean()
                )

        elif self.unseen_level_handling == "Median":
            for c in self.encoded_feature_columns:
                self.unseen_levels_encoding_dict[c] = (
                    X_temp[c].map(self.mappings[c]).median()
                )

        elif self.unseen_level_handling == "Lowest":
            for c in self.encoded_feature_columns:
                self.unseen_levels_encoding_dict[c] = (
                    X_temp[c].map(self.mappings[c]).min()
                )

        elif self.unseen_level_handling == "Highest":
            for c in self.encoded_feature_columns:
                self.unseen_levels_encoding_dict[c] = (
                    X_temp[c].map(self.mappings[c]).max()
                )

        elif isinstance(self.unseen_level_handling, (int, float)):
            for c in self.encoded_feature_columns:
                self.unseen_levels_encoding_dict[c] = float(self.unseen_level_handling)

        return self

    def transform(self, X):
        """Transform method to apply mean response encoding stored in the mappings attribute to
        each column in the columns attribute.

        This method calls the check_mappable_rows method from BaseNominalTransformer to check that
        all rows can be mapped then transform from BaseMappingTransformMixin to apply the
        standard pd.Series.map method.

        N.B. In the mutli-level case, this method briefly overwrites the self.columns attribute, but sets
        it back to the original value at the end.

        Parameters
        ----------
        X : pd.DataFrame
            Data with nominal columns to transform.

        Returns
        -------
        X : pd.DataFrame
            Transformed input X with levels mapped accoriding to mappings dict.

        """
        if self.level:
            for response_level in self.response_levels:
                for column in self.columns:
                    X[column + "_" + response_level] = X[column]
            temp_columns = self.columns
            # Temporarily overwriting self.columns to use BaseMappingTransformMixin
            self.columns = self.mapped_columns

        if self.unseen_level_handling:
            self.check_is_fitted(["mappings"])
            unseen_indices = {}
            for c in self.columns:
                # finding rows with values not in the keys of mappings dictionary
                unseen_indices[c] = X[~X[c].isin(self.mappings[c].keys())].index
            X = BaseMappingTransformMixin.transform(self, X)
            for c in self.columns:
                X.loc[unseen_indices[c], c] = self.unseen_levels_encoding_dict[c]
        else:
            self.check_mappable_rows(X)
            X = BaseMappingTransformMixin.transform(self, X)

        if self.level:
            # Setting self.columns back so that the transformer object is unchanged after transform is called
            self.columns = temp_columns
            X.drop(columns=self.columns, inplace=True)

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

                raise TypeError(f"{self.classname()}: weights_column should be a str")

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

                raise ValueError(
                    f"{self.classname()}: weights column {self.weights_column} not in X"
                )

        response_null_count = y.isnull().sum()

        if response_null_count > 0:

            raise ValueError(
                f"{self.classname()}: y has {response_null_count} null values"
            )

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

                raise ValueError(
                    f"{self.classname()}: column %s has nulls - replace before proceeding"
                    % c
                )

        # Check each field has less than 100 categories/levels
        for c in self.columns:

            levels = X[c].unique().tolist()

            if len(levels) > 100:

                raise ValueError(
                    f"{self.classname()}: column %s has over 100 unique values - consider another type of encoding"
                    % c
                )

        OneHotEncoder.fit(self, X=X[self.columns], y=y)

        return self

    def _get_feature_names(self, input_features, **kwargs):
        """
        Function to access the get_feature_names attribute of the scikit learn attribute. Will return the output columns of the OHE transformer.

        In scikit learn 1.0 "get_feature_names" was deprecated and then replaced with "get_feature_names_out" in version 1.2. The logic in this
        function will call the correct attribute, or raise an error if it can't be found.

        Parameters
        ----------
        input_features : list(str)
            Input columns being transformed by the OHE transformer.

        kwargs : dict
            Keyword arguments to be passed on to the scikit learn attriute.
        """

        if hasattr(self, "get_feature_names"):

            input_columns = self.get_feature_names(
                input_features=input_features, **kwargs
            )

        elif hasattr(self, "get_feature_names_out"):

            input_columns = self.get_feature_names_out(
                input_features=input_features, **kwargs
            )

        else:

            raise AttributeError(
                "Cannot access scikit learn OneHotEncoder get_feature_names method, may be a version issue"
            )

        return input_columns

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

                raise ValueError(
                    f"{self.classname()}: column %s has nulls - replace before proceeding"
                    % c
                )

        X = BaseNominalTransformer.transform(self, X)

        # Apply OHE transform
        X_transformed = OneHotEncoder.transform(self, X[self.columns])

        input_columns = self._get_feature_names(input_features=self.columns)

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
                        f"{self.classname()}: column %s has unseen categories: %s"
                        % (c, unseen_levels)
                    )

        # Drop original columns
        if self.drop_original:

            X.drop(self.columns, axis=1, inplace=True)

        # Concatenate original and new dummy fields
        X_transformed = pd.concat((X, X_transformed), axis=1)

        return X_transformed
