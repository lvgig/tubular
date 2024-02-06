"""This module contains transformers that apply encodings to nominal columns."""
from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from tubular.base import BaseTransformer
from tubular.mapping import BaseMappingTransformMixin


class BaseNominalTransformer(BaseTransformer):
    """Base Transformer extension for nominal transformers.

    Contains columns_set_or_check method which overrides the columns_set_or_check method in BaseTransformer if given
    primacy in inheritance. The difference being that BaseNominalTransformer's columns_set_or_check only selects
    object and categorical columns from X, if the columns attribute is not set by the user.
    """

    def columns_set_or_check(self, X: pd.DataFrame) -> None:
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
                msg = f"{self.classname()}: no object or category columns in X"
                raise ValueError(msg)

            self.columns = columns

        else:
            self.columns_check(X)

    def check_mappable_rows(self, X: pd.DataFrame) -> None:
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
            mappable_rows = X[c].isin(list(self.mappings[c])).sum()

            if mappable_rows < X.shape[0]:
                msg = f"{self.classname()}: nulls would be introduced into column {c} from levels not present in mapping"
                raise ValueError(msg)


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

    def __init__(
        self,
        columns: str | list[str] | None = None,
        start_encoding: int = 0,
        **kwargs: dict[str, bool],
    ) -> None:
        BaseNominalTransformer.__init__(self, columns=columns, **kwargs)

        if not isinstance(start_encoding, int):
            msg = f"{self.classname()}: start_encoding should be an integer"
            raise ValueError(msg)

        self.start_encoding = start_encoding

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
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

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
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

        super(BaseNominalTransformer, self).columns_check(X)

        self.check_mappable_rows(X)

        return BaseMappingTransformMixin.transform(self, X)

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
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
                    + c,
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

    unseen_levels_to_rare : bool, default = True
        If True, unseen levels in new data will be passed to rare, if set to false they will be left unchanged.

    **kwargs
        Arbitrary keyword arguments passed onto BaseTransformer.init method.

    Attributes
    ----------
    cut_off_percent : float
        Cut off percentage (either in terms of number of rows or sum of weight) for a given
        nominal level to be considered rare.

    non_rare_levels : dict
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

    unseen_levels_to_rare : bool
        If True, unseen levels in new data will be passed to rare, if set to false they will be left unchanged.

    training_data_levels : dict[set]
        Dictionary containing the set of values present in the training data for each column in self.columns. It
        will only exist in if unseen_levels_to_rare is set to False.

    """

    def __init__(
        self,
        columns: str | list[str] | None = None,
        cut_off_percent: float = 0.01,
        weight: str | None = None,
        rare_level_name: str | list[str] | None = "rare",
        record_rare_levels: bool = True,
        unseen_levels_to_rare: bool = True,
        **kwargs: dict[str, bool],
    ) -> None:
        super().__init__(columns=columns, **kwargs)

        if not isinstance(cut_off_percent, float):
            msg = f"{self.classname()}: cut_off_percent must be a float"
            raise ValueError(msg)

        if not ((cut_off_percent > 0) & (cut_off_percent < 1)):
            msg = f"{self.classname()}: cut_off_percent must be > 0 and < 1"
            raise ValueError(msg)

        self.cut_off_percent = cut_off_percent

        if weight is not None and not isinstance(weight, str):
            msg = f"{self.classname()}: weight should be a single column (str)"
            raise ValueError(msg)

        self.weight = weight

        self.rare_level_name = rare_level_name

        if not isinstance(record_rare_levels, bool):
            msg = f"{self.classname()}: record_rare_levels must be a bool"
            raise ValueError(msg)

        self.record_rare_levels = record_rare_levels

        if not isinstance(unseen_levels_to_rare, bool):
            msg = f"{self.classname()}: unseen_levels_to_rare must be a bool"
            raise ValueError(msg)

        self.unseen_levels_to_rare = unseen_levels_to_rare

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        """Records non-rare levels for categorical variables.

        When transform is called, only levels records in non_rare_levels during fit will remain
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
            if (X[c].dtype.name != "category") and (
                pd.Series(self.rare_level_name).dtype != X[c].dtypes
            ):
                msg = f"{self.classname()}: rare_level_name must be of the same type of the columns"
                raise ValueError(msg)

        if self.weight is not None and self.weight not in X.columns.to_numpy():
            msg = f"{self.classname()}: weight {self.weight} not in X"
            raise ValueError(msg)

        self.non_rare_levels = {}

        if self.record_rare_levels:
            self.rare_levels_record_ = {}

        if self.weight is None:
            for c in self.columns:
                col_percents = X[c].value_counts(dropna=False) / X.shape[0]

                self.non_rare_levels[c] = list(
                    col_percents.loc[col_percents >= self.cut_off_percent].index.values,
                )

                self.non_rare_levels[c] = sorted(self.non_rare_levels[c], key=str)

                if self.record_rare_levels:
                    self.rare_levels_record_[c] = list(
                        col_percents.loc[
                            col_percents < self.cut_off_percent
                        ].index.values,
                    )

                    self.rare_levels_record_[c] = sorted(
                        self.rare_levels_record_[c],
                        key=str,
                    )

        else:
            for c in self.columns:
                cols_w_percents = X.groupby(c)[self.weight].sum()

                # nulls are excluded from pandas groupby; https://github.com/pandas-dev/pandas/issues/3729
                # so add them back in
                if cols_w_percents.sum() < X[self.weight].sum():
                    cols_w_percents[np.NaN] = X.loc[X[c].isna(), self.weight].sum()

                cols_w_percents = cols_w_percents / X[self.weight].sum()

                self.non_rare_levels[c] = list(
                    cols_w_percents.loc[
                        cols_w_percents >= self.cut_off_percent
                    ].index.values,
                )

                self.non_rare_levels[c] = sorted(self.non_rare_levels[c], key=str)

                if self.record_rare_levels:
                    self.rare_levels_record_[c] = list(
                        cols_w_percents.loc[
                            cols_w_percents < self.cut_off_percent
                        ].index.values,
                    )

                    self.rare_levels_record_[c] = sorted(
                        self.rare_levels_record_[c],
                        key=str,
                    )

        if not self.unseen_levels_to_rare:
            self.training_data_levels = {}
            for c in self.columns:
                self.training_data_levels[c] = set(X[c])

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
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
        X = BaseNominalTransformer.transform(self, X)

        self.check_is_fitted(["non_rare_levels"])

        if not self.unseen_levels_to_rare:
            for c in self.columns:
                unseen_vals = set(X[c]) - set(self.training_data_levels[c])
                for unseen_val in unseen_vals:
                    self.non_rare_levels[c].append(unseen_val)

        for c in self.columns:
            # for categorical dtypes have to set new category for the impute values first
            # and convert back to the categorical type, other it will convert to object
            if "category" in X[c].dtype.name:
                if self.rare_level_name not in X[c].cat.categories:
                    X[c] = X[c].cat.add_categories(self.rare_level_name)

                dtype_before = X[c].dtype

                X[c] = pd.Series(
                    data=np.where(
                        X[c].isin(self.non_rare_levels[c]),
                        X[c],
                        self.rare_level_name,
                    ),
                    index=X.index,
                ).astype(dtype_before)

            else:
                # using np.where converts np.NaN to str value if only one row of data frame is passed
                # instead, using pd.where(), if condition true, keep original value, else replace with self.rare_level_name
                X[c] = X[c].where(
                    X[c].isin(self.non_rare_levels[c]),
                    self.rare_level_name,
                )

        return X


class MeanResponseTransformer(BaseNominalTransformer):
    """Transformer to apply mean response encoding. This converts categorical variables to
    numeric by mapping levels to the mean response for that level.

    For a continuous or binary response the categorical columns specified will have values
    replaced with the mean response for each category.

    For an n > 1 level categorical response, up to n binary responses can be created, which in
    turn can then be used to encode each categorical column specified. This will generate up
    to n * len(columns) new columns, of with names of the form {column}_{response_level}. The
    original columns will be removed from the dataframe. This functionality is controlled using
    the 'level' parameter. Note that the above only works for a n > 1 level categorical response.
    Do not use 'level' parameter for a n = 1 level numerical response. In this case, use the standard
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

    return_type: Literal['float32', 'float64']
        What type to cast return column as, consider exploring float32 to save memory. Defaults to float32.

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
        Created in fit. A nested Dict of {column names : column specific mapping dictionary} pairs.  Column
        specific mapping dictionaries contain {initial value : mapped value} pairs.

    mapped_columns : list
        Only created in the multi-level case. A list of the new columns produced by encoded the columns in self.columns
        against multiple response levels, of the form {column}_{level}.

    transformer_dict : dict
        Only created in the mutli-level case. A dictionary of the form level : transformer containing the mean response
        transformers for each level to be encoded against.

    unseen_levels_encoding_dict: dict
        Dict containing the values (based on chosen unseen_level_handling) derived from the encoded columns to use when handling unseen levels in data passed to transform method.

    return_type: Literal['float32', 'float64']
        What type to cast return column as. Defaults to float32.

    cast_method: Literal[np.float32, np,float64]
        Store the casting method associated to return_type

    """

    def __init__(
        self,
        columns: str | list[str] | None = None,
        weights_column: str | None = None,
        prior: int = 0,
        level: str | list | None = None,
        unseen_level_handling: str | int | float | None = None,
        return_type: Literal["float32", "float64"] = "float32",
        **kwargs: dict[str, bool],
    ) -> None:
        if weights_column is not None and type(weights_column) is not str:
            msg = f"{self.classname()}: weights_column should be a str"
            raise TypeError(msg)

        if type(prior) is not int:
            msg = f"{self.classname()}: prior should be a int"
            raise TypeError(msg)

        if not prior >= 0:
            msg = f"{self.classname()}: prior should be positive int"
            raise ValueError(msg)

        if level and not isinstance(level, str) and not isinstance(level, list):
            msg = f"{self.classname()}: Level should be a NoneType, list or str but got {type(level)}"
            raise TypeError(msg)
        if (
            unseen_level_handling
            and (unseen_level_handling not in ["Mean", "Median", "Lowest", "Highest"])
            and not (isinstance(unseen_level_handling, (int, float)))
        ):
            msg = f"{self.classname()}: unseen_level_handling should be the option: Mean, Median, Lowest, Highest or an arbitrary int/float value"
            raise ValueError(msg)

        if return_type not in ["float64", "float32"]:
            msg = f"{self.classname()}: return_type should be one of: 'float64', 'float32'"
            raise ValueError(msg)

        self.weights_column = weights_column
        self.prior = prior
        self.level = level
        self.unseen_level_handling = unseen_level_handling
        self.return_type = return_type
        if return_type == "float64":
            self.cast_method = np.float64
        else:
            self.cast_method = np.float32
        # TODO: set default prior to None and refactor to only use prior regularisation when it is set?

        BaseNominalTransformer.__init__(self, columns=columns, **kwargs)

    def _prior_regularisation(
        self,
        target_means: pd.Series,
        cat_freq: str,
    ) -> pd.Series:
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

        return (
            target_means.multiply(cat_freq, axis="index")
            + self.global_mean * self.prior
        ).divide(cat_freq + self.prior, axis="index")

    def _fit_binary_response(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        columns: list[str],
    ) -> None:
        """Function to learn the MRE mappings for a given binary or continuous response.

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
        if (
            self.weights_column is not None
            and self.weights_column not in X.columns.to_numpy()
        ):
            msg = f"{self.classname()}: weights column {self.weights_column} not in X"
            raise ValueError(msg)

        response_null_count = y.isna().sum()

        if response_null_count > 0:
            msg = f"{self.classname()}: y has {response_null_count} null values"
            raise ValueError(msg)

        X_y = self._combine_X_y(X, y)
        response_column = "_temporary_response"

        if self.weights_column is None:
            self.global_mean = X_y[response_column].mean()

        else:
            X_y["weighted_response"] = X_y[response_column].multiply(
                X_y[self.weights_column],
            )

            self.global_mean = (
                X_y["weighted_response"].sum() / X_y[self.weights_column].sum()
            )

        for c in columns:
            if self.weights_column is None:
                group_means = X_y.groupby(c)[response_column].mean()

                group_counts = X_y.groupby(c)[response_column].size()

                self.mappings[c] = self._prior_regularisation(
                    group_means,
                    group_counts,
                ).to_dict()

            else:
                groupby_sum = X_y.groupby([c])[
                    ["weighted_response", self.weights_column]
                ].sum()

                group_weight = groupby_sum[self.weights_column]

                group_means = groupby_sum["weighted_response"] / group_weight

                self.mappings[c] = self._prior_regularisation(
                    group_means,
                    group_weight,
                ).to_dict()

            # to_dict changes types
            for key in self.mappings[c]:
                self.mappings[c][key] = self.cast_method(self.mappings[c][key])

    def fit(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
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

                if any(level not in list(y.unique()) for level in self.level):
                    msg = "Levels contains a level to encode against that is not present in the response."
                    raise ValueError(msg)

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
                y_temp = y.apply(
                    lambda x, level=level: x == level if not pd.isna(x) else np.nan,
                )

                self.transformer_dict[level] = self._fit_binary_response(
                    X_temp,
                    y_temp,
                    mapping_columns_for_this_level,
                )

                mapped_columns += mapping_columns_for_this_level

            self.mapped_columns = list(set(mapped_columns) - set(self.columns))
            self.encoded_feature_columns = self.mapped_columns

        else:
            self._fit_binary_response(X, y, self.columns)
            self.encoded_feature_columns = self.columns

        if isinstance(self.unseen_level_handling, (int, float)):
            for c in self.encoded_feature_columns:
                self.unseen_levels_encoding_dict[c] = self.cast_method(
                    self.unseen_level_handling,
                )

        else:
            for c in self.encoded_feature_columns:
                X_temp[c] = X_temp[c].map(self.mappings[c]).astype(self.return_type)

                if self.unseen_level_handling == "Mean":
                    self.unseen_levels_encoding_dict[c] = X_temp[c].mean()

                if self.unseen_level_handling == "Median":
                    self.unseen_levels_encoding_dict[c] = X_temp[c].median()

                if self.unseen_level_handling == "Lowest":
                    self.unseen_levels_encoding_dict[c] = X_temp[c].min()

                if self.unseen_level_handling == "Highest":
                    self.unseen_levels_encoding_dict[c] = X_temp[c].max()

        return self

    def map_imputation_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """maps columns defined by self.columns in X according the the corresponding mapping dictionary contained in self.mappings

        Parameters
        ----------
        X : pd.DataFrame
            Data to with catgeorical variable columns to transform.

        Returns
        -------
        X : pd.DataFrame
            input dataframe with mappings applied
        """
        for c in self.columns:
            X[c] = X[c].map(self.mappings[c]).astype(self.return_type)

        return X

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
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
        X = super().transform(X)

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
            X = self.map_imputation_values(X)
            for c in self.columns:
                X.loc[unseen_indices[c], c] = self.unseen_levels_encoding_dict[c]
        else:
            self.check_mappable_rows(X)
            X = self.map_imputation_values(X)

        if self.level:
            # Setting self.columns back so that the transformer object is unchanged after transform is called
            self.columns = temp_columns
            X = X.drop(columns=self.columns)

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

    def __init__(
        self,
        columns: str | list[str] | None = None,
        weights_column: str | None = None,
        **kwargs: dict[str, bool],
    ) -> None:
        if weights_column is not None and type(weights_column) is not str:
            msg = f"{self.classname()}: weights_column should be a str"
            raise TypeError(msg)

        self.weights_column = weights_column

        BaseNominalTransformer.__init__(self, columns=columns, **kwargs)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
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

        if (
            self.weights_column is not None
            and self.weights_column not in X.columns.to_numpy()
        ):
            msg = f"{self.classname()}: weights column {self.weights_column} not in X"
            raise ValueError(msg)

        response_null_count = y.isna().sum()

        if response_null_count > 0:
            msg = f"{self.classname()}: y has {response_null_count} null values"
            raise ValueError(msg)

        X_y = self._combine_X_y(X, y)
        response_column = "_temporary_response"

        for c in self.columns:
            if self.weights_column is None:
                # get the indexes of the sorted target mean-encoded dict
                _idx_target_mean = list(
                    X_y.groupby([c])[response_column]
                    .mean()
                    .sort_values(ascending=True, kind="mergesort")
                    .index,
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
                    .index,
                )

                # create a dictionary whose keys are the levels of the categorical variable
                # sorted ascending by their target-mean value
                # and whose values are ascending ordinal integers
                ordinal_encoded_dict = {
                    k: _idx_target_mean.index(k) + 1 for k in _idx_target_mean
                }

                self.mappings[c] = ordinal_encoded_dict

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
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
        super(BaseNominalTransformer, self).columns_check(X)

        self.check_mappable_rows(X)

        return BaseMappingTransformMixin.transform(self, X)


class OneHotEncodingTransformer(BaseNominalTransformer, OneHotEncoder):
    """Transformer to convert cetegorical variables into dummy columns.

    Extends the sklearn OneHotEncoder class to provide easy renaming of dummy columns.

    Parameters
    ----------
    columns : str or list of strings or None, default = None
        Names of columns to transform. If the default of None is supplied all object and category
        columns in X are used.

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
        columns: str | list[str] | None = None,
        separator: str = "_",
        drop_original: bool = False,
        copy: bool = True,
        verbose: bool = False,
        dtype: np.int8 = np.int8,
        **kwargs: dict[str, bool],
    ) -> None:
        BaseNominalTransformer.__init__(
            self,
            columns=columns,
            copy=copy,
            verbose=verbose,
        )

        # Set attributes for scikit-learn'S OneHotEncoder
        OneHotEncoder.__init__(
            self,
            sparse=False,
            handle_unknown="ignore",
            dtype=dtype,
            **kwargs,
        )

        # Set other class attrributes
        self.separator = separator
        self.drop_original = drop_original

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
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
            if X[c].isna().sum() > 0:
                raise ValueError(
                    f"{self.classname()}: column %s has nulls - replace before proceeding"
                    % c,
                )

        # Check each field has less than 100 categories/levels
        for c in self.columns:
            levels = X[c].unique().tolist()

            if len(levels) > 100:
                raise ValueError(
                    f"{self.classname()}: column %s has over 100 unique values - consider another type of encoding"
                    % c,
                )

        OneHotEncoder.fit(self, X=X[self.columns], y=y)

        return self

    def _get_feature_names(
        self,
        input_features: list[str],
        **kwargs: dict[str, bool],
    ) -> list[str]:
        """Function to access the get_feature_names attribute of the scikit learn attribute. Will return the output columns of the OHE transformer.

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
                input_features=input_features,
                **kwargs,
            )

        elif hasattr(self, "get_feature_names_out"):
            input_columns = self.get_feature_names_out(
                input_features=input_features,
                **kwargs,
            )

        else:
            msg = "Cannot access scikit learn OneHotEncoder get_feature_names method, may be a version issue"
            raise AttributeError(msg)

        return input_columns

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
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
            if X[c].isna().sum() > 0:
                raise ValueError(
                    f"{self.classname()}: column %s has nulls - replace before proceeding"
                    % c,
                )

        X = BaseNominalTransformer.transform(self, X)

        # Apply OHE transform
        X_transformed = OneHotEncoder.transform(self, X[self.columns])

        input_columns = self._get_feature_names(input_features=self.columns)

        X_transformed = pd.DataFrame(
            X_transformed,
            columns=input_columns,
            index=X.index,
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

            X_transformed = X_transformed.rename(
                columns=dict(zip(old_names, new_names)),
            )

        # Print warning for unseen levels
        if self.verbose:
            for i, c in enumerate(self.columns):
                unseen_levels = set(X[c].unique().tolist()) - set(self.categories_[i])

                if len(unseen_levels) > 0:
                    warnings.warn(
                        f"{self.classname()}: column %s has unseen categories: %s"
                        % (c, unseen_levels),
                        stacklevel=2,
                    )

        # Drop original columns
        if self.drop_original:
            X = X.drop(self.columns, axis=1)

        # Concatenate original and new dummy fields
        return pd.concat((X, X_transformed), axis=1)
