"""This module contains transformers that deal with imputation of missing values."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from tubular.base import BaseTransformer


class BaseImputer(BaseTransformer):
    """Base imputer class containing standard transform method that will use pd.Series.fillna with the
    values in the impute_values_ attribute.

    Other imputers in this module should inherit from this class.
    """

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values with median values calculated from fit method.

        Parameters
        ----------
        X : pd.DataFrame
            Data to impute.

        Returns
        -------
        X : pd.DataFrame
            Transformed input X with nulls imputed with the median value for the specified columns.

        """
        self.check_is_fitted(["impute_values_"])

        X = super().transform(X)

        for c in self.columns:
            X[c] = X[c].fillna(self.impute_values_[c])

        return X


class ArbitraryImputer(BaseImputer):
    """Transformer to impute null values with an arbitrary pre-defined value.

    Parameters
    ----------
    impute_value : int or float or str
        Value to impute nulls with.
    columns : None or str or list, default = None
        Columns to impute, if the default of None is supplied all columns in X are used
        when the transform method is called.
    **kwargs
        Arbitrary keyword arguments passed onto BaseTransformer.init method.

    Attributes
    ----------
    impute_value : int or float or str
        Value to impute nulls with.
    """

    def __init__(
        self,
        impute_value: int | float | str,
        columns: str | list[str] | None = None,
        **kwargs: dict[str, bool],
    ) -> None:
        if columns is None:
            msg = f"{self.classname()}: columns must be specified in init for ArbitraryImputer"
            raise ValueError(msg)

        super().__init__(columns=columns, **kwargs)

        if (
            not isinstance(impute_value, int)
            and not isinstance(impute_value, float)
            and not isinstance(impute_value, str)
        ):
            msg = f"{self.classname()}: impute_value should be a single value (int, float or str)"
            raise ValueError(msg)

        self.impute_values_ = {}
        self.impute_value = impute_value

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values with the supplied impute_value.
        If columns is None all columns in X will be imputed.

        Parameters
        ----------
        X : pd.DataFrame
            Data containing columns to impute.

        Returns
        -------
        X : pd.DataFrame
            Transformed input X with nulls imputed with the specified impute_value, for the specified columns.

        Additions
        ---------
        * Preserving the datatypes of columns
        * Finding the target column dtype and cast imputer values as same dtype
        """
        self.check_is_fitted(["impute_value"])
        self.columns_check(X)

        for c in self.columns:
            if (
                "category" in X[c].dtype.name
                and self.impute_value not in X[c].cat.categories
            ):
                X[c] = X[c].cat.add_categories(
                    self.impute_value,
                )  # add new category

            dtype = X[c].dtype  # get the dtype of column

            X[c] = (
                X[c].fillna(self.impute_values_).astype(dtype)
            )  # casting imputer value as same dtype

            self.impute_values_[
                c
            ] = self.impute_value  # updating impute_values_ attribute

        return super().transform(X)  # impute the values


class MedianImputer(BaseImputer):
    """Transformer to impute missing values with the median of the supplied columns.

    Parameters
    ----------
    columns : None or str or list, default = None
        Columns to impute, if the default of None is supplied all columns in X are used
        when the transform method is called.

    weight: None or str, default=None
        Column containing weights

    **kwargs
        Arbitrary keyword arguments passed onto BaseTransformer.init method.

    Attributes
    ----------
    impute_values_ : dict
        Created during fit method. Dictionary of float / int (median) values of columns
        in the columns attribute. Keys of impute_values_ give the column names.

    """

    def __init__(
        self,
        columns: str | list[str] | None = None,
        weight: str | None = None,
        **kwargs: dict[str, bool],
    ) -> None:
        super().__init__(columns=columns, **kwargs)

        if not isinstance(weight, str) and weight is not None:
            msg = "weight should be str or None"
            raise TypeError(msg)

        self.weight = weight

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        """Calculate median values to impute with from X.

        Parameters
        ----------
        X : pd.DataFrame
            Data to "learn" the median values from.

        y : None or pd.DataFrame or pd.Series, default = None
            Not required.

        """
        super().fit(X, y)

        self.impute_values_ = {}

        if self.weight is not None:
            super().check_weights_column(X, self.weight)

            temp = X.copy()

            for c in self.columns:
                # filter out null rows so their weight doesn't influence calc
                filtered = temp[temp[c].notna()]

                # first sort df by column to be imputed (order of weight column shouldn't matter for median)
                filtered = filtered.sort_values(c)

                # next calculate cumulative weight sums
                cumsum = filtered[self.weight].cumsum()

                # find midpoint
                cutoff = filtered[self.weight].sum() / 2.0

                # find first value >= this point
                median = filtered[c][cumsum >= cutoff].iloc[0]

                self.impute_values_[c] = median

        else:
            for c in self.columns:
                self.impute_values_[c] = X[c].median()

        return self


class MeanImputer(BaseImputer):
    """Transformer to impute missing values with the mean of the supplied columns.

    Parameters
    ----------
    columns : None or str or list, default = None
        Columns to impute, if the default of None is supplied all columns in X are used
        when the transform method is called.

    weights : None or str, default = None
        Column containing weights.

    **kwargs
        Arbitrary keyword arguments passed onto BaseTransformer.init method.

    Attributes
    ----------
    impute_values_ : dict
        Created during fit method. Dictionary of float / int (mean) values of columns
        in the columns attribute. Keys of impute_values_ give the column names.

    """

    def __init__(
        self,
        columns: str | list[str] | None = None,
        weight: str | None = None,
        **kwargs: dict[str, bool],
    ) -> None:
        super().__init__(columns=columns, **kwargs)

        if not isinstance(weight, str) and weight is not None:
            msg = "weight should be str or None"
            raise TypeError(msg)

        self.weight = weight

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        """Calculate mean values to impute with from X.

        Parameters
        ----------
        X : pd.DataFrame
            Data to "learn" the mean values from.

        y : None or pd.DataFrame or pd.Series, default = None
            Not required.

        """
        super().fit(X, y)

        self.impute_values_ = {}

        if self.weight is not None:
            super().check_weights_column(X, self.weight)

            for c in self.columns:
                # filter out null rows so they don't count towards total weight
                filtered = X[X[c].notna()]

                # calculate total weight and total of weighted col
                total_weight = filtered[self.weight].sum()
                total_weighted_col = filtered[c].mul(filtered[self.weight]).sum()

                # find weighted mean and add to dict
                weighted_mean = total_weighted_col / total_weight

                self.impute_values_[c] = weighted_mean

        else:
            for c in self.columns:
                self.impute_values_[c] = X[c].mean()

        return self


class ModeImputer(BaseImputer):
    """Transformer to impute missing values with the mode of the supplied columns.

    If mode is NaN, a warning will be raised.

    Parameters
    ----------
    columns : None or str or list, default = None
        Columns to impute, if the default of None is supplied all columns in X are used
        when the transform method is called.

    weight : str
        Name of weights columns to use if mode should be in terms of sum of weights
        not count of rows.

    **kwargs
        Arbitrary keyword arguments passed onto BaseTransformer.init method.

    Attributes
    ----------
    impute_values_ : dict
        Created during fit method. Dictionary of float / int (mode) values of columns
        in the columns attribute. Keys of impute_values_ give the column names.

    """

    def __init__(
        self,
        columns: str | list[str] | None = None,
        weight: str | None = None,
        **kwargs: dict[str, bool],
    ) -> None:
        super().__init__(columns=columns, **kwargs)

        if weight is not None and not isinstance(weight, str):
            msg = "ModeImputer: weight should be a string or None"
            raise ValueError(msg)

        self.weight = weight

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        """Calculate mode values to impute with from X.

        Parameters
        ----------
        X : pd.DataFrame
            Data to "learn" the mode values from.

        y : None or pd.DataFrame or pd.Series, default = None
            Not required.

        """
        super().fit(X, y)

        self.impute_values_ = {}

        if self.weight is None:
            for c in self.columns:
                mode_value = X[c].mode(dropna=True)

                if len(mode_value) == 0:
                    self.impute_values_[c] = np.nan

                    warnings.warn(
                        f"ModeImputer: The Mode of column {c} is NaN.",
                        stacklevel=2,
                    )

                else:
                    self.impute_values_[c] = mode_value[0]

        else:
            super().check_weights_column(X, self.weight)

            for c in self.columns:
                self.impute_values_[c] = X.groupby(c)[self.weight].sum().idxmax()

        return self


class NearestMeanResponseImputer(BaseImputer):
    """Class to impute missing values with; the value for which the average response is closest
    to the average response for the unknown levels.

    Parameters
    ----------
    columns : None or str or list, default = None
        Columns to impute, if the default of None is supplied all columns in X are used
        when the transform method is called.

    """

    def __init__(
        self,
        columns: str | list[str] | None = None,
        **kwargs: dict[str, bool],
    ) -> None:
        super().__init__(columns=columns, **kwargs)

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        """Calculate mean values to impute with.

        Parameters
        ----------
        X : pd.DataFrame
            Data to fit the transformer on.

        y : pd.Series
            Response column used to determine the value to impute with. The average response for
            each level of every column is calculated. The level which has the closest average response
            to the average response of the unknown levels is selected as the imputation value.

        """
        super().fit(X, y)

        n_nulls = y.isna().sum()

        if n_nulls > 0:
            msg = f"{self.classname()}: y has {n_nulls} null values"
            raise ValueError(msg)

        self.impute_values_ = {}

        X_y = self._combine_X_y(X, y)
        response_column = "_temporary_response"

        for c in self.columns:
            c_nulls = X[c].isna()

            if c_nulls.sum() == 0:
                msg = f"{self.classname()}: Column {c} has no missing values, cannot use this transformer."
                raise ValueError(msg)

            mean_response_by_levels = pd.DataFrame(
                X_y.loc[~c_nulls].groupby(c)[response_column].mean(),
            ).reset_index()

            mean_response_nulls = X_y.loc[c_nulls, response_column].mean()

            mean_response_by_levels["abs_diff_response"] = np.abs(
                mean_response_by_levels[response_column] - mean_response_nulls,
            )

            # take first value having the minimum difference in terms of average response
            self.impute_values_[c] = mean_response_by_levels.loc[
                mean_response_by_levels["abs_diff_response"]
                == mean_response_by_levels["abs_diff_response"].min(),
                c,
            ].to_numpy()[0]

        return self


class NullIndicator(BaseTransformer):
    """Class to create a binary indicator column for null values.

    Parameters
    ----------
    columns : None or str or list, default = None
        Columns to produce indicator columns for, if the default of None is supplied all columns in X are used
        when the transform method is called.

    """

    def __init__(
        self,
        columns: str | list[str] | None = None,
        **kwargs: dict[str, bool],
    ) -> None:
        super().__init__(columns=columns, **kwargs)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create new columns indicating the position of null values for each variable in self.columns.

        Parameters
        ----------
        X : pd.DataFrame
            Data to add indicators to.

        """
        X = super().transform(X)

        for c in self.columns:
            X[f"{c}_nulls"] = X[c].isna().astype(np.int8)

        return X
