"""This module contains a transformer that applies capping to numeric columns."""

from __future__ import annotations

import copy
import warnings

import numpy as np
import pandas as pd

from tubular.base import BaseTransformer


class CappingTransformer(BaseTransformer):
    """Transformer to cap numeric values at both or either minimum and maximum values.

    For max capping any values above the cap value will be set to the cap. Similarly for min capping
    any values below the cap will be set to the cap. Only works for numeric columns.

    Parameters
    ----------
    capping_values : dict or None, default = None
        Dictionary of capping values to apply to each column. The keys in the dict should be the
        column names and each item in the dict should be a list of length 2. Items in the lists
        should be ints or floats or None. The first item in the list is the minimum capping value
        and the second item in the list is the maximum capping value. If None is supplied for
        either value then that capping will not take place for that particular column. Both items
        in the lists cannot be None. Either one of capping_values or quantiles must be supplied.

    quantiles : dict or None, default = None
        Dictionary of quantiles in the range [0, 1] to set capping values at for each column.
        The keys in the dict should be the column names and each item in the dict should be a
        list of length 2. Items in the lists should be ints or floats or None. The first item in the
        list is the lower quantile and the second item is the upper quantile to set the capping
        value from. The fit method calculates the values quantile from the input data X. If None is
        supplied for either value then that capping will not take place for that particular column.
        Both items in the lists cannot be None. Either one of capping_values or quantiles must be
        supplied.

    weights_column : str or None, default = None
        Optional weights column argument that can be used in combination with quantiles. Not used
        if capping_values is supplied. Allows weighted quantiles to be calculated.

    **kwargs
        Arbitrary keyword arguments passed onto BaseTransformer.init method.

    Attributes
    ----------
    capping_values : dict or None
        Capping values to apply to each column, capping_values argument.

    quantiles : dict or None
        Quantiles to set capping values at from input data. Will be empty after init, values
        populated when fit is run.

    weights_column : str or None
        weights_column argument.

    _replacement_values : dict
        Replacement values when capping is applied. Will be a copy of capping_values.

    """

    def __init__(
        self,
        capping_values: dict[str, list[int | float | None]] | None = None,
        quantiles: dict[str, list[int | float]] | None = None,
        weights_column: str | None = None,
        **kwargs: dict[str, bool],
    ) -> None:
        if capping_values is None and quantiles is None:
            msg = f"{self.classname()}: both capping_values and quantiles are None, either supply capping values in the capping_values argument or supply quantiles that can be learnt in the fit method"
            raise ValueError(msg)

        if capping_values is not None and quantiles is not None:
            msg = f"{self.classname()}: both capping_values and quantiles are not None, supply one or the other"
            raise ValueError(msg)

        if capping_values is not None:
            self.check_capping_values_dict(capping_values, "capping_values")

            self.capping_values = capping_values

            super().__init__(columns=list(capping_values.keys()), **kwargs)

        if quantiles is not None:
            self.check_capping_values_dict(quantiles, "quantiles")

            for k, quantile_values in quantiles.items():
                for quantile_value in quantile_values:
                    if (quantile_value is not None) and (
                        quantile_value < 0 or quantile_value > 1
                    ):
                        msg = f"{self.classname()}: quantile values must be in the range [0, 1] but got {quantile_value} for key {k}"
                        raise ValueError(msg)

            self.capping_values = {}

            super().__init__(columns=list(quantiles.keys()), **kwargs)

        self.quantiles = quantiles
        self.weights_column = weights_column
        self._replacement_values = copy.deepcopy(self.capping_values)

    def check_capping_values_dict(
        self,
        capping_values_dict: dict[str, list[int | float | None]],
        dict_name: str,
    ) -> None:
        """Performs checks on a dictionary passed to ."""
        if type(capping_values_dict) is not dict:
            msg = f"{self.classname()}: {dict_name} should be dict of columns and capping values"
            raise TypeError(msg)

        for k, cap_values in capping_values_dict.items():
            if type(k) is not str:
                msg = f"{self.classname()}: all keys in {dict_name} should be str, but got {type(k)}"
                raise TypeError(msg)

            if type(cap_values) is not list:
                msg = f"{self.classname()}: each item in {dict_name} should be a list, but got {type(cap_values)} for key {k}"
                raise TypeError(msg)

            if len(cap_values) != 2:
                msg = f"{self.classname()}: each item in {dict_name} should be length 2, but got {len(cap_values)} for key {k}"
                raise ValueError(msg)

            for cap_value in cap_values:
                if cap_value is not None:
                    if type(cap_value) not in [int, float]:
                        msg = f"{self.classname()}: each item in {dict_name} lists must contain numeric values or None, got {type(cap_value)} for key {k}"
                        raise TypeError(msg)

                    if np.isnan(cap_value) or np.isinf(cap_value):
                        msg = f"{self.classname()}: item in {dict_name} lists contains numpy NaN or Inf values"
                        raise ValueError(msg)

            if all(cap_value is not None for cap_value in cap_values) and (
                cap_values[0] >= cap_values[1]
            ):
                msg = f"{self.classname()}: lower value is greater than or equal to upper value for key {k}"
                raise ValueError(msg)

            if all(cap_value is None for cap_value in cap_values):
                msg = f"{self.classname()}: both values are None for key {k}"
                raise ValueError(msg)

    def fit(self, X: pd.DataFrame, y: None = None) -> CappingTransformer:
        """Learn capping values from input data X.

        Calculates the quantiles to cap at given the quantiles dictionary supplied
        when initialising the transformer. Saves learnt values in the capping_values
        attribute.

        Parameters
        ----------
        X : pd.DataFrame
            A dataframe with required columns to be capped.

        y : None
            Required for pipeline.

        """
        super().fit(X, y)

        if self.quantiles is not None:
            for col in self.columns:
                if self.weights_column is None:
                    cap_values = self.prepare_quantiles(
                        X[col],
                        self.quantiles[col],
                        self.weights_column,
                    )

                else:
                    cap_values = self.prepare_quantiles(
                        X[col],
                        self.quantiles[col],
                        X[self.weights_column],
                    )

                self.capping_values[col] = cap_values

        else:
            warnings.warn(
                f"{self.classname()}: quantiles not set so no fitting done in CappingTransformer",
                stacklevel=2,
            )

        self._replacement_values = copy.deepcopy(self.capping_values)

        return self

    def prepare_quantiles(
        self,
        values: pd.Series | np.array,
        quantiles: list[float],
        sample_weight: pd.Series | np.array | None = None,
    ) -> list[int | float]:
        """Method to call the weighted_quantile method and prepare the outputs.

        If there are no None values in the supplied quantiles then the outputs from weighted_quantile
        are returned as is. If there are then prepare_quantiles removes the None values before
        calling weighted_quantile and adds them back into the output, in the same position, after
        calling.

        Parameters
        ----------
        values : pd.Series or np.array
            A dataframe column with values to calculate quantiles from.

        quantiles : None
            Weighted quantiles to calculate. Must all be between 0 and 1.

        sample_weight : pd.Series or np.array or None, default = None
            Sample weights for each item in values, must be the same lenght as values. If
            not supplied then unit weights will be used.

        Returns
        -------
        interp_quantiles : list
            List containing computed quantiles.

        """
        if quantiles[0] is None:
            quantiles = np.array([quantiles[1]])

            results_no_none = self.weighted_quantile(values, quantiles, sample_weight)

            results = [None] + results_no_none

        elif quantiles[1] is None:
            quantiles = np.array([quantiles[0]])

            results_no_none = self.weighted_quantile(values, quantiles, sample_weight)

            results = results_no_none + [None]

        else:
            results = self.weighted_quantile(values, quantiles, sample_weight)

        return results

    def weighted_quantile(
        self,
        values: pd.Series | np.array,
        quantiles: list[float],
        sample_weight: pd.Series | np.array | None = None,
    ) -> list[int | float]:
        """Method to calculate weighted quantiles.

        This method is adapted from the "Completely vectorized numpy solution" answer from user
        Alleo (https://stackoverflow.com/users/498892/alleo) to the following stackoverflow question;
        https://stackoverflow.com/questions/21844024/weighted-percentile-using-numpy. This
        method is also licenced under the CC-BY-SA terms, as the original code sample posted
        to stackoverflow (pre February 1, 2016) was.

        Method is similar to numpy.percentile, but supports weights. Supplied quantiles should be
        in the range [0, 1]. Method calculates cumulative % of weight for each observation,
        then interpolates between these observations to calculate the desired quantiles. Null values
        in the observations (values) and 0 weight observations are filtered out before
        calculating.

        Parameters
        ----------
        values : pd.Series or np.array
            A dataframe column with values to calculate quantiles from.

        quantiles : None
            Weighted quantiles to calculate. Must all be between 0 and 1.

        sample_weight : pd.Series or np.array or None, default = None
            Sample weights for each item in values, must be the same lenght as values. If
            not supplied then unit weights will be used.

        Returns
        -------
        interp_quantiles : list
            List containing computed quantiles.

        Examples
        --------
        >>> x = CappingTransformer(capping_values={"a": [2, 10]})
        >>> quantiles_to_compute = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        >>> computed_quantiles = x.weighted_quantile(values = [1, 2, 3], sample_weight = [1, 1, 1], quantiles = quantiles_to_compute)
        >>> [round(q, 1) for q in computed_quantiles]
        [1.0, 1.0, 1.0, 1.0, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 3.0]
        >>>
        >>> computed_quantiles = x.weighted_quantile(values = [1, 2, 3], sample_weight = [0, 1, 0], quantiles = quantiles_to_compute)
        >>> [round(q, 1) for q in computed_quantiles]
        [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
        >>>
        >>> computed_quantiles = x.weighted_quantile(values = [1, 2, 3], sample_weight = [1, 1, 0], quantiles = quantiles_to_compute)
        >>> [round(q, 1) for q in computed_quantiles]
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
        >>>
        >>> computed_quantiles = x.weighted_quantile(values = [1, 2, 3, 4, 5], sample_weight = [1, 1, 1, 1, 1], quantiles = quantiles_to_compute)
        >>> [round(q, 1) for q in computed_quantiles]
        [1.0, 1.0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
        >>>
        >>> computed_quantiles = x.weighted_quantile(values = [1, 2, 3, 4, 5], sample_weight = [1, 0, 1, 0, 1], quantiles = [0, 0.5, 1.0])
        >>> [round(q, 1) for q in computed_quantiles]
        [1.0, 2.0, 5.0]

        """
        if sample_weight is None:
            sample_weight = np.ones(len(values))
        else:
            sample_weight = np.array(sample_weight)

        if np.isnan(sample_weight).sum() > 0:
            msg = f"{self.classname()}: null values in sample weights"
            raise ValueError(msg)

        if np.isinf(sample_weight).sum() > 0:
            msg = f"{self.classname()}: infinite values in sample weights"
            raise ValueError(msg)

        if (sample_weight < 0).sum() > 0:
            msg = f"{self.classname()}: negative weights in sample weights"
            raise ValueError(msg)

        if sample_weight.sum() <= 0:
            msg = f"{self.classname()}: total sample weights are not greater than 0"
            raise ValueError(msg)

        values = np.array(values)
        quantiles = np.array(quantiles)

        nan_filter = ~np.isnan(values)
        values = values[nan_filter]
        sample_weight = sample_weight[nan_filter]

        zero_weight_filter = ~(sample_weight == 0)
        values = values[zero_weight_filter]
        sample_weight = sample_weight[zero_weight_filter]

        sorter = np.argsort(values, kind="stable")
        values = values[sorter]
        sample_weight = sample_weight[sorter]

        weighted_quantiles = np.cumsum(sample_weight)
        weighted_quantiles = weighted_quantiles / np.sum(sample_weight)

        return list(np.interp(quantiles, weighted_quantiles, values))

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply capping to columns in X.

        If cap_value_max is set, any values above cap_value_max will be set to cap_value_max. If cap_value_min
        is set any values below cap_value_min will be set to cap_value_min. Only works or numeric columns.

        Parameters
        ----------
        X : pd.DataFrame
            Data to apply capping to.

        Returns
        -------
        X : pd.DataFrame
            Transformed input X with min and max capping applied to the specified columns.

        """
        self.check_is_fitted(["capping_values"])
        self.check_is_fitted(["_replacement_values"])

        if self.capping_values == {}:
            msg = f"{self.classname()}: capping_values attribute is an empty dict - perhaps the fit method has not been run yet"
            raise ValueError(msg)

        if self._replacement_values == {}:
            msg = f"{self.classname()}: _replacement_values attribute is an empty dict - perhaps the fit method has not been run yet"
            raise ValueError(msg)

        X = super().transform(X)

        numeric_column_types = X[self.columns].apply(
            pd.api.types.is_numeric_dtype,
            axis=0,
        )

        if not numeric_column_types.all():
            non_numeric_columns = list(
                numeric_column_types.loc[~numeric_column_types].index,
            )

            msg = f"{self.classname()}: The following columns are not numeric in X; {non_numeric_columns}"
            raise TypeError(msg)

        for col in self.columns:
            cap_value_min = self.capping_values[col][0]
            cap_value_max = self.capping_values[col][1]

            replacement_min = self._replacement_values[col][0]
            replacement_max = self._replacement_values[col][1]

            if cap_value_min is not None:
                X.loc[X[col] < cap_value_min, col] = replacement_min

            if cap_value_max is not None:
                X.loc[X[col] > cap_value_max, col] = replacement_max

        return X


class OutOfRangeNullTransformer(CappingTransformer):
    """Transformer to set values outside of a range to null.

    This transformer sets the cut off values in the same way as
    the CappingTransformer. So either the user can specify them
    directly in the capping_values argument or they can be calculated
    in the fit method, if the user supplies the quantiles argument.

    Parameters
    ----------
    capping_values : dict or None, default = None
        Dictionary of capping values to apply to each column. The keys in the dict should be the
        column names and each item in the dict should be a list of length 2. Items in the lists
        should be ints or floats or None. The first item in the list is the minimum capping value
        and the second item in the list is the maximum capping value. If None is supplied for
        either value then that capping will not take place for that particular column. Both items
        in the lists cannot be None. Either one of capping_values or quantiles must be supplied.

    quantiles : dict or None, default = None
        Dictionary of quantiles to set capping values at for each column. The keys in the dict
        should be the column names and each item in the dict should be a list of length 2. Items
        in the lists should be ints or floats or None. The first item in the list is the lower
        quantile and the second item is the upper quantile to set the capping value from. The fit
        method calculates the values quantile from the input data X. If None is supplied for
        either value then that capping will not take place for that particular column. Both items
        in the lists cannot be None. Either one of capping_values or quantiles must be supplied.

    weights_column : str or None, default = None
        Optional weights column argument that can be used in combination with quantiles. Not used
        if capping_values is supplied. Allows weighted quantiles to be calculated.

    **kwargs
        Arbitrary keyword arguments passed onto BaseTransformer.init method.

    Attributes
    ----------
    capping_values : dict or None
        Capping values to apply to each column, capping_values argument.

    quantiles : dict or None
        Quantiles to set capping values at from input data. Will be empty after init, values
        populated when fit is run.

    weights_column : str or None
        weights_column argument.

    _replacement_values : dict
        Replacement values when capping is applied. This will contain nulls for each column.

    """

    def __init__(
        self,
        capping_values: dict[str, list[int | float | None]] | None = None,
        quantiles: dict[str, list[int | float]] | None = None,
        weights_column: str | None = None,
        **kwargs: dict[str, bool],
    ) -> None:
        super().__init__(
            capping_values=capping_values,
            quantiles=quantiles,
            weights_column=weights_column,
            **kwargs,
        )

        self.set_replacement_values()

    def set_replacement_values(self) -> None:
        """Method to set the _replacement_values to have all null values.

        Keeps the existing keys in the _replacement_values dict and sets all values (except None) in the lists to np.NaN. Any None
        values remain in place.
        """
        for k, replacements_list in self._replacement_values.items():
            null_replacements_list = [
                np.NaN if replace_value is not None else None
                for replace_value in replacements_list
            ]

            self._replacement_values[k] = null_replacements_list

    def fit(self, X: pd.DataFrame, y: None = None) -> OutOfRangeNullTransformer:
        """Learn capping values from input data X.

        Calculates the quantiles to cap at given the quantiles dictionary supplied
        when initialising the transformer. Saves learnt values in the capping_values
        attribute.

        Parameters
        ----------
        X : pd.DataFrame
            A dataframe with required columns to be capped.

        y : None
            Required for pipeline.

        """
        super().fit(X=X, y=y)

        self.set_replacement_values()

        return self
