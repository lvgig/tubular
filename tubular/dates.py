"""
This module contains a transformer that applies capping to numeric columns.
"""

import datetime
import warnings
import numpy as np
import pandas as pd

from tubular.base import BaseTransformer


class DateDiffLeapYearTransformer(BaseTransformer):
    """Transformer to calculate the number of years between two dates.

    Parameters
    ----------
    column_lower : str
        Name of date column to subtract.

    column_upper : str
        Name of date column to subtract from.

    new_column_name : str
        Name for the new year column.

    drop_cols : bool
        Flag for whether to drop the original columns.

    missing_replacement : int/float/str
        Value to output if either the lower date value or the upper date value are
        missing. Default value is None.

    **kwargs
        Arbitrary keyword arguments passed onto BaseTransformer.init method.

    Attributes
    ----------
    column_lower : str
        Name of date column to subtract. This attribute is not for use in any method,
        use 'columns' instead. Here only as a fix to allow string representation of transformer.

    column_upper : str
        Name of date column to subtract from. This attribute is not for use in any method,
        use 'columns instead. Here only as a fix to allow string representation of transformer.

    columns : list
        List containing column names for transformation in format [column_lower, column_upper]

    new_column_name : str
        Column name for the year column calculated in the transform method.

    drop_cols : bool
        Indicator whether to drop old columns during transform method.

    """

    def __init__(
        self,
        column_lower,
        column_upper,
        new_column_name,
        drop_cols,
        missing_replacement=None,
        **kwargs,
    ):

        if not isinstance(column_lower, str):
            raise TypeError("column_lower should be a str")

        if not isinstance(column_upper, str):
            raise TypeError("column_upper should be a str")

        if not isinstance(new_column_name, str):
            raise TypeError("new_column_name should be a str")

        if not isinstance(drop_cols, bool):
            raise TypeError("drop_cols should be a bool")

        if missing_replacement:
            if not type(missing_replacement) in [int, float, str]:
                raise TypeError(
                    "if not None, missing_replacement should be an int, float or string"
                )

        super().__init__(columns=[column_lower, column_upper], **kwargs)

        self.new_column_name = new_column_name
        self.drop_cols = drop_cols
        self.missing_replacement = missing_replacement

        # This attribute is not for use in any method, use 'columns' instead.
        # Here only as a fix to allow string representation of transformer.
        self.column_lower = column_lower
        self.column_upper = column_upper

    def calculate_age(self, row):
        """Function to calculate age from two date columns in a pd.DataFrame.

        This function, although slower than the np.timedelta64 solution (or something
        similar), accounts for leap years to accurately calculate age for all values.

        Parameters
        ----------
        row : pd.Series
            Named pandas series, with lower_date_name and upper_date_name as index values.

        Returns
        -------
        age : int
            Year gap between the upper and lower date values passes.

        """

        if not isinstance(row, pd.Series):
            raise TypeError("row should be a pd.Series")

        if (pd.isnull(row[self.columns[0]])) or (pd.isnull(row[self.columns[1]])):
            return self.missing_replacement

        else:

            if not type(row[self.columns[1]]) in [datetime.date, datetime.datetime]:
                raise TypeError(
                    "upper column values should be datetime.datetime or datetime.date objects"
                )

            if not type(row[self.columns[0]]) in [datetime.date, datetime.datetime]:
                raise TypeError(
                    "lower column values should be datetime.datetime or datetime.date objects"
                )

            age = row[self.columns[1]].year - row[self.columns[0]].year

            if age > 0:
                if (row[self.columns[1]].month, row[self.columns[1]].day) < (
                    row[self.columns[0]].month,
                    row[self.columns[0]].day,
                ):
                    age += -1
            elif age < 0:
                if (row[self.columns[1]].month, row[self.columns[1]].day) > (
                    row[self.columns[0]].month,
                    row[self.columns[0]].day,
                ):
                    age += 1

            return age

    def transform(self, X):
        """Calculate year gap between the two provided columns.

        New column is created under the 'new_column_name', and optionally removes the
        old date columns.

        Parameters
        ----------
        X : pd.DataFrame
            Data containing column_upper and column_lower.

        Returns
        -------
        X : pd.DataFrame
            Transformed data with new_column_name column.

        """

        X = super().transform(X)

        X[self.new_column_name] = X.apply(lambda x: self.calculate_age(x), axis=1)

        if self.drop_cols:
            X.drop(self.columns, axis=1, inplace=True)

        return X


class DateDifferenceTransformer(BaseTransformer):
    """Class to transform calculate the difference between 2 date fields in specified units.

    Parameters
    ----------
    column_lower : str
        Name of first date column, difference will be calculated as column_upper - column_lower
    column_upper : str
        Name of second date column, difference will be calculated as column_upper - column_lower
    new_column_name : str, default = None
        Name given to calculated datediff column. If None then {column_upper}_{column_lower}_datediff_{units}
        will be used.
    units : str, default = 'D'
        Numpy datetime units, accepted values are 'Y', 'M', 'D', 'h', 'm', 's'
    copy : bool, default = True
        Should X be copied prior to transform?
    verbose: bool, default = False
    """

    def __init__(
        self,
        column_lower,
        column_upper,
        new_column_name=None,
        units="D",
        copy=True,
        verbose=False,
    ):

        if not type(column_lower) is str:

            raise TypeError("column_lower must be a str")

        if not type(column_upper) is str:

            raise TypeError("column_upper must be a str")

        columns = [column_lower, column_upper]

        accepted_values_units = [
            "Y",
            "M",
            "D",
            "h",
            "m",
            "s",
        ]

        if not type(units) is str:

            raise TypeError("units must be a str")

        if units not in accepted_values_units:

            raise ValueError(
                f"units must be one of {accepted_values_units}, got {units}"
            )

        self.units = units

        if new_column_name is not None:

            if not type(new_column_name) is str:

                raise TypeError("new_column_name must be a str")

            self.new_column_name = new_column_name

        else:

            self.new_column_name = f"{column_upper}_{column_lower}_datediff_{units}"

        super().__init__(columns=columns, copy=copy, verbose=verbose)

        # This attribute is not for use in any method, use 'columns' instead.
        # Here only as a fix to allow string representation of transformer.
        self.column_lower = column_lower
        self.column_upper = column_upper

    def transform(self, X):
        """Calculate the difference between the given fields in the specified units.

        Parameters
        ----------
        X : pd.DataFrame
            Data to transform.

        """

        X = super().transform(X)

        X[self.new_column_name] = (
            X[self.columns[1]] - X[self.columns[0]]
        ) / np.timedelta64(1, self.units)

        return X


class ToDatetimeTransformer(BaseTransformer):
    """Class to transform convert specified column to datetime.

    Class simply uses the pd.to_datetime method on the specified column.

    Parameters
    ----------
    column : str
        Name of the column to convert to datetime.

    new_column_name : str
        Name given to the new datetime column.

    to_datetime_kwargs : dict, default = {}
        A dictionary of keyword arguments to be passed to the pd.to_datetime method when it is called in transform.

    **kwargs
        Arbitrary keyword arguments passed onto pd.to_datetime().

    """

    def __init__(self, column, new_column_name, to_datetime_kwargs={}, **kwargs):

        if not type(column) is str:

            raise TypeError(
                "column should be a single str giving the column to transform to datetime"
            )

        if not type(new_column_name) is str:

            raise TypeError("new_column_name must be a str")

        if not type(to_datetime_kwargs) is dict:

            raise TypeError(
                f"to_datetime_kwargs should be a dict but got type {type(to_datetime_kwargs)}"
            )

        else:

            for i, k in enumerate(to_datetime_kwargs.keys()):

                if not type(k) is str:

                    raise TypeError(
                        f"unexpected type ({type(k)}) for to_datetime_kwargs key in position {i}, must be str"
                    )

        self.to_datetime_kwargs = to_datetime_kwargs
        self.new_column_name = new_column_name

        # This attribute is not for use in any method, use 'columns' instead.
        # Here only as a fix to allow string representation of transformer.
        self.column = column

        super().__init__(columns=[column], **kwargs)

    def transform(self, X):
        """Convert specified column to datetime using pd.to_datetime.

        Parameters
        ----------
        X : pd.DataFrame
            Data with column to transform.

        """

        X = super().transform(X)

        X[self.new_column_name] = pd.to_datetime(
            X[self.columns[0]], **self.to_datetime_kwargs
        )

        return X


class SeriesDtMethodTransformer(BaseTransformer):
    """Tranformer that applies a pandas.Series.dt method.

    Transformer assigns the output of the method to a new column. It is possible to
    supply other key word arguments to the transform method, which will be passed to the
    pandas.Series.dt method being called.

    Be aware it is possible to supply incompatible arguments to init that will only be
    identified when transform is run. This is because there are many combinations of method, input
    and output sizes. Additionally some methods may only work as expected when called in
    transform with specific key word arguments.

    Parameters
    ----------
    new_column_name : str
        The name of the column to be assigned to the output of running the pandas method in transform.

    pd_method_name : str
        The name of the pandas.Series.dt method to call.

    column : str
        Column to apply the transformer to. If a str is passed this is put into a list. Value passed
        in columns is saved in the columns attribute on the object. Note this has no default value so
        the user has to specify the columns when initialising the transformer. This is avoid likely
        when the user forget to set columns, in this case all columns would be picked up when super
        transform runs.

    pd_method_kwargs : dict, default = {}
        A dictionary of keyword arguments to be passed to the pd.Series.dt method when it is called.

    **kwargs
        Arbitrary keyword arguments passed onto BaseTransformer.__init__().

    Attributes
    ----------
    column : str
        Name of column to apply transformer to. This attribute is not for use in any method,
        use 'columns instead. Here only as a fix to allow string representation of transformer.

    columns : str
        Column name for transformation.

    new_column_name : str
        The name of the column or columns to be assigned to the output of running the
        pandas method in transform.

    pd_method_name : str
        The name of the pandas.DataFrame method to call.

    pd_method_kwargs : dict
        Dictionary of keyword arguments to call the pd.Series.dt method with.

    """

    def __init__(
        self, new_column_name, pd_method_name, column, pd_method_kwargs={}, **kwargs
    ):

        if type(column) is not str:

            raise TypeError(f"column should be a str but got {type(column)}")

        super().__init__(columns=column, **kwargs)

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

            ser = pd.Series([datetime.datetime(2020, 12, 21)])
            getattr(ser.dt, pd_method_name)

        except Exception as err:

            raise AttributeError(
                f"""error accessing "dt.{pd_method_name}" method on pd.Series object - pd_method_name should be a pd.Series.dt method"""
            ) from err

        if callable(getattr(ser.dt, pd_method_name)):

            self._callable = True

        else:

            self._callable = False

        # This attribute is not for use in any method, use 'columns' instead.
        # Here only as a fix to allow string representation of transformer.
        self.column = column

    def transform(self, X):
        """Transform specific column on input pandas.DataFrame (X) using the given pandas.Series.dt method and
        assign the output back to column in X.

        Any keyword arguments set in the pd_method_kwargs attribute are passed onto the pd.Series.dt method
        when calling it.

        Parameters
        ----------
        X : pd.DataFrame
            Data to transform.

        Returns
        -------
        X : pd.DataFrame
            Input X with additional column (self.new_column_name) added. These contain the output of
            running the pd.Series.dt method.

        """

        X = super().transform(X)

        if self._callable:

            X[self.new_column_name] = getattr(
                X[self.columns[0]].dt, self.pd_method_name
            )(**self.pd_method_kwargs)

        else:

            X[self.new_column_name] = getattr(
                X[self.columns[0]].dt, self.pd_method_name
            )

        return X


class BetweenDatesTransformer(BaseTransformer):
    """Transformer to generate a boolean column indicating if one date is between two others.

    If not all column_lower values are less than or equal to column_upper when transform is run
    then a warning will be raised.

    Parameters
    ----------
    column_lower : str
        Name of column containing the lower date range values.

    column_between : str
        Name of column to check if it's values fall between column_lower and column_upper.

    column_upper : str
        Name of column containing the upper date range values.

    new_column_name : str
        Name for new column to be added to X.

    lower_inclusive : bool, defualt = True
        If lower_inclusive is True the comparison to column_lower will be column_lower <=
        column_between, otherwise the comparison will be column_lower < column_between.

    upper_inclusive : bool, defualt = True
        If upper_inclusive is True the comparison to column_upper will be column_between <=
        column_upper, otherwise the comparison will be column_between < column_upper.

    **kwargs
        Arbitrary keyword arguments passed onto BaseTransformer.__init__().

    Attributes
    ----------
    column_lower : str
        Name of date column to subtract. This attribute is not for use in any method,
        use 'columns' instead. Here only as a fix to allow string representation of transformer.

    column_upper : str
        Name of date column to subtract from. This attribute is not for use in any method,
        use 'columns instead. Here only as a fix to allow string representation of transformer.

    column_between : str
        Name of column to check if it's values fall between column_lower and column_upper. This attribute
        is not for use in any method, use 'columns instead. Here only as a fix to allow string representation of transformer.

    columns : list
        Contains the names of the columns to compare in the order [column_lower, column_between
        column_upper].

    new_column_name : str
        new_column_name argument passed when initialising the transformer.

    lower_inclusive : bool
        lower_inclusive argument passed when initialising the transformer.

    upper_inclusive : bool
        upper_inclusive argument passed when initialising the transformer.

    """

    def __init__(
        self,
        column_lower,
        column_between,
        column_upper,
        new_column_name,
        lower_inclusive=True,
        upper_inclusive=True,
        **kwargs,
    ):

        if type(column_lower) is not str:
            raise TypeError("column_lower should be str")

        if type(column_between) is not str:
            raise TypeError("column_between should be str")

        if type(column_upper) is not str:
            raise TypeError("column_upper should be str")

        if type(new_column_name) is not str:
            raise TypeError("new_column_name should be str")

        if type(lower_inclusive) is not bool:
            raise TypeError("lower_inclusive should be a bool")

        if type(upper_inclusive) is not bool:
            raise TypeError("upper_inclusive should be a bool")

        self.new_column_name = new_column_name
        self.lower_inclusive = lower_inclusive
        self.upper_inclusive = upper_inclusive

        super().__init__(columns=[column_lower, column_between, column_upper], **kwargs)

        # This attribute is not for use in any method, use 'columns' instead.
        # Here only as a fix to allow string representation of transformer.
        self.column_lower = column_lower
        self.column_upper = column_upper
        self.column_between = column_between

    def transform(self, X):
        """Transform - creates column indicating if middle date is between the other two

        If not all column_lower values are less than or equal to column_upper when transform is run
        then a warning will be raised.

        Parameters
        ----------
        X : pd.DataFrame
            Data to transform.

        Returns
        -------
        X : pd.DataFrame
            Input X with additional column (self.new_column_name) added. This column is
            boolean and indicates if the middle column is between the other 2.

        """

        X = super().transform(X)

        for col in self.columns:

            if not pd.api.types.is_datetime64_dtype(X[col]):

                raise TypeError(
                    f"{col} should be datetime64[ns] type but got {X[col].dtype}"
                )

        if not (X[self.columns[0]] <= X[self.columns[2]]).all():

            warnings.warn(
                f"not all {self.columns[2]} are greater than or equal to {self.columns[0]}"
            )

        if self.lower_inclusive:

            lower_comparison = X[self.columns[0]] <= X[self.columns[1]]

        else:

            lower_comparison = X[self.columns[0]] < X[self.columns[1]]

        if self.upper_inclusive:

            upper_comparison = X[self.columns[1]] <= X[self.columns[2]]

        else:

            upper_comparison = X[self.columns[1]] < X[self.columns[2]]

        X[self.new_column_name] = lower_comparison & upper_comparison

        return X
